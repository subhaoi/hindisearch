from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException

import typesense
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# IMPORTANT: script-mode imports (python scripts/..). Do NOT use scripts.utils or relative imports.
from scripts.utils import (
    Paths, read_parquet, canonicalize_query_for_search, is_nullish, e5_prefix_text,
    clean_title, query_tokens, roman_query_to_devanagari, iso_to_epoch_seconds,
    location_patterns, derive_locations,
    load_series, article_series, detect_series, classify_query, MAX_QUERY_TOKENS,
    english_glossary, apply_glossary, fold_devanagari, hindi_stem, roman_key,
    DEVANAGARI_RE, HINDI_STOPWORDS, ROMAN_STOPWORD_KEYS,
)
from .ranker_v1 import ranker_v1
from .ranker_v2 import ranker_v2, load_weights, WEIGHTS_PATH
from .db import (
    get_engine, ensure_schema, insert_query, insert_candidates, insert_label, insert_error,
    searches_since, errors_since, label_counts,
)
from .query_entities import detect_entities
from .api_common import (
    SearchRequest, SearchHit, SearchResponse, LabelRequest, QueryLabelRequest,
    _cache_key, _cache_get, _cache_put, build_candidates as merge_candidates, check_vector_size,
)
from . import usage_feed


load_dotenv()

API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

RANKER_VERSION = os.environ.get("RANKER_VERSION", "ranker_v2")
RETRIEVAL_VERSION = os.environ.get("RETRIEVAL_VERSION", "retrieval_v2")
RANKERS = {"ranker_v1", "ranker_v2"}

LEXICAL_TOPK = int(os.environ.get("LEXICAL_TOPK", "80"))
SEM_ARTICLE_TOPK = int(os.environ.get("SEM_ARTICLE_TOPK", "40"))
SEM_CHUNK_TOPK = int(os.environ.get("SEM_CHUNK_TOPK", "80"))
CANDIDATE_CAP = int(os.environ.get("CANDIDATE_CAP", "200"))
# Results per search kept for ranker training; feedback is given on the first pages
LOG_CANDIDATES_TOPN = int(os.environ.get("LOG_CANDIDATES_TOPN", "50"))

TS_COLLECTION = os.environ.get("TYPESENSE_COLLECTION", "idr_articles_hi_v1")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QCOL_ART = os.environ.get("QDRANT_COLLECTION_ARTICLES", "idr_articles_vec_v1")
QCOL_CHK = os.environ.get("QDRANT_COLLECTION_CHUNKS", "idr_chunks_vec_v1")

RAW_ARTICLES_CSV = os.environ.get("RAW_ARTICLES_CSV", "data/raw/articles.csv")

# The model the Qdrant vectors were made with (11_compute_embeddings.py reads the same setting)
MODEL_NAME = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-large")
# torch (default) | onnx. For onnx, point EMBED_ONNX_DIR/EMBED_ONNX_FILE at the output of
# scripts/25_export_onnx_query_encoder.py.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "torch").strip().lower()
EMBED_ONNX_DIR = os.environ.get("EMBED_ONNX_DIR", "models/e5-large-onnx")
EMBED_ONNX_FILE = os.environ.get("EMBED_ONNX_FILE", "onnx/model_qint8_avx2.onnx")


def get_typesense_client() -> typesense.Client:
    host = os.environ.get("TYPESENSE_HOST", "localhost")
    port = os.environ.get("TYPESENSE_PORT", "8108")
    protocol = os.environ.get("TYPESENSE_PROTOCOL", "http")
    api_key = os.environ.get("TYPESENSE_API_KEY")
    if not api_key:
        raise RuntimeError("TYPESENSE_API_KEY not set")
    return typesense.Client(
        {
            "nodes": [{"host": host, "port": port, "protocol": protocol}],
            "api_key": api_key,
            "connection_timeout_seconds": 10,
        }
    )


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def load_query_encoder() -> SentenceTransformer:
    if EMBED_BACKEND == "onnx":
        return SentenceTransformer(
            str(resolve_project_path(EMBED_ONNX_DIR)),
            backend="onnx",
            model_kwargs={"file_name": EMBED_ONNX_FILE},
        )
    return SentenceTransformer(MODEL_NAME)


def tokenize_query(q: str) -> List[str]:
    # Stable token split: Latin words + Devanagari words, ignore 1-char noise. (ranker_v1 only)
    q2 = (q or "").lower()
    toks = re.split(r"[^\wऀ-ॿ]+", q2, flags=re.UNICODE)
    return [t for t in toks if t and len(t) >= 2]


root = Path(".").resolve()
paths = Paths(root=root)

def resolve_project_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (paths.root / path).resolve()


def load_featured_images(csv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not csv_path.exists():
        return mapping

    column_aliases = {"image_featured", "featured_image", "image_url"}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mapping
        target_col = None
        for col in reader.fieldnames:
            if not col:
                continue
            norm = re.sub(r"[\s\-]+", "_", col.strip().lower())
            if norm in column_aliases:
                target_col = col
                break
        if not target_col:
            return mapping

        for row in reader:
            aid = str(row.get("ID") or row.get("id") or "").strip()
            url = row.get(target_col)
            if not aid or not url:
                continue
            mapping[aid] = str(url).strip()
    return mapping


GAZ_PATH = paths.data / "phase_45" / "gazetteer_v1.json"
if not GAZ_PATH.exists():
    raise RuntimeError(f"Missing {GAZ_PATH}. Run: python scripts/20_build_gazetteer.py")

with GAZ_PATH.open("r", encoding="utf-8") as f:
    gazetteer = json.load(f)
if "keys" not in (gazetteer.get("locations_norm") or {}):
    raise RuntimeError(f"{GAZ_PATH} is from the old format. Re-run: python scripts/20_build_gazetteer.py")

# Roman -> Devanagari vocabulary for semantic queries (optional; built by 21_build_translit_vocab.py)
TRANSLIT_VOCAB_PATH = paths.data / "phase_45" / "translit_vocab_v1.json"
translit_vocab: Dict[str, str] = {}
if TRANSLIT_VOCAB_PATH.exists():
    with TRANSLIT_VOCAB_PATH.open("r", encoding="utf-8") as f:
        translit_vocab = json.load(f).get("vocab", {})
else:
    print(f"WARNING: {TRANSLIT_VOCAB_PATH} missing; roman queries will be embedded as typed. "
          "Run: python scripts/21_build_translit_vocab.py")

ARTICLES_PATH = paths.data / "final" / "articles_canonical.parquet"
CHUNKS_PATH = paths.data / "phase_3" / "chunks.parquet"

if not ARTICLES_PATH.exists():
    raise RuntimeError(f"Missing {ARTICLES_PATH}")
if not CHUNKS_PATH.exists():
    raise RuntimeError(f"Missing {CHUNKS_PATH}")

RAW_ARTICLES_PATH = resolve_project_path(RAW_ARTICLES_CSV)
featured_images = load_featured_images(RAW_ARTICLES_PATH)

articles_df = read_parquet(ARTICLES_PATH)


def _as_list(v: Any) -> List[str]:
    """
    Safe conversion for parquet-loaded list-like columns:
    - avoids `v or []` ambiguity for numpy arrays / lists
    """
    if is_nullish(v):
        return []
    if isinstance(v, list):
        return [str(x) for x in v if not is_nullish(x)]
    try:
        return [str(x) for x in list(v) if not is_nullish(x)]
    except Exception:
        return [str(v)]


_gaz_loc = gazetteer.get("locations_norm") or {}
location_aliases = dict(zip(_gaz_loc.get("values") or [], _gaz_loc.get("aliases") or [[] for _ in _gaz_loc.get("values") or []]))
loc_patterns = location_patterns(_gaz_loc.get("values") or [], location_aliases)

series_config = load_series(paths.root / "config" / "series.json")

# English/romanized terms from the synonym groups, used to put the semantic query in Hindi
_syn_path = paths.root / "config" / "synonyms_v1.json"
glossary = english_glossary(json.loads(_syn_path.read_text(encoding="utf-8"))["groups"]) if _syn_path.exists() else {}

articles_meta: Dict[str, Dict[str, Any]] = {}

for _, r in articles_df.iterrows():
    aid = str(r.get("id"))

    cats = _as_list(r.get("categories_raw"))  # display
    tags = _as_list(r.get("tags_raw"))
    locs = _as_list(r.get("locations_raw"))
    contrib = _as_list(r.get("contributors_raw"))

    primary_category = cats[0] if len(cats) > 0 else None

    articles_meta[aid] = {
        "id": aid,
        "url": None if is_nullish(r.get("url")) else str(r.get("url")),
        "title": clean_title(r.get("title_hi")) or None,
        "summary": None if is_nullish(r.get("summary_hi")) else str(r.get("summary_hi")),
        "published_date": None if is_nullish(r.get("published_date")) else str(r.get("published_date")),
        # The canonical parquet has no published_ts column; derive it for recency ranking
        "published_ts": iso_to_epoch_seconds(r.get("published_date")),
        "image_url": featured_images.get(aid),
        # display fields
        "primary_category": primary_category,
        "categories": cats,
        "tags": tags,
        "location": locs,
        "partner_label": None if is_nullish(r.get("partner_label")) else str(r.get("partner_label")),
        "contributors": contrib,
        # norm fields for ranker overlap
        "categories_norm": _as_list(r.get("categories_norm")),
        "tags_norm": _as_list(r.get("tags_norm")),
        "locations_norm": _as_list(r.get("locations_norm")),
        "contributors_norm": _as_list(r.get("contributors_norm")),
        # Tagged + mentioned locations; same derivation as the Typesense locations_all field
        "locations_all": derive_locations(r.get("title_hi"), r.get("summary_hi"), r.get("content_hi"),
                                          _as_list(r.get("locations_norm")), loc_patterns),
        # Same derivation as the Typesense `series` field (config/series.json)
        "series": article_series(r.get("title_hi"), _as_list(r.get("categories_norm")),
                                 None if is_nullish(r.get("multimedia_type")) else str(r.get("multimedia_type")),
                                 series_config),
    }

chunks_df = read_parquet(CHUNKS_PATH)
chunk_text_map = dict(zip(chunks_df["chunk_id"].astype(str), chunks_df["chunk_text"].astype(str)))

app = FastAPI(title="IDR Hybrid Search API (Phase 5)")

ts = get_typesense_client()
qd = get_qdrant_client()
model = load_query_encoder()
check_vector_size(qd, QCOL_CHK, model, MODEL_NAME)
ranker_v2_weights = load_weights()

engine = get_engine()
ensure_schema(engine)


def build_candidates(
    lex_hits: List[Dict[str, Any]],
    sem_art: List[Tuple[str, float]],
    sem_chk: List[Tuple[str, str, float]],
    entity_conf: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    return merge_candidates(lex_hits, sem_art, sem_chk, articles_meta, CANDIDATE_CAP, entity_conf)


def get_article_ids_with(field: str, values: List[str]) -> List[str]:
    wanted = {v.strip().lower() for v in values if v}
    return [
        aid for aid, meta in articles_meta.items()
        if any(x.strip().lower() in wanted for x in (meta.get(field) or []))
    ]


# Words around a topic that say nothing about it ("articles about nrega", "water issues in india")
_QUERY_FILLER = {
    "article", "articles", "story", "stories", "news", "issue", "issues", "problem", "problems",
    "trend", "trends", "india", "indian", "related", "regarding", "idr", "lekh",
}


def glossary_hindi_terms(query: str) -> str:
    """
    The Hindi form of a roman/mixed query when the glossary covers all of it except stopwords,
    filler and numbers ("CSR" -> सीएसआर, "articles about nrega" -> मनरेगा, "NEP 2020" -> शिक्षा नीति).
    Empty when part of the query is unknown ("frontline workers" would become just मजदूर).
    """
    toks = query_tokens(apply_glossary(query, glossary))
    hindi = [t for t in toks if DEVANAGARI_RE.search(t)]
    unknown = [
        t for t in toks
        if not DEVANAGARI_RE.search(t) and not t.isdigit() and t not in _QUERY_FILLER
        and roman_key(t) not in ROMAN_STOPWORD_KEYS
    ]
    if not hindi or unknown:
        return ""
    return " ".join(fold_devanagari(t) for t in hindi if t not in HINDI_STOPWORDS)


# Weight for "the query is an author's name" (स्नेहा, unnayan): strong enough to put that
# author's articles first, which the default contributor boost is not
NAME_QUERY_CONTRIBUTOR_WEIGHT = 1.5


def build_semantic_query(raw_query: str, mode: str, strip_tokens: List[str]) -> str:
    """
    Text to embed: the raw query minus hard-matched entity words (the Qdrant filter already
    handles those), with romanized Hindi converted to Devanagari so it lands in the same
    space as the (Devanagari) article vectors.
    """
    q = raw_query.strip()
    if strip_tokens:
        drop = set(strip_tokens)
        kept = [t for t in query_tokens(q) if t not in drop]
        if kept:  # fall back to the original if stripping removes everything
            q = " ".join(kept)
    if mode in ("roman", "mixed"):
        # Known terms first ("labour law" -> "श्रम कानून"), so word-by-word conversion
        # can't turn English into lookalike Hindi words ("law" -> "लव")
        q = apply_glossary(q, glossary)
        dev = roman_query_to_devanagari(q, translit_vocab)
        if dev:
            q = dev
    return q


# Fusion constant for merging the exact and stemmed keyword lists
LEX_FUSE_K = 20


def typesense_search(canon: Dict[str, Any], filter_by: Optional[str]) -> List[Dict[str, Any]]:
    """
    Keyword retrieval. canon["q"] is canonicalized (folded Devanagari, Latin as match keys).
    English-letter queries also run against the English title/summary translations, and
    Hindi queries also run a stemmed search (बच्चा also finds बच्चों/बच्चे), in the same
    request; the two lists are rank-fused so exact-form matches still come first.
    Metadata (contributors/locations_norm) is Devanagari; *_key fields hold its match keys.
    """
    mode = canon["mode"]
    if mode == "dev":
        query_by = "title_hi,seo_title_hi,summary_hi,content_hi,contributors_norm,locations_norm"
        weights = "6,4,3,1,5,4"
        typos = "1"
    elif mode == "mixed":
        query_by = ("title_hi,seo_title_hi,summary_hi,content_hi,title_roman_norm,seo_title_roman_norm,"
                    "summary_roman_norm,content_mixed_norm,contributors_key,locations_key")
        weights = "6,4,3,1,6,4,3,1,5,4"
        typos = "1"
    else:
        query_by = ("title_roman_norm,seo_title_roman_norm,summary_roman_norm,content_roman_norm,"
                    "contributors_key,locations_key")
        weights = "6,4,3,1,5,4"
        # Romanized spellings drift more than Devanagari ones
        typos = "2,2,2,2,1,1"

    def params(q: str, fields: str, w: str, t: str) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "collection": TS_COLLECTION, "q": q, "query_by": fields, "query_by_weights": w,
            "per_page": LEXICAL_TOPK, "page": 1, "num_typos": t,
        }
        if filter_by:
            p["filter_by"] = filter_by
        return p

    if canon["q"] == "*":
        # Browsing a series or a year: everything that passes the filter, newest first
        browse = params("*", query_by, weights, "0")
        browse["sort_by"] = "published_ts:desc"
        searches = [browse]
    else:
        searches = [params(canon["q"], query_by, weights, typos)]
    if canon["q"] != "*" and mode in ("roman", "mixed") and canon.get("q_en"):
        # English queries against the English translations of titles/summaries
        # (27_translate_articles.py; Typesense stems these fields)
        searches.append(params(canon["q_en"], "title_en,summary_en", "3,1", "1"))
    if canon["q"] != "*" and canon.get("q_gloss"):
        # Known English terms in their Hindi form ("csr" -> सीएसआर, "nrega" -> मनरेगा): headlines
        # use the Hindi spelling, which the roman fields can't match
        searches.append(params(canon["q_gloss"], "title_hi,seo_title_hi,summary_hi,content_hi", "6,4,3,1", "0"))
        searches.append(params(" ".join(hindi_stem(t) for t in canon["q_gloss"].split()),
                               "title_stem,summary_stem,content_stem", "6,3,1", "0"))
    if canon["q"] != "*" and mode == "dev" and canon.get("q_stem") and canon["q_stem"] != canon["q"]:
        # Stems are short; typo tolerance on them mostly adds noise
        searches.append(params(canon["q_stem"], "title_stem,summary_stem,content_stem", "6,3,1", "0"))

    res = ts.multi_search.perform({"searches": searches}, {})
    fused: Dict[str, float] = {}
    best_match: Dict[str, float] = {}
    for r in res.get("results", []):
        if "error" in r:
            raise RuntimeError(f"Typesense search failed: {r.get('error')}")
        for rank, h in enumerate(r.get("hits", []) or [], start=1):
            aid = str((h.get("document") or {}).get("id"))
            fused[aid] = fused.get(aid, 0.0) + 1.0 / (LEX_FUSE_K + rank)
            best_match[aid] = max(best_match.get(aid, 0.0), float(h.get("text_match", 0.0)))
    ordered = sorted(fused, key=lambda a: fused[a], reverse=True)[:LEXICAL_TOPK]
    return [{"article_id": a, "lexical_score": best_match[a]} for a in ordered]


def encode_query(query_semantic: str) -> List[float]:
    return model.encode([e5_prefix_text(query_semantic, "query")], normalize_embeddings=True)[0].tolist()


def _article_filter(article_ids: Optional[List[str]]) -> Optional[qm.Filter]:
    if not article_ids:
        return None
    return qm.Filter(must=[qm.FieldCondition(key="article_id", match=qm.MatchAny(any=article_ids))])


def qdrant_search_articles(q_vec: List[float], article_ids: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    res = qd.search(collection_name=QCOL_ART, query_vector=q_vec, limit=SEM_ARTICLE_TOPK, with_payload=False, query_filter=_article_filter(article_ids))
    return [(str(p.id), float(p.score)) for p in res]


def qdrant_search_chunks(q_vec: List[float], article_ids: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
    res = qd.search(collection_name=QCOL_CHK, query_vector=q_vec, limit=SEM_CHUNK_TOPK, with_payload=True, query_filter=_article_filter(article_ids))
    out: List[Tuple[str, str, float]] = []
    for p in res:
        payload = p.payload or {}
        cid = payload.get("chunk_id")
        aid = payload.get("article_id")
        if cid is None or aid is None:
            continue
        out.append((str(cid), str(aid), float(p.score)))
    return out


def choose_snippet(item: Dict[str, Any]) -> Optional[str]:
    cid = item.get("best_chunk_id")
    if not cid:
        return None
    txt = chunk_text_map.get(str(cid))
    if not txt:
        return None
    snip = " ".join(txt.replace("\n", " ").split())
    return snip[:420]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ranker_version": RANKER_VERSION, "retrieval_version": RETRIEVAL_VERSION}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, background_tasks: BackgroundTasks) -> SearchResponse:
    started = time.perf_counter()
    try:
        return run_search(req, started, background_tasks)
    except HTTPException:
        raise
    except Exception as e:
        # Shown in the admin panel's Problems table
        insert_error(engine, f"{type(e).__name__}: {e}", round(time.perf_counter() - started, 2))
        raise


def run_search(req: SearchRequest, started: float, background_tasks: BackgroundTasks) -> SearchResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    ranker_version = req.ranker or RANKER_VERSION
    if ranker_version not in RANKERS:
        raise HTTPException(status_code=400, detail=f"ranker must be one of {sorted(RANKERS)}")

    kind, _ = classify_query(req.query)
    if kind == "junk":
        # Exploit probes and strings with no letters: nothing to search, and not worth logging
        return SearchResponse(query_id=0, mode="junk", query_used="", query_semantic="", total_results=0,
                              total_pages=1, page=1, per_page=max(1, int(req.per_page)), results=[])

    key = _cache_key(req.query, req.filter_by, ranker_version)
    res = _cache_get(key)
    cached = res is not None
    if res is None:
        res = retrieve(req.query, req.filter_by, ranker_version)
        _cache_put(key, res)
    ranked = res["ranked"]

    per_page = max(1, int(req.per_page))
    page = max(1, int(req.page))
    total_results = len(ranked)
    total_pages = max(1, (total_results + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages

    start = (page - 1) * per_page
    end = min(start + per_page, total_results)

    hits: List[SearchHit] = []
    for item in ranked[start:end]:
        hits.append(
            SearchHit(
                rank=item["rank"],
                id=item["article_id"],
                title=item.get("title"),
                date=item.get("published_date"),
                summary=item.get("summary"),
                url=item.get("url"),
                image_url=item.get("image_url"),
                primary_category=item.get("primary_category"),
                categories=item.get("categories") or [],
                tags=item.get("tags") or [],
                location=item.get("location") or [],
                partner_label=item.get("partner_label"),
                contributors=item.get("contributors") or [],
                score=float(item["score"]),
                snippet=choose_snippet(item),
                features=item["features"] if req.explain else None,
                explanation=item["explanation"] if req.explain else None,
            )
        )

    qid = 0
    if req.log:
        qid = insert_query(
            engine=engine,
            query_raw=req.query,
            query_mode=res["mode"],
            query_used=res["query_used"],
            query_semantic=res["query_semantic"],
            filters={"filter_by": req.filter_by} if req.filter_by else None,
            ranker_version=ranker_version,
            retrieval_version=RETRIEVAL_VERSION,
            meta={**res["meta"], "cached": cached, "took_s": round(time.perf_counter() - started, 2)},
        )
        # The ranked list (for ranker training) is written after the reply has gone out
        background_tasks.add_task(log_candidates, qid, ranked)

    return SearchResponse(
        query_id=qid,
        mode=res["mode"],
        query_used=res["query_used"],
        query_semantic=res["query_semantic"],
        total_results=total_results,
        total_pages=total_pages,
        page=page,
        per_page=per_page,
        results=hits,
    )


def log_candidates(qid: int, ranked: List[Dict[str, Any]]) -> None:
    """Background task: the top LOG_CANDIDATES_TOPN results with their ranking features."""
    topn = min(LOG_CANDIDATES_TOPN, len(ranked))
    to_log: List[Dict[str, Any]] = []
    for item in ranked[:topn]:
        to_log.append(
            {
                "rank": item["rank"],
                "article_id": item["article_id"],
                "url": item.get("url"),
                "title": item.get("title"),
                "published_date": item.get("published_date"),
                "summary": item.get("summary"),
                "primary_category": item.get("primary_category"),
                "categories": item.get("categories") or [],
                "tags": item.get("tags") or [],
                "location": item.get("location") or [],
                "partner_label": item.get("partner_label"),
                "contributors": item.get("contributors") or [],
                "score": float(item["score"]),
                "features": item["features"],
                "explanation": item.get("explanation"),
            }
        )
    try:
        insert_candidates(engine, qid, to_log)
    except Exception as e:  # noqa: BLE001 - the reply has already gone; just report it
        print(f"[candidate_log] query {qid}: {type(e).__name__}: {e}")


def retrieve(query: str, filter_by: Optional[str], ranker_version: str) -> Dict[str, Any]:
    """
    The expensive part of a search: understand the query, retrieve from Typesense and Qdrant,
    rank. Returns what the response and the log need; cached per query (see _search_cache).
    """
    kind, year = classify_query(query)
    # A pasted paragraph: keep the first MAX_QUERY_TOKENS words
    query_text = " ".join(query.split()[:MAX_QUERY_TOKENS])
    extra_filters: List[str] = []
    series_ids: Optional[set] = None
    series: List[str] = []
    if kind == "year":
        # A bare year ("2026") lists that year's articles, newest first
        start = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
        end = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp())
        extra_filters.append(f"published_ts:>={start} && published_ts:<{end}")
        series_ids = {a for a, m in articles_meta.items() if start <= int(m.get("published_ts") or 0) < end}
        if not series_ids:
            # No articles from that year yet ("2027"): show the newest articles instead of an empty page
            extra_filters, series_ids = [], None
        query_text = ""
    else:
        # "फोटो निबंध जलवायु" -> filter to photo essays, search "जलवायु" within them
        series, query_text = detect_series(query_text, series_config)
        if series:
            extra_filters.append(f"series:=[{','.join(series)}]")
            series_ids = {a for a, m in articles_meta.items() if set(m.get("series") or []) & set(series)}

    browse = not query_text.strip()
    if browse:
        mode = canonicalize_query_for_search(query)["mode"]
        canon = {"raw": query, "mode": mode, "q": "*", "q_stem": "*", "q_full": "", "roman_norm": ""}
    else:
        canon = canonicalize_query_for_search(query_text)
        mode = canon["mode"]
        if mode in ("roman", "mixed"):
            canon["q_gloss"] = glossary_hindi_terms(query_text)
    query_used = canon["q"]  # lexical/canonicalized

    entity = detect_entities(query_full=canon["q_full"], gazetteer=gazetteer)
    hard = entity.get("hard", {})

    filter_parts = [f for f in [filter_by, entity.get("filter_by_auto"), *extra_filters] if f]
    filter_final = " && ".join(f"({f})" for f in filter_parts) if len(filter_parts) > 1 else (filter_parts[0] if filter_parts else None)

    hard_contributors = hard.get("contributors_norm") or []
    hard_locations = hard.get("locations_norm") or []
    has_author_entity = bool(hard_contributors)
    has_location_entity = bool(hard_locations)

    # Restrict Qdrant to the same articles the Typesense filter allows.
    qdrant_article_ids: Optional[List[str]] = None
    if has_author_entity or has_location_entity:
        ids: Optional[set] = None
        if has_author_entity:
            ids = set(get_article_ids_with("contributors_norm", hard_contributors))
        if has_location_entity:
            loc_ids = set(get_article_ids_with("locations_all", hard_locations))
            ids = loc_ids if ids is None else (ids & loc_ids) or ids
        qdrant_article_ids = sorted(ids) if ids else None
    if series_ids is not None:
        ids2 = series_ids if qdrant_article_ids is None else (series_ids & set(qdrant_article_ids))
        qdrant_article_ids = sorted(ids2) if ids2 else ["__none__"]

    timings: Dict[str, float] = {}
    t = time.perf_counter()
    lex = typesense_search(canon, filter_by=filter_final)
    timings["t_typesense"] = time.perf_counter() - t
    if browse:
        # Nothing to match semantically; the filtered keyword list is already newest-first
        query_semantic, sem_a, sem_c = "", [], []
    else:
        query_semantic = build_semantic_query(query_text, mode, entity.get("strip_tokens") or [])
        t = time.perf_counter()
        q_vec = encode_query(query_semantic)
        timings["t_embed"] = time.perf_counter() - t
        t = time.perf_counter()
        sem_a = qdrant_search_articles(q_vec, article_ids=qdrant_article_ids)
        sem_c = qdrant_search_chunks(q_vec, article_ids=qdrant_article_ids)
        timings["t_qdrant"] = time.perf_counter() - t

    t = time.perf_counter()
    candidates = build_candidates(lex, sem_a, sem_c, entity_conf=entity.get("confidence"))

    now_ts = int(time.time())
    if ranker_version == "ranker_v1":
        ranked = ranker_v1(candidates, tokenize_query(query_used), now_ts=now_ts,
                           has_author_entity=has_author_entity, has_location_entity=has_location_entity)
    else:
        weights = ranker_v2_weights
        if entity.get("name_query"):
            weights = {**weights, "match_contributor": max(weights.get("match_contributor", 0.0), NAME_QUERY_CONTRIBUTOR_WEIGHT)}
        ranked = ranker_v2(candidates, entity.get("matches", {}), now_ts=now_ts, weights=weights)

    timings["t_rank"] = time.perf_counter() - t

    return {
        "mode": mode,
        "query_used": query_used,
        "query_semantic": query_semantic,
        "ranked": ranked,
        "meta": {
            "lex_n": len(lex),
            "sem_article_n": len(sem_a),
            "sem_chunk_n": len(sem_c),
            "cand_n": len(candidates),
            "entity_matches": entity.get("matches", {}),
            "entity_hard": hard,
            "entity_confidence": entity.get("confidence", {}),
            "filter_by_auto": entity.get("filter_by_auto"),
            "filter_by_final": filter_final,
            "has_author_entity": has_author_entity,
            "has_location_entity": has_location_entity,
            "qdrant_filter_ids_n": len(qdrant_article_ids) if qdrant_article_ids else 0,
            "query_semantic_used": query_semantic,
            "query_kind": kind,
            "series": series,
            "name_query": bool(entity.get("name_query")),
            "query_gloss": canon.get("q_gloss"),
            **{k: round(v, 3) for k, v in timings.items()},
        },
    }


@app.get("/api/usage")
def api_usage(since: float = 0, days: int = 30, x_idr_usage_token: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Usage, searched queries and health for the IDR tools admin panel (see usage_feed.py)."""
    if not usage_feed.token_ok(x_idr_usage_token):
        raise HTTPException(status_code=403, detail="forbidden")
    searches = searches_since(engine, since)

    def safe(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - one missing number must not hide the rest
            return None

    return {
        "events": usage_feed.to_events(searches, errors_since(engine, since)),
        "queries": usage_feed.top_queries(searches, time.time() - max(1, days) * 86400),
        "health": {
            "version": usage_feed.code_version(root),
            "articles": len(articles_meta),
            "typesense_docs": safe(lambda: ts.collections[TS_COLLECTION].retrieve()["num_documents"]),
            "qdrant_articles": safe(lambda: qd.count(QCOL_ART, exact=True).count),
            "qdrant_chunks": safe(lambda: qd.count(QCOL_CHK, exact=True).count),
            "refresh": usage_feed.last_refresh(root),
            "eval": usage_feed.latest_eval(root),
            "labels": {**(safe(lambda: label_counts(engine, RANKER_VERSION)) or {}), "needed": usage_feed.MIN_LABELS},
            "ranker": RANKER_VERSION,
            "weights_learned": WEIGHTS_PATH.exists(),
            "memory": usage_feed.memory(),
            **usage_feed.disk(root),
        },
    }


@app.post("/label")
def label(req: LabelRequest) -> Dict[str, Any]:
    if req.label not in (0, 1):
        raise HTTPException(status_code=400, detail="label must be 0 or 1")
    insert_label(engine, req.query_id, req.article_id, req.label, req.note)
    return {"ok": True}


@app.post("/label_query")
def label_query(req: QueryLabelRequest) -> Dict[str, Any]:
    if req.label != 0:
        raise HTTPException(status_code=400, detail="Only label=0 supported for query-level feedback")
    insert_label(engine, req.query_id, None, 0, req.note)
    return {"ok": True}
