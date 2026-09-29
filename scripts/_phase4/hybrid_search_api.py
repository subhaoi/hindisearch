from __future__ import annotations

import csv
import os
import time
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import typesense
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# IMPORTANT: script-mode imports (python scripts/..). Do NOT use scripts.utils or relative imports.
from scripts.utils import (
    Paths, read_parquet, canonicalize_query_for_search, is_nullish, e5_prefix_text,
    clean_title, query_tokens, roman_query_to_devanagari,
)
from .ranker_v1 import ranker_v1
from .ranker_v2 import ranker_v2, load_weights
from .db import get_engine, ensure_schema, insert_query, insert_candidates, insert_label
from .query_entities import detect_entities


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
LOG_CANDIDATES_TOPN = int(os.environ.get("LOG_CANDIDATES_TOPN", "200"))

TS_COLLECTION = os.environ.get("TYPESENSE_COLLECTION", "idr_articles_hi_v1")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QCOL_ART = os.environ.get("QDRANT_COLLECTION_ARTICLES", "idr_articles_vec_v1")
QCOL_CHK = os.environ.get("QDRANT_COLLECTION_CHUNKS", "idr_chunks_vec_v1")

RAW_ARTICLES_CSV = os.environ.get("RAW_ARTICLES_CSV", "data/raw/articles.csv")

MODEL_NAME = "intfloat/multilingual-e5-large"
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
        "published_ts": int(r.get("published_ts"))
        if "published_ts" in articles_df.columns and not is_nullish(r.get("published_ts"))
        else 0,
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
    }

chunks_df = read_parquet(CHUNKS_PATH)
chunk_text_map = dict(zip(chunks_df["chunk_id"].astype(str), chunks_df["chunk_text"].astype(str)))

app = FastAPI(title="IDR Hybrid Search API (Phase 5)")

ts = get_typesense_client()
qd = get_qdrant_client()
model = load_query_encoder()
ranker_v2_weights = load_weights()

engine = get_engine()
ensure_schema(engine)


class SearchRequest(BaseModel):
    query: str
    filter_by: Optional[str] = None
    per_page: int = 10
    page: int = 1
    explain: bool = False
    # Evaluation hooks: skip Postgres logging / pick a ranker for this request only
    log: bool = True
    ranker: Optional[str] = None


class SearchHit(BaseModel):
    rank: int
    id: str
    title: Optional[str] = None
    date: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[str] = None
    image_url: Optional[str] = None

    primary_category: Optional[str] = None
    categories: List[str] = []
    tags: List[str] = []
    location: List[str] = []
    partner_label: Optional[str] = None
    contributors: List[str] = []

    score: float
    snippet: Optional[str] = None

    features: Optional[Dict[str, Any]] = None
    explanation: Optional[List[Any]] = None


class SearchResponse(BaseModel):
    query_id: int
    mode: str
    query_used: str
    query_semantic: str
    total_results: int
    total_pages: int
    page: int
    per_page: int
    results: List[SearchHit]


class LabelRequest(BaseModel):
    query_id: int
    article_id: Optional[str] = None
    label: int
    note: Optional[str] = None


class QueryLabelRequest(BaseModel):
    query_id: int
    label: int  # only 0 supported here
    note: Optional[str] = None


def get_article_ids_with(field: str, values: List[str]) -> List[str]:
    wanted = {v.strip().lower() for v in values if v}
    return [
        aid for aid, meta in articles_meta.items()
        if any(x.strip().lower() in wanted for x in (meta.get(field) or []))
    ]


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
        dev = roman_query_to_devanagari(q, translit_vocab)
        if dev:
            q = dev
    return q


def typesense_search(query_used: str, mode: str, filter_by: Optional[str]) -> List[Dict[str, Any]]:
    # query_used is canonicalized: Devanagari tokens as-is, Latin tokens as match keys.
    # Metadata (contributors/locations_norm) is Devanagari; *_key fields hold its match keys.
    if mode == "dev":
        query_by = "title_hi,summary_hi,content_hi,contributors_norm,locations_norm"
        weights = "6,3,1,5,4"
        typos = "1"
    elif mode == "mixed":
        query_by = ("title_hi,summary_hi,content_hi,title_roman_norm,summary_roman_norm,"
                    "content_mixed_norm,contributors_key,locations_key")
        weights = "6,3,1,6,3,1,5,4"
        typos = "1"
    else:
        query_by = "title_roman_norm,summary_roman_norm,content_roman_norm,contributors_key,locations_key"
        weights = "6,3,1,5,4"
        # Romanized spellings drift more than Devanagari ones
        typos = "2,2,2,1,1"

    params: Dict[str, Any] = {
        "q": query_used,
        "query_by": query_by,
        "query_by_weights": weights,
        "per_page": LEXICAL_TOPK,
        "page": 1,
        "num_typos": typos,
    }
    if filter_by:
        params["filter_by"] = filter_by

    res = ts.collections[TS_COLLECTION].documents.search(params)
    hits = res.get("hits", []) or []
    out: List[Dict[str, Any]] = []
    for h in hits:
        doc = h.get("document", {}) or {}
        out.append({"article_id": str(doc.get("id")), "lexical_score": float(h.get("text_match", 0.0))})
    return out


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


def build_candidates(
    lex_hits: List[Dict[str, Any]],
    sem_art: List[Tuple[str, float]],
    sem_chk: List[Tuple[str, str, float]],
    entity_conf: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    cand: Dict[str, Dict[str, Any]] = {}

    # Inputs arrive best-first, so the first time an article is seen gives its rank.
    for rank, x in enumerate(lex_hits, start=1):
        aid = x["article_id"]
        c = cand.setdefault(aid, {})
        c["lexical_score"] = max(float(c.get("lexical_score", 0.0)), float(x.get("lexical_score", 0.0)))
        c.setdefault("lex_rank", rank)
        c["src_lexical"] = True

    for rank, (aid, s) in enumerate(sem_art, start=1):
        c = cand.setdefault(aid, {})
        c["sem_article"] = max(float(c.get("sem_article", 0.0)), float(s))
        c.setdefault("sem_article_rank", rank)
        c["src_sem_article"] = True

    # Chunk rank is per article: rank among distinct articles by their best chunk
    chunk_article_rank = 0
    for cid, aid, s in sem_chk:
        c = cand.setdefault(aid, {})
        if "sem_chunk_rank" not in c:
            chunk_article_rank += 1
            c["sem_chunk_rank"] = chunk_article_rank
        best = float(c.get("sem_chunk", 0.0))
        if float(s) > best:
            c["sem_chunk"] = float(s)
            c["best_chunk_id"] = cid
        c["src_sem_chunk"] = True

    out: List[Dict[str, Any]] = []
    for aid, c in cand.items():
        m = articles_meta.get(aid, {})
        out.append(
            {
                "article_id": aid,
                "url": m.get("url"),
                "title": m.get("title"),
                "summary": m.get("summary"),
                "published_date": m.get("published_date"),
                "published_ts": int(m.get("published_ts") or 0),
                "image_url": m.get("image_url"),
                "primary_category": m.get("primary_category"),
                "categories": m.get("categories") if isinstance(m.get("categories"), list) else [],
                "tags": m.get("tags") if isinstance(m.get("tags"), list) else [],
                "location": m.get("location") if isinstance(m.get("location"), list) else [],
                "partner_label": m.get("partner_label"),
                "contributors": m.get("contributors") if isinstance(m.get("contributors"), list) else [],
                "categories_norm": m.get("categories_norm") if isinstance(m.get("categories_norm"), list) else [],
                "tags_norm": m.get("tags_norm") if isinstance(m.get("tags_norm"), list) else [],
                "locations_norm": m.get("locations_norm") if isinstance(m.get("locations_norm"), list) else [],
                "contributors_norm": m.get("contributors_norm") if isinstance(m.get("contributors_norm"), list) else [],
                "lexical_score": float(c.get("lexical_score", 0.0)),
                "sem_article": float(c.get("sem_article", 0.0)),
                "sem_chunk": float(c.get("sem_chunk", 0.0)),
                "lex_rank": c.get("lex_rank"),
                "sem_article_rank": c.get("sem_article_rank"),
                "sem_chunk_rank": c.get("sem_chunk_rank"),
                "best_chunk_id": c.get("best_chunk_id"),
                "entity_conf": entity_conf or {},
            }
        )

    # Keep the best-ranked articles from any retriever when capping
    out.sort(key=lambda z: min(z.get("lex_rank") or 10**6, z.get("sem_chunk_rank") or 10**6, z.get("sem_article_rank") or 10**6))
    return out[:CANDIDATE_CAP]


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
def search(req: SearchRequest) -> SearchResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    ranker_version = req.ranker or RANKER_VERSION
    if ranker_version not in RANKERS:
        raise HTTPException(status_code=400, detail=f"ranker must be one of {sorted(RANKERS)}")

    canon = canonicalize_query_for_search(req.query)
    mode = canon["mode"]
    query_used = canon["q"]  # lexical/canonicalized

    entity = detect_entities(query_full=canon["q_full"], gazetteer=gazetteer)
    hard = entity.get("hard", {})

    filter_final = req.filter_by
    auto_f = entity.get("filter_by_auto")
    if auto_f:
        filter_final = f"({filter_final}) && ({auto_f})" if filter_final else auto_f

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
            loc_ids = set(get_article_ids_with("locations_norm", hard_locations))
            ids = loc_ids if ids is None else (ids & loc_ids) or ids
        qdrant_article_ids = sorted(ids) if ids else None

    query_semantic = build_semantic_query(req.query, mode, entity.get("strip_tokens") or [])

    lex = typesense_search(query_used=query_used, mode=mode, filter_by=filter_final)
    q_vec = encode_query(query_semantic)
    sem_a = qdrant_search_articles(q_vec, article_ids=qdrant_article_ids)
    sem_c = qdrant_search_chunks(q_vec, article_ids=qdrant_article_ids)

    candidates = build_candidates(lex, sem_a, sem_c, entity_conf=entity.get("confidence"))

    now_ts = int(time.time())
    if ranker_version == "ranker_v1":
        ranked = ranker_v1(candidates, tokenize_query(query_used), now_ts=now_ts,
                           has_author_entity=has_author_entity, has_location_entity=has_location_entity)
    else:
        ranked = ranker_v2(candidates, entity.get("matches", {}), now_ts=now_ts, weights=ranker_v2_weights)

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
            query_mode=mode,
            query_used=query_used,
            query_semantic=query_semantic,
            filters={"filter_by": req.filter_by} if req.filter_by else None,
            ranker_version=ranker_version,
            retrieval_version=RETRIEVAL_VERSION,
            meta={
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
            },
        )

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
        insert_candidates(engine, qid, to_log)

    return SearchResponse(
        query_id=qid,
        mode=mode,
        query_used=query_used,
        query_semantic=query_semantic,
        total_results=total_results,
        total_pages=total_pages,
        page=page,
        per_page=per_page,
        results=hits,
    )


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
