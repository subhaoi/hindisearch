"""
English site search API (idronline.org). Same contract as the Hindi API (api_common), its own
indexes, model and log database, all set by EN_* variables in .env:

  EN_API_PORT                    8001
  EN_TYPESENSE_COLLECTION        idr_en_articles_v1
  EN_QDRANT_COLLECTION_ARTICLES  idr_en_articles_vec_v1
  EN_QDRANT_COLLECTION_CHUNKS    idr_en_chunks_vec_v1
  EN_EMBED_MODEL                 the model the vectors were made with (scripts/en/03_embed_english.py)
  EN_DATABASE_URL                its own Postgres database, so the two sites' logs stay apart
  EN_RANKER_WEIGHTS              learned weights; missing -> ranker_v2 defaults

Typesense/Qdrant hosts and keys are shared with the Hindi API.

  uvicorn scripts._phase4.english_search_api:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException

import pandas as pd
import typesense
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from scripts.embedding_models import prefixes
from scripts.english import LocationIndex, as_list, leaves, primary_category, s
from scripts.utils import classify_query, iso_to_epoch_seconds, MAX_QUERY_TOKENS
from .api_common import (
    SearchRequest, SearchHit, SearchResponse, LabelRequest, QueryLabelRequest,
    _cache_key, _cache_get, _cache_put, build_candidates, check_vector_size,
)
from .db import (
    get_engine, ensure_schema, insert_query, insert_candidates, insert_label, insert_error,
    searches_since, errors_since, label_counts,
)
from .english_entities import STOPWORDS, Gazetteer, detect_entities, detect_post_type
from .ranker_v2 import ranker_v2, load_weights
from . import usage_feed

load_dotenv()

RANKER_VERSION = "ranker_v2"
RETRIEVAL_VERSION = os.environ.get("EN_RETRIEVAL_VERSION", "en_retrieval_v1")
LEXICAL_TOPK = int(os.environ.get("LEXICAL_TOPK", "80"))
SEM_ARTICLE_TOPK = int(os.environ.get("SEM_ARTICLE_TOPK", "40"))
SEM_CHUNK_TOPK = int(os.environ.get("SEM_CHUNK_TOPK", "80"))
CANDIDATE_CAP = int(os.environ.get("CANDIDATE_CAP", "200"))
LOG_CANDIDATES_TOPN = int(os.environ.get("LOG_CANDIDATES_TOPN", "50"))

TS_COLLECTION = os.environ.get("EN_TYPESENSE_COLLECTION", "idr_en_articles_v1")
QCOL_ART = os.environ.get("EN_QDRANT_COLLECTION_ARTICLES", "idr_en_articles_vec_v1")
QCOL_CHK = os.environ.get("EN_QDRANT_COLLECTION_CHUNKS", "idr_en_chunks_vec_v1")
MODEL_NAME = os.environ.get("EN_EMBED_MODEL", "")
DATABASE_URL = os.environ.get("EN_DATABASE_URL", "")
WEIGHTS_PATH = Path(os.environ.get("EN_RANKER_WEIGHTS", "data/en/phase_4/ranker_v2_weights.json"))
if not MODEL_NAME or not DATABASE_URL:
    raise RuntimeError("Set EN_EMBED_MODEL and EN_DATABASE_URL in .env (see .env.example)")

# Same weight as the Hindi API for "the query is an author's name"
NAME_QUERY_CONTRIBUTOR_WEIGHT = 1.5

root = Path(".").resolve()
DATA = root / "data" / "en"

articles_df = pd.read_parquet(DATA / "final" / "articles_canonical.parquet")
chunks_df = pd.read_parquet(DATA / "phase_3" / "chunks.parquet")
chunk_text_map = dict(zip(chunks_df["chunk_id"].astype(str), chunks_df["chunk_text"].astype(str)))
_locs = LocationIndex(articles_df)

articles_meta: Dict[str, Dict[str, Any]] = {}
for _, r in articles_df.iterrows():
    cats = as_list(r["categories"])
    aid = str(r["id"])
    articles_meta[aid] = {
        "id": aid,
        "url": s(r["url"]) or None,
        "title": s(r["title"]) or None,
        "summary": s(r["summary"]) or None,
        "published_date": s(r["published_date"]) or None,
        "published_ts": iso_to_epoch_seconds(r["published_date"]),
        "image_url": s(r["image_url"]) or None,
        "post_type": s(r["post_type"]) or None,
        "primary_category": primary_category(cats),
        "categories": leaves(cats),
        "tags": as_list(r["tags"]),
        "location": leaves(as_list(r["locations"])),
        "partner_label": None,
        "contributors": as_list(r["contributors"]),
        # Ranker features compare these with the query's entities (same names as Hindi)
        "categories_norm": leaves(cats),
        "tags_norm": as_list(r["tags"]),
        "locations_norm": leaves(as_list(r["locations"])),
        "locations_all": _locs.for_article(r),
        "contributors_norm": as_list(r["contributors"]),
    }
gazetteer = Gazetteer(
    {a: {"contributors": m["contributors"], "locations_all": m["locations_all"],
         "categories": m["categories"], "tags": m["tags"]} for a, m in articles_meta.items()},
    {a: f"{m['title'] or ''} {m['summary'] or ''}" for a, m in articles_meta.items()},
    location_aliases=_locs.aliases,
)

QUERY_PREFIX, _, _TRUST = prefixes(MODEL_NAME)

app = FastAPI(title="IDR English Search API")
ts = typesense.Client({
    "nodes": [{"host": os.environ.get("TYPESENSE_HOST", "localhost"), "port": os.environ.get("TYPESENSE_PORT", "8108"),
               "protocol": os.environ.get("TYPESENSE_PROTOCOL", "http")}],
    "api_key": os.environ["TYPESENSE_API_KEY"], "connection_timeout_seconds": 10,
})
qd = QdrantClient(host=os.environ.get("QDRANT_HOST", "localhost"), port=int(os.environ.get("QDRANT_PORT", "6333")))
model = SentenceTransformer(MODEL_NAME, trust_remote_code=_TRUST)
check_vector_size(qd, QCOL_CHK, model, MODEL_NAME)
ranker_weights = load_weights(WEIGHTS_PATH)
engine = get_engine(DATABASE_URL)
ensure_schema(engine)


class EnglishSearchHit(SearchHit):
    # article | feature | ground_up_story | post | podcast
    post_type: Optional[str] = None


class EnglishSearchResponse(SearchResponse):
    results: List[EnglishSearchHit]


def typesense_search(q: str, filter_by: Optional[str]) -> List[Dict[str, Any]]:
    p: Dict[str, Any] = {
        "collection": TS_COLLECTION, "q": q, "per_page": LEXICAL_TOPK, "page": 1,
        "query_by": "title,seo_title,summary,content,contributors,locations_all,tags",
        "query_by_weights": "6,4,3,1,5,4,2",
        "num_typos": "2,2,2,1,1,1,1",
    }
    if q == "*":
        p["sort_by"] = "published_ts:desc"
    if filter_by:
        p["filter_by"] = filter_by
    res = ts.multi_search.perform({"searches": [p]}, {})["results"][0]
    if "error" in res:
        raise RuntimeError(f"Typesense search failed: {res.get('error')}")
    return [{"article_id": str(h["document"]["id"]), "lexical_score": float(h.get("text_match", 0.0))}
            for h in res.get("hits", []) or []]


def _article_filter(article_ids: Optional[List[str]]) -> Optional[qm.Filter]:
    if not article_ids:
        return None
    return qm.Filter(must=[qm.FieldCondition(key="article_id", match=qm.MatchAny(any=article_ids))])


def vector_search(q_vec: List[float], article_ids: Optional[List[str]]) -> Tuple[list, list]:
    flt = _article_filter(article_ids)
    arts = qd.search(collection_name=QCOL_ART, query_vector=q_vec, limit=SEM_ARTICLE_TOPK, with_payload=False, query_filter=flt)
    chks = qd.search(collection_name=QCOL_CHK, query_vector=q_vec, limit=SEM_CHUNK_TOPK, with_payload=True, query_filter=flt)
    return ([(str(p.id), float(p.score)) for p in arts],
            [(str(p.payload["chunk_id"]), str(p.payload["article_id"]), float(p.score)) for p in chks
             if p.payload and p.payload.get("chunk_id") is not None])


def ids_with(field: str, values: List[str]) -> set:
    wanted = set(values)
    return {a for a, m in articles_meta.items() if wanted & set(m.get(field) or [])}


def choose_snippet(item: Dict[str, Any]) -> Optional[str]:
    txt = chunk_text_map.get(str(item.get("best_chunk_id") or ""))
    return " ".join(txt.split())[:420] if txt else None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "site": "en", "model": MODEL_NAME, "ranker_version": RANKER_VERSION,
            "retrieval_version": RETRIEVAL_VERSION}


@app.post("/search", response_model=EnglishSearchResponse)
def search(req: SearchRequest, background_tasks: BackgroundTasks) -> EnglishSearchResponse:
    started = time.perf_counter()
    try:
        return run_search(req, started, background_tasks)
    except HTTPException:
        raise
    except Exception as e:
        insert_error(engine, f"{type(e).__name__}: {e}", round(time.perf_counter() - started, 2))
        raise


def run_search(req: SearchRequest, started: float, background_tasks: BackgroundTasks) -> EnglishSearchResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    if req.ranker and req.ranker != RANKER_VERSION:
        raise HTTPException(status_code=400, detail=f"ranker must be {RANKER_VERSION}")
    per_page = max(1, int(req.per_page))
    if classify_query(req.query)[0] == "junk":
        return EnglishSearchResponse(query_id=0, mode="junk", query_used="", query_semantic="", total_results=0,
                                     total_pages=1, page=1, per_page=per_page, results=[])

    key = _cache_key(req.query, req.filter_by, RANKER_VERSION)
    res = _cache_get(key)
    cached = res is not None
    if res is None:
        res = retrieve(req.query, req.filter_by)
        _cache_put(key, res)
    ranked = res["ranked"]

    total = len(ranked)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(max(1, int(req.page)), total_pages)
    start = (page - 1) * per_page
    hits = [
        EnglishSearchHit(
            rank=it["rank"], id=it["article_id"], title=it.get("title"), date=it.get("published_date"),
            summary=it.get("summary"), url=it.get("url"), image_url=it.get("image_url"),
            primary_category=it.get("primary_category"), categories=it.get("categories") or [],
            tags=it.get("tags") or [], location=it.get("location") or [], partner_label=None,
            contributors=it.get("contributors") or [], score=float(it["score"]), snippet=choose_snippet(it),
            post_type=it.get("post_type"),
            features=it["features"] if req.explain else None,
            explanation=it["explanation"] if req.explain else None,
        )
        for it in ranked[start:start + per_page]
    ]

    qid = 0
    if req.log:
        qid = insert_query(
            engine=engine, query_raw=req.query, query_mode="en", query_used=res["query_used"],
            query_semantic=res["query_semantic"], filters={"filter_by": req.filter_by} if req.filter_by else None,
            ranker_version=RANKER_VERSION, retrieval_version=RETRIEVAL_VERSION,
            meta={**res["meta"], "cached": cached, "took_s": round(time.perf_counter() - started, 2)},
        )
        background_tasks.add_task(log_candidates, qid, ranked)

    return EnglishSearchResponse(query_id=qid, mode="en", query_used=res["query_used"], query_semantic=res["query_semantic"],
                                 total_results=total, total_pages=total_pages, page=page, per_page=per_page, results=hits)


def log_candidates(qid: int, ranked: List[Dict[str, Any]]) -> None:
    keep = ("rank", "article_id", "url", "title", "published_date", "summary", "primary_category", "categories",
            "tags", "location", "partner_label", "contributors", "features", "explanation")
    rows = [{**{k: it.get(k) for k in keep}, "score": float(it["score"])} for it in ranked[:LOG_CANDIDATES_TOPN]]
    try:
        insert_candidates(engine, qid, rows)
    except Exception as e:  # noqa: BLE001 - the reply has already gone; just report it
        print(f"[candidate_log] query {qid}: {type(e).__name__}: {e}")


def retrieve(query: str, filter_by: Optional[str]) -> Dict[str, Any]:
    """Understand the query, retrieve from Typesense and Qdrant, rank. Cached per query."""
    kind, year = classify_query(query)
    text = " ".join(query.split()[:MAX_QUERY_TOKENS])
    filters: List[str] = [filter_by] if filter_by else []
    allowed: Optional[set] = None
    post_types: List[str] = []
    if kind == "year":
        start = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
        end = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp())
        ids = {a for a, m in articles_meta.items() if start <= m["published_ts"] < end}
        if ids:  # a future year shows the newest articles instead of nothing
            filters.append(f"published_ts:>={start} && published_ts:<{end}")
            allowed = ids
        text = ""
    else:
        # "podcast on climate" -> podcasts only, searched for "on climate"
        post_types, text = detect_post_type(text)
        if post_types:
            filters.append(f"post_type:=[{','.join(post_types)}]")
            allowed = {a for a, m in articles_meta.items() if m["post_type"] in post_types}

    entity = detect_entities(text, gazetteer) if text else {"matches": {}, "hard": {}, "confidence": {}}
    hard = entity.get("hard", {})
    if entity.get("filter_by_auto"):
        filters.append(entity["filter_by_auto"])
    filter_final = " && ".join(f"({f})" for f in filters) if len(filters) > 1 else (filters[0] if filters else None)

    ids: Optional[set] = None
    if hard.get("contributors_norm"):
        ids = ids_with("contributors_norm", hard["contributors_norm"])
    if hard.get("locations_norm"):
        loc_ids = ids_with("locations_all", hard["locations_norm"])
        ids = loc_ids if ids is None else (ids & loc_ids) or ids
    if allowed is not None:
        ids = allowed if ids is None else (ids & allowed)
    qdrant_ids = sorted(ids) if ids else (["__none__"] if ids is not None else None)

    timings: Dict[str, float] = {}
    t = time.perf_counter()
    lex = typesense_search(text or "*", filter_final)
    timings["t_typesense"] = time.perf_counter() - t
    query_semantic, sem_a, sem_c = "", [], []
    if text:
        drop = set(entity.get("strip_tokens") or [])
        kept = [w for w in text.split() if w.lower().strip(".,;:!?\"'()") not in drop]
        if drop:
            # "water crisis in maharashtra" -> "water crisis", not "water crisis in"
            while kept and kept[-1].lower() in STOPWORDS:
                kept.pop()
            while kept and kept[0].lower() in STOPWORDS:
                kept.pop(0)
        query_semantic = " ".join(kept) if kept else text
        t = time.perf_counter()
        q_vec = model.encode([QUERY_PREFIX + query_semantic], normalize_embeddings=True)[0].tolist()
        timings["t_embed"] = time.perf_counter() - t
        t = time.perf_counter()
        sem_a, sem_c = vector_search(q_vec, qdrant_ids)
        timings["t_qdrant"] = time.perf_counter() - t

    t = time.perf_counter()
    candidates = build_candidates(lex, sem_a, sem_c, articles_meta, CANDIDATE_CAP, entity.get("confidence"))
    weights = ranker_weights
    if entity.get("name_query"):
        weights = {**weights, "match_contributor": max(weights.get("match_contributor", 0.0), NAME_QUERY_CONTRIBUTOR_WEIGHT)}
    ranked = ranker_v2(candidates, entity.get("matches", {}), now_ts=int(time.time()), weights=weights)
    timings["t_rank"] = time.perf_counter() - t

    return {
        "query_used": text or "*",
        "query_semantic": query_semantic,
        "ranked": ranked,
        "meta": {
            "lex_n": len(lex), "sem_article_n": len(sem_a), "sem_chunk_n": len(sem_c), "cand_n": len(candidates),
            "entity_matches": entity.get("matches", {}), "entity_hard": hard,
            "filter_by_final": filter_final, "post_types": post_types, "query_kind": kind,
            "name_query": bool(entity.get("name_query")),
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
            # The English weekly refresh and test set are not set up yet
            "refresh": None,
            "eval": None,
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
