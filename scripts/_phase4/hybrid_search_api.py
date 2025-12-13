from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import typesense
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from scripts.utils import Paths, read_parquet, canonicalize_query_for_search, is_nullish
from .ranker_v1 import ranker_v1
from .db import get_engine, ensure_schema, insert_query, insert_candidates, insert_label

from .query_entities import detect_entities
import json

load_dotenv()

API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

RANKER_VERSION = os.environ.get("RANKER_VERSION", "ranker_v1")
RETRIEVAL_VERSION = os.environ.get("RETRIEVAL_VERSION", "retrieval_v1")

LEXICAL_TOPK = int(os.environ.get("LEXICAL_TOPK", "80"))
SEM_ARTICLE_TOPK = int(os.environ.get("SEM_ARTICLE_TOPK", "40"))
SEM_CHUNK_TOPK = int(os.environ.get("SEM_CHUNK_TOPK", "80"))
CANDIDATE_CAP = int(os.environ.get("CANDIDATE_CAP", "200"))
LOG_CANDIDATES_TOPN = int(os.environ.get("LOG_CANDIDATES_TOPN", "200"))

TS_COLLECTION = os.environ.get("TYPESENSE_COLLECTION", os.environ.get("TYPESENSE_COLLECTION", "idr_articles_hi_v1"))
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QCOL_ART = os.environ.get("QDRANT_COLLECTION_ARTICLES", "idr_articles_vec_v1")
QCOL_CHK = os.environ.get("QDRANT_COLLECTION_CHUNKS", "idr_chunks_vec_v1")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


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


def tokenize_query(q: str) -> List[str]:
    # Stable, punctuation-aware token split for both scripts (lightweight)
    import re
    q2 = (q or "").lower()
    toks = re.split(r"[^\w\u0900-\u097F]+", q2, flags=re.UNICODE)
    return [t for t in toks if t and len(t) >= 2]


root = Path(".").resolve()
paths = Paths(root=root)

GAZ_PATH = paths.data / "phase_45" / "gazetteer_v1.json"
if not GAZ_PATH.exists():
    raise RuntimeError(f"Missing {GAZ_PATH}. Run: python scripts/20_build_gazetteer.py")

with GAZ_PATH.open("r", encoding="utf-8") as f:
    gazetteer = json.load(f)

ARTICLES_PATH = paths.data / "final" / "articles_canonical.parquet"
CHUNKS_PATH = paths.data / "phase_3" / "chunks.parquet"

if not ARTICLES_PATH.exists():
    raise RuntimeError(f"Missing {ARTICLES_PATH}")
if not CHUNKS_PATH.exists():
    raise RuntimeError(f"Missing {CHUNKS_PATH}")

articles_df = read_parquet(ARTICLES_PATH)
articles_meta: Dict[str, Dict[str, Any]] = {}
def _as_list(v):
    if is_nullish(v):
        return []
    try:
        return list(v)
    except Exception:
        return [str(v)]

for _, r in articles_df.iterrows():
    aid = str(r.get("id"))
    cats = _as_list(r.get("categories_raw"))  # human-readable
    tags = _as_list(r.get("tags_raw"))
    locs = _as_list(r.get("locations_raw"))
    contrib = _as_list(r.get("contributors_raw"))

    primary_category = cats[0] if len(cats) > 0 else None

    articles_meta[aid] = {
        "id": aid,
        "url": None if is_nullish(r.get("url")) else str(r.get("url")),
        "title": None if is_nullish(r.get("title_hi")) else str(r.get("title_hi")),
        "summary": None if is_nullish(r.get("summary_hi")) else str(r.get("summary_hi")),
        "published_date": None if is_nullish(r.get("published_date")) else str(r.get("published_date")),
        "published_ts": int(r.get("published_ts")) if "published_ts" in articles_df.columns and not is_nullish(r.get("published_ts")) else 0,

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

app = FastAPI(title="IDR Hybrid Search API (Phase 4)")

ts = get_typesense_client()
qd = get_qdrant_client()
model = SentenceTransformer(MODEL_NAME)

engine = get_engine()
ensure_schema(engine)


class SearchRequest(BaseModel):
    query: str
    filter_by: Optional[str] = None
    per_page: int = 10
    explain: bool = False


class SearchHit(BaseModel):
    rank: int
    id: str
    title: Optional[str] = None
    date: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[str] = None

    primary_category: Optional[str] = None
    categories: List[str] = []
    tags: List[str] = []
    location: List[str] = []
    partner_label: Optional[str] = None
    contributors: List[str] = []

    score: float
    snippet: Optional[str] = None

    # keep internal optional
    features: Optional[Dict[str, Any]] = None
    explanation: Optional[List[Any]] = None


class SearchResponse(BaseModel):
    query_id: int
    mode: str
    query_used: str
    query_semantic: str
    results: List[SearchHit]


class LabelRequest(BaseModel):
    query_id: int
    article_id: Optional[str] = None
    label: int
    note: Optional[str] = None

class QueryLabelRequest(BaseModel):
    query_id: int
    label: int  # only 0 supported here (nothing relevant)
    note: Optional[str] = None

def typesense_search(query_used: str, mode: str, filter_by: Optional[str]) -> List[Dict[str, Any]]:
    if mode == "dev":
        query_by = "title_hi,summary_hi,content_hi"
    else:
        query_by = "title_roman_norm,summary_roman_norm,content_roman_norm"

    params: Dict[str, Any] = {
        "q": query_used,
        "query_by": query_by,
        "query_by_weights": "6,3,1",
        "per_page": LEXICAL_TOPK,
        "page": 1,
        "num_typos": 1,
    }
    if filter_by:
        params["filter_by"] = filter_by

    res = ts.collections[TS_COLLECTION].documents.search(params)
    hits = res.get("hits", []) or []
    out = []
    for h in hits:
        doc = h.get("document", {}) or {}
        out.append(
            {
                "article_id": str(doc.get("id")),
                "lexical_score": float(h.get("text_match", 0.0)),
            }
        )
    return out


def qdrant_search_articles(query_semantic: str) -> List[Tuple[str, float]]:
    q_vec = model.encode([query_semantic], normalize_embeddings=True)[0].tolist()
    res = qd.search(collection_name=QCOL_ART, query_vector=q_vec, limit=SEM_ARTICLE_TOPK, with_payload=False)
    return [(str(p.id), float(p.score)) for p in res]


def qdrant_search_chunks(query_semantic: str) -> List[Tuple[str, str, float]]:
    q_vec = model.encode([query_semantic], normalize_embeddings=True)[0].tolist()
    res = qd.search(collection_name=QCOL_CHK, query_vector=q_vec, limit=SEM_CHUNK_TOPK, with_payload=True)
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
) -> List[Dict[str, Any]]:
    cand: Dict[str, Dict[str, Any]] = {}

    # lexical
    for x in lex_hits:
        aid = x["article_id"]
        c = cand.setdefault(aid, {})
        c["lexical_score"] = max(float(c.get("lexical_score", 0.0)), float(x.get("lexical_score", 0.0)))
        c["src_lexical"] = True

    # semantic articles
    for aid, s in sem_art:
        c = cand.setdefault(aid, {})
        c["sem_article"] = max(float(c.get("sem_article", 0.0)), float(s))
        c["src_sem_article"] = True

    # semantic chunks (best snippet chunk)
    for cid, aid, s in sem_chk:
        c = cand.setdefault(aid, {})
        best = float(c.get("sem_chunk", 0.0))
        if float(s) > best:
            c["sem_chunk"] = float(s)
            c["best_chunk_id"] = cid
        c["src_sem_chunk"] = True

    out: List[Dict[str, Any]] = []
    for aid, c in cand.items():
        m = articles_meta.get(aid, {})
        out.append({
                    "article_id": aid,
                    "url": m.get("url"),
                    "title": m.get("title"),
                    "summary": m.get("summary"),
                    "published_date": m.get("published_date"),
                    "published_ts": int(m.get("published_ts") or 0),

                    "primary_category": m.get("primary_category"),
                    "categories": m.get("categories") or [],
                    "tags": m.get("tags") or [],
                    "location": m.get("location") or [],
                    "partner_label": m.get("partner_label"),
                    "contributors": m.get("contributors") or [],

                    "categories_norm": m.get("categories_norm") or [],
                    "tags_norm": m.get("tags_norm") or [],
                    "locations_norm": m.get("locations_norm") or [],
                    "contributors_norm": m.get("contributors_norm") or [],

                    "lexical_score": float(c.get("lexical_score", 0.0)),
                    "sem_article": float(c.get("sem_article", 0.0)),
                    "sem_chunk": float(c.get("sem_chunk", 0.0)),
                    "best_chunk_id": c.get("best_chunk_id"),
                })

    out.sort(key=lambda z: (z.get("lexical_score", 0.0) + z.get("sem_chunk", 0.0) + z.get("sem_article", 0.0)), reverse=True)
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

    canon = canonicalize_query_for_search(req.query)
    mode = canon["mode"]
    query_used = canon["q"]  # lexical
    query_semantic = req.query.strip()  # semantic uses raw

    entity = detect_entities(query_used=query_used, mode=mode, gazetteer=gazetteer)

    filter_final = req.filter_by
    if entity.get("filter_by_auto"):
        if filter_final:
            filter_final = f"({filter_final}) && ({entity['filter_by_auto']})"
        else:
            filter_final = entity["filter_by_auto"]

    lex = typesense_search(query_used=query_used, mode=mode, filter_by=filter_final)

    sem_a = qdrant_search_articles(query_semantic=query_semantic)
    sem_c = qdrant_search_chunks(query_semantic=query_semantic)

    candidates = build_candidates(lex, sem_a, sem_c)

    q_tokens = tokenize_query(query_used)
    now_ts = int(time.time())
    ranked = ranker_v1(candidates, q_tokens, now_ts=now_ts)

    hits: List[SearchHit] = []
    for item in ranked[: max(1, req.per_page)]:
        hits.append(
            SearchHit(
                rank=item["rank"],
                article_id=item["article_id"],
                title=item.get("title"),
                url=item.get("url"),
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

    qid = insert_query(
        engine=engine,
        query_raw=req.query,
        query_mode=mode,
        query_used=query_used,
        query_semantic=query_semantic,
        filters={"filter_by": req.filter_by} if req.filter_by else None,
        ranker_version=RANKER_VERSION,
        retrieval_version=RETRIEVAL_VERSION,
        meta={
            "lex_n": len(lex),
            "sem_article_n": len(sem_a),
            "sem_chunk_n": len(sem_c),
            "cand_n": len(candidates),
            "entity_matches": entity.get("matches", {}),
            "entity_confidence": entity.get("confidence", {}),
            "filter_by_auto": entity.get("filter_by_auto"),
            "filter_by_final": filter_final,
            }
        )

    # Log top-N ranked candidates (training-grade)
    topn = min(LOG_CANDIDATES_TOPN, len(ranked))
    to_log = []
    for item in ranked[:topn]:
        to_log.append({
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
        })
    insert_candidates(engine, qid, to_log)

    return SearchResponse(query_id=qid, mode=mode, query_used=query_used, query_semantic=query_semantic, results=hits)


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
    # store as labels row with article_id NULL
    insert_label(engine, req.query_id, None, 0, req.note)
    return {"ok": True}