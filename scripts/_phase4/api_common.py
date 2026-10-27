"""
Shared by the Hindi and English search APIs: the request/response contract (one for both sites,
so a frontend works with either), merging retriever results into candidates, and the in-memory
cache of ranked results.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


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


def build_candidates(
    lex_hits: List[Dict[str, Any]],
    sem_art: List[Tuple[str, float]],
    sem_chk: List[Tuple[str, str, float]],
    articles_meta: Dict[str, Dict[str, Any]],
    cap: int,
    entity_conf: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """One record per article from the three retrievers: its rank in each, raw scores, metadata."""
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
                "post_type": m.get("post_type"),
                "categories_norm": m.get("categories_norm") if isinstance(m.get("categories_norm"), list) else [],
                "tags_norm": m.get("tags_norm") if isinstance(m.get("tags_norm"), list) else [],
                "locations_norm": m.get("locations_norm") if isinstance(m.get("locations_norm"), list) else [],
                "locations_all": m.get("locations_all") if isinstance(m.get("locations_all"), list) else [],
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
    return out[:cap]


def check_vector_size(qd: Any, collection: str, model: Any, model_name: str) -> None:
    """
    Refuse to start when the query model and the stored vectors can't be compared (a model
    changed in .env without re-embedding). Qdrant being unreachable is not decided here.
    """
    try:
        stored = qd.get_collection(collection).config.params.vectors.size
    except Exception:  # noqa: BLE001 - reported by the first search instead
        return
    dim = model.get_sentence_embedding_dimension()
    if dim and stored and int(dim) != int(stored):
        raise RuntimeError(f"{model_name} makes {dim}-dim vectors but {collection} holds {stored}-dim ones: "
                           "re-embed and re-index with the same model, or fix the model setting")


# Ranked results per query, kept until the API restarts. The weekly refresh restarts it, so a
# cached result never outlives the data it came from. Repeated searches ("2026" and "2027" are
# about a third of all searches) then skip retrieval entirely.
_search_cache: "OrderedDict[Tuple[str, str, str], Dict[str, Any]]" = OrderedDict()
_search_cache_lock = threading.Lock()


def _cache_key(query: str, filter_by: Optional[str], ranker_version: str) -> Tuple[str, str, str]:
    return " ".join(query.split()).casefold(), filter_by or "", ranker_version


def _cache_get(key: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
    with _search_cache_lock:
        hit = _search_cache.get(key)
        if hit is not None:
            _search_cache.move_to_end(key)
        return hit


def _cache_size() -> int:
    # Read when used, not at import: the APIs load .env after importing this module
    return int(os.environ.get("SEARCH_CACHE_SIZE", "128"))


def _cache_put(key: Tuple[str, str, str], value: Dict[str, Any]) -> None:
    size = _cache_size()
    if size <= 0:
        return
    with _search_cache_lock:
        _search_cache[key] = value
        _search_cache.move_to_end(key)
        while len(_search_cache) > size:
            _search_cache.popitem(last=False)
