from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reciprocal rank fusion constant (standard value; larger = flatter).
RRF_K = 60

# Each feature is in [0, 1]; score = sum(weight * feature).
FEATURE_NAMES = [
    "rrf_lex",          # keyword rank in Typesense
    "rrf_sem_chunk",    # rank of the article's best chunk in Qdrant
    "rrf_sem_article",  # article-vector rank in Qdrant
    "match_location",   # article is tagged with a location detected in the query
    "match_contributor",
    "match_category",
    "match_tag",
    "recency",
]

DEFAULT_WEIGHTS: Dict[str, float] = {
    "rrf_lex": 1.0,
    "rrf_sem_chunk": 1.0,
    "rrf_sem_article": 0.5,
    "match_location": 0.25,
    "match_contributor": 0.40,
    "match_category": 0.08,
    "match_tag": 0.08,
    "recency": 0.05,
}

RECENCY_HALF_LIFE_DAYS = 730.0

# Written by scripts/22_train_ranker.py. Missing file -> DEFAULT_WEIGHTS.
WEIGHTS_PATH = Path(os.environ.get("RANKER_V2_WEIGHTS", "data/phase_4/ranker_v2_weights.json"))


def load_weights(path: Path = WEIGHTS_PATH) -> Dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_WEIGHTS)
    with path.open("r", encoding="utf-8") as f:
        learned = json.load(f).get("weights", {})
    return {k: float(learned.get(k, DEFAULT_WEIGHTS[k])) for k in FEATURE_NAMES}


def rrf(rank: Optional[int]) -> float:
    # Normalized so rank 1 -> 1.0 and "not retrieved" -> 0.0
    if not rank:
        return 0.0
    return (RRF_K + 1) / (RRF_K + rank)


def recency_score(published_ts: int, now_ts: int) -> float:
    if published_ts <= 0 or now_ts <= 0:
        return 0.0
    age_days = max(0.0, (now_ts - published_ts) / 86400.0)
    return math.pow(0.5, age_days / RECENCY_HALF_LIFE_DAYS)


def _overlap(article_values: List[str], query_values: List[str]) -> float:
    if not article_values or not query_values:
        return 0.0
    return 1.0 if set(article_values) & set(query_values) else 0.0


def compute_features(c: Dict[str, Any], entity_matches: Dict[str, List[str]], now_ts: int) -> Dict[str, float]:
    return {
        "rrf_lex": rrf(c.get("lex_rank")),
        "rrf_sem_chunk": rrf(c.get("sem_chunk_rank")),
        "rrf_sem_article": rrf(c.get("sem_article_rank")),
        "match_location": _overlap(c.get("locations_norm") or [], entity_matches.get("locations_norm") or []),
        "match_contributor": _overlap(c.get("contributors_norm") or [], entity_matches.get("contributors_norm") or []),
        "match_category": _overlap(c.get("categories_norm") or [], entity_matches.get("categories_norm") or []),
        "match_tag": _overlap(c.get("tags_norm") or [], entity_matches.get("tags_norm") or []),
        "recency": recency_score(int(c.get("published_ts", 0) or 0), now_ts),
    }


def ranker_v2(
    candidates: List[Dict[str, Any]],
    entity_matches: Dict[str, List[str]],
    now_ts: int,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Rank fusion instead of min-max score blending: each retriever contributes by rank,
    so an article missing from one retriever is not treated as its worst match, and
    Typesense's large bucketed text_match values cannot dominate. Entity matches are
    per article (does this article carry the detected location/author/tag?).
    """
    w = weights or DEFAULT_WEIGHTS
    out: List[Dict[str, Any]] = []
    for c in candidates:
        feats = compute_features(c, entity_matches, now_ts)
        parts = {k: w.get(k, 0.0) * feats[k] for k in FEATURE_NAMES}
        score = sum(parts.values())
        explain = sorted(parts.items(), key=lambda kv: kv[1], reverse=True)[:4]

        features: Dict[str, Any] = dict(feats)
        features.update(
            {
                "lex_rank": c.get("lex_rank"),
                "sem_chunk_rank": c.get("sem_chunk_rank"),
                "sem_article_rank": c.get("sem_article_rank"),
                "lexical_score_raw": float(c.get("lexical_score", 0.0)),
                "sem_article_raw": float(c.get("sem_article", 0.0)),
                "sem_chunk_raw": float(c.get("sem_chunk", 0.0)),
                "best_chunk_id": c.get("best_chunk_id"),
            }
        )
        out.append({**c, "score": float(score), "features": features, "explanation": explain})

    out.sort(key=lambda x: x["score"], reverse=True)
    for r, item in enumerate(out, start=1):
        item["rank"] = r
    return out
