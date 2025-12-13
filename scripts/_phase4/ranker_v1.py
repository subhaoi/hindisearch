from __future__ import annotations

from typing import Any, Dict, List


def minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def safe_overlap_count(query_tokens: List[str], field_tokens: List[str]) -> int:
    if not query_tokens or not field_tokens:
        return 0
    qs = set(t for t in query_tokens if t)
    fs = set(t for t in field_tokens if t)
    return len(qs.intersection(fs))


def recency_score(published_ts: int, now_ts: int) -> float:
    if published_ts <= 0 or now_ts <= 0:
        return 0.0
    age_days = max(0.0, (now_ts - published_ts) / 86400.0)
    return float(max(0.0, 1.0 - (age_days / 1095.0)))


def ranker_v1(candidates: List[Dict[str, Any]], query_tokens: List[str], now_ts: int) -> List[Dict[str, Any]]:
    lex = [float(c.get("lexical_score", 0.0)) for c in candidates]
    sa = [float(c.get("sem_article", 0.0)) for c in candidates]
    sc = [float(c.get("sem_chunk", 0.0)) for c in candidates]

    lex_n = minmax_norm(lex)
    sa_n = minmax_norm(sa)
    sc_n = minmax_norm(sc)

    # Conservative: lexical dominates top-3
    W_LEX = 1.0
    W_SC = 0.40
    W_SA = 0.18
    W_TAG = 0.12
    W_CAT = 0.10
    W_LOC = 0.15
    W_CONTRIB = 0.06
    W_REC = 0.08

    out: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        tags = c.get("tags_norm") or []
        cats = c.get("categories_norm") or []
        locs = c.get("locations_norm") or []
        contrib = c.get("contributors_norm") or []

        tag_ov = safe_overlap_count(query_tokens, tags)
        cat_ov = safe_overlap_count(query_tokens, cats)
        loc_ov = safe_overlap_count(query_tokens, locs)
        con_ov = safe_overlap_count(query_tokens, contrib)

        tag_feat = min(1.0, tag_ov / 2.0)
        cat_feat = min(1.0, cat_ov / 2.0)
        loc_feat = min(1.0, loc_ov / 1.0)
        con_feat = min(1.0, con_ov / 1.0)

        rec = recency_score(int(c.get("published_ts", 0) or 0), now_ts)

        score_parts = {
            "lex": W_LEX * lex_n[i],
            "sem_chunk": W_SC * sc_n[i],
            "sem_article": W_SA * sa_n[i],
            "tag_overlap": W_TAG * tag_feat,
            "cat_overlap": W_CAT * cat_feat,
            "loc_overlap": W_LOC * loc_feat,
            "contrib_overlap": W_CONTRIB * con_feat,
            "recency": W_REC * rec,
        }
        score = sum(score_parts.values())

        explain = sorted(score_parts.items(), key=lambda kv: kv[1], reverse=True)[:4]

        features = {
            "lexical_score_raw": float(c.get("lexical_score", 0.0)),
            "sem_article_raw": float(c.get("sem_article", 0.0)),
            "sem_chunk_raw": float(c.get("sem_chunk", 0.0)),
            "lex_norm": lex_n[i],
            "sem_article_norm": sa_n[i],
            "sem_chunk_norm": sc_n[i],
            "tag_overlap_count": tag_ov,
            "cat_overlap_count": cat_ov,
            "loc_overlap_count": loc_ov,
            "contrib_overlap_count": con_ov,
            "recency": rec,
            "best_chunk_id": c.get("best_chunk_id"),
            "src_lexical": bool(c.get("src_lexical", False)),
            "src_sem_article": bool(c.get("src_sem_article", False)),
            "src_sem_chunk": bool(c.get("src_sem_chunk", False)),
        }

        out.append({**c, "score": float(score), "features": features, "explanation": explain})

    out.sort(key=lambda x: x["score"], reverse=True)
    for r, item in enumerate(out, start=1):
        item["rank"] = r
    return out
