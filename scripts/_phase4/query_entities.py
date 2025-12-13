from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _norm_ws(s: str) -> str:
    s2 = (s or "").strip().lower()
    s2 = re.sub(r"\s+", " ", s2)
    return s2


def roman_norm(s: str) -> str:
    t = _norm_ws(s)
    t = re.sub(r"aa+", "aa", t)
    t = re.sub(r"ee+", "ee", t)
    t = re.sub(r"ii+", "ii", t)
    t = re.sub(r"oo+", "oo", t)
    t = re.sub(r"uu+", "uu", t)
    return t


def tokenize_loose(q: str) -> List[str]:
    q2 = _norm_ws(q)
    toks = re.split(r"[^\w\u0900-\u097F]+", q2, flags=re.UNICODE)
    return [t for t in toks if t and len(t) >= 2]


def _safe_ts_backtick(s: str) -> str:
    # Typesense filter strings use backticks for string literals.
    # Escape any backticks defensively.
    return str(s).replace("`", "\\`")


def _build_in_filter(field: str, values: List[str]) -> Optional[str]:
    if not values:
        return None
    # field:=[`a`,`b`]
    v = ",".join([f"`{_safe_ts_backtick(x)}`" for x in values])
    return f"{field}:=[{v}]"


def detect_entities(
    query_used: str,
    mode: str,
    gazetteer: Dict[str, Any],
    max_per_field: int = 3,
) -> Dict[str, Any]:
    """
    Returns:
      {
        matches: {field: [values...]},
        confidence: {field: int},
        filter_by_auto: str|None
      }
    Confidence heuristic:
      - phrase substring match => +2
      - token match => +1
    """
    q_used = _norm_ws(query_used)
    q_tokens = tokenize_loose(q_used)

    # For roman mode, also use roman_norm for matching
    q_roman = roman_norm(q_used) if mode != "dev" else ""

    matches: Dict[str, List[str]] = {}
    conf: Dict[str, int] = {}

    def scan(field: str, allow_token: bool) -> None:
        vals = gazetteer.get(field, {}).get("values", []) or []
        vals_r = gazetteer.get(field, {}).get("values_roman_norm", []) or []

        got: List[str] = []
        score = 0

        # Phrase match (longest-first ordering already)
        for i, v in enumerate(vals):
            if len(got) >= max_per_field:
                break
            v_norm = _norm_ws(v)
            if not v_norm:
                continue

            if mode == "dev":
                if v_norm in q_used:
                    got.append(v)
                    score += 2
            else:
                # roman mode: match either raw (sometimes contributors are latin in metadata)
                if v_norm in q_used:
                    got.append(v)
                    score += 2
                else:
                    vr = vals_r[i] if i < len(vals_r) else roman_norm(v_norm)
                    if vr and vr in q_roman:
                        got.append(v)
                        score += 2

        # Token match fallback (optional)
        if allow_token and len(got) < max_per_field:
            qtok = set(q_tokens)
            for v in vals:
                if len(got) >= max_per_field:
                    break
                vtok = set(tokenize_loose(v))
                if not vtok:
                    continue
                if len(qtok.intersection(vtok)) > 0:
                    if v not in got:
                        got.append(v)
                        score += 1

        if got:
            matches[field] = got
            conf[field] = score

    # Locations: allow token matching, strong signal
    scan("locations_norm", allow_token=True)
    # Contributors: phrase-only (avoid false positives)
    scan("contributors_norm", allow_token=False)
    # Categories/tags: allow token matching but treated as soft unless very confident
    scan("categories_norm", allow_token=True)
    scan("tags_norm", allow_token=True)

    # Decide auto filter_by (conservative)
    filters: List[str] = []

    # Apply location filter if any
    if matches.get("locations_norm"):
        f = _build_in_filter("locations_norm", matches["locations_norm"])
        if f:
            filters.append(f)

    # Apply contributor filter only if strong (>=2 implies phrase match)
    if matches.get("contributors_norm") and conf.get("contributors_norm", 0) >= 2:
        f = _build_in_filter("contributors_norm", matches["contributors_norm"])
        if f:
            filters.append(f)

    # Categories/tags: only hard-filter if strong confidence (>=4 means likely multiple phrase hits)
    if matches.get("categories_norm") and conf.get("categories_norm", 0) >= 4:
        f = _build_in_filter("categories_norm", matches["categories_norm"])
        if f:
            filters.append(f)

    if matches.get("tags_norm") and conf.get("tags_norm", 0) >= 4:
        f = _build_in_filter("tags_norm", matches["tags_norm"])
        if f:
            filters.append(f)

    auto_filter = " && ".join(filters) if filters else None

    return {
        "matches": matches,
        "confidence": conf,
        "filter_by_auto": auto_filter,
    }
