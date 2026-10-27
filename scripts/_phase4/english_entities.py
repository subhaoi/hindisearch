"""
Names in English queries: authors, locations, categories and tags of the English site, found
as whole-word phrases. Same output shape as query_entities.detect_entities (Hindi), so the
ranker and filters work unchanged. Field keys keep the Hindi names (contributors_norm, ...):
they are the ranker's feature names; the values are the English metadata as written.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from scripts.utils import query_tokens

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of",
    "on", "or", "the", "to", "what", "with", "about", "into", "its", "why", "who", "does", "do",
}

# meta field -> entity key used by ranker_v2
FIELDS = {"contributors": "contributors_norm", "locations_all": "locations_norm",
          "categories": "categories_norm", "tags": "tags_norm"}

# Typesense fields the hard filters use
FILTER_FIELDS = {"contributors_norm": "contributors", "locations_norm": "locations_all"}

# Tagged/mentioned in more than this share of articles ("India"): a boost, never a filter
BROAD_SHARE = 0.20

# A one-word query that is part of an author's name boosts their articles, unless the word is
# common in headlines/summaries (a surname like "Rice" or "Hope" is also an ordinary word)
COMMON_WORD_SHARE = 0.01


class Gazetteer:
    def __init__(self, articles_meta: Dict[str, Dict[str, Any]], headline_text: Dict[str, str],
                 location_aliases: Optional[Dict[str, str]] = None) -> None:
        self.n = max(1, len(articles_meta))
        self.values: Dict[str, List[tuple]] = {}
        for field in FIELDS:
            counts = Counter(v for m in articles_meta.values() for v in (m.get(field) or []))
            vals = []
            for v, dc in counts.most_common():
                toks = tuple(query_tokens(v))
                if not toks or all(t in STOPWORDS for t in toks) or (len(toks) == 1 and len(toks[0]) < 3):
                    continue
                vals.append((v, toks, dc))
            if field == "locations_all":
                # Other spellings ("orissa") match as their place, with its article count
                dc_of = {v: dc for v, _, dc in vals}
                vals += [(canon, tuple(query_tokens(alias)), dc_of.get(canon, 0))
                         for alias, canon in (location_aliases or {}).items() if canon in dc_of]
            self.values[field] = vals
        df: Counter = Counter()
        for text in headline_text.values():
            df.update(set(query_tokens(text)))
        self.common = {t for t, c in df.items() if c / self.n > COMMON_WORD_SHARE}


def _contains(seq: List[str], sub: tuple) -> List[int]:
    """Start positions where `sub` occurs in `seq` as consecutive tokens."""
    k = len(sub)
    return [i for i in range(len(seq) - k + 1) if tuple(seq[i:i + k]) == sub]


def _is_part(words: List[str], name: tuple) -> bool:
    """`words` are consecutive words of `name`, but not all of it (a full name is a phrase match)."""
    n = len(words)
    return 0 < n < len(name) and any(list(name[i:i + n]) == words for i in range(len(name) - n + 1))


def _filter(field: str, values: List[str]) -> str:
    return f"{field}:=[{','.join('`' + v.replace('`', '') + '`' for v in values)}]"


def detect_entities(query: str, gaz: Gazetteer, max_per_field: int = 3) -> Dict[str, Any]:
    toks = query_tokens(query)
    matches: Dict[str, List[str]] = {}
    conf: Dict[str, int] = {}
    hard: Dict[str, List[str]] = {}
    strip: Set[int] = set()
    name_query = False

    for field, key in FIELDS.items():
        got, strong = [], []
        for v, vt, dc in gaz.values[field]:
            if len(got) >= max_per_field:
                break
            pos = _contains(toks, vt)
            if not pos:
                continue
            got.append(v)
            broad = dc / gaz.n > BROAD_SHARE
            # Single-word author names collide with ordinary words; tags/categories only boost
            is_strong = (key == "contributors_norm" and len(vt) >= 2) or (key == "locations_norm" and not broad)
            if is_strong:
                strong.append(v)
                for p in pos:
                    strip.update(range(p, p + len(vt)))
        if key == "contributors_norm" and not got and len(toks) == 1 and len(toks[0]) >= 4 \
                and toks[0] not in STOPWORDS and toks[0] not in gaz.common:
            partial = [v for v, vt, _ in gaz.values[field] if toks[0] in vt]
            if 1 <= len(partial) <= max_per_field:
                got, name_query = partial, True
        # Several words that are consecutive words of an author's longer name ("deep jyoti" ->
        # Deep Jyoti Sonu Brahma); not when every word is common in headlines ("india
        # development" is a topic, not the author India Development Review)
        if key == "contributors_norm" and not got and len(toks) >= 2 \
                and not any(t in STOPWORDS for t in toks) and not all(t in gaz.common for t in toks):
            partial = [v for v, vt, _ in gaz.values[field] if _is_part(toks, vt)]
            if 1 <= len(partial) <= max_per_field:
                got, name_query = partial, True
        if got:
            matches[key] = got
            conf[key] = 2 * len(got)
        if strong and key in FILTER_FIELDS:
            hard[key] = strong

    filters = [_filter(FILTER_FIELDS[k], v) for k, v in hard.items()]
    return {
        "matches": matches,
        "confidence": conf,
        "hard": hard,
        "filter_by_auto": " && ".join(filters) if filters else None,
        "strip_tokens": [toks[i] for i in sorted(strip)],
        "name_query": name_query,
    }


_POST_TYPE_WORDS = [
    (re.compile(r"\bpodcasts?\b", re.IGNORECASE), "podcast"),
    (re.compile(r"\bground[\s-]?up( stor(y|ies))?\b", re.IGNORECASE), "ground_up_story"),
]


def detect_post_type(query: str) -> tuple:
    """("podcast on climate") -> (["podcast"], "on climate"): content-type words become a filter."""
    found = []
    for rx, pt in _POST_TYPE_WORDS:
        if rx.search(query):
            found.append(pt)
            query = rx.sub(" ", query)
    rest = [t for t in query.split() if t.lower() not in STOPWORDS]
    return found, (" ".join(query.split()) if rest else "")
