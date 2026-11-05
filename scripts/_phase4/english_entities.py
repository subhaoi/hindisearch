"""
Names in English queries: authors, locations, categories and tags of the English site, found
as whole-word phrases. Same output shape as query_entities.detect_entities (Hindi), so the
ranker and filters work unchanged. Field keys keep the Hindi names (contributors_norm, ...):
they are the ranker's feature names; the values are the English metadata as written.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from scripts.utils import query_tokens as _query_tokens

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

# Regions people search by: the states (and tag) an article must be tagged with to count
REGIONS = [
    (re.compile(r"\b(?:north\s*-?\s*east(?:ern)?(?:\s+india)?|ne\s+india|seven\s+sisters)\b", re.IGNORECASE),
     {"Arunachal Pradesh", "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Sikkim", "Tripura", "Northeast India"}),
]


def detect_region(query: str) -> tuple:
    """("education status in northeast india") -> ({states...}, "education status in")."""
    for rx, places in REGIONS:
        m = rx.search(query)
        if m:
            return places, " ".join(rx.sub(" ", query, count=1).split())
    return None, query

# Bylines that are IDR itself: not a person to filter on or show a card for
HOUSE_NAMES = {"India Development Review", "IDR"}


def unaccent(text: str) -> str:
    """"Drèze" -> "Dreze", so a name typed without accents still matches."""
    return "".join(c for c in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(c))


def query_tokens(text: str) -> List[str]:
    return _query_tokens(unaccent(text))


class Gazetteer:
    def __init__(self, articles_meta: Dict[str, Dict[str, Any]], headline_text: Dict[str, str],
                 location_aliases: Optional[Dict[str, str]] = None) -> None:
        self.n = max(1, len(articles_meta))
        self.values: Dict[str, List[tuple]] = {}
        for field in FIELDS:
            counts = Counter(v for m in articles_meta.values() for v in (m.get(field) or []))
            vals = []
            for v, dc in counts.most_common():
                if field == "contributors" and v in HOUSE_NAMES:
                    continue
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
    single_name = None

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
                # A rare word that is part of exactly one person's name ("kuvethilu"): that person
                if len(partial) == 1 and len(toks[0]) >= 5:
                    single_name = partial[0]
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
        "single_name": single_name,
    }


# How people ask rather than what they ask about: "articles about tamil nadu", "show me ...",
# "what has idr published on ...". Left in, "articles" becomes the whole meaning of the search
# once the place turns into a filter, and "about" matches every "Read more about ...".
_REQUEST_PHRASES = [
    re.compile(r"^what\s+(?:has|have|did|does)\s+idr\s+(?:published|publish|written|write|covered|cover|said)"
               r"(?:\s+(?:on|about|by|regarding))?\s+", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?(?:show\s+me|find\s+me|search\s+for|looking\s+for|i\s+(?:want|need)\s+to\s+"
               r"(?:read|find|see)(?:\s+about)?)\s+", re.IGNORECASE),
    re.compile(r"^(?:(?:all|any|some|the|idr'?s?)\s+)*(?:articles?|stories|pieces?|posts?|content|anything|something|"
               r"writings?|reads?|resources?)\s+(?:about|on|regarding|related\s+to|relating\s+to|around|covering|"
               r"of|in|from|by|for|mentioning)\s+", re.IGNORECASE),
]


# What a search can be narrowed to besides the content type: a series or format by its category
# ("Features>IDR Interviews" -> "IDR Interviews") or podcast series ("On the Contrary>Season 1").
# Humour is one too: asked for, it is what the person wants; otherwise it is ranked lower.
SERIES = [
    (re.compile(r"\bidr\s+interviews?\b|\binterviews?\s+with\b|\binterviews?\s*$", re.IGNORECASE), "IDR Interviews"),
    (re.compile(r"\bidr\s+explains?\b|\bexplainers?\b", re.IGNORECASE), "IDR Explains"),
    (re.compile(r"\bphoto\s*-?\s*(?:essays?|stor(?:y|ies))\b", re.IGNORECASE), "Photo Essays"),
    (re.compile(r"\ba\s+day\s+in\s+the\s+life(?:\s+of)?\b", re.IGNORECASE), "A day in the life of"),
    (re.compile(r"\bfailure\s+files\b", re.IGNORECASE), "Failure Files"),
    (re.compile(r"\bon\s+the\s+contrary\b", re.IGNORECASE), "On the Contrary"),
    (re.compile(r"\bwell-?\s?being\s+series\b", re.IGNORECASE), "The Well-being Series"),
    (re.compile(r"\b(?:humou?r|humorous|funny|memes?|cartoons?|comics?|satire|jokes?)\b", re.IGNORECASE), "Humour"),
]

# "latest articles on climate": newest first, searched as "articles on climate"
_RECENT_RE = re.compile(r"\b(?:latest|recent|recently|newest|most\s+recent|this\s+(?:week|month|year))\b", re.IGNORECASE)

# Words that only say "content": left over around a name or place they mean nothing
FILLER = {"articles", "article", "stories", "story", "pieces", "piece", "posts", "post", "content", "news",
          "updates", "writings", "writing"}


def detect_series(query: str) -> tuple:
    """("idr interviews with farmers") -> (["IDR Interviews"], "farmers"): series words become a filter."""
    found = []
    for rx, name in SERIES:
        if rx.search(query):
            found.append(name)
            query = rx.sub(" ", query)
    rest = [t for t in query.split() if t.lower() not in STOPWORDS]
    return found, (" ".join(query.split()) if rest else "")


def detect_recent(query: str) -> tuple:
    """("latest articles on climate") -> (True, "articles on climate")."""
    rest = " ".join(_RECENT_RE.sub(" ", query).split())
    return (True, rest) if rest != " ".join(query.split()) else (False, query)


def is_filler(word: str) -> bool:
    w = word.lower().strip(".,;:!?\"'()")
    return w in STOPWORDS or w in FILLER or w in ("'s", "’s")


def strip_request(query: str) -> str:
    """"articles about tamil nadu" -> "tamil nadu"; unchanged when nothing would be left."""
    q = " ".join(query.split())
    for rx in _REQUEST_PHRASES:
        rest = rx.sub("", q, count=1)
        if rest.strip():
            q = rest
    return q


def keyword_text(text: str, named: bool = False) -> str:
    """
    The search without STOPWORDS for the keyword index ("about" is in half the site); as is if only
    those. With a name or place found, FILLER goes too ("rachita vora articles" -> "rachita vora").
    """
    words = [w for w in text.split() if not (is_filler(w) if named else w.lower().strip(".,;:!?\"'()") in STOPWORDS)]
    return " ".join(words) if words else text


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
