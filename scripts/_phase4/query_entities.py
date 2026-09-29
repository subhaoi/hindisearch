from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from scripts.utils import query_tokens, text_to_key, fold_devanagari, HINDI_STOPWORDS, ROMAN_STOPWORD_KEYS


ENTITY_FIELDS = ["locations_norm", "contributors_norm", "categories_norm", "tags_norm"]

# Fields whose phrase matches may become hard filters. Tags/categories are incomplete
# labels, so they only ever boost (see ranker_v2).
HARD_FILTER_FIELDS = ["locations_norm", "contributors_norm"]

# A location tagged on more than this share of the corpus (e.g. भारत, ~50%) is too broad
# to filter on; it is used as a ranking boost instead.
BROAD_LOCATION_SHARE = 0.20


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
    query_full: str,
    gazetteer: Dict[str, Any],
    max_per_field: int = 3,
) -> Dict[str, Any]:
    """
    Match the query against gazetteer values (locations, contributors, categories, tags).

    A value matches when its text appears in the query as a whole-word phrase, either
    literally (Devanagari/Latin) or via phonetic match keys, so "bihar", "बिहार" and
    "uttar pradesh" all hit the Devanagari metadata. Multi-word names whose words all
    appear (any order) are weaker "token" matches.

    Returns:
      {
        matches:         {field: [values...]}     all matches (ranking boosts)
        phrase_matches:  {field: [values...]}
        confidence:      {field: int}             +2 per phrase match, +1 per token match
        hard:            {field: [values...]}     strong matches used to filter
        filter_by_auto:  str|None                 Typesense filter built from `hard`
        strip_tokens:    [query tokens]           tokens covered by `hard` matches
      }
    """
    # Folded so ज़/ज and ँ/ं spellings match the same gazetteer value
    toks = [fold_devanagari(t) for t in query_tokens(query_full)]
    tok_keys = [text_to_key(t) for t in toks]
    q_raw = " " + " ".join(toks) + " "
    q_key = " " + " ".join(k for k in tok_keys if k) + " "
    tok_set = set(toks)
    key_set = {k for k in tok_keys if k}
    corpus_size = int(gazetteer.get("corpus_size") or 0)

    matches: Dict[str, List[str]] = {}
    phrase_matches: Dict[str, List[str]] = {}
    conf: Dict[str, int] = {}
    hard: Dict[str, List[str]] = {}
    strip: Set[int] = set()

    for field in ENTITY_FIELDS:
        g = gazetteer.get(field) or {}
        values = g.get("values") or []
        texts = g.get("match_text") or values
        keys_list = g.get("keys") or [[] for _ in values]
        counts = g.get("doc_count") or [0 for _ in values]

        got: List[str] = []
        phrase: List[str] = []
        strong: List[str] = []
        score = 0

        for v, mt, keys, dc in zip(values, texts, keys_list, counts):
            if len(got) >= max_per_field:
                break
            mt_toks = [fold_devanagari(t) for t in query_tokens(mt)]
            if not mt_toks:
                continue
            mt_norm = " ".join(mt_toks)

            is_phrase = f" {mt_norm} " in q_raw or any(
                len(k) >= 3 and f" {k} " in q_key for k in keys
            )
            if is_phrase:
                got.append(v)
                phrase.append(v)
                score += 2

                if field == "locations_norm":
                    broad = corpus_size > 0 and dc / corpus_size > BROAD_LOCATION_SHARE
                    is_strong = not broad
                elif field == "contributors_norm":
                    # Single-word names (शांति, रवि) collide with ordinary words
                    is_strong = len(mt_toks) >= 2
                else:
                    is_strong = False
                if is_strong:
                    strong.append(v)
                    val_keys = {kt for k in keys for kt in k.split()}
                    for i, (t, tk) in enumerate(zip(toks, tok_keys)):
                        if t in mt_toks or tk in val_keys:
                            strip.add(i)
                continue

            # Token fallback: every content word of a multi-word name present, any order
            # ("jammu kashmir" -> जम्मू और कश्मीर)
            content = {t for t in mt_toks if t not in HINDI_STOPWORDS}
            k0 = {k for k in (keys[0].split() if keys else []) if k not in ROMAN_STOPWORD_KEYS}
            if len(mt_toks) >= 2 and ((len(content) >= 2 and content <= tok_set) or (len(k0) >= 2 and k0 <= key_set)):
                got.append(v)
                score += 1
                broad = corpus_size > 0 and dc / corpus_size > BROAD_LOCATION_SHARE
                if field == "locations_norm" and not broad:
                    strong.append(v)
                    for i, (t, tk) in enumerate(zip(toks, tok_keys)):
                        if t in content or tk in k0:
                            strip.add(i)

        if got:
            matches[field] = got
            conf[field] = score
        if phrase:
            phrase_matches[field] = phrase
        if strong and field in HARD_FILTER_FIELDS:
            hard[field] = strong

    # Locations filter on locations_all (tags + mentions in the text); tags alone miss many articles
    filter_field = {"locations_norm": "locations_all", "contributors_norm": "contributors_norm"}
    filters = [f for f in (_build_in_filter(filter_field[fld], hard[fld]) for fld in HARD_FILTER_FIELDS if fld in hard) if f]

    return {
        "matches": matches,
        "phrase_matches": phrase_matches,
        "confidence": conf,
        "hard": hard,
        "filter_by_auto": " && ".join(filters) if filters else None,
        "strip_tokens": [toks[i] for i in sorted(strip)],
    }
