"""
English site (idronline.org): how its articles become search text, chunks and metadata.
Shared by the English pipeline (scripts/en/), the GPU evaluation and the English search API,
so what gets indexed and what the API reasons about can never drift apart.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from scripts.utils import is_nullish, iso_to_epoch_seconds

CHUNK_WORDS = 220

# Locations tagged on more than this share of articles ("India") are too broad to filter on
BROAD_LOCATION_SHARE = 0.20


def s(v: Any) -> str:
    return "" if is_nullish(v) else str(v).strip()


def as_list(v: Any) -> List[str]:
    """Parquet list columns load as numpy arrays."""
    if v is None or isinstance(v, (str, bytes, float, int)):
        return []
    try:
        return [str(x).strip() for x in list(v) if not is_nullish(x) and str(x).strip()]
    except TypeError:
        return []


def leaf(v: str) -> str:
    """"India>Maharashtra" -> "Maharashtra", "Sectors>Health & Well-being" -> "Health & Well-being"."""
    return v.split(">")[-1].strip()


def leaves(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(leaf(v) for v in values if leaf(v)))


def primary_category(categories: List[str]) -> Optional[str]:
    """The first sector/theme ("Sectors>Livelihoods"); channel labels like "Instagram" only as a fallback."""
    topical = [c for c in categories if c.split(">")[0] in ("Sectors", "Themes")]
    pick = (topical or categories or [None])[0]
    return leaf(pick) if pick else None


def article_text(title: Any, seo_title: Any, summary: Any) -> str:
    """What the article vector is made from: headline, SEO title, summary."""
    return "\n\n".join(x for x in [s(title), s(seo_title), s(summary)] if x)


def english_chunks(articles: pd.DataFrame, words: int = CHUNK_WORDS) -> pd.DataFrame:
    """
    Paragraph-packed chunks of about `words` words; the first starts with title + summary
    (as for Hindi), so a short post is still one complete chunk.
    """
    rows = []
    for r in articles.itertuples():
        paras = [p.strip() for p in re.split(r"\n\s*\n", s(r.content)) if p.strip()]
        head = "\n\n".join(x for x in [s(r.title), s(r.summary)] if x)
        chunks, cur, n = [], [head] if head else [], len(head.split())
        for p in paras:
            w = len(p.split())
            if cur and n + w > words:
                chunks.append("\n\n".join(cur))
                cur, n = [], 0
            cur.append(p)
            n += w
        if cur:
            chunks.append("\n\n".join(cur))
        for i, c in enumerate(chunks or [head]):
            rows.append({
                "chunk_id": f"{r.id}_{i}", "article_id": str(r.id), "chunk_index": i,
                "url": s(r.url) or None, "published_date": s(r.published_date) or None,
                "published_ts": iso_to_epoch_seconds(r.published_date), "chunk_tokens": len(c.split()),
                "chunk_text": c,
            })
    return pd.DataFrame(rows)


def location_names(articles: pd.DataFrame) -> Dict[str, int]:
    """Tagged location leaves -> number of articles tagged with them."""
    counts: Dict[str, int] = {}
    for v in articles["locations"]:
        for loc in leaves(as_list(v)):
            counts[loc] = counts.get(loc, 0) + 1
    return counts


def location_pattern(names: Iterable[str]) -> Optional[re.Pattern]:
    names = sorted({n for n in names if len(n) >= 3}, key=len, reverse=True)
    if not names:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)


def locations_all(tagged: List[str], text: str, pattern: Optional[re.Pattern], canonical: Dict[str, str]) -> List[str]:
    """Tagged locations plus known locations named in the text (many articles are tagged only "India")."""
    found = list(tagged)
    if pattern is not None:
        found += [canonical[m.group(1).casefold()] for m in pattern.finditer(text or "")]
    return list(dict.fromkeys(found))


class LocationIndex:
    """Tagged location names of the corpus, to find them in article text (same in index and API)."""

    def __init__(self, articles: pd.DataFrame) -> None:
        self.counts = location_names(articles)
        self.canonical = {n.casefold(): n for n in self.counts}
        self.pattern = location_pattern(self.counts)

    def for_article(self, row: Any) -> List[str]:
        text = " ".join([s(row["title"]), s(row["summary"]), s(row["content"])])
        return locations_all(leaves(as_list(row["locations"])), text, self.pattern, self.canonical)
