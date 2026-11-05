"""
English site (idronline.org): how its articles become search text, chunks and metadata.
Shared by the English pipeline (scripts/en/), the GPU evaluation and the English search API,
so what gets indexed and what the API reasons about can never drift apart.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from scripts.utils import is_nullish, iso_to_epoch_seconds

CHUNK_WORDS = 220

# Locations tagged on more than this share of articles ("India") are too broad to filter on
BROAD_LOCATION_SHARE = 0.20

# Shared with the Hindi site: Devanagari place -> other spellings, English ones among them
LOCATION_ALIASES = Path(__file__).resolve().parents[1] / "config" / "location_aliases.json"


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


# Category pages on idronline.org: /<parent>/<slug>/ ("Sectors>Rights" -> /sectors/rights/), or /<slug>/
# for a top-level one ("Humour" -> /humour/). Channel and admin labels have no page worth linking.
CATEGORY_NO_PAGE = {"Instagram", "Uncategorized", "IDR Select", "Commissioned content"}
# Where the site's slug isn't the name made lowercase and hyphenated
CATEGORY_SLUGS = {"Health & Well-being": "health", "Fundraising & Communications": "fundraising-and-communications"}


def category_slug(name: str) -> str:
    n = leaf(name)
    return CATEGORY_SLUGS.get(n) or re.sub(r"[^a-z0-9]+", "-", n.lower().replace("'", "").replace("&", "")).strip("-")


def category_url(category: str) -> Optional[str]:
    """Page for a category path as in the exports ("Sectors>Rights"); None when it has none."""
    if not category or leaf(category) in CATEGORY_NO_PAGE:
        return None
    parent = category.split(">")[0].strip() if ">" in category else ""
    return "https://idronline.org/" + (f"{category_slug(parent)}/" if parent else "") + f"{category_slug(category)}/"


def contributor_url(name: str) -> str:
    """
    Profile page on idronline.org, slug made the way WordPress makes it: accents dropped (Drèze ->
    dreze), apostrophes removed (D’Souza -> dsouza), everything else a hyphen (J.V. -> j-v).
    """
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"['’‘]", "", plain)).strip("-")
    return f"https://idronline.org/contributor/{slug}/"


def primary_category_path(categories: List[str], url: Any = None) -> Optional[str]:
    """
    The site's own primary category, read from the address (/article/<slug>/..., /features/<slug>/...);
    otherwise the first sector/theme ("Sectors>Livelihoods"), channel labels like "Instagram" last.
    """
    m = re.match(r"https?://[^/]+/(?:article|features)/([^/]+)/[^/]+", s(url))
    if m:
        for c in categories:
            if category_slug(c) == m.group(1):
                return c
    topical = [c for c in categories if c.split(">")[0] in ("Sectors", "Themes")]
    return (topical or categories or [None])[0]


def primary_category(categories: List[str], url: Any = None) -> Optional[str]:
    pick = primary_category_path(categories, url)
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


def english_location_aliases(places: Iterable[str], path: Path = LOCATION_ALIASES) -> Dict[str, str]:
    """Other English spellings of the site's places from the shared alias file:
    {"orissa": "Odisha", "kashmir": "Jammu and Kashmir", ...}."""
    if not path.exists():
        return {}
    by_name = {p.casefold(): p for p in places}
    out: Dict[str, str] = {}
    for names in json.loads(path.read_text(encoding="utf-8")).get("aliases", {}).values():
        latin = [n for n in names if n.isascii()]
        canon = next((by_name[n.casefold()] for n in latin if n.casefold() in by_name), None)
        if canon:
            out.update({n.casefold(): canon for n in latin if n.casefold() != canon.casefold()})
    return out


class LocationIndex:
    """Tagged location names of the corpus and their other spellings, to find them in article
    text and in queries (same in index and API)."""

    def __init__(self, articles: pd.DataFrame) -> None:
        self.counts = location_names(articles)
        self.aliases = english_location_aliases(self.counts)
        self.canonical = {**{n.casefold(): n for n in self.counts}, **self.aliases}
        self.pattern = location_pattern(list(self.counts) + list(self.aliases))

    def for_article(self, row: Any) -> List[str]:
        text = " ".join([s(row["title"]), s(row["summary"]), s(row["content"])])
        return locations_all(leaves(as_list(row["locations"])), text, self.pattern, self.canonical)
