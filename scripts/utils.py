## `scripts/utils.py`

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ftfy
import pandas as pd
import regex as reg
from bs4 import BeautifulSoup
from dateutil import parser as dateparser


# --------- I/O helpers ---------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)

def is_nullish(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


# --------- Text cleaning / normalization ---------

_SHORTCODE_RE = re.compile(r"\[[^\]]+\]")  # crude: strips WP shortcodes like [caption], [gallery], etc.
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")


def strip_wp_html_to_text(html: str) -> Tuple[str, Dict[str, Any]]:
    """
    Convert WordPress HTML to readable plain text while preserving paragraph boundaries.
    Returns (text, stats).
    """
    if html is None:
        return "", {"ok": True, "reason": "null"}

    original_len = len(html)

    # Remove script/style blocks first (if present as literal HTML)
    cleaned = _SCRIPT_STYLE_RE.sub(" ", html)

    # Remove WP shortcodes
    cleaned = _SHORTCODE_RE.sub(" ", cleaned)

    # Parse HTML
    try:
        soup = BeautifulSoup(cleaned, "lxml")
        # Replace <br> with newlines
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Convert list items to newline-prefixed bullets
        for li in soup.find_all("li"):
            li.insert_before("\n- ")

        # Ensure paragraphs and headings break lines
        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6"]):
            tag.insert_before("\n\n")

        text = soup.get_text(separator=" ", strip=False)
        ok = True
        reason = "parsed"
    except Exception as e:
        # fallback: brutal strip of tags
        text = re.sub(r"(?s)<[^>]+>", " ", cleaned)
        ok = False
        reason = f"fallback_strip_tags: {type(e).__name__}"

    # Normalize whitespace but preserve paragraphs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    stats = {
        "ok": ok,
        "reason": reason,
        "original_len": original_len,
        "clean_len": len(text),
    }
    return text, stats


def normalize_devanagari_text(s: Optional[str]) -> Optional[str]:
    """
    Conservative normalization: Unicode normalization, ftfy fixes, whitespace normalization.
    Does not translate or alter meaning.
    """
    if is_nullish(s):
        return None

    s2 = ftfy.fix_text(str(s))
    s2 = unicodedata.normalize("NFKC", s2)

    # Remove zero-width chars
    s2 = s2.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")

    # Normalize whitespace
    s2 = s2.replace("\r\n", "\n").replace("\r", "\n")
    s2 = re.sub(r"[ \t]+", " ", s2)
    s2 = re.sub(r"\n{3,}", "\n\n", s2).strip()

    # Normalize common punctuation spacing
    s2 = re.sub(r"\s+([।,;:!?])", r"\1", s2)
    s2 = re.sub(r"([।,;:!?])([^\s\n])", r"\1 \2", s2)

    return s2


DEVANAGARI_RE = reg.compile(r"\p{Devanagari}")
LATIN_RE = reg.compile(r"\p{Latin}")

def script_stats(s: Optional[str]) -> Dict[str, Any]:
    if is_nullish(s):
        return {"len": 0, "dev_pct": 0.0, "latin_pct": 0.0}
    t = str(s)
    n = len(t)
    if n == 0:
        return {"len": 0, "dev_pct": 0.0, "latin_pct": 0.0}
    dev = len(DEVANAGARI_RE.findall(t))
    lat = len(LATIN_RE.findall(t))
    return {"len": n, "dev_pct": dev / n, "latin_pct": lat / n}


# --------- Multi-value parsing ---------

def split_pipe_field(val: Any) -> List[str]:
    """
    Split a pipe-separated field into a list of strings.
    """
    if is_nullish(val):
        return []
    s = str(val)
    parts = [p.strip() for p in s.split("|")]
    return [p for p in parts if p != ""]


def normalize_token_list(tokens: List[str]) -> List[str]:
    """
    Reversible normalization for matching: lowercase + unicode normalize + strip.
    """
    out: List[str] = []
    for t in tokens:
        t2 = ftfy.fix_text(t)
        t2 = unicodedata.normalize("NFKC", t2)
        t2 = t2.strip().lower()
        if t2:
            out.append(t2)
    return out


# --------- Date parsing ---------

def parse_date_to_iso(val: Any) -> Optional[str]:
    """
    Parse the 'Date' column into ISO-8601 date-time string when possible.
    Keeps None if missing/unparseable; logs will capture unparseable values.
    """
    if is_nullish(val):
        return None
    s = str(val).strip()
    try:
        dt = dateparser.parse(s, fuzzy=True)
        if dt is None:
            return None
        # Use full ISO string; keep timezone if present, else naive ISO
        return dt.isoformat()
    except Exception:
        return None


@dataclass
class Paths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def stage(self, name: str) -> Path:
        return self.data / name

# --------- Typesense helpers ---------

def iso_to_epoch_seconds(iso: Optional[str]) -> int:
    """
    Convert ISO string -> epoch seconds. Returns 0 if missing/unparseable.
    Typesense sorting needs numeric.
    """
    if is_nullish(iso):
        return 0
    try:
        dt = dateparser.parse(str(iso), fuzzy=True)
        if dt is None:
            return 0
        # If timezone-naive, treat as UTC-like ordering; only relative ordering matters here.
        return int(dt.timestamp())
    except Exception:
        return 0


# --------- Romanization for indexing (Devanagari -> Roman) ---------

def devanagari_to_roman_hk(s: Optional[str]) -> str:
    """
    Deterministic Devanagari -> Roman (Harvard-Kyoto) using indic-transliteration.
    Then normalize for matching.
    """
    if is_nullish(s):
        return ""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        roman = transliterate(str(s), sanscript.DEVANAGARI, sanscript.HK)
    except Exception:
        # Fallback: return raw; still normalized below
        roman = str(s)

    return roman_normalize_for_index(roman)


_ROMAN_SPACE_RE = re.compile(r"\s+")
_ROMAN_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")

def roman_normalize_for_index(s: Optional[str]) -> str:
    """
    Roman normalization used for indexed roman fields.
    Goal: collapse user spelling variance into stable forms.
    """
    if is_nullish(s):
        return ""
    t = str(s).lower().strip()
    t = _ROMAN_NON_ALNUM_RE.sub(" ", t)
    t = _ROMAN_SPACE_RE.sub(" ", t).strip()

    # Collapse repeated vowels
    t = re.sub(r"a{2,}", "a", t)
    t = re.sub(r"i{2,}", "i", t)
    t = re.sub(r"u{2,}", "u", t)
    t = re.sub(r"e{2,}", "e", t)
    t = re.sub(r"o{2,}", "o", t)

    # Common roman drift rules (conservative)
    t = t.replace("v", "w")
    t = re.sub(r"\b(yojna|yojana|yojnaa)\b", "yojana", t)

    t = _ROMAN_SPACE_RE.sub(" ", t).strip()
    return t


def is_query_devanagari(q: str) -> bool:
    ss = script_stats(q)
    return ss["dev_pct"] > 0.02


def canonicalize_query_for_search(raw_query: str) -> dict:
    """
    For Phase 2 routing only:
    - If Devanagari: use normalized Hindi query.
    - Else: use roman-normalized query.
    """
    raw = "" if is_nullish(raw_query) else str(raw_query)
    if is_query_devanagari(raw):
        dev = normalize_devanagari_text(raw) or raw
        return {"raw": raw, "mode": "dev", "q": dev, "roman_norm": ""}
    roman_norm = roman_normalize_for_index(raw)
    return {"raw": raw, "mode": "roman", "q": roman_norm, "roman_norm": roman_norm}
