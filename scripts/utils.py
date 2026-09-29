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


_TITLE_SITE_SUFFIX_RE = re.compile(r"\s*[\-–]\s*(आईडीआर|आईडीआऱ|IDR)\s*$", re.IGNORECASE)
# Trailing " | ..." segments that are site/section labels, not part of the title
_TITLE_LABEL_SEGMENTS = {"आईडीआर", "आईडीआऱ", "idr", "हल्का-फुल्का", ""}


def clean_title(t: Optional[str]) -> str:
    """Drop site/section suffixes: "शीर्षक | हल्का-फुल्का | आईडीआर" -> "शीर्षक"."""
    if is_nullish(t):
        return ""
    parts = [p.strip() for p in str(t).replace("\u200b", "").split("|")]
    while len(parts) > 1 and parts[-1].lower() in _TITLE_LABEL_SEGMENTS:
        parts.pop()
    return _TITLE_SITE_SUFFIX_RE.sub("", " | ".join(parts)).strip()


# --------- Devanagari folding + light stemming (lexical matching only) ---------

_NUKTA_PRECOMPOSED = {chr(0x0958 + i): b for i, b in enumerate("कखगजडढफय")}


def fold_devanagari(s: str) -> str:
    """
    Collapse spelling variants that differ only by nukta or chandrabindu, so
    ज़रूरत/जरूरत and गाँव/गांव index and match as one word. Lexical fields only.
    """
    s = "".join(_NUKTA_PRECOMPOSED.get(ch, ch) for ch in s)
    return s.replace("\u093c", "").replace("ँ", "ं")


# Noun/adjective inflections, longest first (light stemmer after Ramanathan & Rao, 2003)
# Nasal consonant + virama before a consonant is the same sound as anusvara: आन्दोलन = आंदोलन
_NASAL_CONJUNCT_RE = re.compile("[ङञणनम]\u094d(?=[\u0915-\u0939])")
# Long/short i and u are often confused in typing (घरेलु/घरेलू, जाती/जाति)
_VOWEL_LENGTH = str.maketrans({"ी": "ि", "ू": "ु", "ई": "इ", "ऊ": "उ"})


def loose_fold(word: str) -> str:
    """Stronger folding for the stemmed fields only; exact fields keep the spelling."""
    return _NASAL_CONJUNCT_RE.sub("ं", fold_devanagari(word)).translate(_VOWEL_LENGTH)


# Noun/adjective inflections after loose_fold, longest first (light stemmer after Ramanathan & Rao, 2003)
_HINDI_SUFFIXES = sorted(
    ["ियों", "ियां", "ाओं", "ाएं", "ुओं", "ुएं", "ओं", "एं", "ों", "ें", "ां", "िं", "े", "ा", "ो", "ि", "ु"],
    key=len,
    reverse=True,
)


def hindi_stem(word: str) -> str:
    """बच्चा/बच्चे/बच्चों -> बच्च, महिला/महिलाओं/महिलाएं -> महिल, आन्दोलन/आंदोलन -> आंदोलन."""
    word = loose_fold(word)
    for suf in _HINDI_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


def stem_text(text: Optional[str]) -> str:
    """Folded + stemmed Devanagari tokens (Latin tokens kept as-is), for the *_stem index fields."""
    if is_nullish(text):
        return ""
    return " ".join(
        hindi_stem(fold_devanagari(t)) if DEVANAGARI_RE.search(t) else t for t in query_tokens(str(text))
    )


# --------- Locations mentioned in an article (tags are incomplete) ---------

def location_patterns(values: List[str], aliases: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    """[(location value, [folded token strings to look for])] from gazetteer values + aliases."""
    out: List[Tuple[str, List[str]]] = []
    for v in values:
        pats = []
        for name in [v] + list(aliases.get(v, [])):
            p = " ".join(fold_devanagari(t) for t in query_tokens(name))
            if p and p not in pats:
                pats.append(p)
        out.append((v, pats))
    return out


def derive_locations(
    title: Optional[str],
    summary: Optional[str],
    content: Optional[str],
    tagged: List[str],
    patterns: List[Tuple[str, List[str]]],
    min_content_mentions: int = 3,
) -> List[str]:
    """
    Tagged locations plus any location named in the title/summary, or at least
    `min_content_mentions` times in the body. Many articles about a state are only
    tagged भारत (e.g. 26 untagged articles mention असम 3+ times vs 20 tagged).
    Two body mentions are often passing references; three or more usually are not.
    """
    def norm(t: Optional[str]) -> str:
        return " " + " ".join(fold_devanagari(x) for x in query_tokens(t or "")) + " "

    head = norm(f"{title or ''} {summary or ''}")
    body = norm(content)
    out = list(dict.fromkeys(tagged))
    for value, pats in patterns:
        if value in out:
            continue
        if any(f" {p} " in head for p in pats) or sum(body.count(f" {p} ") for p in pats) >= min_content_mentions:
            out.append(value)
    return out


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


# --------- Romanization (Devanagari -> colloquial Roman) + phonetic match keys ---------
#
# Roman Typesense fields, roman queries and gazetteer entries are all reduced to a
# "match key": a lossy phonetic form that collapses common spelling variants
# (shiksha/siksha, kisaan/kisan, panchaayat/panchayat). Devanagari is romanized the
# way people type Hindi (श→sh, च→ch, ं→n, schwa deletion), not with a scholarly
# scheme like Harvard-Kyoto (which gives zikSA, kisAna, paMcAyata).

_DEV_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # precomposed nukta forms (U+0958..U+095F)
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z",
    "ड़": "d", "ढ़": "dh", "फ़": "f", "य़": "y",
}
_NUKTA = "़"
_VIRAMA = "्"
# ड़/ढ़ are usually typed d/dh (ladki, padhai)
_NUKTA_MAP = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f", "य": "y"}
_DEV_IND_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ॠ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e", "ऎ": "e", "ऒ": "o", "ॐ": "om",
}
_DEV_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "ॄ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o",
}
_DEV_CODA = {"ं": "M", "ँ": "n", "ः": "h"}  # "M" resolved in _syllables_to_roman
_DEV_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}


def _syllables_to_roman(syls: List[List[Any]], schwa: int) -> str:
    """
    syls: [consonant_cluster, vowel, coda, has_inherent_a, is_conjunct]
    schwa: 0 = keep every inherent 'a', 1 = drop word-final, 2 = also drop medial (VC_CV rule).
    """
    n = len(syls)
    if n == 0:
        return ""
    if schwa >= 1 and n > 1:
        last = syls[-1]
        # Keep the final 'a' after a conjunct (स्वास्थ्य -> swasthya)
        if last[3] and not last[4] and not last[2]:
            last[1] = ""
    if schwa >= 2:
        for i in range(n - 2, 0, -1):
            s, prev, nxt = syls[i], syls[i - 1], syls[i + 1]
            # Not before a conjunct: परिवर्तन -> parivartan, not parivrtan
            if s[3] and s[1] == "a" and not s[4] and not s[2] and prev[1] and nxt[0] and nxt[1] and not nxt[4]:
                s[1] = ""
    out: List[str] = []
    for i, (cons, vowel, coda, inherent, _) in enumerate(syls):
        # "M" marks anusvara: word-final after inherent 'a' it is typed 'm' (स्वयं -> swayam)
        coda = coda.replace("M", "m" if (i == n - 1 and inherent and vowel == "a") else "n")
        out.append(cons + vowel + coda)
    return "".join(out)


def devanagari_to_roman(s: Optional[str], schwa: int = 2) -> str:
    """
    Devanagari -> Roman as Hindi is commonly typed (किसान -> kisaan, शिक्षा -> shikshaa,
    सरकार -> sarkaar). Non-Devanagari text passes through unchanged.
    """
    if is_nullish(s):
        return ""
    text = str(s)
    out: List[str] = []
    syls: List[List[Any]] = []
    cluster = ""
    ncons = 0

    def flush() -> None:
        nonlocal syls, cluster, ncons
        if cluster:  # word ends in a virama
            syls.append([cluster, "", "", False, False])
        if syls:
            out.append(_syllables_to_roman(syls, schwa))
        syls, cluster, ncons = [], "", 0

    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        cons = _DEV_CONSONANTS.get(ch)
        if cons is not None:
            i += 1
            if i < n and text[i] == _NUKTA:
                cons = _NUKTA_MAP.get(ch, cons)
                i += 1
            if i < n and text[i] == _VIRAMA:
                cluster += cons
                ncons += 1
                i += 1
                continue
            cluster = (cluster + cons).replace("jn", "gy")  # ज्ञ -> gy
            ncons += 1
            if i < n and text[i] in _DEV_MATRAS:
                vowel, inherent = _DEV_MATRAS[text[i]], False
                i += 1
            else:
                vowel, inherent = "a", True
            coda = ""
            while i < n and text[i] in _DEV_CODA:
                coda += _DEV_CODA[text[i]]
                i += 1
            syls.append([cluster, vowel, coda, inherent, ncons > 1])
            cluster, ncons = "", 0
        elif ch in _DEV_IND_VOWELS or ch in _DEV_MATRAS:
            if cluster:
                syls.append([cluster, "", "", False, False])
                cluster, ncons = "", 0
            vowel = _DEV_IND_VOWELS.get(ch) or _DEV_MATRAS[ch]
            i += 1
            coda = ""
            while i < n and text[i] in _DEV_CODA:
                coda += _DEV_CODA[text[i]]
                i += 1
            syls.append(["", vowel, coda, False, False])
        elif ch in _DEV_CODA:
            if syls:
                syls[-1][2] += _DEV_CODA[ch]
            i += 1
        elif ch in (_NUKTA, _VIRAMA, "‌", "‍"):
            i += 1
        else:
            flush()
            out.append(_DEV_DIGITS.get(ch, " " if ch in "।॥" else ch))
            i += 1
    flush()
    return "".join(out)


_ROMAN_SPACE_RE = re.compile(r"\s+")
_ROMAN_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_ROMAN_KEY_RULES = [
    (re.compile(r"chh"), "ch"),
    (re.compile(r"c+h?c*h"), "ch"),  # bachcha/bacha/bachha/baccha
    (re.compile(r"x"), "ks"),
    (re.compile(r"sh"), "s"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"w"), "v"),
    (re.compile(r"z"), "j"),
    (re.compile(r"q"), "k"),
    (re.compile(r"ee"), "i"),
    (re.compile(r"oo"), "u"),
    (re.compile(r"ou"), "au"),
    # ai/e and au/o are typed interchangeably (kaise/kese, yaun/yon, aur/or);
    # collapse long vowels first so भाई (bhaaee) and "bhai" meet
    (re.compile(r"([aeiou])\1+"), r"\1"),
    (re.compile(r"ai"), "e"),
    (re.compile(r"au"), "o"),
    # Anusvara before a consonant is typed m or n (samvidhan/sanvidhan, sampark/sanpark)
    (re.compile(r"m(?=[bcdfgjklnpqrstvwxz])"), "n"),
    (re.compile(r"([a-z])\1+"), r"\1"),
]


def roman_key(s: Optional[str]) -> str:
    """
    Lossy phonetic key for Roman text, applied identically to indexed text and queries:
    shiksha/siksha -> siksa, kisaan/kisan -> kisan, swasthya/svasthya -> svasthya.
    """
    if is_nullish(s):
        return ""
    t = unicodedata.normalize("NFKD", str(s).lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = _ROMAN_NON_ALNUM_RE.sub(" ", t)
    for rx, rep in _ROMAN_KEY_RULES:
        t = rx.sub(rep, t)
    return _ROMAN_SPACE_RE.sub(" ", t).strip()


def text_to_key(s: Optional[str], schwa: int = 2) -> str:
    """Any script -> match key. Used for Typesense roman fields, queries and the gazetteer."""
    return roman_key(devanagari_to_roman(s, schwa=schwa))


def text_to_key_variants(s: Optional[str]) -> List[str]:
    """Keys with and without schwa deletion (users type both 'yojna' and 'yojana')."""
    out: List[str] = []
    for schwa in (2, 1, 0):
        k = text_to_key(s, schwa=schwa)
        if k and k not in out:
            out.append(k)
    return out


# --------- Series / formats (config/series.json) ---------

def load_series(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f).get("series", [])


def article_series(
    title: Optional[str], categories: List[str], multimedia_type: Optional[str], series: List[Dict[str, Any]]
) -> List[str]:
    """Names of the series an article belongs to (headline regex, category or media type)."""
    folded = fold_devanagari(clean_title(title))
    cats = set(categories or [])
    out = []
    for sr in series:
        hit = (
            (sr.get("title_regex") and re.search(fold_devanagari(sr["title_regex"]), folded))
            or (cats & set(sr.get("categories", [])))
            or (sr.get("multimedia_type") and str(multimedia_type or "") == sr["multimedia_type"])
        )
        if hit:
            out.append(sr["name"])
    return out


def _trigger_tokens(trigger: str) -> List[str]:
    return [fold_devanagari(t) if DEVANAGARI_RE.search(t) else roman_key(t) for t in query_tokens(trigger)]


def detect_series(query: str, series: List[Dict[str, Any]]) -> Tuple[List[str], str]:
    """
    ("फोटो निबंध जलवायु", ...) -> (["photo_essay"], "जलवायु"). Matches the longest trigger
    first; Latin triggers compare as phonetic keys so "saral kosh"/"saral kosh" variants meet.
    Returns the series names and the query with trigger words removed.
    """
    raw = query_tokens(query)
    norm = [fold_devanagari(t) if DEVANAGARI_RE.search(t) else roman_key(t) for t in raw]
    triggers = sorted(
        ((sr["name"], _trigger_tokens(t)) for sr in series for t in sr.get("triggers", [])),
        key=lambda x: -len(x[1]),
    )
    used = [False] * len(norm)
    found: List[str] = []
    for name, tt in triggers:
        n = len(tt)
        if not n:
            continue
        for i in range(len(norm) - n + 1):
            if norm[i:i + n] == tt and not any(used[i:i + n]):
                used[i:i + n] = [True] * n
                if name not in found:
                    found.append(name)
    rest = " ".join(t for t, u in zip(raw, used) if not u)
    return found, rest


# --------- Queries that are not searches ---------

_YEAR_RE = re.compile(r"^\s*((?:19|20)\d\d)\s*$")
# Exploit probes / pasted markup seen in the logs (/index/\think\app/invokefunction, @import url(...))
_PROBE_RE = re.compile(r"(://|^\s*/|\\|@import|<\s*script|\.php\b|\bunion\s+select\b)", re.IGNORECASE)
MAX_QUERY_TOKENS = 30


def classify_query(q: str) -> Tuple[str, Optional[int]]:
    """
    ("year", 2026) for a bare year, ("junk", None) for probes and strings with no letters,
    else ("search", None).
    """
    m = _YEAR_RE.match(q or "")
    if m:
        return "year", int(m.group(1))
    if _PROBE_RE.search(q or "") or not reg.search(r"\p{L}", q or ""):
        return "junk", None
    return "search", None


# --------- Query canonicalization ---------

HINDI_STOPWORDS = {
    "के", "का", "की", "को", "में", "मे", "से", "पर", "और", "है", "हैं", "था", "थे", "थी",
    "भी", "एक", "यह", "वह", "ये", "वे", "इस", "उस", "इन", "उन", "ने", "लिए", "लिये", "तो",
    "ही", "या", "क्या", "कैसे", "क्यों", "कि", "जो", "हो", "होता", "होती", "होते", "कर",
    "करना", "करने", "किया", "गया", "गई", "रहा", "रही", "रहे", "द्वारा", "तक", "अपने", "अपनी",
}
_ROMAN_HINDI_STOPWORDS = [
    "ke", "ka", "ki", "ko", "me", "mein", "main", "se", "par", "aur", "hai", "hain", "tha", "the",
    "thi", "bhi", "ek", "yah", "yeh", "ye", "vah", "voh", "wo", "is", "us", "in", "un", "ne",
    "liye", "lie", "to", "hi", "ya", "kya", "kaise", "kyon", "kyu", "jo", "ho", "kar", "karna",
    "karne", "kiya", "gaya", "dwara", "tak", "apne", "apni",
]
_ENGLISH_STOPWORDS = [
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "of", "on", "or",
    "that", "what", "with", "about", "into", "its",
]
ROMAN_STOPWORD_KEYS = {roman_key(w) for w in _ROMAN_HINDI_STOPWORDS + _ENGLISH_STOPWORDS}

_QUERY_TOKEN_RE = reg.compile(r"[\p{L}\p{M}\p{N}]+")


def query_tokens(s: Optional[str]) -> List[str]:
    if is_nullish(s):
        return []
    return _QUERY_TOKEN_RE.findall(str(s).lower())


def is_query_devanagari(q: str) -> bool:
    ss = script_stats(q)
    return ss["dev_pct"] > 0.02


def is_query_mixed(q: str) -> bool:
    ss = script_stats(q)
    return ss["dev_pct"] > 0.02 and ss["latin_pct"] > 0.02


def canonicalize_query_for_search(raw_query: str) -> dict:
    """
    Returns:
      mode:   dev | roman | mixed
      q:      lexical query for Typesense. Devanagari tokens stay as-is, Latin tokens become
              match keys (to hit *_roman_norm / *_key fields); stopwords are dropped.
      q_stem: `q` with Hindi inflections stripped (searched against the *_stem fields).
      q_full: normalized query with every token kept (entity detection uses this).
    """
    raw = "" if is_nullish(raw_query) else str(raw_query)
    if is_query_mixed(raw):
        mode = "mixed"
    elif is_query_devanagari(raw):
        mode = "dev"
    else:
        mode = "roman"

    toks = query_tokens(normalize_devanagari_text(raw) or raw)

    def lexical(tok: str) -> str:
        return fold_devanagari(tok) if DEVANAGARI_RE.search(tok) else roman_key(tok)

    def stemmed(tok: str) -> str:
        return hindi_stem(tok) if DEVANAGARI_RE.search(tok) else tok

    def is_stop(tok: str) -> bool:
        return tok in HINDI_STOPWORDS if DEVANAGARI_RE.search(tok) else roman_key(tok) in ROMAN_STOPWORD_KEYS

    lex = [lexical(t) for t in toks if not is_stop(t)]
    if not lex:  # query was only stopwords
        lex = [lexical(t) for t in toks]
    lex = [t for t in lex if t]

    return {
        "raw": raw,
        "mode": mode,
        "q": " ".join(lex),
        # Same tokens with Hindi inflections stripped, for the *_stem fields
        "q_stem": " ".join(stemmed(t) for t in lex),
        # Latin words as typed (minus stopwords), for the English title_en/summary_en fields
        "q_en": " ".join(t for t in toks if t.isascii() and roman_key(t) not in ROMAN_STOPWORD_KEYS),
        "q_full": " ".join(toks),
        "roman_norm": text_to_key(raw),
    }


# Common English words: never transliterate these into Hindi.
ENGLISH_COMMON_WORDS = set(_ENGLISH_STOPWORDS) | {
    "the", "this", "these", "those", "was", "were", "has", "have", "had", "not", "no", "can",
    "will", "who", "why", "when", "where", "which", "all", "more", "most", "new", "our", "your",
    "their", "his", "her", "we", "you", "they", "it", "do", "does", "did", "than", "then",
    "health", "education", "school", "women", "woman", "girls", "children", "child", "rural",
    "urban", "water", "climate", "change", "impact", "funding", "fund", "ngo", "ngos", "csr",
    "policy", "government", "community", "communities", "training", "workers", "worker",
    "development", "livelihood", "livelihoods", "gender", "data", "social", "sector", "india",
    "program", "programme", "scheme", "rights", "farmers", "agriculture", "nutrition",
    "sanitation", "migration", "tribal", "disability", "leadership", "philanthropy",
}


def english_glossary(synonym_groups: List[List[str]]) -> Dict[str, str]:
    """
    {latin phrase: Hindi term} from the synonym groups (config/synonyms_v1.json): every
    Latin-letter term maps to the first Devanagari term of its group, e.g.
    "labour law" -> "श्रम कानून", "fpo" -> "एफपीओ", "andolan" -> "आंदोलन".
    """
    out: Dict[str, str] = {}
    for grp in synonym_groups:
        hindi = next((t for t in grp if DEVANAGARI_RE.search(t)), None)
        if not hindi:
            continue
        for t in grp:
            if not DEVANAGARI_RE.search(t):
                key = " ".join(query_tokens(t))
                if key:
                    out.setdefault(key, hindi)
    return out


def apply_glossary(query: str, glossary: Dict[str, str]) -> str:
    """Replace known English/romanized phrases with their Hindi term, longest phrase first."""
    toks = query_tokens(query)
    if not glossary or not toks:
        return query
    max_len = max(len(k.split()) for k in glossary)
    out: List[str] = []
    i = 0
    while i < len(toks):
        for n in range(min(max_len, len(toks) - i), 0, -1):
            hit = glossary.get(" ".join(toks[i:i + n]))
            if hit:
                out.append(hit)
                i += n
                break
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def roman_query_to_devanagari(query: str, vocab: Dict[str, str], min_hit_ratio: float = 0.5) -> Optional[str]:
    """
    Transliterate the Latin tokens of a roman/mixed query into Devanagari using a
    corpus-derived vocabulary {match_key: devanagari_word} (see 21_build_translit_vocab.py).
    Returns None when fewer than `min_hit_ratio` of the Latin tokens look like romanized
    Hindi, i.e. the query is probably English and is better embedded as-is.
    """
    if not vocab or is_nullish(query):
        return None
    toks = query_tokens(query)
    latin = [t for t in toks if not DEVANAGARI_RE.search(t)]
    if not latin:
        return None
    hits = 0
    out: List[str] = []
    for t in toks:
        if DEVANAGARI_RE.search(t) or t in ENGLISH_COMMON_WORDS or len(t) < 2:
            out.append(t)
            continue
        dev = vocab.get(roman_key(t))
        if dev:
            hits += 1
            out.append(dev)
        else:
            out.append(t)
    if hits == 0 or hits / len(latin) < min_hit_ratio:
        return None
    return " ".join(out)

# --------- Phase 3: chunking + embeddings helpers ---------

def build_tokenizer_for_mpnet():
    """
    Tokenizer for intfloat/multilingual-e5-large.
    Used only for approximate chunk sizing.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")


def e5_prefix_text(text: str, kind: str) -> str:
    """
    Apply E5 recommended prefixes: "query: " or "passage: ".
    """
    t = "" if is_nullish(text) else str(text).strip()
    prefix = "query: " if kind == "query" else "passage: "
    return prefix + t


def count_tokens(tokenizer, text: str) -> int:
    if is_nullish(text):
        return 0
    return len(tokenizer.encode(str(text), add_special_tokens=False))


def safe_join(parts: List[str], sep: str = "\n\n") -> str:
    parts2 = []
    for p in parts:
        if not is_nullish(p):
            s = str(p).strip()
            if s:
                parts2.append(s)
    return sep.join(parts2)
