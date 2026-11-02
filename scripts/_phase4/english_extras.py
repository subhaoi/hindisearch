"""
English search helpers that work on the words of a query before searching:

  Speller     corrects misspelt words ("uneployment" -> unemployment) and author names
              ("saba koli dave" -> Saba Kohli Dave), using only the site's own words and authors,
              so IDR's vocabulary (anganwadi, MGNREGA) is never "corrected"
  QuickLinks  site pages for searches about the site itself ("contact", "who do i write to if i
              want to publish a piece"), from config/en_quick_links.json
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

WORD_RE = re.compile(r"[a-z]+")

# Words worth correcting must be this long: shorter ones are too often names or words the site
# hasn't used ("peth", "nino", "kele"), and a one-letter change makes another real word
MIN_WORD = 6
# Edits allowed: one up to this length, two beyond (checked on real searches: "decison",
# "hostory", "burntout" need one; "indegeneous" needs two; "labnya" -> "lanka" must not happen)
ONE_EDIT_UP_TO = 8
# A correction must be a word the site uses at least this often (rare words are often typos too)
MIN_COUNT = 3


def fold(text: str) -> str:
    """Lowercase ASCII: accents dropped (Drèze -> dreze), everything but letters, digits and spaces removed."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def distance(a: str, b: str, limit: int = 3) -> int:
    """Damerau-Levenshtein (optimal string alignment) distance; returns limit + 1 when larger."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: List[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def _deletes(word: str) -> Set[str]:
    return {word[:i] + word[i + 1:] for i in range(len(word))}


class Speller:
    def __init__(self, texts: Iterable[str], names: Iterable[str], stopwords: Set[str]) -> None:
        self.counts: Counter = Counter()
        for t in texts:
            self.counts.update(WORD_RE.findall(fold(t)))
        self.names = {fold(n): n for n in names if fold(n)}
        for n in self.names:
            self.counts.update(n.split())
        self.stop = stopwords
        # Each common word under itself and its one-letter deletions: a query word reaches it
        # through its own deletions (up to two), the usual way to find words within two edits fast
        self.index: Dict[str, List[str]] = defaultdict(list)
        for w, c in self.counts.items():
            if c >= MIN_COUNT and len(w) >= MIN_WORD - 1:
                for v in _deletes(w) | {w}:
                    self.index[v].append(w)
        self.names_by_shape: Dict[tuple, List[str]] = defaultdict(list)
        for n in self.names:
            toks = n.split()
            self.names_by_shape[(len(toks), toks[0][0])].append(n)

    def correct_word(self, w: str) -> Optional[str]:
        if len(w) < MIN_WORD or w in self.counts or w in self.stop:
            return None
        limit = 1 if len(w) <= ONE_EDIT_UP_TO else 2
        variants = {w} | _deletes(w)
        variants |= {d2 for d in list(variants) for d2 in _deletes(d)}
        best = None
        for c in {c for v in variants for c in self.index.get(v, [])}:
            d = distance(w, c, limit)
            if d <= limit:
                key = (d, -self.counts[c])
                if best is None or key < best[0]:
                    best = (key, c)
        return best[1] if best else None

    def correct_name(self, q: str) -> Optional[str]:
        """An author's name, spelt slightly differently: every word within one or two edits."""
        toks = q.split()
        if not 2 <= len(toks) <= 4 or q in self.names:
            return None
        best = None
        for n in self.names_by_shape.get((len(toks), toks[0][0]), []):
            ds = [distance(a, b, 2) for a, b in zip(toks, n.split())]
            if all(d <= (1 if len(a) <= 4 else 2) for d, a in zip(ds, toks)):
                total = sum(ds)
                if best is None or total < best[0]:
                    best = (total, n)
        return self.names[best[1]] if best else None

    def correct(self, query: str) -> Optional[str]:
        """The corrected query, or None when nothing needed correcting."""
        # "re-wilding" is the site's "rewilding", not "re" + "wilding"
        joined = re.sub(r"\b([a-z]+)-([a-z]+)\b",
                        lambda m: m.group(1) + m.group(2) if m.group(1) + m.group(2) in self.counts else m.group(0),
                        query.lower())
        q = fold(joined)
        if not q or not re.search(r"[a-z]", q):
            return None
        name = self.correct_name(q)
        if name:
            return name
        out, changed = [], joined != query.lower()
        for w in q.split():
            c = self.correct_word(w) if w.isalpha() else None
            out.append(c or w)
            changed |= bool(c)
        return " ".join(out) if changed else None


class QuickLinks:
    def __init__(self, path: Path) -> None:
        cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"links": [], "filler": []}
        self.filler = {fold(w) for w in cfg.get("filler", [])}
        self.links = [{**link, "_words": sorted({fold(w) for w in link["words"]}, key=len, reverse=True)}
                      for link in cfg.get("links", [])]

    def match(self, query: str) -> List[Dict[str, str]]:
        """Links whose words make up the whole search, apart from filler words."""
        q = fold(query)
        found = []
        for link in self.links:
            rest, hit = f" {q} ", False
            for w in link["_words"]:   # longest first, so "job board" is taken before "job"
                if f" {w} " in rest:
                    rest, hit = rest.replace(f" {w} ", " "), True
            if hit and all(t in self.filler for t in rest.split()):
                found.append({"title": link["title"], "url": link["url"]})
        return found
