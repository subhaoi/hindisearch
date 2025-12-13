from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from utils import Paths, ensure_dir, read_parquet, write_json, is_nullish


def roman_norm(s: str) -> str:
    """
    Minimal roman normalizer (fast, safe). Not transliteration.
    Used only to match roman queries against metadata strings that are already roman-ish.
    """
    t = (s or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    # vowel collapse: karoonga/karunga -> karoonga-ish similarity
    t = re.sub(r"aa+", "aa", t)
    t = re.sub(r"ee+", "ee", t)
    t = re.sub(r"ii+", "ii", t)
    t = re.sub(r"oo+", "oo", t)
    t = re.sub(r"uu+", "uu", t)
    return t


def collect_unique(df, col: str) -> List[str]:
    vals: Set[str] = set()
    if col not in df.columns:
        return []
    for v in df[col].tolist():
        if is_nullish(v):
            continue
        try:
            # v may be list/np.array/tuple
            for item in list(v):
                if is_nullish(item):
                    continue
                s = str(item).strip()
                if s:
                    vals.add(s)
        except Exception:
            # fallback: treat as scalar
            s = str(v).strip()
            if s:
                vals.add(s)
    # Prefer longer strings first for longest-match scanning
    return sorted(vals, key=lambda x: (-len(x), x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/final/articles_canonical.parquet")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="data/phase_45/gazetteer_v1.json")
    args = ap.parse_args()

    paths = Paths(root=Path(args.root).resolve())
    ensure_dir((paths.data / "phase_45"))

    df = read_parquet(Path(args.input).resolve())

    gaz: Dict[str, Any] = {}
    for field in ["locations_norm", "categories_norm", "tags_norm", "contributors_norm"]:
        items = collect_unique(df, field)
        gaz[field] = {
            "values": items,
            "values_roman_norm": [roman_norm(x) for x in items],
        }

    out_path = Path(args.out).resolve()
    write_json(out_path, gaz)
    print(f"Wrote: {out_path}")
    for k in gaz:
        print(k, "count:", len(gaz[k]["values"]))


if __name__ == "__main__":
    main()
