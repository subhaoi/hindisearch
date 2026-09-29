from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

import json

from utils import (
    Paths, ensure_dir, read_parquet, write_json, is_nullish, text_to_key_variants,
    location_patterns, derive_locations,
)


def match_text(value: str) -> str:
    """
    Text a query has to contain to match this value. Hierarchical categories like
    "क्षेत्र>कृषि" match on their leaf ("कृषि").
    """
    return value.split(">")[-1].strip()


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
    ap.add_argument("--aliases", default="config/location_aliases.json", help="Extra names per location value")
    args = ap.parse_args()

    paths = Paths(root=Path(args.root).resolve())
    ensure_dir((paths.data / "phase_45"))

    df = read_parquet(Path(args.input).resolve())

    aliases: Dict[str, List[str]] = {}
    alias_path = Path(args.aliases)
    if alias_path.exists():
        aliases = json.loads(alias_path.read_text(encoding="utf-8")).get("aliases", {})

    # Locations an article is about = tags + mentions (see utils.derive_locations). Counted
    # on that set so "broad location" (> 20% of articles) matches what the filter would use.
    loc_values = collect_unique(df, "locations_norm")
    patterns = location_patterns(loc_values, aliases)
    locations_all = [
        derive_locations(r.get("title_hi"), r.get("summary_hi"), r.get("content_hi"),
                         [str(x) for x in (r.get("locations_norm") if r.get("locations_norm") is not None else [])], patterns)
        for _, r in df.iterrows()
    ]

    gaz: Dict[str, Any] = {"corpus_size": int(len(df))}
    for field in ["locations_norm", "categories_norm", "tags_norm", "contributors_norm"]:
        items = collect_unique(df, field)
        doc_count: Dict[str, int] = {}
        per_article = locations_all if field == "locations_norm" else (df[field].tolist() if field in df.columns else [])
        for v in per_article:
            if is_nullish(v):
                continue
            for item in set(str(x).strip() for x in list(v) if not is_nullish(x)):
                doc_count[item] = doc_count.get(item, 0) + 1
        gaz[field] = {
            "values": items,
            "match_text": [match_text(x) for x in items],
            # Phonetic match keys (with/without schwa deletion) so Roman and Devanagari
            # queries both match: "bihar" / "बिहार" -> "bihar"
            "keys": [
                list(dict.fromkeys(k for name in [match_text(x)] + aliases.get(x, []) for k in text_to_key_variants(name)))
                for x in items
            ],
            "doc_count": [doc_count.get(x, 0) for x in items],
        }
        if field == "locations_norm":
            gaz[field]["aliases"] = [aliases.get(x, []) for x in items]

    out_path = Path(args.out).resolve()
    write_json(out_path, gaz)
    print(f"Wrote: {out_path}")
    for k, v in gaz.items():
        if isinstance(v, dict):
            print(k, "count:", len(v["values"]))


if __name__ == "__main__":
    main()
