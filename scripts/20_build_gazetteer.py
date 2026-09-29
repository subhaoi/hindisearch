from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

from utils import Paths, ensure_dir, read_parquet, write_json, is_nullish, text_to_key_variants


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
    args = ap.parse_args()

    paths = Paths(root=Path(args.root).resolve())
    ensure_dir((paths.data / "phase_45"))

    df = read_parquet(Path(args.input).resolve())

    gaz: Dict[str, Any] = {"corpus_size": int(len(df))}
    for field in ["locations_norm", "categories_norm", "tags_norm", "contributors_norm"]:
        items = collect_unique(df, field)
        doc_count: Dict[str, int] = {}
        for v in df[field].tolist() if field in df.columns else []:
            if is_nullish(v):
                continue
            for item in set(str(x).strip() for x in list(v) if not is_nullish(x)):
                doc_count[item] = doc_count.get(item, 0) + 1
        gaz[field] = {
            "values": items,
            "match_text": [match_text(x) for x in items],
            # Phonetic match keys (with/without schwa deletion) so Roman and Devanagari
            # queries both match: "bihar" / "बिहार" -> "bihar"
            "keys": [text_to_key_variants(match_text(x)) for x in items],
            "doc_count": [doc_count.get(x, 0) for x in items],
        }

    out_path = Path(args.out).resolve()
    write_json(out_path, gaz)
    print(f"Wrote: {out_path}")
    for k, v in gaz.items():
        if isinstance(v, dict):
            print(k, "count:", len(v["values"]))


if __name__ == "__main__":
    main()
