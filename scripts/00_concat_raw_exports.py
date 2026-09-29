from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from utils import Paths, ensure_dir


IMAGE_COL = "Image Featured"

DEFAULT_INPUTS = [
    "data/raw/Articles-Export-2024-January-25-0205.csv",
    "data/raw/Features-Export-2024-January-25-0214.csv",
    "data/raw/Ground-Up-Stories-Export-2024-January-25-0217.csv",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="List of CSVs to concatenate (default: three WP exports)",
    )
    ap.add_argument("--output", default="data/raw/articles.csv", help="Output CSV path")
    ap.add_argument(
        "--carry-images-from",
        default=None,
        help="Earlier combined CSV; fills missing 'Image Featured' URLs by ID (the WP exports may not include the column)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    input_paths: List[Path] = [Path(p).resolve() if Path(p).is_absolute() else (paths.root / p).resolve() for p in args.inputs]
    output_path = Path(args.output).resolve() if Path(args.output).is_absolute() else (paths.root / args.output).resolve()

    dfs: List[pd.DataFrame] = []
    for p in input_paths:
        df = load_csv(p)
        dfs.append(df)
        print(f"Loaded {len(df):>4} rows from {p}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)

    if args.carry_images_from:
        prev_path = Path(args.carry_images_from).resolve()
        if prev_path.exists():
            prev = load_csv(prev_path)
            if IMAGE_COL in prev.columns:
                prev_map = dict(zip(prev["ID"].astype(str).str.strip(), prev[IMAGE_COL]))
                if IMAGE_COL not in combined.columns:
                    combined[IMAGE_COL] = ""
                missing = combined[IMAGE_COL].fillna("").astype(str).str.strip() == ""
                combined.loc[missing, IMAGE_COL] = combined.loc[missing, "ID"].astype(str).str.strip().map(prev_map).fillna("")
                print(f"Carried {int((combined.loc[missing, IMAGE_COL] != '').sum())} image URLs from {prev_path}")
        if IMAGE_COL not in combined.columns or (combined[IMAGE_COL].fillna("") == "").any():
            n = len(combined) if IMAGE_COL not in combined.columns else int((combined[IMAGE_COL].fillna("") == "").sum())
            print(f"WARNING: {n} articles have no '{IMAGE_COL}'. Add the field to the WP All Export templates.")

    ensure_dir(output_path.parent)
    combined.to_csv(output_path, index=False)

    print(f"Wrote combined CSV: {output_path}")
    print(f"Total rows: {len(combined)} | Unique IDs: {combined['ID'].nunique()}")


if __name__ == "__main__":
    main()
