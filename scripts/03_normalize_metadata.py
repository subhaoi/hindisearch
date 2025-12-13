from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from utils import (
    Paths, read_parquet, write_parquet, write_json,
    split_pipe_field, normalize_token_list, parse_date_to_iso, is_nullish
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to stage_2/text_cleaned.parquet")
    ap.add_argument("--root", default=".", help="Project root (default .)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    df = read_parquet(Path(args.input).resolve())

    mv_logs: Dict[str, Any] = {
        "rows": len(df),
        "examples": [],
        "empty_fields_counts": {},
    }

    def split_and_norm(col: str) -> None:
        raw_list = []
        norm_list = []
        empty_count = 0
        for v in df[col].tolist() if col in df.columns else [None] * len(df):
            items = split_pipe_field(v)
            if len(items) == 0:
                empty_count += 1
            raw_list.append(items)
            norm_list.append(normalize_token_list(items))
        mv_logs["empty_fields_counts"][col] = empty_count
        df[f"{col.lower()}_raw"] = raw_list
        df[f"{col.lower()}_norm"] = norm_list

    # Multi-value fields are pipe-separated
    for c in ["Categories", "Locations", "Tags", "Contributors"]:
        if c not in df.columns:
            df[c] = None
        split_and_norm(c)

    # Parse date to ISO
    if "Date" not in df.columns:
        df["Date"] = None
    df["published_date"] = df["Date"].apply(parse_date_to_iso)

    # Canonical minimal field names for downstream systems
    df["id"] = df["ID"].apply(lambda x: str(x).strip() if not is_nullish(x) else None)
    df["url"] = df["Permalink"].apply(lambda x: str(x).strip() if not is_nullish(x) else None)

    df["article_type"] = df.get("wph_article_type")
    df["multimedia_type"] = df.get("wph_multimedia_type")
    df["partner_label"] = df.get("wph_partner_label")

    # Write stage_3 output
    out_path = paths.stage("stage_3") / "metadata_normalized.parquet"
    write_parquet(df, out_path)
    write_json(paths.logs / "multivalue_logs.json", mv_logs)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {paths.logs / 'multivalue_logs.json'}")


if __name__ == "__main__":
    main()
