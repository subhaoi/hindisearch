from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils import Paths, read_parquet, write_json, is_nullish, script_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to stage_3/metadata_normalized.parquet")
    ap.add_argument("--root", default=".", help="Project root (default .)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    df = read_parquet(Path(args.input).resolve())

    report: Dict[str, Any] = {
        "rows": len(df),
        "null_counts": {},
        "basic": {},
        "script_stats_samples": [],
    }

    # Basic null counts
    for c in ["id", "url", "title_hi", "summary_hi", "content_hi", "published_date"]:
        if c in df.columns:
            report["null_counts"][c] = int(df[c].isna().sum())
        else:
            report["null_counts"][c] = None

    # Content length distribution
    if "content_hi_len" in df.columns:
        report["basic"]["content_len_min"] = int(df["content_hi_len"].min())
        report["basic"]["content_len_p25"] = int(df["content_hi_len"].quantile(0.25))
        report["basic"]["content_len_med"] = int(df["content_hi_len"].quantile(0.50))
        report["basic"]["content_len_p75"] = int(df["content_hi_len"].quantile(0.75))
        report["basic"]["content_len_max"] = int(df["content_hi_len"].max())

    # Script contamination samples (first 50)
    if "content_hi" in df.columns:
        for i, row in df.head(50).iterrows():
            s = row.get("content_hi")
            ss = script_stats(s)
            report["script_stats_samples"].append({
                "id": row.get("id"),
                "dev_pct": ss["dev_pct"],
                "latin_pct": ss["latin_pct"],
                "snippet": ("" if is_nullish(s) else str(s)[:160]),
            })

    out_path = paths.logs / "quality_report.json"
    write_json(out_path, report)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
