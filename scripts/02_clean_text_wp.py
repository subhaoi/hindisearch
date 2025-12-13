from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from utils import Paths, read_parquet, write_parquet, write_json, strip_wp_html_to_text, normalize_devanagari_text, script_stats, is_nullish


def choose_title(row: pd.Series) -> str:
    yoast = row.get("_yoast_wpseo_title")
    title = row.get("Title")
    if not is_nullish(yoast):
        return str(yoast)
    if not is_nullish(title):
        return str(title)
    return ""


def choose_summary(row: pd.Series, cleaned_content: str) -> str:
    wph = row.get("wph_summary")
    yoast_desc = row.get("_yoast_wpseo_metadesc")
    if not is_nullish(wph):
        return str(wph)
    if not is_nullish(yoast_desc):
        return str(yoast_desc)
    # Fallback: first N chars of cleaned content
    return (cleaned_content or "")[:400]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to stage_1/schema_validated.parquet")
    ap.add_argument("--root", default=".", help="Project root (default .)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    in_path = Path(args.input).resolve()
    df = read_parquet(in_path)

    wp_logs: Dict[str, Any] = {
        "rows": len(df),
        "failed_parse_count": 0,
        "very_short_content_rows": [],
        "script_contamination_examples": [],
    }

    content_clean: List[str] = []
    content_stats: List[Dict[str, Any]] = []
    titles: List[str] = []
    summaries: List[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning WP HTML"):
        raw_html = row.get("Content")
        raw_html = "" if is_nullish(raw_html) else str(raw_html)

        cleaned, stats = strip_wp_html_to_text(raw_html)

        title = choose_title(row)
        summary = choose_summary(row, cleaned)

        # Normalize Hindi text representation
        title_hi = normalize_devanagari_text(title) or ""
        summary_hi = normalize_devanagari_text(summary) or ""
        content_hi = normalize_devanagari_text(cleaned) or ""

        # Log short content / failures
        if not stats.get("ok", True):
            wp_logs["failed_parse_count"] += 1

        if len(content_hi) < 200:
            wp_logs["very_short_content_rows"].append({
                "id": row.get("ID"),
                "url": row.get("Permalink"),
                "clean_len": len(content_hi),
                "reason": "clean_len<200",
            })

        # Script contamination sample (store a few)
        ss = script_stats(content_hi)
        if ss["latin_pct"] > 0.10 and len(wp_logs["script_contamination_examples"]) < 25:
            wp_logs["script_contamination_examples"].append({
                "id": row.get("ID"),
                "latin_pct": ss["latin_pct"],
                "dev_pct": ss["dev_pct"],
                "snippet": content_hi[:200],
            })

        content_clean.append(content_hi)
        content_stats.append(stats)
        titles.append(title_hi)
        summaries.append(summary_hi)

    out = df.copy()
    out["title_hi"] = titles
    out["summary_hi"] = summaries
    out["content_hi"] = content_clean

    # Keep a compact content length column for later analysis
    out["content_hi_len"] = out["content_hi"].apply(lambda x: 0 if is_nullish(x) else len(str(x)))

    # Save stage output
    out_path = paths.stage("stage_2") / "text_cleaned.parquet"
    write_parquet(out, out_path)

    # Save logs
    write_json(paths.logs / "wp_strip_logs.json", wp_logs)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {paths.logs / 'wp_strip_logs.json'}")


if __name__ == "__main__":
    main()
