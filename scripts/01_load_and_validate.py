from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from utils import Paths, ensure_dir, write_json, write_parquet, is_nullish


EXPECTED_COLUMNS = [
    "ID", "Date", "Title", "Content", "Permalink", "Categories",
    "Locations", "Tags", "Contributors", "wph_summary", "wph_article_type",
    "wph_multimedia_type", "_yoast_wpseo_title", "_yoast_wpseo_metadesc",
    "wph_partner_label",
]


def load_csv_defensively(csv_path: Path) -> pd.DataFrame:
    """
    Defensive CSV read for WP exports:
    - keeps all columns as strings initially
    - tolerates embedded newlines in Content (as long as quoting is correct)
    """
    # Read everything as string to avoid dtype chaos.
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,  # so empty becomes ""
        na_values=[],
        encoding="utf-8",
        engine="python",  # more tolerant
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw articles.csv")
    ap.add_argument("--root", default=".", help="Project root (default .)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    ensure_dir(paths.stage("raw"))
    ensure_dir(paths.stage("stage_1"))
    ensure_dir(paths.logs)

    load_errors: Dict[str, Any] = {"missing_columns": [], "notes": []}

    df = load_csv_defensively(in_path)

    # Column checks
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        load_errors["missing_columns"] = missing
        # We do not hard fail; continue with available columns so pipeline can adapt.

    # Trim whitespace in column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist
    for req in ["ID", "Permalink", "Content", "Title", "Date"]:
        if req not in df.columns:
            df[req] = ""

    # Normalize empty strings to None for key fields (not all yet)
    for c in df.columns:
        df[c] = df[c].apply(lambda x: None if isinstance(x, str) and x.strip() == "" else x)

    # Enforce ID as string, ensure no null IDs
    df["ID"] = df["ID"].apply(lambda x: str(x).strip() if not is_nullish(x) else None)

    null_id_rows = df["ID"].isna().sum()
    if null_id_rows > 0:
        load_errors["notes"].append(f"Rows with null ID: {int(null_id_rows)}")

    # Drop rows with null ID (cannot be indexed later), but log count.
    df_valid = df.dropna(subset=["ID"]).copy()

    # Ensure permalink exists; keep row but log
    null_url_rows = df_valid["Permalink"].isna().sum()
    if null_url_rows > 0:
        load_errors["notes"].append(f"Rows with null Permalink: {int(null_url_rows)}")

    # Save stage outputs
    out_loaded = paths.stage("stage_1") / "loaded.parquet"
    write_parquet(df_valid, out_loaded)

    out_schema = paths.stage("stage_1") / "schema_validated.parquet"
    # For stage 1, loaded and schema_validated are identical but kept separate for clarity.
    write_parquet(df_valid, out_schema)

    write_json(paths.logs / "load_errors.json", load_errors)

    print(f"Wrote: {out_loaded}")
    print(f"Wrote: {out_schema}")
    print(f"Wrote: {paths.logs / 'load_errors.json'}")
    print(f"Rows in CSV: {len(df)} | Rows kept (non-null ID): {len(df_valid)}")


if __name__ == "__main__":
    main()
