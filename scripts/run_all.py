from __future__ import annotations

import argparse
from pathlib import Path

from utils import Paths, ensure_dir, write_parquet, read_parquet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw articles.csv")
    ap.add_argument("--root", default=".", help="Project root (default .)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    # Ensure dirs
    for d in ["raw", "stage_1", "stage_2", "stage_3", "final"]:
        ensure_dir(paths.stage(d))
    ensure_dir(paths.logs)

    # 01
    stage1_schema = paths.stage("stage_1") / "schema_validated.parquet"
    if not stage1_schema.exists():
        import subprocess, sys
        subprocess.check_call([sys.executable, str(root / "scripts" / "01_load_and_validate.py"),
                               "--input", str(Path(args.input).resolve()), "--root", str(root)])

    # 02
    stage2_clean = paths.stage("stage_2") / "text_cleaned.parquet"
    if not stage2_clean.exists():
        import subprocess, sys
        subprocess.check_call([sys.executable, str(root / "scripts" / "02_clean_text_wp.py"),
                               "--input", str(stage1_schema), "--root", str(root)])

    # 03
    stage3_meta = paths.stage("stage_3") / "metadata_normalized.parquet"
    if not stage3_meta.exists():
        import subprocess, sys
        subprocess.check_call([sys.executable, str(root / "scripts" / "03_normalize_metadata.py"),
                               "--input", str(stage2_clean), "--root", str(root)])

    # 04
    import subprocess, sys
    subprocess.check_call([sys.executable, str(root / "scripts" / "04_quality_checks.py"),
                           "--input", str(stage3_meta), "--root", str(root)])

    # Final canonical output: copy stage_3 to final with a stable name
    final_path = paths.stage("final") / "articles_canonical.parquet"
    df = read_parquet(stage3_meta)

    # Keep a focused set of columns for downstream
    keep_cols = [
        "id", "published_date", "url",
        "title_hi", "summary_hi", "content_hi",
        "categories_raw", "locations_raw", "tags_raw", "contributors_raw",
        "categories_norm", "locations_norm", "tags_norm", "contributors_norm",
        "article_type", "multimedia_type", "partner_label",
        "content_hi_len",
        # keep originals for traceability if present
        "ID", "Date", "Title", "Content", "Permalink", "Categories", "Locations", "Tags", "Contributors",
        "wph_summary", "wph_article_type", "wph_multimedia_type",
        "_yoast_wpseo_title", "_yoast_wpseo_metadesc", "wph_partner_label",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_final = df[keep_cols].copy()
    write_parquet(df_final, final_path)

    print(f"Wrote: {final_path}")
    print("Phase-1 pipeline complete.")


if __name__ == "__main__":
    main()
