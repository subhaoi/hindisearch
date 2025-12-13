from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
import typesense
from tqdm import tqdm

from utils import (
    Paths, read_parquet, ensure_dir, write_json,
    is_nullish, iso_to_epoch_seconds, devanagari_to_roman_hk
)


def get_client() -> typesense.Client:
    load_dotenv()
    host = os.environ.get("TYPESENSE_HOST", "localhost")
    port = os.environ.get("TYPESENSE_PORT", "8108")
    protocol = os.environ.get("TYPESENSE_PROTOCOL", "http")
    api_key = os.environ.get("TYPESENSE_API_KEY")
    if not api_key:
        raise RuntimeError("TYPESENSE_API_KEY not set in .env")

    return typesense.Client({
        "nodes": [{"host": host, "port": port, "protocol": protocol}],
        "api_key": api_key,
        "connection_timeout_seconds": 30,
    })


def safe_list(val: Any) -> List[str]:
    if is_nullish(val):
        return []
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val if not is_nullish(x)]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to data/final/articles_canonical.parquet")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)
    ensure_dir(paths.logs)

    load_dotenv()
    collection = os.environ.get("TYPESENSE_COLLECTION", "idr_articles_hi_v1")

    df = read_parquet(Path(args.input).resolve())
    client = get_client()

    report: Dict[str, Any] = {"rows": len(df), "indexed": 0, "failed": 0, "failures": []}

    docs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        published_date = None if is_nullish(row.get("published_date")) else str(row.get("published_date"))
        published_ts = iso_to_epoch_seconds(published_date)

        title_hi = "" if is_nullish(row.get("title_hi")) else str(row.get("title_hi"))
        summary_hi = "" if is_nullish(row.get("summary_hi")) else str(row.get("summary_hi"))
        content_hi = "" if is_nullish(row.get("content_hi")) else str(row.get("content_hi"))

        doc = {
            "id": str(row.get("id")),
            "url": "" if is_nullish(row.get("url")) else str(row.get("url")),
            "published_date": published_date,
            "published_ts": published_ts,

            "title_hi": title_hi,
            "summary_hi": summary_hi,
            "content_hi": content_hi,

            # Romanized fields for Roman queries
            "title_roman_norm": devanagari_to_roman_hk(title_hi),
            "summary_roman_norm": devanagari_to_roman_hk(summary_hi),
            "content_roman_norm": devanagari_to_roman_hk(content_hi),

            "categories_norm": safe_list(row.get("categories_norm")),
            "tags_norm": safe_list(row.get("tags_norm")),
            "locations_norm": safe_list(row.get("locations_norm")),
            "contributors_norm": safe_list(row.get("contributors_norm")),

            "article_type": None if is_nullish(row.get("article_type")) else str(row.get("article_type")),
            "multimedia_type": None if is_nullish(row.get("multimedia_type")) else str(row.get("multimedia_type")),
            "partner_label": None if is_nullish(row.get("partner_label")) else str(row.get("partner_label")),
        }
        docs.append(doc)

    bs = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(docs), bs), desc="Indexing into Typesense"):
        batch = docs[i:i + bs]
        try:
            res = client.collections[collection].documents.import_(batch, {"action": "upsert"})
            for line in str(res).splitlines():
                if '"success":true' in line:
                    report["indexed"] += 1
                else:
                    report["failed"] += 1
                    if len(report["failures"]) < 50:
                        report["failures"].append(line)
        except Exception as e:
            report["failed"] += len(batch)
            if len(report["failures"]) < 50:
                report["failures"].append(f"Batch {i}-{i+len(batch)} failed: {type(e).__name__}: {e}")

    out = paths.logs / "phase2_ingest_report.json"
    write_json(out, report)
    print(f"Wrote: {out}")
    print(f"Indexed: {report['indexed']} | Failed: {report['failed']}")


if __name__ == "__main__":
    main()
