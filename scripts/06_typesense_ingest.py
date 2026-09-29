from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
import typesense
from tqdm import tqdm

from utils import (
    Paths, read_parquet, ensure_dir, write_json,
    is_nullish, iso_to_epoch_seconds, text_to_key, clean_title
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
    # Parquet list columns load as numpy arrays, not lists
    if val is None or isinstance(val, (str, bytes, float, int)):
        return []
    try:
        return [str(x) for x in list(val) if not is_nullish(x)]
    except TypeError:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to data/final/articles_canonical.parquet")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--prune", action="store_true", help="Delete indexed documents that are no longer in --input")
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

        title_hi = clean_title(row.get("title_hi"))
        summary_hi = "" if is_nullish(row.get("summary_hi")) else str(row.get("summary_hi"))
        content_hi = "" if is_nullish(row.get("content_hi")) else str(row.get("content_hi"))
        content_key = text_to_key(content_hi)

        doc = {
            "id": str(row.get("id")),
            "url": "" if is_nullish(row.get("url")) else str(row.get("url")),
            "published_date": published_date,
            "published_ts": published_ts,

            "title_hi": title_hi,
            "summary_hi": summary_hi,
            "content_hi": content_hi,

            # Romanized match keys for Roman queries
            "title_roman_norm": text_to_key(title_hi),
            "summary_roman_norm": text_to_key(summary_hi),
            "content_roman_norm": content_key,

            # Mixed: original + romanized for mixed-script queries
            "content_mixed_norm": f"{content_hi}\n\n{content_key}".strip(),

            "categories_norm": safe_list(row.get("categories_norm")),
            "tags_norm": safe_list(row.get("tags_norm")),
            "locations_norm": safe_list(row.get("locations_norm")),
            "contributors_norm": safe_list(row.get("contributors_norm")),
            "locations_key": [k for k in (text_to_key(x) for x in safe_list(row.get("locations_norm"))) if k],
            "contributors_key": [k for k in (text_to_key(x) for x in safe_list(row.get("contributors_norm"))) if k],

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
            # The client returns a list of dicts for list input (JSONL text for string input)
            results = res if isinstance(res, list) else [json.loads(l) for l in str(res).splitlines() if l.strip()]
            for r in results:
                if r.get("success"):
                    report["indexed"] += 1
                else:
                    report["failed"] += 1
                    if len(report["failures"]) < 50:
                        report["failures"].append(r)
        except Exception as e:
            report["failed"] += len(batch)
            if len(report["failures"]) < 50:
                report["failures"].append(f"Batch {i}-{i+len(batch)} failed: {type(e).__name__}: {e}")

    if args.prune:
        keep = {d["id"] for d in docs}
        exported = client.collections[collection].documents.export({"include_fields": "id"})
        stale = [json.loads(l)["id"] for l in str(exported).splitlines() if l.strip()]
        stale = [i for i in stale if i not in keep]
        for sid in stale:
            client.collections[collection].documents[sid].delete()
        report["pruned"] = len(stale)
        print(f"Pruned {len(stale)} documents no longer in the input")

    out = paths.logs / "phase2_ingest_report.json"
    write_json(out, report)
    print(f"Wrote: {out}")
    print(f"Indexed: {report['indexed']} | Failed: {report['failed']}")
    if report["failed"]:
        raise SystemExit(f"{report['failed']} documents failed to index; see {out}")


if __name__ == "__main__":
    main()
