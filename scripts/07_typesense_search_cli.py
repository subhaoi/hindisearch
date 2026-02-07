from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import typesense

from utils import Paths, canonicalize_query_for_search, write_json


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
        "connection_timeout_seconds": 10,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="User query (Roman or Devanagari)")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--per-page", type=int, default=10)
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--filter", default="", help="Typesense filter_by string (optional)")
    ap.add_argument("--sort", default="", help="e.g. published_ts:desc (optional)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    load_dotenv()
    collection = os.environ.get("TYPESENSE_COLLECTION", "idr_articles_hi_v1")
    client = get_client()

    canon = canonicalize_query_for_search(args.q)
    q = canon["q"]

    if canon["mode"] == "dev":
        query_by = "title_hi,summary_hi,content_hi"
        weights = "6,3,1"
    elif canon["mode"] == "mixed":
        query_by = "title_hi,summary_hi,content_hi,content_mixed_norm"
        weights = "6,3,1,1"
    else:
        query_by = "title_roman_norm,summary_roman_norm,content_roman_norm"
        weights = "6,3,1"

    search_parameters: Dict[str, Any] = {
        "q": q,
        "query_by": query_by,
        "query_by_weights": weights,
        "per_page": args.per_page,
        "page": args.page,
        "num_typos": 1,
    }
    if args.filter.strip():
        search_parameters["filter_by"] = args.filter.strip()
    if args.sort.strip():
        search_parameters["sort_by"] = args.sort.strip()

    res = client.collections[collection].documents.search(search_parameters)

    hits = res.get("hits", []) or []
    print(f"mode: {canon['mode']}")
    print(f"query(raw): {canon['raw']}")
    print(f"query(used): {q}")
    print(f"found: {res.get('found', 0)} | showing: {len(hits)}")
    print("-" * 80)

    for i, h in enumerate(hits, start=1):
        doc = h.get("document", {})
        print(f"{i}. {doc.get('title_hi','')}")
        print(f"   url: {doc.get('url','')}")
        print(f"   published_date: {doc.get('published_date','')}")
        print(f"   text_match: {h.get('text_match','')}")
        print()

    out = {
        "mode": canon["mode"],
        "query_raw": canon["raw"],
        "query_used": q,
        "filter": args.filter,
        "sort": args.sort,
        "found": res.get("found", 0),
        "hits": [
            {
                "id": (h.get("document") or {}).get("id"),
                "title": (h.get("document") or {}).get("title_hi"),
                "url": (h.get("document") or {}).get("url"),
                "published_date": (h.get("document") or {}).get("published_date"),
                "text_match": h.get("text_match"),
            }
            for h in hits
        ],
    }
    write_json(paths.logs / "phase2_search_smoketest.json", out)


if __name__ == "__main__":
    main()
