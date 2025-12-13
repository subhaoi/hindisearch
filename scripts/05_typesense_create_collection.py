from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import typesense

from utils import Paths, ensure_dir, write_json


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
    ap.add_argument("--root", default=".", help="Project root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)
    ensure_dir(paths.stage("phase_2"))

    load_dotenv()
    collection = os.environ.get("TYPESENSE_COLLECTION", "idr_articles_hi_v1")

    schema: Dict[str, Any] = {
        "name": collection,
        "fields": [
            {"name": "id", "type": "string"},
            {"name": "url", "type": "string"},
            {"name": "published_ts", "type": "int64"},  # numeric sorting
            {"name": "published_date", "type": "string", "optional": True},

            # Hindi fields
            {"name": "title_hi", "type": "string"},
            {"name": "summary_hi", "type": "string", "optional": True},
            {"name": "content_hi", "type": "string", "optional": True},

            # Romanized + normalized fields (for Roman queries)
            {"name": "title_roman_norm", "type": "string", "optional": True},
            {"name": "summary_roman_norm", "type": "string", "optional": True},
            {"name": "content_roman_norm", "type": "string", "optional": True},

            # Facets / filters
            {"name": "categories_norm", "type": "string[]", "facet": True, "optional": True},
            {"name": "tags_norm", "type": "string[]", "facet": True, "optional": True},
            {"name": "locations_norm", "type": "string[]", "facet": True, "optional": True},
            {"name": "contributors_norm", "type": "string[]", "facet": True, "optional": True},

            {"name": "article_type", "type": "string", "facet": True, "optional": True},
            {"name": "multimedia_type", "type": "string", "facet": True, "optional": True},
            {"name": "partner_label", "type": "string", "facet": True, "optional": True},
        ],
        "default_sorting_field": "published_ts",
    }

    client = get_client()

    # Rebuild-friendly
    try:
        client.collections[collection].retrieve()
        client.collections[collection].delete()
        print(f"Deleted existing collection: {collection}")
    except Exception:
        pass

    created = client.collections.create(schema)
    print(f"Created collection: {created.get('name')}")

    out_schema = paths.stage("phase_2") / "typesense_schema.json"
    write_json(out_schema, schema)
    print(f"Wrote schema: {out_schema}")


if __name__ == "__main__":
    main()
