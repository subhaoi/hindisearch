from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import typesense

from utils import Paths, write_json


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

    load_dotenv()
    collection = os.environ.get("TYPESENSE_COLLECTION", "idr_articles_hi_v1")

    client = get_client()
    schema = client.collections[collection].retrieve()

    out = paths.stage("phase_2") / "typesense_schema.json"
    write_json(out, schema)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
