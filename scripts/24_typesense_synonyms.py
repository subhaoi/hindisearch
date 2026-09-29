from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
import typesense

from utils import Paths, write_json, normalize_devanagari_text, text_to_key, DEVANAGARI_RE

SYNONYM_ID_PREFIX = "syn_v1_"


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


def expand_group(group: List[str]) -> List[str]:
    """
    Devanagari terms as-is (for *_hi fields) plus every term's roman match key (for the
    *_roman_norm fields that roman queries search), so 'women' also reaches 'mahila'.
    """
    out: List[str] = []
    for term in group:
        forms = [text_to_key(term)]
        if DEVANAGARI_RE.search(term):
            forms.insert(0, normalize_devanagari_text(term) or term)
        for f in forms:
            if f and f not in out:
                out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--synonyms", default="config/synonyms_v1.json")
    ap.add_argument("--dry-run", action="store_true", help="Print expanded groups without writing")
    args = ap.parse_args()

    paths = Paths(root=Path(args.root).resolve())
    load_dotenv()
    collection = os.environ.get("TYPESENSE_COLLECTION", "idr_articles_hi_v1")

    with Path(args.synonyms).resolve().open("r", encoding="utf-8") as f:
        groups = json.load(f)["groups"]
    expanded = [expand_group(g) for g in groups]

    if args.dry_run:
        for g in expanded:
            print(" | ".join(g))
        return

    client = get_client()
    syn_api = client.collections[collection].synonyms

    # Replace our previous synonym set; leave any hand-made synonyms alone.
    existing = syn_api.retrieve().get("synonyms", []) or []
    removed = 0
    for s in existing:
        sid = str(s.get("id", ""))
        if sid.startswith(SYNONYM_ID_PREFIX):
            syn_api[sid].delete()
            removed += 1

    for i, terms in enumerate(expanded):
        syn_api.upsert(f"{SYNONYM_ID_PREFIX}{i:03d}", {"synonyms": terms})

    report = {"collection": collection, "removed": removed, "upserted": len(expanded), "groups": expanded}
    out = paths.logs / "typesense_synonyms_report.json"
    write_json(out, report)
    print(f"Removed {removed} old synonym groups, upserted {len(expanded)} into {collection}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
