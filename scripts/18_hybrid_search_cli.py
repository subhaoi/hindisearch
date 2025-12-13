from __future__ import annotations

import argparse
import json
import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    payload = {"query": args.q, "per_page": args.k, "explain": bool(args.explain), "filter_by": args.filter}
    r = requests.post(args.host.rstrip("/") + "/search", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    print("query_id:", data["query_id"])
    print("mode:", data["mode"])
    print("query_used (lexical):", data["query_used"])
    print("query_semantic:", data["query_semantic"])
    print("-" * 80)
    for hit in data["results"]:
        print(f'{hit["rank"]}. {hit.get("title","")}')
        print("   url:", hit.get("url",""))
        print("   score:", hit.get("score",""))
        if hit.get("snippet"):
            print("   snippet:", hit["snippet"][:240])
        print()

    print(json.dumps(data, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
