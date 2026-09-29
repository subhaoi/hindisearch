from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy import text

from utils import Paths, ensure_dir, write_json
from _phase4.db import get_engine

JUDGMENTS_SQL = """
SELECT DISTINCT ON (lower(trim(q.query_raw)), l.article_id)
       lower(trim(q.query_raw)) AS query_key, q.query_raw, l.article_id, l.label
FROM labels l
JOIN query_log q ON q.id = l.query_id
WHERE l.article_id IS NOT NULL
ORDER BY lower(trim(q.query_raw)), l.article_id, l.created_at DESC
"""


def load_judgments() -> Dict[str, Dict[str, Any]]:
    """{query_key: {"query": raw text, "pos": set(article ids), "neg": set(article ids)}} (latest label wins)."""
    engine = get_engine()
    out: Dict[str, Dict[str, Any]] = {}
    with engine.connect() as conn:
        for row in conn.execute(text(JUDGMENTS_SQL)):
            j = out.setdefault(row.query_key, {"query": row.query_raw, "pos": set(), "neg": set()})
            (j["pos"] if int(row.label) == 1 else j["neg"]).add(str(row.article_id))
    return out


def run_query(host: str, query: str, depth: int, ranker: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"query": query, "per_page": depth, "page": 1, "log": False}
    if ranker:
        payload["ranker"] = ranker
    r = requests.post(host.rstrip("/") + "/search", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return {"mode": data.get("mode"), "ids": [str(h["id"]) for h in data.get("results", [])]}


def metrics(ids: List[str], pos: set, neg: set, k: int) -> Dict[str, float]:
    dcg = sum(1.0 / math.log2(i + 2) for i, a in enumerate(ids[:k]) if a in pos)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(pos), k)))
    first = next((i + 1 for i, a in enumerate(ids) if a in pos), None)
    top = ids[:k]
    return {
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "recall@depth": len(pos & set(ids)) / len(pos) if pos else 0.0,
        f"wrong@{k}": float(sum(1 for a in top if a in neg)),        # known-bad results in top k
        f"judged@{k}": sum(1 for a in top if a in pos or a in neg) / max(1, len(top)),
    }


def mean(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]}


def print_table(title: str, by_group: Dict[str, List[Dict[str, float]]]) -> None:
    print(f"\n{title}")
    names = list(next(iter(by_group.values()))[0].keys()) if by_group else []
    print(f"{'group':<10}{'n':>5}" + "".join(f"{m:>15}" for m in names))
    for g, rows in by_group.items():
        avg = mean(rows)
        print(f"{g:<10}{len(rows):>5}" + "".join(f"{avg[m]:>15.3f}" for m in names))


def main() -> None:
    """
    Offline evaluation against the labels collected through the feedback UI.

    Every labelled query (plus the core query set) is replayed through POST /search with
    log=false, so evaluation does not pollute query_log. Unlabelled results count as not
    relevant, so absolute numbers are pessimistic; compare runs rather than reading them
    in isolation. `judged@k` shows how much of the top k has labels at all.

    Typical use:
      python scripts/23_evaluate_search.py --save runs/before.json
      ...deploy changes...
      python scripts/23_evaluate_search.py --save runs/after.json --compare runs/before.json
      python scripts/23_evaluate_search.py --ranker ranker_v1   # A/B rankers on one server
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--depth", type=int, default=50, help="Results fetched per query (recall/MRR depth)")
    ap.add_argument("--ranker", default=None, help="Override ranker for this run (ranker_v1 | ranker_v2)")
    ap.add_argument("--core", default="data/phase_4/core_queries.json", help="Unlabelled queries to snapshot")
    ap.add_argument("--save", default=None, help="Write this run to a JSON file")
    ap.add_argument("--compare", default=None, help="Earlier run JSON to diff against")
    args = ap.parse_args()

    paths = Paths(root=Path(args.root).resolve())
    judgments = load_judgments()
    judged = {k: v for k, v in judgments.items() if v["pos"]}
    print(f"Labelled queries: {len(judgments)} | with at least one relevant article: {len(judged)}")

    run: Dict[str, Any] = {"meta": vars(args), "queries": {}}
    by_mode: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for key, j in judged.items():
        res = run_query(args.host, j["query"], args.depth, args.ranker)
        m = metrics(res["ids"], j["pos"], j["neg"], args.k)
        run["queries"][key] = {"query": j["query"], "mode": res["mode"], "ids": res["ids"], "metrics": m}
        by_mode["all"].append(m)
        by_mode[res["mode"] or "?"].append(m)

    if by_mode:
        print_table("Labelled queries (mean)", by_mode)

    core_path = Path(args.core)
    if core_path.exists():
        with core_path.open("r", encoding="utf-8") as f:
            core = json.load(f).get("queries", [])
        run["core"] = {q: run_query(args.host, q, args.k, args.ranker)["ids"] for q in core}

    if args.compare:
        with Path(args.compare).open("r", encoding="utf-8") as f:
            prev = json.load(f)
        print(f"\nChanges vs {args.compare}")
        ndcg = f"ndcg@{args.k}"
        deltas = []
        for key, cur in run["queries"].items():
            old = prev.get("queries", {}).get(key)
            if old:
                deltas.append((cur["metrics"][ndcg] - old["metrics"][ndcg], cur["query"]))
        if deltas:
            print(f"  mean {ndcg} delta over {len(deltas)} shared queries: {sum(d for d, _ in deltas) / len(deltas):+.3f}")
            for d, q in sorted(deltas)[:10]:
                if d < 0:
                    print(f"  worse  {d:+.3f}  {q}")
            for d, q in sorted(deltas, reverse=True)[:10]:
                if d > 0:
                    print(f"  better {d:+.3f}  {q}")
        for q, ids in (run.get("core") or {}).items():
            old_ids = (prev.get("core") or {}).get(q)
            if old_ids is not None:
                overlap = len(set(ids) & set(old_ids)) / max(1, len(set(ids) | set(old_ids)))
                print(f"  core top-{args.k} overlap {overlap:.2f}  {q}")

    if args.save:
        out = Path(args.save).resolve()
        ensure_dir(out.parent)
        write_json(out, run)
        print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
