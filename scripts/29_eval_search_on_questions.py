from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests


def first_hit(api: str, query: str, gold: str, k: int) -> Dict[str, Any]:
    t = time.perf_counter()
    r = requests.post(f"{api}/search", json={"query": query, "per_page": k, "log": False}, timeout=120)
    r.raise_for_status()
    ids = [str(h["id"]) for h in r.json().get("results", [])]
    return {"rank": ids.index(gold) + 1 if gold in ids else 0, "seconds": time.perf_counter() - t}


def summary(ranks: pd.Series) -> Dict[str, float]:
    return {"n": int(len(ranks)), "recall@1": float((ranks == 1).mean()), "recall@10": float((ranks > 0).mean()),
            "mrr@10": float((1.0 / ranks.where(ranks > 0)).fillna(0).mean())}


def main() -> None:
    """
    The whole search (keywords, entities, vectors, ranking) on generated test questions: does the
    question's article come back in the top 10? Runs against a live API with log=false, so it
    measures exactly what readers get, unlike gpu/31_eval_vectors.py (the model alone).

      python scripts/29_eval_search_on_questions.py --api http://localhost:8000 --tag current
    Compare runs with the same --sample and --seed.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--split", default="data/synthetic/split_hi.parquet")
    ap.add_argument("--sets", default="test_q,test_new")
    ap.add_argument("--sample", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    q = pd.read_parquet(args.split)
    q = q[q["split"].isin(args.sets.split(","))]
    q = q.sample(min(args.sample, len(q)), random_state=args.seed).reset_index(drop=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        got: List[Dict[str, Any]] = list(pool.map(
            lambda r: first_hit(args.api, r[0], str(r[1]), 10), zip(q["query"], q["article_id"])))
    q["rank"] = [g["rank"] for g in got]
    q["seconds"] = [g["seconds"] for g in got]

    result = {"tag": args.tag, "api": args.api, "at": datetime.now(timezone.utc).isoformat(),
              "seconds_median": float(q["seconds"].median()), "overall": summary(q["rank"]),
              "by_set": {k: summary(g["rank"]) for k, g in q.groupby("split")},
              "by_style": {k: summary(g["rank"]) for k, g in q.groupby("style")}}
    out = Path(args.out or f"runs/gpu/search_{Path(args.split).stem}_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    o = result["overall"]
    print(f"{args.tag}: recall@1 {o['recall@1']:.3f}  recall@10 {o['recall@10']:.3f}  mrr@10 {o['mrr@10']:.3f}"
          f"  | median {result['seconds_median'] * 1000:.0f} ms/search")
    for k, v in {**result["by_set"], **result["by_style"]}.items():
        print(f"   {k:15} n={v['n']:5}  recall@10 {v['recall@10']:.3f}  mrr@10 {v['mrr@10']:.3f}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
