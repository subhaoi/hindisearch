from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests

from utils import read_parquet, fold_devanagari, clean_title, is_nullish


def as_list(v: Any) -> List[str]:
    if v is None or isinstance(v, (str, bytes, float, int)):
        return []
    try:
        return [str(x) for x in list(v) if not is_nullish(x)]
    except TypeError:
        return []


def resolve(expect: Dict[str, Any], articles: pd.DataFrame) -> Optional[Set[str]]:
    """Article ids an expectation points to, computed from the current articles."""
    fold = lambda s: fold_devanagari(clean_title("" if is_nullish(s) else str(s)))  # noqa: E731
    seo = articles["seo_title_hi"] if "seo_title_hi" in articles.columns else pd.Series("", index=articles.index)
    titles = articles["title_hi"].map(fold) + " || " + seo.map(fold)
    ids = articles["id"].astype(str)
    if "title_contains" in expect:
        return set(ids[titles.str.contains(fold(expect["title_contains"]), regex=False)])
    if "title_contains_any" in expect:
        mask = pd.Series(False, index=articles.index)
        for n in expect["title_contains_any"]:
            mask |= titles.str.contains(fold(n), regex=False)
        return set(ids[mask])
    if "contributor" in expect:
        return set(ids[articles["contributors_norm"].map(lambda v: expect["contributor"] in as_list(v))])
    if "contributor_like" in expect:
        part = expect["contributor_like"]
        return set(ids[articles["contributors_norm"].map(lambda v: any(part in x for x in as_list(v)))])
    if "category" in expect:
        return set(ids[articles["categories_norm"].map(lambda v: expect["category"] in as_list(v))])
    if "multimedia_type" in expect:
        return set(ids[articles["multimedia_type"].astype(str) == expect["multimedia_type"]])
    return None


def score(ids: List[str], rel: Set[str], k: int = 10) -> Dict[str, float]:
    first = next((r + 1 for r, i in enumerate(ids) if i in rel), None)
    hits = sum(1 for i in ids[:k] if i in rel)
    return {
        "rank": first or 0,
        "mrr": 1.0 / first if first else 0.0,
        # Normalized by what is achievable, so an author with 3 articles can reach 1.0
        f"p@{k}": hits / max(1, min(k, len(rel))),
    }


def main() -> None:
    """
    Free, repeatable quality check. Runs queries whose right answers are known from the
    data (pasted titles, author names, series) against POST /search with log=false.

      python scripts/26_eval_rules.py --save runs/before.json
      ...deploy a change, restart the API...
      python scripts/26_eval_rules.py --save runs/after.json --compare runs/before.json

    known_item: MRR (1.0 = the article is #1). author/series: precision@10.
    no_answer: how many results come back for off-topic/junk queries (lower is better).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--testset", default="config/eval/queries_v1.json")
    ap.add_argument("--articles", default="data/final/articles_canonical.parquet")
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    articles = read_parquet(Path(args.articles))
    run: Dict[str, Any] = {"meta": {"at": datetime.now(timezone.utc).isoformat(), "host": args.host}, "queries": {}}

    for set_name, block in testset["sets"].items():
        rows = []
        for q in block["queries"]:
            r = requests.post(args.host.rstrip("/") + "/search",
                              json={"query": q["query"], "per_page": 50, "page": 1, "log": False}, timeout=120)
            r.raise_for_status()
            data = r.json()
            ids = [str(h["id"]) for h in data.get("results", [])]
            rec: Dict[str, Any] = {"set": set_name, "top": ids[:10], "total": data.get("total_results", 0)}
            if q["expect"].get("no_good_match"):
                rec["metrics"] = {"results": float(data.get("total_results", 0))}
            else:
                rel = resolve(q["expect"], articles)
                if not rel:
                    rec["skipped"] = "no matching article in the current data"
                else:
                    rec["metrics"] = score(ids, rel)
                    rec["expected"] = sorted(rel)[:20]
            run["queries"][q["query"]] = rec
            rows.append((q["query"], rec))

        scored = [rec["metrics"] for _, rec in rows if "metrics" in rec]
        if not scored:
            continue
        avg = {m: sum(x[m] for x in scored) / len(scored) for m in scored[0]}
        print(f"\n== {set_name} ({len(scored)} scored)  " + "  ".join(f"{m}={v:.2f}" for m, v in avg.items() if m != "rank"))
        for query, rec in rows:
            m = rec.get("metrics")
            if rec.get("skipped"):
                print(f"   skip  {query}  ({rec['skipped']})")
            elif set_name == "known_item" and m["rank"] != 1:
                print(f"   rank {m['rank'] or '-':>3}  {query}")
            elif set_name in ("author", "series_format") and m["p@10"] < 0.5:
                print(f"   p@10 {m['p@10']:.2f}  {query}")

    if args.compare:
        prev = json.loads(Path(args.compare).read_text(encoding="utf-8"))["queries"]
        print(f"\n== changes vs {args.compare}")
        for query, rec in run["queries"].items():
            old = prev.get(query, {}).get("metrics")
            new = rec.get("metrics")
            if not old or not new:
                continue
            key = "mrr" if "mrr" in new else "results"
            d = new[key] - old[key]
            if abs(d) > 1e-9:
                better = d > 0 if key == "mrr" else d < 0
                print(f"   {'better' if better else 'worse ':6} {key} {old[key]:.2f} -> {new[key]:.2f}  {query}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nSaved: {args.save}")


if __name__ == "__main__":
    main()
