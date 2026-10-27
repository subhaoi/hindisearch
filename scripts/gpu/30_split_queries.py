from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main() -> None:
    """
    Split generated queries (28_generate_queries.py) into training and test sets, once,
    with a fixed seed, so every later comparison uses the same test queries.

      train     used for fine-tuning / ranker training
      test_q    other queries for articles that also have training queries
                (the site's existing articles: does the model find them better?)
      test_new  all queries of articles never used in training
                (stands in for next week's new articles: does it still generalise?)

    Test queries are never used for anything else.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/synthetic/queries.jsonl")
    ap.add_argument("--out", default="data/synthetic/split_hi.parquet")
    ap.add_argument("--new-articles", type=float, default=0.15, help="Share of articles held out entirely")
    ap.add_argument("--test-share", type=float, default=0.25, help="Share of each remaining article's queries held out")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.queries).read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows).drop_duplicates(["article_id", "query"])
    rng = random.Random(args.seed)
    articles = sorted(df["article_id"].astype(str).unique())
    rng.shuffle(articles)
    held_out = set(articles[: int(len(articles) * args.new_articles)])

    split = []
    by_article = defaultdict(list)
    for i, r in df.iterrows():
        by_article[str(r["article_id"])].append(i)
    for aid, idx in by_article.items():
        if aid in held_out:
            split += [(i, "test_new") for i in idx]
            continue
        idx = idx[:]
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * args.test_share)) if len(idx) > 2 else 0
        split += [(i, "test_q") for i in idx[:n_test]] + [(i, "train") for i in idx[n_test:]]
    df["split"] = pd.Series(dict(split))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(df.groupby("split").agg(queries=("query", "size"), articles=("article_id", "nunique")).to_string())
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
