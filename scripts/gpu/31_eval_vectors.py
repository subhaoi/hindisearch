from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import (load_corpus, load_model, encode, prefixes, rank_articles, metrics, gold_index,
                    chunk_article_index)


def cpu_latency_ms(name: str, queries: list, threads: int = 2) -> float:
    """Query encoding time on `threads` CPU threads, roughly the search server (t3.large: 2 vCPU)."""
    import torch
    torch.set_num_threads(threads)
    m = load_model(name, device="cpu")
    qp = prefixes(name)[0]
    m.encode([qp + queries[0]])
    t0 = time.perf_counter()
    for q in queries:
        m.encode([qp + q])
    return (time.perf_counter() - t0) * 1000 / len(queries)


def main() -> None:
    """
    How well one embedding model finds each test query's article, scored the way the search
    API fuses its vector results (best chunk + article vector). Vector search only: no keywords,
    no ranking boosts, so it isolates the model. Writes runs/gpu/<lang>_<model>.json.

      python scripts/gpu/31_eval_vectors.py --lang hi --model intfloat/multilingual-e5-large
      python scripts/gpu/31_eval_vectors.py --lang hi --model models/e5-large-idr-ft
      python scripts/gpu/31_eval_vectors.py --lang en --model BAAI/bge-base-en-v1.5 --cpu-latency
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi", choices=["hi", "en"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default=None, help="Default data/synthetic/split_<lang>.parquet")
    ap.add_argument("--sets", default="test_q,test_new")
    ap.add_argument("--cpu-latency", action="store_true", help="Also time query encoding on 2 CPU threads")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    split_path = Path(args.split or f"data/synthetic/split_{args.lang}.parquet")
    q = pd.read_parquet(split_path)
    q = q[q["split"].isin(args.sets.split(","))].reset_index(drop=True)
    articles, chunks = load_corpus(args.lang)
    chunk_art = chunk_article_index(articles, chunks)
    keep = chunk_art >= 0
    chunks, chunk_art = chunks[keep].reset_index(drop=True), chunk_art[keep]

    t0 = time.time()
    model = load_model(args.model)
    qp, pp, _ = prefixes(args.model)
    art_vecs = encode(model, articles["article_text"].tolist(), pp)
    chunk_vecs = encode(model, chunks["chunk_text"].tolist(), pp)
    q_vecs = encode(model, q["query"].tolist(), qp, batch_size=256)
    encode_s = time.time() - t0

    gold, ok = gold_index(articles, q["article_id"])
    top = np.full((len(q), 10), -1)
    for s in range(0, len(q), 512):
        top[s:s + 512] = rank_articles(q_vecs[s:s + 512], art_vecs, chunk_vecs, chunk_art, len(articles))
    q = q[ok].reset_index(drop=True)
    top, gold = top[ok], gold[ok]

    result = {"lang": args.lang, "model": args.model, "at": datetime.now(timezone.utc).isoformat(),
              "articles": len(articles), "chunks": len(chunks), "encode_seconds": round(encode_s, 1),
              "overall": metrics(top, gold), "by_set": {}, "by_style": {}}
    for name, g in q.groupby("split"):
        result["by_set"][name] = metrics(top[g.index], gold[g.index])
    for name, g in q.groupby("style"):
        result["by_style"][name] = metrics(top[g.index], gold[g.index])
    if args.cpu_latency:
        del model
        result["cpu_ms_per_query"] = round(cpu_latency_ms(args.model, q["query"].sample(30, random_state=1).tolist()), 1)

    out = Path(args.out or f"runs/gpu/{args.lang}_{Path(args.model).name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    o = result["overall"]
    print(f"{args.model}: recall@1 {o['recall@1']:.3f}  recall@10 {o['recall@10']:.3f}  mrr@10 {o['mrr@10']:.3f}"
          + (f"  | CPU {result['cpu_ms_per_query']} ms/query" if args.cpu_latency else ""))
    for k, v in result["by_set"].items():
        print(f"   {k:9} n={v['n']:5}  recall@10 {v['recall@10']:.3f}  mrr@10 {v['mrr@10']:.3f}")
    for k, v in result["by_style"].items():
        print(f"   {k:15} n={v['n']:5}  recall@10 {v['recall@10']:.3f}  mrr@10 {v['mrr@10']:.3f}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
