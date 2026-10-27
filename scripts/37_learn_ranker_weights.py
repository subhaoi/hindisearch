from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

from _phase4.ranker_v2 import DEFAULT_WEIGHTS, FEATURE_NAMES


def candidates(api: str, query: str, depth: int) -> List[Dict[str, Any]]:
    r = requests.post(f"{api}/search", json={"query": query, "per_page": depth, "explain": True, "log": False},
                      timeout=120)
    r.raise_for_status()
    return [{"id": str(h["id"]), **{f: float((h.get("features") or {}).get(f) or 0.0) for f in FEATURE_NAMES}}
            for h in r.json().get("results", [])]


def collect(api: str, q: pd.DataFrame, depth: int, workers: int) -> List[Dict[str, Any]]:
    """Per question: candidate ids and their ranker_v2 features (as the API computes them)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        got = list(pool.map(lambda x: candidates(api, x, depth), q["query"]))
    out = []
    for (query, gold, style, split), cands in zip(q[["query", "article_id", "style", "split"]].itertuples(index=False), got):
        if cands:
            out.append({"query": query, "gold": str(gold), "style": style, "split": split,
                        "ids": [c["id"] for c in cands], "X": np.array([[c[f] for f in FEATURE_NAMES] for c in cands])})
    return out


def ranks(data: List[Dict[str, Any]], w: np.ndarray) -> np.ndarray:
    """Rank of each question's source article under weights w (0 = not in the candidates)."""
    out = np.zeros(len(data), dtype=int)
    for i, d in enumerate(data):
        if d["gold"] in d["ids"]:
            s = d["X"] @ w
            g = d["ids"].index(d["gold"])
            # Ties go against the source article (no credit for lucky ordering)
            out[i] = int((s >= s[g]).sum())
    return out


def summary(r: np.ndarray) -> Dict[str, float]:
    return {"n": int(len(r)), "recall@1": float((r == 1).mean()), "recall@10": float(((r > 0) & (r <= 10)).mean()),
            "mrr@10": float(np.where((r > 0) & (r <= 10), 1.0 / np.maximum(r, 1), 0).mean())}


def objective(data: List[Dict[str, Any]], w: np.ndarray) -> float:
    s = summary(ranks(data, w))
    return s["mrr@10"] + 0.5 * s["recall@10"]


def coordinate_search(data: List[Dict[str, Any]], start: np.ndarray, rounds: int = 4) -> np.ndarray:
    """Improve one weight at a time over a grid of multiples; rrf_lex stays 1.0 (scores are relative)."""
    w = start.copy()
    best = objective(data, w)
    factors = [0.0, 0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0]
    for _ in range(rounds):
        improved = False
        for i, name in enumerate(FEATURE_NAMES):
            if name == "rrf_lex":
                continue
            base = w[i] if w[i] > 0 else 0.05
            for f in factors:
                trial = w.copy()
                trial[i] = base * f
                val = objective(data, trial)
                if val > best + 1e-6:
                    best, w, improved = val, trial, True
        if not improved:
            break
    return w


def graded(data: List[Dict[str, Any]], w: np.ndarray, grades: Dict[tuple, int]) -> Dict[str, float]:
    """Using LLM relevance grades: share of questions with a relevant (2) article in the top 10 and
    at rank 1, and how much of the top 10 was graded at all (ungraded counts as not relevant)."""
    top1 = top10 = judged = shown = 0
    for d in data:
        order = [d["ids"][j] for j in np.argsort(-(d["X"] @ w), kind="stable")[:10]]
        labels = [grades.get((d["query"], a)) for a in order]
        shown += len(order)
        judged += sum(lab is not None for lab in labels)
        top10 += any(lab == 2 for lab in labels)
        top1 += bool(labels) and labels[0] == 2
    n = max(1, len(data))
    return {"n": len(data), "relevant_in_top10": top10 / n, "relevant_at_1": top1 / n, "graded_share": judged / max(1, shown)}


def main() -> None:
    """
    Learn ranker_v2 weights from generated questions instead of hand labels: collect each
    candidate's features from a running API (explain=true), search for the weights that put the
    question's source article highest on TRAINING questions, then compare default and learned
    weights on the TEST questions (strict: the source article; graded: any article the LLM judged
    relevant, from data/judgments/<lang>.parquet). Writes the weights file only with --write.

      python scripts/37_learn_ranker_weights.py --api http://localhost:8000
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--split", default="data/synthetic/split_hi.parquet")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--train-sample", type=int, default=3000)
    ap.add_argument("--test-sample", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--articles", default=None, help="Keep only questions about articles still on the site")
    ap.add_argument("--report", default=None)
    ap.add_argument("--features", default=None, help="Save collected features here, or reuse them if the file exists")
    ap.add_argument("--write", default=None, help="Weights file to write (e.g. data/phase_4/ranker_v2_weights.json)")
    args = ap.parse_args()

    q = pd.read_parquet(args.split)
    if args.articles:
        q = q[q["article_id"].astype(str).isin(set(pd.read_parquet(args.articles)["id"].astype(str)))]
    train_q = q[q["split"] == "train"].sample(min(args.train_sample, (q["split"] == "train").sum()), random_state=args.seed)
    # Same test questions as 29_eval_search_on_questions.py (same seed and sample size)
    test_all = q[q["split"].isin(["test_q", "test_new"])]
    test_q = test_all.sample(min(args.test_sample, len(test_all)), random_state=args.seed)

    fpath = Path(args.features) if args.features else None
    if fpath and fpath.exists():
        train, test = pickle.loads(fpath.read_bytes())
        print(f"Reusing features from {fpath}: {len(train)} training + {len(test)} test questions")
    else:
        print(f"Collecting features: {len(train_q)} training + {len(test_q)} test questions", flush=True)
        train = collect(args.api, train_q, args.depth, args.workers)
        test = collect(args.api, test_q, args.depth, args.workers)
        if fpath:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(pickle.dumps((train, test)))

    w0 = np.array([DEFAULT_WEIGHTS[f] for f in FEATURE_NAMES])
    w1 = coordinate_search(train, w0)
    learned = {f: round(float(v), 4) for f, v in zip(FEATURE_NAMES, w1)}

    gpath = Path(f"data/judgments/{args.lang}.parquet")
    grades: Dict[tuple, int] = {}
    if gpath.exists():
        g = pd.read_parquet(gpath)
        grades = {(qq, str(a)): int(lab) for qq, a, lab in zip(g["query"], g["article_id"], g["label"])}

    report: Dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(), "api": args.api, "depth": args.depth,
        "default_weights": dict(DEFAULT_WEIGHTS), "learned_weights": learned,
        "train": {"default": summary(ranks(train, w0)), "learned": summary(ranks(train, w1))},
        "test": {"default": summary(ranks(test, w0)), "learned": summary(ranks(test, w1))},
        "test_graded": {"default": graded(test, w0, grades), "learned": graded(test, w1, grades)} if grades else None,
        "test_by_style": {
            st: {"default": summary(ranks([d for d in test if d["style"] == st], w0)),
                 "learned": summary(ranks([d for d in test if d["style"] == st], w1))}
            for st in sorted({d["style"] for d in test})},
    }
    out = Path(args.report or f"runs/gpu/ranker_weights_{args.lang}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    print("Learned weights:", learned)
    for part in ("train", "test"):
        for k in ("default", "learned"):
            s = report[part][k]
            print(f"  {part:5} {k:8} recall@1 {s['recall@1']:.3f}  recall@10 {s['recall@10']:.3f}  mrr@10 {s['mrr@10']:.3f}")
    if report["test_graded"]:
        for k in ("default", "learned"):
            s = report["test_graded"][k]
            print(f"  graded {k:8} relevant in top 10 {s['relevant_in_top10']:.3f}  relevant at 1 {s['relevant_at_1']:.3f}"
                  f"  (top-10 graded: {s['graded_share']:.0%})")
    print(f"Wrote: {out}")
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(json.dumps({"weights": learned, "source": str(out),
                                                "trained_at": report["at"]}, indent=1), encoding="utf-8")
        print(f"Wrote weights: {args.write}")


if __name__ == "__main__":
    main()
