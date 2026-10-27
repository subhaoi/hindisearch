from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import classify_query, clean_title, is_nullish  # noqa: E402

LABELS = {0: "not relevant", 1: "partly relevant", 2: "relevant"}
FIELDS = {"hi": ("title_hi", "summary_hi", "data/final/articles_canonical.parquet"),
          "en": ("title", "summary", "data/en/final/articles_canonical.parquet")}


def s(v: Any) -> str:
    return "" if is_nullish(v) else str(v).strip()


def pairs_from_logs(searches: str, results: str) -> pd.DataFrame:
    """Each distinct real search (latest run) with its top 10, leaving out years, junk and empty queries."""
    sq = pd.read_csv(searches)
    sq = sq[sq["query_raw"].fillna("").str.strip() != ""]
    sq = sq[sq["query_raw"].map(lambda q: classify_query(str(q))[0] == "search")]
    sq = sq.assign(k=sq["query_raw"].str.split().str.join(" ").str.casefold())
    latest = sq.sort_values("query_id").drop_duplicates("k", keep="last")
    res = pd.read_csv(results, usecols=["query_id", "article_id"])
    out = latest[["query_id", "query_raw"]].merge(res, on="query_id")
    return out.rename(columns={"query_raw": "query"})[["query", "article_id"]].assign(source="logs")


def judge_group(query: str, arts: List[Dict[str, str]], args: argparse.Namespace) -> List[int]:
    items = "\n\n".join(f"[{i + 1}] Title: {a['title']}\nSummary: {a['summary'][:350]}" for i, a in enumerate(arts))
    prompt = f"""You judge search results for {args.site}.
A reader typed this into the site's search box: "{query}"

For each article below, judge how well it answers what the reader is looking for:
2 = relevant: a good result for this search
1 = partly relevant: related, but not really what they want
0 = not relevant

Judge the article's topic, not whether it repeats the exact words. Short or vague searches
(one word) are relevant for articles mainly about that topic.

{items}

Answer as JSON: {{"judgments": [{{"n": 1, "label": 2}}, ...]}} with one entry per article."""
    r = requests.post(f"{args.ollama}/api/generate", timeout=900, json={
        "model": args.model, "prompt": prompt, "format": "json", "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 400},
        **({"think": False} if args.model.startswith("qwen3") else {})})
    r.raise_for_status()
    got = {int(j["n"]): int(j["label"]) for j in json.loads(r.json()["response"]).get("judgments", [])}
    return [got.get(i + 1, -1) for i in range(len(arts))]


def main() -> None:
    """
    Relevance judgments by a local model for (query, article) pairs, cached in
    data/judgments/<lang>.parquet so every pair is judged once. Used to score rankings on
    real searches (not only generated ones) and as ranker training labels.

      python scripts/gpu/33_judge.py --lang hi --from-logs        # real logged searches, their top 10
      python scripts/gpu/33_judge.py --lang hi --pairs more.parquet  # e.g. a new model's top 10
    Spot-check a sample against your own judgment before relying on the labels.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi", choices=sorted(FIELDS))
    ap.add_argument("--from-logs", action="store_true")
    ap.add_argument("--searches", default="data/exports/searches_all.csv")
    ap.add_argument("--results", default="data/exports/results_top10_all.csv")
    ap.add_argument("--pairs", default=None, help="Parquet with query, article_id")
    ap.add_argument("--model", default="gemma3:12b")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.site = "IDR Hindi, a website about the Indian development sector" if args.lang == "hi" else \
        "IDR (India Development Review), a website about the Indian development sector"

    pairs = pairs_from_logs(args.searches, args.results) if args.from_logs else pd.read_parquet(args.pairs)
    pairs["article_id"] = pairs["article_id"].astype(str)
    cache_path = Path(f"data/judgments/{args.lang}.parquet")
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["query", "article_id", "label", "model"])
    done = set(zip(cache["query"], cache["article_id"].astype(str)))
    todo = pairs[[(q, a) not in done for q, a in zip(pairs["query"], pairs["article_id"])]].drop_duplicates(["query", "article_id"])

    t_col, s_col, path = FIELDS[args.lang]
    arts = pd.read_parquet(path)
    info = {str(r["id"]): {"title": clean_title(s(r[t_col])), "summary": s(r[s_col])} for _, r in arts.iterrows()}
    todo = todo[todo["article_id"].isin(info)]
    groups = [(q, g["article_id"].tolist()[i:i + 10]) for q, g in todo.groupby("query") for i in range(0, len(g), 10)]
    print(f"Pairs: {len(pairs)} | already judged: {len(pairs) - len(todo)} | to judge: {len(todo)} in {len(groups)} prompts")

    new_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(judge_group, q, [info[a] for a in ids], args): (q, ids) for q, ids in groups}
        for i, f in enumerate(as_completed(futs), 1):
            q, ids = futs[f]
            try:
                labels = f.result()
            except (requests.RequestException, ValueError, KeyError) as e:
                print(f"  skipped {q!r}: {e}")
                continue
            new_rows += [{"query": q, "article_id": a, "label": lab, "model": args.model}
                         for a, lab in zip(ids, labels) if lab in LABELS]
            if i % 50 == 0 or i == len(groups):
                print(f"  [{i}/{len(groups)}] judged", flush=True)
                out = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                out.to_parquet(cache_path, index=False)
    out = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_parquet(cache_path, index=False)
    print(out["label"].map(LABELS).value_counts().to_string())
    print(f"Wrote: {cache_path} ({len(out)} judgments)")


if __name__ == "__main__":
    main()
