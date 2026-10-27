from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import requests

from utils import read_parquet, clean_title, is_nullish, query_tokens

# Offline, one-time data preparation on a laptop: an open model served by Ollama
# (https://ollama.com) writes the searches each article could answer. The search service
# never calls a model; the output is used to build a large test set and ranker training data.

# Per site: the article columns, the site's name for the prompt, and how many queries of each
# kind per article. Hindi: roughly the mix in the search logs (short words dominate; about 60%
# of real searches are typed in English letters). English: no logs yet, so the usual mix of
# keywords, phrases, questions and names, plus Indian terms in English letters.
LANGS: Dict[str, Dict[str, Any]] = {
    "hi": {
        "site": "IDR Hindi, a website about the Indian development sector",
        "fields": {"title": "title_hi", "summary": "summary_hi", "content": "content_hi",
                   "authors": "contributors_norm", "places": "locations_norm", "tags": "tags_norm"},
        "styles": {
            "hindi_keywords": ("1-3 Hindi words in Devanagari, the way people type in a search box", 5),
            "hinglish": ("1-4 Hindi words typed in English letters (e.g. 'mahila swasthya', 'jal sankat')", 5),
            "english": ("1-4 English words (e.g. 'tribal land rights')", 4),
            "question": ("a short natural question in Hindi (Devanagari) that this article answers", 2),
            "entity": ("a person, place, organisation, scheme or law named in the article, alone or with one word", 2),
            "misspelt": ("a Hindi or Hinglish search with a common typing mistake", 2),
        },
    },
    "en": {
        "site": "IDR (India Development Review), an English website about the Indian development sector",
        "fields": {"title": "title", "summary": "summary", "content": "content",
                   "authors": "contributors", "places": "locations", "tags": "tags"},
        "styles": {
            "keywords": ("1-3 English words, the way people type in a search box (e.g. 'csr funding')", 6),
            "phrase": ("a natural 3-6 word search (e.g. 'how to measure nonprofit impact')", 4),
            "question": ("a short question this article answers", 3),
            "entity": ("a person, place, organisation, scheme or law named in the article, alone or with one word", 3),
            "indian_term": ("an Indian term as Indians type it in English letters (e.g. 'anganwadi workers', 'gram sabha')", 2),
            "misspelt": ("a search with a common typing mistake", 2),
        },
    },
}


def as_text(v: Any) -> str:
    return "" if is_nullish(v) else str(v).strip()


def as_list(v: Any) -> List[str]:
    if v is None or isinstance(v, (str, bytes, float, int)):
        return []
    try:
        return [str(x) for x in list(v) if not is_nullish(x)]
    except TypeError:
        return []


def build_prompt(row: Dict[str, Any], max_chars: int, lang: Dict[str, Any]) -> str:
    fld = lang["fields"]
    wanted = "\n".join(f"- {n} x {name}: {desc}" for name, (desc, n) in lang["styles"].items())
    meta = []
    if as_list(row.get(fld["authors"])):
        meta.append("Authors: " + ", ".join(as_list(row.get(fld["authors"]))))
    if as_list(row.get(fld["places"])):
        meta.append("Places: " + ", ".join(as_list(row.get(fld["places"]))))
    if as_list(row.get(fld["tags"])):
        meta.append("Tags: " + ", ".join(as_list(row.get(fld["tags"]))[:8]))
    first_style = next(iter(lang["styles"]))
    return f"""You help build a test set for the search box of {lang["site"]}.
Read the article and write the searches that real readers would type to find it.

Rules:
- Real searches are short. Do not copy the headline.
- Each search must be something this article is a good answer to.
- No numbering, no quotes, no explanations.

Write exactly these searches:
{wanted}

Title: {clean_title(as_text(row.get(fld["title"])))}
Summary: {as_text(row.get(fld["summary"]))}
{chr(10).join(meta)}
Article (start):
{as_text(row.get(fld["content"]))[:max_chars]}

Answer as JSON: {{"queries": [{{"q": "...", "style": "{first_style}"}}, ...]}}"""


def clean_queries(raw: List[Dict[str, Any]], title: str, max_words: int, styles: Dict[str, Any]) -> List[Dict[str, str]]:
    """Drop empty, overlong, duplicate and headline-copy queries."""
    seen: Set[str] = set()
    title_key = " ".join(query_tokens(clean_title(title)))
    out = []
    for item in raw:
        q = " ".join(str(item.get("q", "")).replace('"', " ").split()).strip(" .,-")
        style = str(item.get("style", ""))
        key = " ".join(query_tokens(q))
        if not key or style not in styles or key in seen or key == title_key:
            continue
        if len(key.split()) > max_words:
            continue
        seen.add(key)
        out.append({"query": q, "style": style})
    return out


def generate(row: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, str]]:
    r = requests.post(
        f"{args.ollama.rstrip('/')}/api/generate",
        json={
            "model": args.model,
            "prompt": build_prompt(row, args.max_chars, LANGS[args.lang]),
            # Plain JSON mode: a full JSON-schema grammar makes Gemma (262k-word vocabulary) crawl
            # on the CPU-side sampler; styles are checked in clean_queries instead
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.7, "num_ctx": 8192, "num_predict": 1200},
            # Qwen3 reasons step by step by default; not needed to list searches, and slow
            **({"think": False} if args.model.startswith("qwen3") else {}),
        },
        timeout=900,
    )
    r.raise_for_status()
    raw = json.loads(r.json().get("response") or "{}").get("queries", [])
    lang = LANGS[args.lang]
    return clean_queries(raw, as_text(row.get(lang["fields"]["title"])), args.max_words, lang["styles"])


def done_ids(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    ids = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(str(json.loads(line)["article_id"]))
            except (ValueError, KeyError):
                continue
    return ids


def main() -> None:
    """
    Writes data/synthetic/queries.jsonl: one line per query
      {"article_id", "query", "style", "model", "created_at"}
    Safe to stop (Ctrl+C) and re-run: articles already in the file are skipped.

      ollama pull gemma3:4b
      python scripts/28_generate_queries.py --limit 5        # try it, check the output
      python scripts/28_generate_queries.py                  # everything (runs for hours)
      OLLAMA_NUM_PARALLEL=4 ollama serve & ... --workers 4   # on a GPU
      ... --lang en --articles data/en/final/articles_canonical.parquet --out data/synthetic/queries_en.jsonl
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi", choices=sorted(LANGS), help="Which site the articles are from")
    ap.add_argument("--articles", default="data/final/articles_canonical.parquet")
    ap.add_argument("--out", default="data/synthetic/queries.jsonl")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--max-chars", type=int, default=2500, help="Article text sent per prompt (speed vs context)")
    ap.add_argument("--max-words", type=int, default=12, help="Drop longer queries")
    ap.add_argument("--limit", type=int, default=0, help="Only this many new articles (0 = all)")
    ap.add_argument("--newest-first", action="store_true", help="Start with the newest articles")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel requests; on a GPU use 4 and start Ollama with OLLAMA_NUM_PARALLEL=4")
    args = ap.parse_args()

    df = read_parquet(Path(args.articles))
    if args.newest_first and "published_date" in df.columns:
        df = df.sort_values("published_date", ascending=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(out_path)
    todo = [r for r in df.to_dict("records") if str(r.get("id")) not in skip]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Articles: {len(df)} | already done: {len(skip)} | to do now: {len(todo)} | model: {args.model}")

    started = time.time()
    n_queries = 0
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = {pool.submit(generate, row, args): row for row in todo}
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for i, fut in enumerate(as_completed(futures), start=1):
                row = futures[fut]
                try:
                    qs = fut.result()
                except (requests.RequestException, ValueError) as e:
                    print(f"  {row.get('id')}: skipped ({e}); re-run to retry")
                    continue
                now = datetime.now(timezone.utc).isoformat()
                # Only this thread writes; workers just call the model
                for q in qs:
                    f.write(json.dumps({"article_id": str(row.get("id")), **q, "model": args.model, "created_at": now},
                                       ensure_ascii=False) + "\n")
                f.flush()
                n_queries += len(qs)
                per = (time.time() - started) / i
                print(f"[{i}/{len(todo)}] {row.get('id')}: {len(qs)} queries"
                      f" | about {per * (len(todo) - i) / 3600:.1f} h left", flush=True)
    except KeyboardInterrupt:
        # Don't wait for the queued articles; finished ones are already saved
        pool.shutdown(wait=False, cancel_futures=True)
        print("Stopped. Re-run the same command to continue.")
        return
    pool.shutdown()
    print(f"Done: {n_queries} queries written to {out_path}")


if __name__ == "__main__":
    main()
