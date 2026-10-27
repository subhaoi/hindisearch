from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.english import english_chunks  # noqa: E402


def main() -> None:
    """
    English articles -> passages for the vector index (scripts/english.py: about 220 words,
    whole paragraphs, the first one headed by title + summary).

      python scripts/en/02_chunk_english.py
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default="data/en/final/articles_canonical.parquet")
    ap.add_argument("--out", default="data/en/phase_3/chunks.parquet")
    args = ap.parse_args()

    chunks = english_chunks(pd.read_parquet(args.articles))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(args.out, index=False)
    print(f"Chunks: {len(chunks)} from {chunks['article_id'].nunique()} articles "
          f"(median {int(chunks['chunk_tokens'].median())} words)")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
