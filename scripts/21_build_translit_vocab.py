from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from utils import (
    Paths, ensure_dir, read_parquet, write_json, is_nullish,
    query_tokens, text_to_key_variants, DEVANAGARI_RE,
)


def main() -> None:
    """
    Build a roman match key -> Devanagari word map from the corpus, used by the search API
    to turn romanized Hindi queries ("kisan andolan") into Devanagari (किसान आंदोलन) before
    embedding. Every word is indexed under its keys with and without schwa deletion, so both
    "yojna" and "yojana" resolve to योजना.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/final/articles_canonical.parquet")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="data/phase_45/translit_vocab_v1.json")
    args = ap.parse_args()

    paths = Paths(root=Path(args.root).resolve())
    ensure_dir(paths.data / "phase_45")
    df = read_parquet(Path(args.input).resolve())

    word_freq: Counter = Counter()
    for col in ["title_hi", "summary_hi", "content_hi"]:
        if col not in df.columns:
            continue
        for text in tqdm(df[col].tolist(), desc=f"Counting words in {col}"):
            if is_nullish(text):
                continue
            word_freq.update(t for t in query_tokens(str(text)) if DEVANAGARI_RE.search(t))

    # key -> Counter(word). The primary (schwa-deleted) spelling outweighs the variants.
    by_key: Dict[str, Counter] = defaultdict(Counter)
    for word, freq in tqdm(word_freq.items(), desc="Building keys"):
        for i, key in enumerate(text_to_key_variants(word)):
            if len(key) >= 2:
                by_key[key][word] += freq * (3 if i == 0 else 1)

    vocab = {key: words.most_common(1)[0][0] for key, words in by_key.items()}

    out_path = Path(args.out).resolve()
    write_json(out_path, {"version": "translit_vocab_v1", "words": len(word_freq), "vocab": vocab})
    print(f"Wrote: {out_path}")
    print(f"Distinct Devanagari words: {len(word_freq)} | keys: {len(vocab)}")


if __name__ == "__main__":
    main()
