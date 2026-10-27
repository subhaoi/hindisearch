from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from embedding_models import prefixes  # noqa: E402
from scripts.english import article_text  # noqa: E402
from scripts.utils import iso_to_epoch_seconds  # noqa: E402

# Unchanged-text reuse and resumable checkpoints, shared with the Hindi embedding step
_embed = importlib.import_module("11_compute_embeddings")


def main() -> None:
    """
    Article and passage vectors for the English site, with the model the English search API
    uses (EN_EMBED_MODEL). Unchanged texts keep their vectors, so the weekly run only embeds
    new or edited articles; a different model re-embeds everything.

      python scripts/en/03_embed_english.py                                 # EN_EMBED_MODEL
      python scripts/en/03_embed_english.py --model BAAI/bge-base-en-v1.5
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("EN_EMBED_MODEL"))
    ap.add_argument("--articles", default="data/en/final/articles_canonical.parquet")
    ap.add_argument("--chunks", default="data/en/phase_3/chunks.parquet")
    ap.add_argument("--out-dir", default="data/en/phase_3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--no-reuse", action="store_true")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("Set EN_EMBED_MODEL or pass --model")

    from sentence_transformers import SentenceTransformer
    import torch

    _, passage_prefix, trust = prefixes(args.model)
    articles = pd.read_parquet(args.articles)
    chunks = pd.read_parquet(args.chunks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _model: Dict[str, SentenceTransformer] = {}

    def get_model() -> SentenceTransformer:
        if "m" not in _model:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                torch.set_num_threads(max(1, os.cpu_count() or 1))
            m = SentenceTransformer(args.model, device=device, trust_remote_code=trust)
            m.max_seq_length = min(args.max_seq, m.max_seq_length or args.max_seq)
            _model["m"] = m
        return _model["m"]

    art_path, chk_path = out_dir / "article_vectors.parquet", out_dir / "chunk_vectors.parquet"
    prev_art = {} if args.no_reuse else _embed.load_previous_vectors(art_path, "id", args.model)
    prev_chk = {} if args.no_reuse else _embed.load_previous_vectors(chk_path, "chunk_id", args.model)

    art_ids = articles["id"].astype(str).tolist()
    art_texts = [passage_prefix + article_text(t, st, su)
                 for t, st, su in zip(articles["title"], articles["seo_title"], articles["summary"])]
    art_vecs, art_hashes, n_art = _embed.embed_with_reuse(
        get_model, art_ids, art_texts, prev_art, args.batch_size, "Articles",
        out_dir / ".embed_checkpoint_articles.parquet", args.model)
    art_out = pd.DataFrame({
        "id": art_ids, "url": articles["url"], "published_date": articles["published_date"],
        "published_ts": [iso_to_epoch_seconds(d) for d in articles["published_date"]],
        "vector": art_vecs, "text_sha1": art_hashes, "model": args.model,
    })
    art_out.to_parquet(art_path, index=False)

    chk_ids = chunks["chunk_id"].astype(str).tolist()
    chk_texts = [passage_prefix + t for t in chunks["chunk_text"].fillna("").astype(str)]
    chk_vecs, chk_hashes, n_chk = _embed.embed_with_reuse(
        get_model, chk_ids, chk_texts, prev_chk, args.batch_size, "Chunks",
        out_dir / ".embed_checkpoint_chunks.parquet", args.model)
    chk_out = chunks.drop(columns=["chunk_text"]).assign(vector=chk_vecs, text_sha1=chk_hashes, model=args.model)
    chk_out.to_parquet(chk_path, index=False)

    for ck in out_dir.glob(".embed_checkpoint_*.parquet"):
        ck.unlink()
    print(f"Model: {args.model} | dim {len(art_vecs[0]) if art_vecs else 0} | "
          f"embedded {n_art} articles, {n_chk} chunks (rest reused)")
    print(f"Wrote: {art_path}\nWrote: {chk_path}")


if __name__ == "__main__":
    main()
