from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from utils import Paths, ensure_dir, read_parquet, write_parquet, write_json, is_nullish, safe_join


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--articles", default="data/final/articles_canonical.parquet", help="Canonical articles parquet")
    ap.add_argument("--chunks", default="data/phase_3/chunks.parquet", help="Chunks parquet")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    ensure_dir(paths.data / "phase_3")
    ensure_dir(paths.logs)

    articles = read_parquet(Path(args.articles).resolve())
    chunks = read_parquet(Path(args.chunks).resolve())

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    report: Dict[str, Any] = {
        "model": model_name,
        "dim": model.get_sentence_embedding_dimension(),
        "articles": len(articles),
        "chunks": len(chunks),
        "batch_size": args.batch_size,
    }

    # Article vectors: title + summary (high-level intent)
    article_texts: List[str] = []
    article_ids: List[str] = []
    for _, r in articles.iterrows():
        aid = str(r.get("id"))
        title = "" if is_nullish(r.get("title_hi")) else str(r.get("title_hi"))
        summary = "" if is_nullish(r.get("summary_hi")) else str(r.get("summary_hi"))
        txt = safe_join([title, summary], sep="\n\n").strip()
        if not txt:
            txt = title.strip()
        article_ids.append(aid)
        article_texts.append(txt)

    art_vecs = embed_texts(model, article_texts, batch_size=args.batch_size)

    art_out = articles[["id", "url", "published_date"]].copy()
    if "published_ts" in articles.columns:
        art_out["published_ts"] = articles["published_ts"]
    else:
        art_out["published_ts"] = 0

    # Store as list[float] for parquet simplicity
    art_out["vector"] = [v.astype(np.float32).tolist() for v in art_vecs]
    art_path = paths.data / "phase_3" / "article_vectors.parquet"
    write_parquet(art_out, art_path)

    # Chunk vectors
    chunk_texts = chunks["chunk_text"].fillna("").astype(str).tolist()
    chunk_ids = chunks["chunk_id"].astype(str).tolist()

    chk_vecs = embed_texts(model, chunk_texts, batch_size=args.batch_size)

    chk_out = chunks[["chunk_id", "article_id", "chunk_index", "url", "published_date", "published_ts", "title_hi", "chunk_tokens"]].copy()
    chk_out["vector"] = [v.astype(np.float32).tolist() for v in chk_vecs]
    chk_path = paths.data / "phase_3" / "chunk_vectors.parquet"
    write_parquet(chk_out, chk_path)

    report["article_vectors_path"] = str(art_path)
    report["chunk_vectors_path"] = str(chk_path)

    write_json(paths.logs / "phase3_embedding_report.json", report)

    print(f"Wrote: {art_path}")
    print(f"Wrote: {chk_path}")
    print(f"Wrote: {paths.logs / 'phase3_embedding_report.json'}")


if __name__ == "__main__":
    main()
