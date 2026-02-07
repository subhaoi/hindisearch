from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from utils import (
    Paths,
    ensure_dir,
    read_parquet,
    write_parquet,
    write_json,
    is_nullish,
    safe_join,
    build_tokenizer_for_mpnet,
    e5_prefix_text,
)


def truncate_to_max_tokens(tokenizer, text: str, hard_max_tokens: int) -> Tuple[str, bool, int]:
    """
    Returns (possibly truncated_text, was_truncated, token_count_after)
    """
    t = "" if is_nullish(text) else str(text)
    ids = tokenizer.encode(t, add_special_tokens=False)
    if len(ids) <= hard_max_tokens:
        return t, False, len(ids)
    ids2 = ids[:hard_max_tokens]
    t2 = tokenizer.decode(ids2).strip()
    return t2, True, len(ids2)


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--articles", default="data/final/articles_canonical.parquet", help="Canonical articles parquet")
    ap.add_argument("--chunks", default="data/phase_3/chunks.parquet", help="Chunks parquet")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hard-max-tokens", type=int, default=384, help="Hard cap for E5 (<=512)")
    args = ap.parse_args()

    if args.hard_max_tokens > 512:
        raise ValueError("hard-max-tokens must be <= 512 for intfloat/multilingual-e5-large")

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    ensure_dir(paths.data / "phase_3")
    ensure_dir(paths.logs)

    articles = read_parquet(Path(args.articles).resolve())
    chunks = read_parquet(Path(args.chunks).resolve())

    model_name = "intfloat/multilingual-e5-large"
    model = SentenceTransformer(model_name)
    tokenizer = build_tokenizer_for_mpnet()

    report: Dict[str, Any] = {
        "model": model_name,
        "dim": model.get_sentence_embedding_dimension(),
        "articles": len(articles),
        "chunks": len(chunks),
        "batch_size": args.batch_size,
        "hard_max_tokens": args.hard_max_tokens,
        "article_truncated": 0,
        "chunk_truncated": 0,
        "examples": [],
    }

    # ---- Article vectors: title + summary ----
    article_texts: List[str] = []
    article_ids: List[str] = []
    article_trunc_flags: List[bool] = []

    for _, r in articles.iterrows():
        aid = str(r.get("id"))
        title = "" if is_nullish(r.get("title_hi")) else str(r.get("title_hi"))
        summary = "" if is_nullish(r.get("summary_hi")) else str(r.get("summary_hi"))
        txt = safe_join([title, summary], sep="\n\n").strip()
        if not txt:
            txt = title.strip()

        txt2, was_trunc, tok_ct = truncate_to_max_tokens(tokenizer, txt, args.hard_max_tokens)
        txt2 = e5_prefix_text(txt2, "passage")
        if was_trunc:
            report["article_truncated"] += 1
            if len(report["examples"]) < 10:
                report["examples"].append({"type": "article", "id": aid, "tokens": tok_ct, "text_preview": txt2[:120]})

        article_ids.append(aid)
        article_texts.append(txt2)
        article_trunc_flags.append(was_trunc)

    print(f"Embedding articles: {len(article_texts)} texts")
    art_vecs = embed_texts(model, article_texts, batch_size=args.batch_size)

    art_out = articles[["id", "url", "published_date"]].copy()
    if "published_ts" in articles.columns:
        art_out["published_ts"] = articles["published_ts"]
    else:
        art_out["published_ts"] = 0

    art_out["vector"] = [v.astype(np.float32).tolist() for v in art_vecs]
    art_out["was_truncated"] = article_trunc_flags

    art_path = paths.data / "phase_3" / "article_vectors.parquet"
    write_parquet(art_out, art_path)

    # ---- Chunk vectors ----
    chunk_ids = chunks["chunk_id"].astype(str).tolist()
    raw_chunk_texts = chunks["chunk_text"].fillna("").astype(str).tolist()

    chunk_texts: List[str] = []
    chunk_trunc_flags: List[bool] = []

    for cid, txt in zip(chunk_ids, raw_chunk_texts):
        txt2, was_trunc, tok_ct = truncate_to_max_tokens(tokenizer, txt, args.hard_max_tokens)
        txt2 = e5_prefix_text(txt2, "passage")
        if was_trunc:
            report["chunk_truncated"] += 1
            if len(report["examples"]) < 10:
                report["examples"].append({"type": "chunk", "id": cid, "tokens": tok_ct, "text_preview": txt2[:120]})
        chunk_texts.append(txt2)
        chunk_trunc_flags.append(was_trunc)

    print(f"Embedding chunks: {len(chunk_texts)} texts")
    chk_vecs = embed_texts(model, chunk_texts, batch_size=args.batch_size)

    chk_out = chunks[[
        "chunk_id", "article_id", "chunk_index", "url", "published_date",
        "published_ts", "title_hi", "chunk_tokens"
    ]].copy()
    chk_out["vector"] = [v.astype(np.float32).tolist() for v in chk_vecs]
    chk_out["was_truncated"] = chunk_trunc_flags

    chk_path = paths.data / "phase_3" / "chunk_vectors.parquet"
    write_parquet(chk_out, chk_path)

    report["article_vectors_path"] = str(art_path)
    report["chunk_vectors_path"] = str(chk_path)

    write_json(paths.logs / "phase3_embedding_report.json", report)

    print(f"Wrote: {art_path}")
    print(f"Wrote: {chk_path}")
    print(f"Wrote: {paths.logs / 'phase3_embedding_report.json'}")
    print(f"Articles truncated: {report['article_truncated']}")
    print(f"Chunks truncated: {report['chunk_truncated']}")


if __name__ == "__main__":
    main()
