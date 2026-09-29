from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        show_progress_bar=False,  # embed_with_reuse prints line-based progress (readable in logs)
        normalize_embeddings=True,
    )


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_previous_vectors(path: Path, id_col: str, model_name: str) -> Dict[Tuple[str, str], List[float]]:
    """
    {(id, sha1 of the exact text embedded): vector} from an earlier run, so unchanged
    articles/chunks are not re-embedded. Older files without hashes are ignored.
    """
    if not path.exists():
        return {}
    prev = read_parquet(path)
    if "text_sha1" not in prev.columns or "model" not in prev.columns:
        return {}
    prev = prev[prev["model"] == model_name]
    return {(str(i), str(h)): v for i, h, v in zip(prev[id_col], prev["text_sha1"], prev["vector"])}


CHECKPOINT_BLOCK = 512


def load_checkpoint(path: Path, model_name: str) -> Dict[Tuple[str, str], Any]:
    if not path.exists():
        return {}
    ck = read_parquet(path)
    ck = ck[ck["model"] == model_name]
    return {(str(i), str(h)): v for i, h, v in zip(ck["id"], ck["text_sha1"], ck["vector"])}


def embed_with_reuse(
    get_model,
    ids: List[str],
    texts: List[str],
    previous: Dict[Tuple[str, str], Any],
    batch_size: int,
    label: str,
    checkpoint: Path,
    model_name: str,
) -> Tuple[List[List[float]], List[str], int]:
    """
    Returns (vectors, text hashes, number newly embedded). Embeds in blocks and saves a
    checkpoint after each, so an interrupted run resumes instead of starting over.
    """
    hashes = [text_sha1(t) for t in texts]
    previous = {**previous, **load_checkpoint(checkpoint, model_name)}
    vectors: List[Optional[Any]] = [previous.get((i, h)) for i, h in zip(ids, hashes)]
    todo = [k for k, v in enumerate(vectors) if v is None]
    print(f"{label}: {len(texts)} total | reused {len(texts) - len(todo)} | embedding {len(todo)}", flush=True)
    done: List[int] = []
    t0 = time.time()
    for b in range(0, len(todo), CHECKPOINT_BLOCK):
        block = todo[b:b + CHECKPOINT_BLOCK]
        new_vecs = embed_texts(get_model(), [texts[k] for k in block], batch_size=batch_size)
        for k, v in zip(block, new_vecs):
            vectors[k] = v
        done.extend(block)
        write_parquet(pd.DataFrame({
            "id": [ids[k] for k in done], "text_sha1": [hashes[k] for k in done],
            "vector": [np.asarray(vectors[k], dtype=np.float32).tolist() for k in done], "model": model_name,
        }), checkpoint)
        rate = len(done) / max(1e-9, time.time() - t0)
        left = (len(todo) - len(done)) / rate if rate else 0
        print(f"{label}: {len(done)}/{len(todo)} embedded | {rate:.2f}/s | about {left / 60:.0f} min left", flush=True)
    return [np.asarray(v, dtype=np.float32).tolist() for v in vectors], hashes, len(todo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--articles", default="data/final/articles_canonical.parquet", help="Canonical articles parquet")
    ap.add_argument("--chunks", default="data/phase_3/chunks.parquet", help="Chunks parquet")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hard-max-tokens", type=int, default=384, help="Hard cap for E5 (<=512)")
    ap.add_argument("--no-reuse", action="store_true", help="Re-embed everything instead of reusing unchanged vectors")
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
    # Load the model only if something actually needs embedding (saves ~2 GB RAM on quiet weeks)
    _model: Dict[str, SentenceTransformer] = {}

    def get_model() -> SentenceTransformer:
        if "m" not in _model:
            try:
                import torch
                # PyTorch defaults to physical cores; a t3.large's 2 vCPUs are 1 hyperthreaded core
                torch.set_num_threads(max(1, os.cpu_count() or 1))
            except ImportError:
                pass
            _model["m"] = SentenceTransformer(model_name)
        return _model["m"]

    art_path = paths.data / "phase_3" / "article_vectors.parquet"
    chk_path = paths.data / "phase_3" / "chunk_vectors.parquet"
    prev_art = {} if args.no_reuse else load_previous_vectors(art_path, "id", model_name)
    prev_chk = {} if args.no_reuse else load_previous_vectors(chk_path, "chunk_id", model_name)
    tokenizer = build_tokenizer_for_mpnet()

    report: Dict[str, Any] = {
        "model": model_name,
        "dim": 1024,
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
        seo_title = "" if is_nullish(r.get("seo_title_hi")) else str(r.get("seo_title_hi"))
        summary = "" if is_nullish(r.get("summary_hi")) else str(r.get("summary_hi"))
        # Headline + SEO title (often names the topic/place) + summary
        txt = safe_join([title, seo_title, summary], sep="\n\n").strip()
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

    art_vecs, art_hashes, report["article_embedded"] = embed_with_reuse(
        get_model, article_ids, article_texts, prev_art, args.batch_size, "Articles",
        paths.data / "phase_3" / ".embed_checkpoint_articles.parquet", model_name,
    )

    art_out = articles[["id", "url", "published_date"]].copy()
    if "published_ts" in articles.columns:
        art_out["published_ts"] = articles["published_ts"]
    else:
        art_out["published_ts"] = 0

    art_out["vector"] = art_vecs
    art_out["was_truncated"] = article_trunc_flags
    art_out["text_sha1"] = art_hashes
    art_out["model"] = model_name

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

    chk_vecs, chk_hashes, report["chunk_embedded"] = embed_with_reuse(
        get_model, chunk_ids, chunk_texts, prev_chk, args.batch_size, "Chunks",
        paths.data / "phase_3" / ".embed_checkpoint_chunks.parquet", model_name,
    )

    chk_out = chunks[[
        "chunk_id", "article_id", "chunk_index", "url", "published_date",
        "published_ts", "title_hi", "chunk_tokens"
    ]].copy()
    chk_out["vector"] = chk_vecs
    chk_out["was_truncated"] = chunk_trunc_flags
    chk_out["text_sha1"] = chk_hashes
    chk_out["model"] = model_name

    write_parquet(chk_out, chk_path)

    report["article_vectors_path"] = str(art_path)
    report["chunk_vectors_path"] = str(chk_path)

    write_json(paths.logs / "phase3_embedding_report.json", report)
    # Final files are written; checkpoints are no longer needed
    for ck in (paths.data / "phase_3").glob(".embed_checkpoint_*.parquet"):
        ck.unlink()

    print(f"Wrote: {art_path}")
    print(f"Wrote: {chk_path}")
    print(f"Wrote: {paths.logs / 'phase3_embedding_report.json'}")
    print(f"Articles truncated: {report['article_truncated']}")
    print(f"Chunks truncated: {report['chunk_truncated']}")


if __name__ == "__main__":
    main()
