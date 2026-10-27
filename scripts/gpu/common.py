"""Shared helpers for the GPU-session scripts: corpora, model prefixes, encoding, scoring."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import clean_title, is_nullish  # noqa: E402
from embedding_models import PREFIXES, prefixes  # noqa: E402,F401
from scripts.english import article_text, english_chunks  # noqa: E402

# Fusion as in the search API (ranker_v2): normalised reciprocal rank, chunks weight 1.0,
# article vectors 0.5
RRF_K = 60
W_CHUNK, W_ARTICLE = 1.0, 0.5


def _s(v) -> str:
    return "" if is_nullish(v) else str(v).strip()


def load_corpus(lang: str, root: Path = Path(".")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(articles[id, article_text], chunks[chunk_id, article_id, chunk_text]) for one site."""
    if lang == "hi":
        a = pd.read_parquet(root / "data/final/articles_canonical.parquet")
        a["article_text"] = [
            "\n\n".join(x for x in [clean_title(_s(t)), clean_title(_s(st)), _s(su)] if x)
            for t, st, su in zip(a["title_hi"], a.get("seo_title_hi", pd.Series("", index=a.index)), a["summary_hi"])
        ]
        c = pd.read_parquet(root / "data/phase_3/chunks.parquet")[["chunk_id", "article_id", "chunk_text"]]
    else:
        a = pd.read_parquet(root / "data/en/final/articles_canonical.parquet")
        a["article_text"] = [article_text(t, st, su) for t, st, su in zip(a["title"], a["seo_title"], a["summary"])]
        c = english_chunks(a)
    a["id"] = a["id"].astype(str)
    c["article_id"] = c["article_id"].astype(str)
    return a[["id", "article_text"]].reset_index(drop=True), c.reset_index(drop=True)


def load_model(name: str, device: str = "cuda", max_seq_length: int = 512):
    from sentence_transformers import SentenceTransformer
    _, _, trust = prefixes(name)
    m = SentenceTransformer(name, device=device, trust_remote_code=trust)
    m.max_seq_length = min(max_seq_length, m.max_seq_length or max_seq_length)
    if device == "cuda":
        m.half()
    return m


def encode(model, texts: List[str], prefix: str, batch_size: int = 64) -> np.ndarray:
    return model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def rank_articles(q: np.ndarray, art_vecs: np.ndarray, chunk_vecs: np.ndarray, chunk_art: np.ndarray,
                  n_articles: int, top: int = 10) -> np.ndarray:
    """Top-`top` article indexes per query, fusing chunk-max and article similarity like the API."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = torch.from_numpy(q).to(dev)
    ca = torch.from_numpy(chunk_art).to(dev)
    sim_c = Q @ torch.from_numpy(chunk_vecs).to(dev).T                       # [q, chunks]
    best = torch.full((Q.shape[0], n_articles), -2.0, device=dev)
    best = best.scatter_reduce(1, ca.expand_as(sim_c), sim_c, reduce="amax")  # best chunk per article
    sim_a = Q @ torch.from_numpy(art_vecs).to(dev).T                         # [q, articles]

    def rrf(scores):
        order = scores.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(1, scores.shape[1] + 1, device=dev).expand_as(order))
        return (RRF_K + 1) / (RRF_K + ranks.float())

    fused = W_CHUNK * rrf(best) + W_ARTICLE * rrf(sim_a)
    return fused.topk(top, dim=1).indices.cpu().numpy()


def metrics(top_idx: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    hit = top_idx == gold[:, None]
    first = np.where(hit.any(1), hit.argmax(1) + 1, 0)
    return {"n": int(len(gold)), "recall@1": float((first == 1).mean()), "recall@10": float((first > 0).mean()),
            "mrr@10": float(np.where(first > 0, 1.0 / np.maximum(first, 1), 0).mean())}


def gold_index(articles: pd.DataFrame, ids: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """(row index of each query's article, mask of queries whose article is in the corpus)."""
    pos = {a: i for i, a in enumerate(articles["id"])}
    idx = ids.astype(str).map(pos)
    return idx.fillna(-1).astype(int).to_numpy(), idx.notna().to_numpy()


def chunk_article_index(articles: pd.DataFrame, chunks: pd.DataFrame) -> Optional[np.ndarray]:
    pos = {a: i for i, a in enumerate(articles["id"])}
    return chunks["article_id"].map(pos).fillna(-1).astype(np.int64).to_numpy()
