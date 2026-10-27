from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import encode, load_model, prefixes, clean_title, _s

BASE = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"   # Apache-2.0, multilingual incl. Hindi
MAX_LEN = 320
PASSAGE_CHARS = 900


class Corpus:
    """Article text and each article's passages with the search's embedding model, to pick the
    passage a query matches best (what the API shows as the snippet, from best_chunk_id)."""

    def __init__(self, embed_model: str) -> None:
        import torch
        arts = pd.read_parquet("data/final/articles_canonical.parquet")
        self.ids = arts["id"].astype(str).tolist()
        self.pos = {a: i for i, a in enumerate(self.ids)}
        self.head = [clean_title(_s(t)) + "\n" + _s(s) for t, s in zip(arts["title_hi"], arts["summary_hi"])]
        ch = pd.read_parquet("data/phase_3/chunks.parquet")
        ch = ch[ch["article_id"].astype(str).isin(self.pos)].reset_index(drop=True)
        self.chunk_text = ch["chunk_text"].astype(str).tolist()
        self.chunk_art = torch.tensor([self.pos[str(a)] for a in ch["article_id"]], device="cuda")
        self.model = load_model(embed_model)
        self.qp, pp, _ = prefixes(embed_model)
        self.cv = torch.from_numpy(encode(self.model, self.chunk_text, pp)).cuda()

    def query_vecs(self, queries: List[str]):
        import torch
        return torch.from_numpy(encode(self.model, queries, self.qp, batch_size=256)).cuda()

    def best_chunks(self, qv) -> "torch.Tensor":
        """[queries, articles]: index of each article's best chunk, and the article's best score."""
        import torch
        sims = qv @ self.cv.T
        n_art = len(self.ids)
        best = torch.full((sims.shape[0], n_art), -2.0, device="cuda").scatter_reduce(
            1, self.chunk_art.expand_as(sims), sims, reduce="amax")
        # argmax chunk per article: the chunk whose score equals the article max
        is_best = sims == best.gather(1, self.chunk_art.expand_as(sims))
        idx = torch.full((sims.shape[0], n_art), -1, device="cuda", dtype=torch.long)
        cols = torch.arange(sims.shape[1], device="cuda").expand_as(sims)
        idx = idx.scatter_reduce(1, self.chunk_art.expand_as(sims), torch.where(is_best, cols, -1), reduce="amax")
        return best, idx

    def passage(self, art: int, chunk: int) -> str:
        body = self.chunk_text[chunk] if chunk >= 0 else ""
        return (self.head[art] + "\n" + body)[:PASSAGE_CHARS]


def build_pairs(corpus: Corpus, q: pd.DataFrame, n_hard: int, margin: float, seed: int) -> List[tuple]:
    rng = random.Random(seed)
    pairs = []
    for s in range(0, len(q), 256):
        part = q.iloc[s:s + 256]
        best, idx = corpus.best_chunks(corpus.query_vecs(part["query"].tolist()))
        for j, (query, aid) in enumerate(zip(part["query"], part["article_id"].astype(str))):
            gold = corpus.pos.get(aid)
            if gold is None:
                continue
            row, ci = best[j], idx[j]
            p_score = float(row[gold])
            pairs.append((query, corpus.passage(gold, int(ci[gold])), 1.0))
            order = row.argsort(descending=True)[:60].tolist()
            hard = [a for a in order if a != gold and float(row[a]) < p_score - margin][:n_hard]
            others = [a for a in order[20:60] if a != gold and a not in hard]
            negs = hard + ([rng.choice(others)] if others else [])
            pairs += [(query, corpus.passage(a, int(ci[a])), 0.0) for a in negs]
    return pairs


def train(args: argparse.Namespace) -> None:
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader
    split = pd.read_parquet("data/synthetic/split_hi.parquet")
    tr = split[split["split"] == "train"].reset_index(drop=True)
    corpus = Corpus(args.embed_model)
    pairs = build_pairs(corpus, tr, args.hard, args.margin, args.seed)
    print(f"Training pairs: {len(pairs)} ({sum(p[2] for p in pairs):.0f} positive) from {len(tr)} questions", flush=True)
    random.Random(args.seed).shuffle(pairs)
    model = CrossEncoder(BASE, num_labels=1, max_length=MAX_LEN, device="cuda")
    if args.layers:
        # Keep the first N transformer layers: about N/12 of the CPU time per candidate
        enc = model.model.base_model.encoder
        enc.layer = enc.layer[:args.layers]
        model.model.config.num_hidden_layers = args.layers
    loader = DataLoader([InputExample(texts=[a, b], label=y) for a, b, y in pairs], shuffle=True, batch_size=args.batch)
    model.fit(train_dataloader=loader, epochs=args.epochs, warmup_steps=int(0.1 * len(loader) * args.epochs),
              optimizer_params={"lr": args.lr}, use_amp=True, show_progress_bar=False)
    model.save(args.out)
    Path(args.out, "idr_training.json").write_text(json.dumps({**vars(args), "pairs": len(pairs)}, indent=1))
    print(f"Saved: {args.out}")


def metrics(ranks: np.ndarray) -> Dict[str, float]:
    return {"n": int(len(ranks)), "recall@1": float((ranks == 1).mean()), "recall@10": float(((ranks > 0) & (ranks <= 10)).mean()),
            "mrr@10": float(np.where((ranks > 0) & (ranks <= 10), 1.0 / np.maximum(ranks, 1), 0).mean())}


def evaluate(args: argparse.Namespace) -> None:
    from sentence_transformers.cross_encoder import CrossEncoder
    res = pd.read_parquet(args.results)
    corpus = Corpus(args.embed_model)
    models = {name: CrossEncoder(path, num_labels=1, max_length=args.max_len, device="cuda")
              for name, path in [("base", BASE)] + ([("trained", args.out)] if Path(args.out).exists() else [])}
    out: Dict[str, Dict] = {"questions": len(res)}
    ranks: Dict[str, List[int]] = {}
    for s in range(0, len(res), 128):
        part = res.iloc[s:s + 128]
        best, idx = corpus.best_chunks(corpus.query_vecs(part["query"].tolist()))
        for j, r in enumerate(part.itertuples()):
            cands = [a for a in (str(x) for x in list(r.top)[:args.depth]) if a in corpus.pos]
            gold = str(r.article_id)
            orig = {a: i + 1 for i, a in enumerate(cands)}
            ranks.setdefault("search", []).append(orig.get(gold, 0))
            if not cands:
                for k in list(ranks):
                    if k != "search":
                        ranks[k].append(0)
                continue
            texts = [[r.query, corpus.passage(corpus.pos[a], -1 if args.no_passage else int(idx[j][corpus.pos[a]]))]
                     for a in cands]
            for name, m in models.items():
                sc = m.predict(texts, batch_size=64, show_progress_bar=False)
                rr = {a: i + 1 for i, a in enumerate(np.array(cands)[np.argsort(-sc)])}
                ranks.setdefault(f"{name}_only", []).append(rr.get(gold, 0))
                for w in (0.3, 0.5, 0.7):
                    fused = {a: w * 61 / (60 + orig[a]) + (1 - w) * 61 / (60 + rr[a]) for a in cands}
                    order = sorted(cands, key=lambda a: -fused[a])
                    ranks.setdefault(f"{name}_blend{int(w * 10)}", []).append(order.index(gold) + 1 if gold in order else 0)
    styles = res["style"].to_numpy()
    for k, v in ranks.items():
        v = np.array(v)
        out[k] = {"all": metrics(v), **{st: metrics(v[styles == st]) for st in sorted(set(styles))}}
    # CPU time to rerank one search's candidates on 2 threads (as on the search server)
    import torch
    torch.set_num_threads(2)
    m = CrossEncoder(args.out if Path(args.out).exists() else BASE, num_labels=1, max_length=args.max_len, device="cpu")
    sample = [[res["query"].iloc[0], corpus.passage(i, -1)] for i in range(30)]
    m.predict(sample[:5])
    for n in (20, 30):
        t = time.perf_counter()
        for _ in range(3):
            m.predict(sample[:n], batch_size=n)
        out[f"cpu_ms_{n}"] = round((time.perf_counter() - t) * 1000 / 3, 1)
    Path(args.report).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in ranks:
        a = out[k]["all"]
        print(f"{k:16} recall@1 {a['recall@1']:.3f}  recall@10 {a['recall@10']:.3f}  mrr@10 {a['mrr@10']:.3f}")
    print(f"CPU (2 threads, torch fp32): {out['cpu_ms_20']} ms for 20, {out['cpu_ms_30']} ms for 30 candidates")
    print(f"Wrote: {args.report}")


def main() -> None:
    """
    A small cross-encoder that rereads a search's top candidates (query + title, summary, best
    passage) and scores how well each answers it, trained on the training questions only.

      python scripts/gpu/35_reranker.py train
      python scripts/gpu/35_reranker.py eval --results runs/gpu/search_split_hi_..._top30.results.parquet
    eval reorders each test question's top --depth results and compares, on the same questions:
    the search as is, the reranker alone, and blends of both rankings.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--embed-model", default="models/e5-base-hi-idr-v1")
    ap.add_argument("--out", default="models/hi-reranker-v1")
    ap.add_argument("--hard", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--layers", type=int, default=0, help="Keep only the first N layers (0 = all 12)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--results", default=None)
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--report", default="runs/gpu/hi_reranker_eval.json")
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="Tokens read per candidate at eval time")
    ap.add_argument("--no-passage", action="store_true", help="Title + summary only (no best passage)")
    args = ap.parse_args()
    train(args) if args.mode == "train" else evaluate(args)


if __name__ == "__main__":
    main()
