from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_corpus, load_model, encode, prefixes, chunk_article_index


def build_triplets(model_name: str, q: pd.DataFrame, articles: pd.DataFrame, chunks: pd.DataFrame,
                   chunk_art: np.ndarray, neg_from: int, neg_to: int, margin: float, seed: int,
                   fallback_from: int = 100, fallback_to: int = 400) -> pd.DataFrame:
    """
    (query, positive passage, hard negative passage) per training query, mined with the model
    being fine-tuned:
      positive  the source article's chunk that the model already finds closest to the query
      negative  a chunk from another article ranked neg_from..neg_to for the query, skipped when
                its score is within `margin` of the positive (probably relevant too: many queries
                fit several articles, and training those as wrong would teach the model mistakes);
                when none qualifies, a chunk ranked fallback_from..fallback_to instead
    """
    import torch
    model = load_model(model_name)
    qp, pp, _ = prefixes(model_name)
    cv = torch.from_numpy(encode(model, chunks["chunk_text"].tolist(), pp)).cuda()
    qv = torch.from_numpy(encode(model, q["query"].tolist(), qp, batch_size=256)).cuda()
    art_of_chunk = torch.from_numpy(chunk_art).cuda()
    pos_art = {a: i for i, a in enumerate(articles["id"])}
    rng = random.Random(seed)
    out = []
    fallbacks = 0
    for s in range(0, len(q), 256):
        sims = qv[s:s + 256] @ cv.T
        for j in range(sims.shape[0]):
            row = q.iloc[s + j]
            gold = pos_art.get(str(row["article_id"]))
            if gold is None:
                continue
            own = (art_of_chunk == gold)
            if not own.any():
                continue
            own_scores = torch.where(own, sims[j], torch.tensor(-2.0, device=sims.device))
            p_idx = int(own_scores.argmax())
            p_score = float(own_scores[p_idx])
            order = torch.where(~own, sims[j], torch.tensor(-2.0, device=sims.device)).argsort(descending=True)
            cands = [int(i) for i in order[neg_from:neg_to] if float(sims[j][i]) < p_score - margin]
            if not cands:
                # Usually a query the model gets wrong (its own article scores below the others):
                # the most useful ones to train on, so keep it with a less similar negative
                cands = [int(i) for i in order[fallback_from:fallback_to]]
                fallbacks += 1
            out.append({"anchor": qp + row["query"], "positive": pp + chunks["chunk_text"].iloc[p_idx],
                        "negative": pp + chunks["chunk_text"].iloc[rng.choice(cands)]})
    del model
    torch.cuda.empty_cache()
    print(f"Negatives from ranks {fallback_from}-{fallback_to} (no hard one qualified): {fallbacks}")
    return pd.DataFrame(out)


def main() -> None:
    """
    Fine-tune an embedding model on the training split of the generated queries (never the
    test splits), with hard negatives. Compare afterwards with 31_eval_vectors.py on the same
    test sets; adopt only if it clearly beats the base model on both test_q and test_new.

      python scripts/gpu/32_finetune.py --lang hi --base intfloat/multilingual-e5-large --out models/e5-large-idr-ft
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi", choices=["hi", "en"])
    ap.add_argument("--base", default="intfloat/multilingual-e5-large")
    ap.add_argument("--out", default="models/e5-large-idr-ft")
    ap.add_argument("--split", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=64, help="In-batch negatives per step (cached, so memory-safe)")
    ap.add_argument("--neg-from", type=int, default=10)
    ap.add_argument("--neg-to", type=int, default=60)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--fallback-from", type=int, default=100)
    ap.add_argument("--fallback-to", type=int, default=400)
    ap.add_argument("--max-seq", type=int, default=384)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    split = pd.read_parquet(args.split or f"data/synthetic/split_{args.lang}.parquet")
    train_q = split[split["split"] == "train"].reset_index(drop=True)
    articles, chunks = load_corpus(args.lang)
    chunk_art = chunk_article_index(articles, chunks)
    keep = chunk_art >= 0
    chunks, chunk_art = chunks[keep].reset_index(drop=True), chunk_art[keep]

    trip = build_triplets(args.base, train_q, articles, chunks, chunk_art, args.neg_from, args.neg_to,
                          args.margin, args.seed, args.fallback_from, args.fallback_to)
    print(f"Training triplets: {len(trip)} from {len(train_q)} training queries")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trip.head(20).to_json(out / "sample_triplets.json", orient="records", force_ascii=False, indent=1)

    _, _, trust = prefixes(args.base)
    model = SentenceTransformer(args.base, device="cuda", trust_remote_code=trust)
    model.max_seq_length = args.max_seq
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
        learning_rate=args.lr, warmup_ratio=0.1, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=20, save_strategy="no", report_to="none", seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=Dataset.from_pandas(trip), loss=loss)
    trainer.train()
    model.save(str(out))
    (out / "idr_training.json").write_text(json.dumps({**vars(args), "triplets": len(trip)}, indent=1), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
