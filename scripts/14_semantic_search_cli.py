from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from utils import Paths, canonicalize_query_for_search, read_parquet, e5_prefix_text


def get_qdrant() -> QdrantClient:
    load_dotenv()
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Query (Roman or Devanagari)")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--chunks-parquet", default="data/phase_3/chunks.parquet")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    load_dotenv()
    c_chunks = os.environ.get("QDRANT_COLLECTION_CHUNKS", "idr_chunks_vec_v1")

    canon = canonicalize_query_for_search(args.q)
    q_text = canon["q"]  # dev query or roman_norm query

    model_name = "intfloat/multilingual-e5-large"
    model = SentenceTransformer(model_name)
    q_vec = model.encode([e5_prefix_text(q_text, "query")], normalize_embeddings=True)[0].tolist()

    client = get_qdrant()
    res = client.search(
        collection_name=c_chunks,
        query_vector=q_vec,
        limit=args.topk,
        with_payload=True,
    )

    # Load chunks parquet to print chunk text (payload does not include chunk_text)
    chunks_df = read_parquet(Path(args.chunks_parquet).resolve())
    chunk_text_map = dict(zip(chunks_df["chunk_id"].astype(str), chunks_df["chunk_text"].astype(str)))

    print(f"mode: {canon['mode']}")
    print(f"query(raw): {canon['raw']}")
    print(f"query(used): {q_text}")
    print("-" * 80)

    for i, p in enumerate(res, start=1):
        payload = p.payload or {}
        cid = str(payload.get("chunk_id", p.id))
        title = payload.get("title_hi", "")
        url = payload.get("url", "")
        score = p.score
        snippet = chunk_text_map.get(cid, "")[:350].replace("\n", " ").strip()
        print(f"{i}. score={score:.4f}")
        print(f"   {title}")
        print(f"   url: {url}")
        print(f"   chunk_id: {cid}")
        print(f"   snippet: {snippet}")
        print()


if __name__ == "__main__":
    main()
