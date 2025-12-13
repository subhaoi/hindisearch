from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from tqdm import tqdm
import hashlib
import struct

from utils import Paths, ensure_dir, read_parquet, write_json, is_nullish


def stable_uint64_from_str(s: str) -> int:
    h = hashlib.sha1(s.encode("utf-8")).digest()
    # take first 8 bytes as unsigned 64-bit int
    return struct.unpack(">Q", h[:8])[0]


def get_client() -> QdrantClient:
    load_dotenv()
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--articles", default="data/phase_3/article_vectors.parquet")
    ap.add_argument("--chunks", default="data/phase_3/chunk_vectors.parquet")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    ensure_dir(paths.logs)

    load_dotenv()
    c_articles = os.environ.get("QDRANT_COLLECTION_ARTICLES", "idr_articles_vec_v1")
    c_chunks = os.environ.get("QDRANT_COLLECTION_CHUNKS", "idr_chunks_vec_v1")

    client = get_client()

    articles = read_parquet(Path(args.articles).resolve())
    chunks = read_parquet(Path(args.chunks).resolve())

    report: Dict[str, Any] = {
        "articles_rows": len(articles),
        "chunks_rows": len(chunks),
        "articles_upserted": 0,
        "chunks_upserted": 0,
        "batch_size": args.batch_size,
        "failures": [],
    }

    # Articles: point id = article id (string)
    bs = max(1, int(args.batch_size))
    for i in tqdm(range(0, len(articles), bs), desc="Upserting article vectors"):
        batch = articles.iloc[i:i+bs]
        points: List[qm.PointStruct] = []
        for _, r in batch.iterrows():
            aid = str(r.get("id"))
            vec = r.get("vector")
            if is_nullish(vec):
                continue
            payload = {
                "article_id": aid,
                "url": None if is_nullish(r.get("url")) else str(r.get("url")),
                "published_date": None if is_nullish(r.get("published_date")) else str(r.get("published_date")),
                "published_ts": int(r.get("published_ts")) if not is_nullish(r.get("published_ts")) else 0,
            }
            points.append(qm.PointStruct(id=int(aid), vector=vec, payload=payload))
        try:
            client.upsert(collection_name=c_articles, points=points)
            report["articles_upserted"] += len(points)
        except Exception as e:
            if len(report["failures"]) < 50:
                report["failures"].append(f"articles batch {i}: {type(e).__name__}: {e}")

    # Chunks: point id = chunk_id
    for i in tqdm(range(0, len(chunks), bs), desc="Upserting chunk vectors"):
        batch = chunks.iloc[i:i+bs]
        points = []
        for _, r in batch.iterrows():
            cid = str(r.get("chunk_id"))
            vec = r.get("vector")
            if is_nullish(vec):
                continue
            payload = {
                "chunk_id": cid,
                "article_id": str(r.get("article_id")),
                "chunk_index": int(r.get("chunk_index")) if not is_nullish(r.get("chunk_index")) else 0,
                "url": None if is_nullish(r.get("url")) else str(r.get("url")),
                "published_date": None if is_nullish(r.get("published_date")) else str(r.get("published_date")),
                "published_ts": int(r.get("published_ts")) if not is_nullish(r.get("published_ts")) else 0,
                "title_hi": None if is_nullish(r.get("title_hi")) else str(r.get("title_hi")),
                "chunk_tokens": int(r.get("chunk_tokens")) if not is_nullish(r.get("chunk_tokens")) else 0,
            }
            pid = stable_uint64_from_str(cid)
            points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))
        try:
            client.upsert(collection_name=c_chunks, points=points)
            report["chunks_upserted"] += len(points)
        except Exception as e:
            if len(report["failures"]) < 50:
                report["failures"].append(f"chunks batch {i}: {type(e).__name__}: {e}")

    out = paths.logs / "phase3_qdrant_ingest_report.json"
    write_json(out, report)
    print(f"Wrote: {out}")
    print(f"Articles upserted: {report['articles_upserted']}")
    print(f"Chunks upserted: {report['chunks_upserted']}")


if __name__ == "__main__":
    main()
