from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from utils import Paths, ensure_dir


def get_client() -> QdrantClient:
    load_dotenv()
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def recreate_collection(client: QdrantClient, name: str, dim: int) -> None:
    # drop if exists
    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=dim,
            distance=qm.Distance.COSINE,
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--dim", type=int, default=768)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    paths = Paths(root=root)

    ensure_dir(paths.data / "phase_3")

    load_dotenv()
    c_articles = os.environ.get("QDRANT_COLLECTION_ARTICLES", "idr_articles_vec_v1")
    c_chunks = os.environ.get("QDRANT_COLLECTION_CHUNKS", "idr_chunks_vec_v1")

    client = get_client()

    recreate_collection(client, c_articles, args.dim)
    recreate_collection(client, c_chunks, args.dim)

    print(f"Recreated collection: {c_articles} (dim={args.dim})")
    print(f"Recreated collection: {c_chunks} (dim={args.dim})")


if __name__ == "__main__":
    main()
