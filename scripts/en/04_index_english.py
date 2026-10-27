from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.english import (  # noqa: E402
    LocationIndex, as_list, leaves, primary_category, s,
)
from scripts.utils import iso_to_epoch_seconds  # noqa: E402

TS_DEFAULT = "idr_en_articles_v1"
QA_DEFAULT = "idr_en_articles_vec_v1"
QC_DEFAULT = "idr_en_chunks_vec_v1"


def ts_schema(name: str) -> Dict[str, Any]:
    text = {"type": "string", "optional": True, "stem": True}
    facet = {"type": "string[]", "optional": True, "facet": True}
    return {
        "name": name,
        "fields": [
            {"name": "title", "type": "string", "stem": True},
            {"name": "seo_title", **text},
            {"name": "summary", **text},
            {"name": "content", **text},
            {"name": "post_type", "type": "string", "facet": True},
            {"name": "categories", **facet},
            {"name": "tags", **facet},
            {"name": "locations", **facet},
            # Tagged + named in the text (scripts/english.py); what location filters use
            {"name": "locations_all", **facet},
            {"name": "contributors", **facet},
            {"name": "podcast_series", **facet},
            {"name": "published_ts", "type": "int64"},
        ],
        "default_sorting_field": "published_ts",
    }


def ts_documents(articles: pd.DataFrame) -> List[Dict[str, Any]]:
    locs = LocationIndex(articles)
    docs = []
    for _, r in articles.iterrows():
        cats = as_list(r["categories"])
        docs.append({
            "id": str(r["id"]), "title": s(r["title"]), "seo_title": s(r["seo_title"]),
            "summary": s(r["summary"]), "content": s(r["content"]), "post_type": s(r["post_type"]),
            "categories": leaves(cats), "primary_category": primary_category(cats),
            "tags": as_list(r["tags"]), "locations": leaves(as_list(r["locations"])),
            "locations_all": locs.for_article(r), "contributors": as_list(r["contributors"]),
            "podcast_series": as_list(r["podcast_series"]),
            "published_ts": iso_to_epoch_seconds(r["published_date"]), "published_date": s(r["published_date"]),
            "url": s(r["url"]), "image_url": s(r["image_url"]),
        })
    return docs


def index_typesense(articles: pd.DataFrame, collection: str, recreate: bool, prune: bool) -> Dict[str, Any]:
    import typesense
    client = typesense.Client({
        "nodes": [{"host": os.environ.get("TYPESENSE_HOST", "localhost"), "port": os.environ.get("TYPESENSE_PORT", "8108"),
                   "protocol": os.environ.get("TYPESENSE_PROTOCOL", "http")}],
        "api_key": os.environ["TYPESENSE_API_KEY"], "connection_timeout_seconds": 30,
    })
    existing = {c["name"] for c in client.collections.retrieve()}
    if collection in existing and recreate:
        client.collections[collection].delete()
        existing.discard(collection)
    if collection not in existing:
        client.collections.create(ts_schema(collection))
        print(f"Created Typesense collection {collection}")

    docs = ts_documents(articles)
    report: Dict[str, Any] = {"indexed": 0, "failed": 0, "failures": []}
    for i in range(0, len(docs), 100):
        res = client.collections[collection].documents.import_(docs[i:i + 100], {"action": "upsert"})
        for r in res if isinstance(res, list) else [json.loads(x) for x in str(res).splitlines() if x.strip()]:
            report["indexed" if r.get("success") else "failed"] += 1
            if not r.get("success") and len(report["failures"]) < 20:
                report["failures"].append(r)
    if prune:
        keep = {d["id"] for d in docs}
        exported = client.collections[collection].documents.export({"include_fields": "id"})
        stale = [i for i in (json.loads(x)["id"] for x in str(exported).splitlines() if x.strip()) if i not in keep]
        for sid in stale:
            client.collections[collection].documents[sid].delete()
        report["pruned"] = len(stale)
    print(f"Typesense {collection}: indexed {report['indexed']} | failed {report['failed']} | pruned {report.get('pruned', 0)}")
    return report


SYNONYM_PREFIX = "en_syn_"


def load_synonyms(collection: str, path: Path) -> int:
    """Replace this script's synonym groups in the collection with those in `path`."""
    import typesense
    client = typesense.Client({
        "nodes": [{"host": os.environ.get("TYPESENSE_HOST", "localhost"), "port": os.environ.get("TYPESENSE_PORT", "8108"),
                   "protocol": os.environ.get("TYPESENSE_PROTOCOL", "http")}],
        "api_key": os.environ["TYPESENSE_API_KEY"], "connection_timeout_seconds": 30,
    })
    syn = client.collections[collection].synonyms
    for old in syn.retrieve().get("synonyms", []) or []:
        if str(old.get("id", "")).startswith(SYNONYM_PREFIX):
            syn[old["id"]].delete()
    groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
    for i, terms in enumerate(groups):
        syn.upsert(f"{SYNONYM_PREFIX}{i:03d}", {"synonyms": [t.lower() for t in terms]})
    print(f"Synonyms: {len(groups)} groups loaded into {collection}")
    return len(groups)


def point_id(chunk_id: str) -> int:
    # Same scheme as the Hindi chunks (13_qdrant_ingest.py)
    return struct.unpack(">Q", hashlib.sha1(chunk_id.encode("utf-8")).digest()[:8])[0]


def index_qdrant(art_vecs: pd.DataFrame, chk_vecs: pd.DataFrame, c_art: str, c_chk: str,
                 recreate: bool, prune: bool) -> Dict[str, Any]:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    client = QdrantClient(host=os.environ.get("QDRANT_HOST", "localhost"), port=int(os.environ.get("QDRANT_PORT", "6333")))
    dim = len(art_vecs["vector"].iloc[0])
    report: Dict[str, Any] = {"dim": dim}
    for name in (c_art, c_chk):
        if client.collection_exists(name):
            size = client.get_collection(name).config.params.vectors.size
            if recreate or size != dim:
                if size != dim and not recreate:
                    raise SystemExit(f"{name} holds {size}-dim vectors, these are {dim}: new model? Re-run with --recreate")
                client.delete_collection(name)
        if not client.collection_exists(name):
            client.create_collection(name, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
            client.create_payload_index(name, "article_id", qm.PayloadSchemaType.KEYWORD)
            print(f"Created Qdrant collection {name} (dim {dim})")

    def upsert(name: str, points: List[qm.PointStruct]) -> None:
        for i in range(0, len(points), 256):
            client.upsert(name, points=points[i:i + 256])

    upsert(c_art, [qm.PointStruct(id=int(r.id), vector=list(r.vector), payload={
        "article_id": str(r.id), "url": r.url, "published_ts": int(r.published_ts)}) for r in art_vecs.itertuples()])
    upsert(c_chk, [qm.PointStruct(id=point_id(r.chunk_id), vector=list(r.vector), payload={
        "chunk_id": r.chunk_id, "article_id": str(r.article_id), "chunk_index": int(r.chunk_index),
        "url": r.url, "published_ts": int(r.published_ts)}) for r in chk_vecs.itertuples()])
    report["articles"], report["chunks"] = len(art_vecs), len(chk_vecs)

    if prune:
        for name, keep in ((c_art, {int(i) for i in art_vecs["id"]}), (c_chk, {point_id(c) for c in chk_vecs["chunk_id"]})):
            stale, offset = [], None
            while True:
                pts, offset = client.scroll(name, limit=1000, offset=offset, with_payload=False, with_vectors=False)
                stale += [int(p.id) for p in pts if int(p.id) not in keep]
                if offset is None:
                    break
            for i in range(0, len(stale), 1000):
                client.delete(name, points_selector=qm.PointIdsList(points=stale[i:i + 1000]))
            report[f"pruned_{name}"] = len(stale)
    print(f"Qdrant: {report['articles']} article and {report['chunks']} chunk vectors (dim {dim})")
    return report


def main() -> None:
    """
    Load the English site into its own Typesense and Qdrant collections (EN_* settings in .env;
    the Hindi collections are never touched). Creates missing collections, upserts, and with
    --prune deletes what is no longer in the inputs. --recreate drops and rebuilds them (after a
    schema or model change).

      python scripts/en/04_index_english.py --prune
    """
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", default="data/en/final/articles_canonical.parquet")
    ap.add_argument("--vectors-dir", default="data/en/phase_3")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--synonyms", default="config/synonyms_en_v1.json")
    ap.add_argument("--report", default="logs/en_index_report.json")
    args = ap.parse_args()

    ts_name = os.environ.get("EN_TYPESENSE_COLLECTION", TS_DEFAULT)
    c_art = os.environ.get("EN_QDRANT_COLLECTION_ARTICLES", QA_DEFAULT)
    c_chk = os.environ.get("EN_QDRANT_COLLECTION_CHUNKS", QC_DEFAULT)
    hindi = {os.environ.get("TYPESENSE_COLLECTION", "idr_articles_hi_v1"),
             os.environ.get("QDRANT_COLLECTION_ARTICLES", "idr_articles_vec_v1"),
             os.environ.get("QDRANT_COLLECTION_CHUNKS", "idr_chunks_vec_v1")}
    if hindi & {ts_name, c_art, c_chk}:
        raise SystemExit(f"English collection names overlap the Hindi ones: {hindi & {ts_name, c_art, c_chk}}")

    articles = pd.read_parquet(args.articles)
    vd = Path(args.vectors_dir)
    art_vecs = pd.read_parquet(vd / "article_vectors.parquet")
    chk_vecs = pd.read_parquet(vd / "chunk_vectors.parquet")
    model = os.environ.get("EN_EMBED_MODEL")
    if model and set(art_vecs["model"]) != {model}:
        raise SystemExit(f"Vectors are from {sorted(set(art_vecs['model']))}, EN_EMBED_MODEL is {model}: "
                         "re-run scripts/en/03_embed_english.py")

    report = {"typesense": index_typesense(articles, ts_name, args.recreate, args.prune),
              "qdrant": index_qdrant(art_vecs, chk_vecs, c_art, c_chk, args.recreate, args.prune),
              "model": sorted(set(art_vecs["model"]))}
    if args.synonyms and Path(args.synonyms).exists():
        report["synonym_groups"] = load_synonyms(ts_name, Path(args.synonyms))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if report["typesense"]["failed"]:
        raise SystemExit(f"{report['typesense']['failed']} documents failed to index; see {args.report}")


if __name__ == "__main__":
    main()
