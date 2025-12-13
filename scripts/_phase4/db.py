from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_database_url() -> str:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set in .env")
    return url


def get_engine() -> Engine:
    return create_engine(get_database_url(), pool_pre_ping=True)


DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS query_log (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      query_raw TEXT NOT NULL,
      query_mode TEXT NOT NULL,
      query_used TEXT NOT NULL,
      query_semantic TEXT NOT NULL,
      filters JSONB NULL,
      ranker_version TEXT NOT NULL,
      retrieval_version TEXT NOT NULL,
      meta JSONB NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate_log (
      id BIGSERIAL PRIMARY KEY,
      query_id BIGINT NOT NULL REFERENCES query_log(id) ON DELETE CASCADE,
      rank INT NOT NULL,
      article_id TEXT NOT NULL,
      url TEXT NULL,
      title TEXT NULL,
      score DOUBLE PRECISION NOT NULL,
      features JSONB NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_candidate_query_id ON candidate_log(query_id)",
    "CREATE INDEX IF NOT EXISTS idx_candidate_article_id ON candidate_log(article_id)",
    """
    CREATE TABLE IF NOT EXISTS labels (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      query_id BIGINT NOT NULL REFERENCES query_log(id) ON DELETE CASCADE,
      article_id TEXT NULL,
      label INT NOT NULL,
      note TEXT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_labels_query_id ON labels(query_id)",
]


def ensure_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        for stmt in DDL_STATEMENTS:
            conn.execute(text(stmt))


def insert_query(
    engine: Engine,
    query_raw: str,
    query_mode: str,
    query_used: str,
    query_semantic: str,
    filters: Optional[Dict[str, Any]],
    ranker_version: str,
    retrieval_version: str,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    q = text(
        """
        INSERT INTO query_log(query_raw, query_mode, query_used, query_semantic, filters, ranker_version, retrieval_version, meta)
        VALUES (:qr, :qm, :qu, :qs, :filters::jsonb, :rv, :tv, :meta::jsonb)
        RETURNING id
        """
    )
    with engine.begin() as conn:
        res = conn.execute(
            q,
            {
                "qr": query_raw,
                "qm": query_mode,
                "qu": query_used,
                "qs": query_semantic,
                "filters": json.dumps(filters) if filters is not None else None,
                "rv": ranker_version,
                "tv": retrieval_version,
                "meta": json.dumps(meta) if meta is not None else None,
            },
        ).fetchone()
        return int(res[0])


def insert_candidates(engine: Engine, query_id: int, ranked: List[Dict[str, Any]]) -> None:
    stmt = text(
        """
        INSERT INTO candidate_log(query_id, rank, article_id, url, title, score, features)
        VALUES (:qid, :rank, :aid, :url, :title, :score, :features::jsonb)
        """
    )
    with engine.begin() as conn:
        for r in ranked:
            conn.execute(
                stmt,
                {
                    "qid": query_id,
                    "rank": int(r["rank"]),
                    "aid": str(r["article_id"]),
                    "url": r.get("url"),
                    "title": r.get("title"),
                    "score": float(r["score"]),
                    "features": json.dumps(r["features"]),
                },
            )


def insert_label(engine: Engine, query_id: int, article_id: Optional[str], label: int, note: Optional[str]) -> None:
    stmt = text(
        """
        INSERT INTO labels(query_id, article_id, label, note)
        VALUES (:qid, :aid, :lab, :note)
        """
    )
    with engine.begin() as conn:
        conn.execute(
            stmt,
            {"qid": query_id, "aid": article_id, "lab": int(label), "note": note},
        )
