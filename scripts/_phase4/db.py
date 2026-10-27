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


def get_engine(url: Optional[str] = None) -> Engine:
    """DATABASE_URL by default; the English API passes its own database (EN_DATABASE_URL)."""
    return create_engine(url or get_database_url(), pool_pre_ping=True)


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
        published_date TEXT NULL,
        summary TEXT NULL,
        primary_category TEXT NULL,
        categories JSONB NULL,
        tags JSONB NULL,
        location JSONB NULL,
        partner_label TEXT NULL,
        contributors JSONB NULL,
        score DOUBLE PRECISION NOT NULL,
        features JSONB NOT NULL,
        explanation JSONB NULL
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
    "CREATE INDEX IF NOT EXISTS idx_query_log_created_at ON query_log(created_at)",
    # Searches that crashed, for the admin panel's Problems table (the error only, no query text)
    """
    CREATE TABLE IF NOT EXISTS search_errors (
      id BIGSERIAL PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      seconds DOUBLE PRECISION NULL,
      error TEXT NOT NULL
    )
    """,
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
        INSERT INTO query_log(
          query_raw, query_mode, query_used, query_semantic,
          filters, ranker_version, retrieval_version, meta
        )
        VALUES (
          :qr, :qm, :qu, :qs,
          CAST(:filters AS jsonb), :rv, :tv, CAST(:meta AS jsonb)
        )
        RETURNING id
        """
    )

    payload = {
        "qr": query_raw,
        "qm": query_mode,
        "qu": query_used,
        "qs": query_semantic,
        "filters": json.dumps(filters) if filters is not None else None,
        "rv": ranker_version,
        "tv": retrieval_version,
        "meta": json.dumps(meta) if meta is not None else None,
    }

    with engine.begin() as conn:
        res = conn.execute(q, payload).fetchone()
        return int(res[0])



def insert_candidates(engine: Engine, query_id: int, ranked: List[Dict[str, Any]]) -> None:
    stmt = text(
        """
        INSERT INTO candidate_log(
          query_id, rank, article_id, url, title, published_date, summary,
          primary_category, categories, tags, location, partner_label, contributors,
          score, features, explanation
        )
        VALUES (
          :qid, :rank, :aid, :url, :title, :published_date, :summary,
          :primary_category, CAST(:categories AS jsonb), CAST(:tags AS jsonb), CAST(:location AS jsonb),
          :partner_label, CAST(:contributors AS jsonb),
          :score, CAST(:features AS jsonb), CAST(:explanation AS jsonb)
        )
        """
    )
    if not ranked:
        return
    # One executemany instead of a round trip per row
    with engine.begin() as conn:
        conn.execute(
            stmt,
            [
                {
                    "qid": query_id,
                    "rank": int(r["rank"]),
                    "aid": str(r["article_id"]),
                    "url": r.get("url"),
                    "title": r.get("title"),
                    "published_date": r.get("published_date"),
                    "summary": r.get("summary"),
                    "primary_category": r.get("primary_category"),
                    "categories": json.dumps(r.get("categories") or []),
                    "tags": json.dumps(r.get("tags") or []),
                    "location": json.dumps(r.get("location") or []),
                    "partner_label": r.get("partner_label"),
                    "contributors": json.dumps(r.get("contributors") or []),
                    "score": float(r["score"]),
                    "features": json.dumps(r["features"]),
                    "explanation": json.dumps(r.get("explanation")),
                }
                for r in ranked
            ],
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


def insert_error(engine: Engine, error: str, seconds: Optional[float]) -> None:
    """Never raises: a failing log write must not hide the original error."""
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO search_errors(seconds, error) VALUES (:s, :e)"),
                         {"s": seconds, "e": str(error)[:500]})
    except Exception as e:  # noqa: BLE001
        print(f"[search_errors] could not record: {type(e).__name__}: {e}")


# --------- Read side, for the admin panel feed (usage_feed.py) ---------

def searches_since(engine: Engine, since_ts: float) -> List[Dict[str, Any]]:
    """Logged searches since a unix time, oldest first: ts, query_raw, query_mode, meta."""
    stmt = text(
        """
        SELECT extract(epoch FROM created_at) AS ts, query_raw, query_mode, meta
        FROM query_log WHERE created_at >= to_timestamp(:since) ORDER BY created_at
        """
    )
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt, {"since": since_ts})]


def errors_since(engine: Engine, since_ts: float) -> List[Dict[str, Any]]:
    stmt = text(
        """
        SELECT extract(epoch FROM created_at) AS ts, seconds, error
        FROM search_errors WHERE created_at >= to_timestamp(:since) ORDER BY created_at
        """
    )
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt, {"since": since_ts})]


def label_counts(engine: Engine, ranker: str) -> Dict[str, int]:
    """All feedback clicks, and the labelled results the ranker can train on (as 22_train_ranker counts them)."""
    stmt = text(
        """
        SELECT (SELECT count(*) FROM labels) AS total,
               (SELECT count(*) FROM (
                  SELECT DISTINCT l.query_id, l.article_id
                  FROM labels l
                  JOIN query_log q ON q.id = l.query_id
                  JOIN candidate_log c ON c.query_id = l.query_id AND c.article_id = l.article_id
                  WHERE l.article_id IS NOT NULL AND q.ranker_version = :ranker) t) AS trainable
        """
    )
    with engine.connect() as conn:
        r = conn.execute(stmt, {"ranker": ranker}).fetchone()
        return {"total": int(r[0]), "trainable": int(r[1])}
