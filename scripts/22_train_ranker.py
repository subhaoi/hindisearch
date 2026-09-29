from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sqlalchemy import text

from utils import Paths, write_json
from _phase4.db import get_engine
from _phase4.ranker_v2 import FEATURE_NAMES, DEFAULT_WEIGHTS

TRAINING_SQL = """
SELECT DISTINCT ON (l.query_id, l.article_id)
       l.query_id, l.label, c.features
FROM labels l
JOIN query_log q ON q.id = l.query_id
JOIN candidate_log c ON c.query_id = l.query_id AND c.article_id = l.article_id
WHERE l.article_id IS NOT NULL AND q.ranker_version = :ranker
ORDER BY l.query_id, l.article_id, l.created_at DESC
"""


def load_rows(ranker: str) -> List[Dict[str, Any]]:
    engine = get_engine()
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(TRAINING_SQL), {"ranker": ranker})]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def main() -> None:
    """
    Learn ranker_v2 feature weights from feedback-UI labels (logistic regression on the
    features logged in candidate_log). Only writes the weights file when the learned
    weights beat the defaults on held-out queries (grouped CV), unless --force.
    The API picks the file up on restart.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--ranker", default="ranker_v2", help="Train on labels of queries served by this ranker")
    ap.add_argument("--min-labels", type=int, default=200)
    ap.add_argument("--out", default="data/phase_4/ranker_v2_weights.json")
    ap.add_argument("--force", action="store_true", help="Write weights even if they don't beat the defaults")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    paths = Paths(root=Path(args.root).resolve())
    rows = load_rows(args.ranker)
    if len(rows) < args.min_labels:
        raise SystemExit(f"Only {len(rows)} labelled results for {args.ranker}; need {args.min_labels}. Collect more labels first.")

    X = np.array([[float((r["features"] or {}).get(f) or 0.0) for f in FEATURE_NAMES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    groups = np.array([int(r["query_id"]) for r in rows])
    if len(set(y.tolist())) < 2:
        raise SystemExit("Labels are all one class; need both relevant and not-relevant examples.")

    n_splits = min(5, len(set(groups.tolist())))
    default_w = np.array([DEFAULT_WEIGHTS[f] for f in FEATURE_NAMES])
    learned_scores = np.zeros(len(y))
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(set(y[train_idx].tolist())) < 2:
            learned_scores[test_idx] = X[test_idx] @ default_w
            continue
        m = LogisticRegression(class_weight="balanced", max_iter=1000)
        m.fit(X[train_idx], y[train_idx])
        learned_scores[test_idx] = m.decision_function(X[test_idx])

    auc_default = auc(y, X @ default_w)
    auc_learned = auc(y, learned_scores)
    print(f"Labelled results: {len(y)} ({int(y.sum())} relevant) over {len(set(groups.tolist()))} queries")
    print(f"Held-out AUC  default weights: {auc_default:.3f} | learned weights: {auc_learned:.3f}")

    final = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y)
    coef = final.coef_[0]
    # Rescale to the default weights' total magnitude; ranking only depends on direction.
    coef = coef * (np.abs(default_w).sum() / max(1e-9, np.abs(coef).sum()))
    weights = {f: round(float(w), 4) for f, w in zip(FEATURE_NAMES, coef)}
    for f in FEATURE_NAMES:
        flag = "  (negative)" if weights[f] < 0 else ""
        print(f"  {f:<18} default {DEFAULT_WEIGHTS[f]:>6.3f}  learned {weights[f]:>7.3f}{flag}")

    if not args.force and not (auc_learned > auc_default + 0.005):
        print("Learned weights do not beat the defaults on held-out queries; not writing. Use --force to override.")
        return

    out = Path(args.out).resolve()
    write_json(out, {
        "weights": weights,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on_ranker": args.ranker,
        "n_labels": int(len(y)),
        "n_queries": int(len(set(groups.tolist()))),
        "cv_auc_default": auc_default,
        "cv_auc_learned": auc_learned,
    })
    print(f"Wrote: {out}  (restart the API to use it)")
    write_json(paths.logs / "ranker_training_report.json", json.loads(out.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
