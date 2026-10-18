"""
Feed for the IDR tools admin panel (tools.idronline.org/admin/): GET /api/usage on the search API.

The panel fetches it the same way as Image Search's /api/usage, with an X-IDR-Usage-Token
header derived from the shared SECRET_KEY (in ~/idr-tools.env on this server, never in git).

Returns
  events   one row per search in the panel's usage format (counts and settings only)
  queries  what people searched for, aggregated over the chosen period, with anything that
           looks like an email address or phone number left out
  health   index sizes, last weekly refresh, latest test scores, labels, memory, disk, version
"""
from __future__ import annotations

import glob
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# Copied from email_format_tool/shared/idr_usage.py (export_token / token_ok); keep in step with it.
def export_token() -> Optional[str]:
    key = os.environ.get("SECRET_KEY", "")
    return hmac.new(key.encode(), b"idr-usage-export", hashlib.sha256).hexdigest() if key else None


def token_ok(given: Optional[str]) -> bool:
    expected = export_token()
    return bool(expected and given and hmac.compare_digest(expected, given))


TOOL = "hindi-search"
TOP_N = 15
MIN_LABELS = 200  # 22_train_ranker.py --min-labels

_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"(?:\d[\s\-().]*){8,}")
_MAX_SHOWN_CHARS = 80


def shareable(query: str) -> Optional[str]:
    """The query as shown on the panel, or None when it may contain personal details."""
    q = " ".join(str(query or "").split())
    if not q or _EMAIL_RE.search(q) or _PHONE_RE.search(q):
        return None
    return q if len(q) <= _MAX_SHOWN_CHARS else q[:_MAX_SHOWN_CHARS - 1] + "…"


def _meta(row: Dict[str, Any]) -> Dict[str, Any]:
    m = row.get("meta")
    if isinstance(m, str):
        try:
            return json.loads(m)
        except ValueError:
            return {}
    return m or {}


def search_kind(meta: Dict[str, Any]) -> str:
    """year | series | author | place | search"""
    if meta.get("query_kind") == "year":
        return "year"
    if meta.get("series"):
        return "series"
    if meta.get("has_author_entity") or meta.get("name_query"):
        return "author"
    if meta.get("has_location_entity"):
        return "place"
    return "search"


def to_events(searches: List[Dict[str, Any]], errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Searches and crashes as the panel's usage rows (no query text)."""
    out = []
    for r in searches:
        m = _meta(r)
        out.append({
            "ts": float(r["ts"]), "tool": TOOL, "ok": 1, "seconds": m.get("took_s"), "error": None,
            "details": {
                "mode": r.get("query_mode"),
                "kind": search_kind(m),
                "results": int(m.get("cand_n") or 0),
                "keyword_match": int(m.get("lex_n") or 0) > 0,
            },
        })
    for e in errors:
        out.append({"ts": float(e["ts"]), "tool": TOOL, "ok": 0, "seconds": e.get("seconds"),
                    "error": e.get("error"), "details": {}})
    return sorted(out, key=lambda e: e["ts"])


# A search whose words appear in this many articles or fewer is worth a look: often fine
# (a rare name or place), sometimes a missing synonym, alias or spelling (csr, third eye)
FEW_MATCHES = 5


def top_queries(searches: List[Dict[str, Any]], since_ts: float) -> Dict[str, Any]:
    """
    Most searched, and searches whose words matched few articles (see FEW_MATCHES) with that
    number. Year, series, author and place searches are left out of the second list: few
    matches is expected there.
    """
    shown: Counter = Counter()
    few: Counter = Counter()
    matched: Dict[str, int] = {}
    hidden = 0
    for r in searches:
        if float(r["ts"]) < since_ts:
            continue
        q = shareable(r.get("query_raw"))
        if q is None:
            hidden += 1
            continue
        key = q.lower()
        shown[key] += 1
        m = _meta(r)
        lex = int(m.get("lex_n") or 0)
        if search_kind(m) == "search" and lex <= FEW_MATCHES:
            few[key] += 1
            matched[key] = min(lex, matched.get(key, lex))
    return {
        "top": shown.most_common(TOP_N),
        "few_matches": [(q, n, matched[q]) for q, n in few.most_common(TOP_N)],
        "few_matches_limit": FEW_MATCHES,
        "hidden": hidden,
    }


def last_refresh(root: Path) -> Optional[Dict[str, Any]]:
    """Summary of logs/refresh/last_status.json from the weekly refresh."""
    try:
        s = json.loads((root / "logs" / "refresh" / "last_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {
        "at": (s.get("finished_at") or s.get("started_at") or "")[:16].replace("T", " "),
        "ok": bool(s.get("ok")), "outcome": s.get("outcome"),
        "articles": s.get("articles"), "added": s.get("articles_added"), "removed": s.get("articles_removed"),
        "failed_step": s.get("failed_step"), "error": (s.get("error") or "")[:300] or None,
    }


# Main score per test set (26_eval_rules.py): higher is better except no_answer
_EVAL_METRIC = {"known_item": "mrr", "author": "p@10", "series_format": "p@10", "topic": "p@10", "no_answer": "results"}


def latest_eval(root: Path) -> Optional[Dict[str, Any]]:
    """Averages from the newest runs/*.json saved by 26_eval_rules.py --save."""
    files = sorted(glob.glob(str(root / "runs" / "*.json")), key=os.path.getmtime)
    if not files:
        return None
    try:
        run = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    sets: Dict[str, List[float]] = {}
    for rec in run.get("queries", {}).values():
        metric = _EVAL_METRIC.get(rec.get("set"))
        if metric and metric in (rec.get("metrics") or {}):
            sets.setdefault(rec["set"], []).append(float(rec["metrics"][metric]))
    return {
        "file": Path(files[-1]).name,
        "at": (run.get("meta", {}).get("at") or "")[:16].replace("T", " "),
        "sets": [{"set": k, "metric": _EVAL_METRIC[k], "value": round(sum(v) / len(v), 2), "n": len(v)}
                 for k, v in sets.items()],
    }


def memory() -> Optional[Dict[str, Any]]:
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        total = int(info["MemTotal"].split()[0]) * 1024
        avail = int(info["MemAvailable"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        return None
    return {"used_pct": round((total - avail) / total * 100), "free_gb": round(avail / 1e9, 1)}


def code_version(root: Path) -> Optional[Dict[str, str]]:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h|%cd|%s", "--date=format:%Y-%m-%d %H:%M"],
                             cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        commit, date, subject = out.split("|", 2)
        return {"commit": commit, "date": date, "subject": subject[:80]}
    except Exception:  # noqa: BLE001
        return None


def disk(root: Path) -> Dict[str, Any]:
    d = shutil.disk_usage(root)
    return {"disk_used_pct": round(d.used / d.total * 100), "disk_free_gb": round(d.free / 1e9, 1)}
