"""
Weekly refresh: download the WP All Export CSVs, rebuild every pipeline stage, update the
live Typesense/Qdrant indexes in place, and restart the search API.

Safety model
  1. Downloads are validated before anything else is touched.
  2. Current artifacts are backed up (data/backups/<run_id>/) before the rebuild.
  3. All offline stages (Phase 1, chunking, embeddings, gazetteer, vocab) run first. If any
     fails, the backup is restored and the live indexes / API are never touched.
  4. Only then are the live indexes updated (upsert + prune, no collection drops) and the
     API restarted and smoke-tested.

Logs: logs/refresh/<run_id>.log (full output), logs/refresh/last_status.json,
logs/refresh/history.jsonl. Optional failure alerts via alert_webhook_url (Slack-style
{"text": ...} POST).

Config: config/refresh.json (git-ignored; copy config/refresh.example.json).

Usage
  python scripts/refresh_weekly.py                 # normal weekly run (cron)
  python scripts/refresh_weekly.py --skip-fetch    # rebuild from data/raw/exports/*.csv
  python scripts/refresh_weekly.py --force         # rebuild even if the exports are unchanged
  python scripts/refresh_weekly.py --no-restart    # don't restart the API
  python scripts/refresh_weekly.py --skip-fetch --recreate-typesense
                                                   # after a Typesense schema change (drops and
                                                   # rebuilds the collection: ~1 min of degraded search)
"""

from __future__ import annotations

import argparse
import fcntl
import filecmp
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
LOG_DIR = ROOT / "logs" / "refresh"
EXPORTS_DIR = ROOT / "data" / "raw" / "exports"
COMBINED_CSV = ROOT / "data" / "raw" / "articles.csv"
CANONICAL = ROOT / "data" / "final" / "articles_canonical.parquet"
BACKUP_ROOT = ROOT / "data" / "backups"
WEIGHTS = ROOT / "data" / "phase_4" / "ranker_v2_weights.json"

# Everything a rebuild overwrites; backed up before and restored on an offline failure.
BACKUP_PATHS = [
    "data/raw/articles.csv",
    "data/raw/exports",
    "data/final/articles_canonical.parquet",
    "data/phase_3/chunks.parquet",
    "data/phase_3/article_vectors.parquet",
    "data/phase_3/chunk_vectors.parquet",
    "data/phase_45/gazetteer_v1.json",
    "data/phase_45/translit_vocab_v1.json",
    "data/phase_3/translations.parquet",
    "data/phase_3/translation_cache.parquet",
    "data/phase_4/ranker_v2_weights.json",
]

REQUIRED_COLUMNS = ["ID", "Date", "Title", "Content", "Permalink"]


class StepFailed(Exception):
    pass


class RefreshRun:
    def __init__(self, cfg: Dict[str, Any], args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{self.run_id}.log"
        self._log = self.log_path.open("a", encoding="utf-8")
        self.backup_dir = BACKUP_ROOT / self.run_id
        self.status: Dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "outcome": None,
            "failed_step": None,
            "error": None,
            "steps": [],
            "log": str(self.log_path),
        }

    # ---------- logging ----------

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def step(self, name: str, cmd: List[str], fatal: bool = True) -> int:
        """Run a pipeline script, streaming its output into the run log."""
        self.log(f"=== {name}: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            # tqdm redraws with \r; keep only the final state of each line
            self._log.write("    " + line.rstrip("\n").split("\r")[-1] + "\n")
        self._log.flush()
        rc = proc.wait()
        secs = round(time.time() - t0, 1)
        self.status["steps"].append({"name": name, "rc": rc, "seconds": secs})
        self.log(f"=== {name}: {'ok' if rc == 0 else f'exit code {rc}'} ({secs}s)")
        if rc != 0 and fatal:
            raise StepFailed(name)
        return rc

    def py(self, script: str, *args: str) -> List[str]:
        return [sys.executable, str(SCRIPTS / script), *args]

    # ---------- download ----------

    def fetch_export(self, exp: Dict[str, Any], dest: Path) -> None:
        name = exp["name"]
        timeout = 120

        r = requests.get(exp["trigger_url"], timeout=timeout)
        self.log(f"[{name}] trigger -> HTTP {r.status_code}: {r.text.strip()[:300]}")
        if r.status_code != 200:
            raise StepFailed(f"fetch:{name}:trigger")

        done_patterns = [p.lower() for p in self.cfg.get("done_patterns", ["complete"])]
        interval = int(self.cfg.get("poll_interval_seconds", 120))
        deadline = time.time() + 60 * int(self.cfg.get("poll_timeout_minutes", 60))
        seen_progress = False
        time.sleep(5)
        while True:
            r = requests.get(exp["processing_url"], timeout=timeout)
            msg = r.text.strip()
            self.log(f"[{name}] processing -> HTTP {r.status_code}: {msg[:300]}")
            if r.status_code != 200:
                raise StepFailed(f"fetch:{name}:processing")
            low = msg.lower()
            if any(p in low for p in done_patterns):
                break
            if "not triggered" in low:
                # After progress this means the run finished and was reset; before, the trigger failed
                if seen_progress:
                    break
                raise StepFailed(f"fetch:{name}:not-triggered")
            seen_progress = True
            if time.time() > deadline:
                raise StepFailed(f"fetch:{name}:timeout")
            time.sleep(interval)

        tmp = dest.with_suffix(".part")
        with requests.get(exp["file_url"], timeout=600, stream=True) as r:
            if r.status_code != 200:
                raise StepFailed(f"fetch:{name}:download HTTP {r.status_code}")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
        self.log(f"[{name}] downloaded {dest.stat().st_size:,} bytes")

    def validate_export(self, name: str, path: Path) -> int:
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
        except Exception as e:
            raise StepFailed(f"validate:{name}: not a readable CSV ({type(e).__name__}: {e})")
        df.columns = [c.lstrip("﻿") for c in df.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise StepFailed(f"validate:{name}: missing columns {missing}")
        if len(df) == 0 or (df["ID"].str.strip() == "").all():
            raise StepFailed(f"validate:{name}: no rows")
        if "Image Featured" not in df.columns:
            self.log(f"[{name}] WARNING: no 'Image Featured' column; image URLs carried over from last run")

        prev = EXPORTS_DIR / f"{name}.csv"
        if prev.exists():
            prev_rows = len(pd.read_csv(prev, dtype=str, keep_default_na=False, na_values=[]))
            ratio = float(self.cfg.get("min_rows_ratio", 0.9))
            if len(df) < ratio * prev_rows:
                raise StepFailed(
                    f"validate:{name}: {len(df)} rows vs {prev_rows} last time (< {ratio:.0%}). "
                    "Is the export set to 'only new posts' or filtered?"
                )
            self.log(f"[{name}] {len(df)} rows (last run: {prev_rows})")
        else:
            self.log(f"[{name}] {len(df)} rows (first run for this export)")
        return len(df)

    # ---------- backup ----------

    def backup(self) -> None:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        present = []
        for rel in BACKUP_PATHS:
            src = ROOT / rel
            if not src.exists():
                continue
            dst = self.backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            present.append(rel)
        (self.backup_dir / "manifest.json").write_text(json.dumps({"present": present}, indent=2))
        self.log(f"Backed up {len(present)} artifacts to {self.backup_dir}")

        keep = int(self.cfg.get("keep_backups", 4))
        old = sorted(p for p in BACKUP_ROOT.iterdir() if p.is_dir())[:-keep] if keep > 0 else []
        for p in old:
            shutil.rmtree(p, ignore_errors=True)
            self.log(f"Removed old backup {p.name}")

    def restore(self) -> None:
        manifest = json.loads((self.backup_dir / "manifest.json").read_text())
        present = set(manifest["present"])
        for rel in BACKUP_PATHS:
            dst = ROOT / rel
            if dst.is_dir():
                shutil.rmtree(dst)
            elif dst.exists():
                dst.unlink()
            if rel in present:
                src = self.backup_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        self.log(f"Restored artifacts from {self.backup_dir}")

    # ---------- checks ----------

    def check_canonical(self) -> None:
        old_path = self.backup_dir / "data/final/articles_canonical.parquet"
        new_ids = set(pd.read_parquet(CANONICAL, columns=["id"])["id"].astype(str))
        self.status["articles"] = len(new_ids)
        if not old_path.exists():
            self.log(f"Canonical articles: {len(new_ids)} (no previous version)")
            return
        old_ids = set(pd.read_parquet(old_path, columns=["id"])["id"].astype(str))
        added, removed = sorted(new_ids - old_ids), sorted(old_ids - new_ids)
        self.status.update({"articles_added": len(added), "articles_removed": len(removed)})
        self.log(f"Canonical articles: {len(new_ids)} (was {len(old_ids)}; +{len(added)} / -{len(removed)})")
        if added:
            self.log(f"  added ids: {', '.join(added[:50])}{' ...' if len(added) > 50 else ''}")
        if removed:
            self.log(f"  removed ids: {', '.join(removed[:50])}{' ...' if len(removed) > 50 else ''}")
        ratio = float(self.cfg.get("min_rows_ratio", 0.9))
        if len(new_ids) < ratio * len(old_ids):
            raise StepFailed(f"canonical: {len(new_ids)} articles vs {len(old_ids)} before (< {ratio:.0%})")

    def log_memory(self) -> None:
        try:
            info = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
            avail_gb = int(info["MemAvailable"].strip().split()[0]) / 1024 / 1024
            self.log(f"Memory available before embedding: {avail_gb:.1f} GB")
            if avail_gb < 3:
                self.log("WARNING: under 3 GB free; loading the embedding model may OOM")
        except Exception:
            pass

    def restart_and_check_api(self) -> None:
        cmd = self.cfg.get("restart_cmd")
        if not cmd:
            self.log("No restart_cmd configured; restart the API manually to load the new data")
            return
        self.log(f"=== restart API: {cmd}")
        r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.log(r.stdout + r.stderr)
            raise StepFailed("restart_api")

        base = self.cfg.get("api_base", "http://localhost:8000").rstrip("/")
        deadline = time.time() + int(self.cfg.get("health_timeout_seconds", 600))
        while True:
            try:
                if requests.get(f"{base}/health", timeout=5).json().get("ok"):
                    break
            except Exception:
                pass
            if time.time() > deadline:
                raise StepFailed("api_health: API did not come back up")
            time.sleep(10)
        self.log("API is up")

        for q in self.cfg.get("smoke_queries", []):
            r = requests.post(f"{base}/search", json={"query": q, "per_page": 3, "log": False}, timeout=120)
            n = r.json().get("total_results", 0) if r.status_code == 200 else 0
            self.log(f"Smoke query {q!r}: HTTP {r.status_code}, {n} results")
            if r.status_code != 200 or n == 0:
                raise StepFailed(f"smoke_query:{q}")

    # ---------- main flow ----------

    def run(self) -> int:
        live_touched = False
        backed_up = False
        try:
            # 1. Download + validate into a staging dir (nothing live is touched)
            incoming_root = ROOT / "data" / "raw" / "incoming"
            if incoming_root.exists():  # leftovers from earlier failed runs (kept for debugging until now)
                for p in incoming_root.iterdir():
                    shutil.rmtree(p, ignore_errors=True)
            incoming = incoming_root / self.run_id
            incoming.mkdir(parents=True, exist_ok=True)
            names = [e["name"] for e in self.cfg["exports"]]
            if self.args.skip_fetch:
                self.log("Skipping download; using data/raw/exports/*.csv")
                for n in names:
                    shutil.copy2(EXPORTS_DIR / f"{n}.csv", incoming / f"{n}.csv")
            else:
                for exp in self.cfg["exports"]:
                    self.fetch_export(exp, incoming / f"{exp['name']}.csv")
            for n in names:
                self.validate_export(n, incoming / f"{n}.csv")

            unchanged = all(
                (EXPORTS_DIR / f"{n}.csv").exists() and filecmp.cmp(incoming / f"{n}.csv", EXPORTS_DIR / f"{n}.csv", shallow=False)
                for n in names
            )
            if unchanged and not self.args.force and not self.args.skip_fetch and not self.args.recreate_typesense:
                self.log("Exports unchanged since the last run; skipping the rebuild")
                weights_before = WEIGHTS.stat().st_mtime if WEIGHTS.exists() else None
                self.step("train ranker", self.py("22_train_ranker.py"), fatal=False)
                weights_after = WEIGHTS.stat().st_mtime if WEIGHTS.exists() else None
                if weights_after != weights_before and not self.args.no_restart:
                    self.restart_and_check_api()
                shutil.rmtree(incoming, ignore_errors=True)
                return self.finish(True, "unchanged")

            # 2. Back up, then promote the downloads
            self.backup()
            backed_up = True
            EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            for n in names:
                shutil.copy2(incoming / f"{n}.csv", EXPORTS_DIR / f"{n}.csv")
            shutil.rmtree(incoming, ignore_errors=True)

            # 3. Offline rebuild
            chunking = self.cfg.get("chunking", {})
            prev_combined = self.backup_dir / "data/raw/articles.csv"
            concat = self.py(
                "00_concat_raw_exports.py",
                "--inputs", *[str(EXPORTS_DIR / f"{n}.csv") for n in names],
                "--output", str(COMBINED_CSV),
            )
            if prev_combined.exists():
                concat += ["--carry-images-from", str(prev_combined)]
            self.step("concat exports", concat)
            self.step("phase 1 (clean + normalize)", self.py("run_all.py", "--input", str(COMBINED_CSV), "--force"))
            self.check_canonical()
            self.step("chunk", self.py(
                "10_chunk_articles.py",
                "--max-tokens", str(chunking.get("max_tokens", 200)),
                "--overlap-tokens", str(chunking.get("overlap_tokens", 40)),
                "--hard-max-tokens", str(chunking.get("hard_max_tokens", 384)),
            ))
            self.log_memory()
            self.step("embed (incremental)", self.py(
                "11_compute_embeddings.py", "--hard-max-tokens", str(chunking.get("hard_max_tokens", 384)),
            ))
            self.step("gazetteer", self.py("20_build_gazetteer.py"))
            self.step("transliteration vocab", self.py("21_build_translit_vocab.py"))
            # English titles/summaries (new or edited articles only). Not fatal: without it the
            # new articles just lack English fields until the next successful run.
            if self.step("translate to English", self.py("27_translate_articles.py"), fatal=False) != 0:
                self.log("WARNING: translation failed (HF_TOKEN missing or model not accepted?); "
                         "English fields keep last run's values, new articles have none")

            # 4. Live updates (in place: upsert + prune)
            live_touched = True
            if self.args.recreate_typesense:
                self.step("typesense recreate collection", self.py("05_typesense_create_collection.py"))
            self.step("typesense ingest", self.py("06_typesense_ingest.py", "--input", str(CANONICAL), "--prune"))
            self.step("qdrant ingest", self.py("13_qdrant_ingest.py", "--prune"))
            if self.step("typesense synonyms", self.py("24_typesense_synonyms.py"), fatal=False) != 0:
                self.log("WARNING: synonyms update failed; search works, synonyms may be incomplete")
            if self.step("train ranker", self.py("22_train_ranker.py"), fatal=False) != 0:
                self.log("Ranker training skipped (usually: not enough labels yet); keeping current weights")

            if self.args.no_restart:
                self.log("--no-restart: restart the API to load the new data")
            else:
                self.restart_and_check_api()
            return self.finish(True, "rebuilt")

        except Exception as e:
            failed = str(e) if isinstance(e, StepFailed) else f"{type(e).__name__}: {e}"
            self.log(f"FAILED: {failed}")
            self.log(traceback.format_exc())
            self.status["failed_step"] = failed
            if backed_up and not live_touched:
                try:
                    self.restore()
                    self.status["restored_backup"] = True
                except Exception as re:
                    self.log(f"Restore failed: {re}")
            elif live_touched:
                self.log(
                    "Live indexes may be partially updated. Fix the cause and re-run with --skip-fetch; "
                    f"previous artifacts are in {self.backup_dir}"
                )
            return self.finish(False, "failed")

    def finish(self, ok: bool, outcome: str) -> int:
        self.status.update({"ok": ok, "outcome": outcome, "finished_at": datetime.now(timezone.utc).isoformat()})
        (LOG_DIR / "last_status.json").write_text(json.dumps(self.status, indent=2, ensure_ascii=False))
        with (LOG_DIR / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.status, ensure_ascii=False) + "\n")
        self.log(f"Refresh {outcome.upper()} (log: {self.log_path})")
        if not ok:
            self.alert()
        return 0 if ok else 1

    def alert(self) -> None:
        url = self.cfg.get("alert_webhook_url")
        if not url:
            return
        text = (f"Hindi search weekly refresh FAILED at step '{self.status['failed_step']}'. "
                f"Log: {self.log_path}")
        try:
            requests.post(url, json={"text": text}, timeout=15)
        except Exception as e:
            self.log(f"Alert webhook failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly data refresh for the Hindi search stack")
    ap.add_argument("--config", default=str(ROOT / "config" / "refresh.json"))
    ap.add_argument("--skip-fetch", action="store_true", help="Rebuild from data/raw/exports/*.csv")
    ap.add_argument("--force", action="store_true", help="Rebuild even if the exports are unchanged")
    ap.add_argument("--no-restart", action="store_true", help="Don't restart the API at the end")
    ap.add_argument("--recreate-typesense", action="store_true",
                    help="Drop and recreate the Typesense collection before ingest (needed after schema changes)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock = (LOG_DIR / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another refresh is already running; exiting.", flush=True)
        sys.exit(0)

    sys.exit(RefreshRun(cfg, args).run())


if __name__ == "__main__":
    main()
