"""
Weekly refresh for the English site search: download the five WP All Export CSVs (articles,
features, ground_up, posts, podcasts), rebuild the English data, update the English Typesense
and Qdrant collections in place, and restart the English API. The Hindi refresh
(scripts/refresh_weekly.py) is separate and untouched; this one waits while it runs.

Safety model (same as the Hindi refresh)
  1. Downloads are validated before anything else is touched.
  2. Current English artifacts are backed up (data/en/backups/<run_id>/) before the rebuild.
  3. The offline stages (prepare, chunk, embed) run first. If any fails, the backup is restored
     and the live collections / API are never touched.
  4. Only then are the live collections updated (upsert + prune, no drops) and the API
     restarted and smoke-tested.

Logs: logs/refresh_en/<run_id>.log, logs/refresh_en/last_status.json (shown in the admin panel),
logs/refresh_en/history.jsonl. Optional failure alerts via alert_webhook_url.

Config: config/refresh_en.json (git-ignored; copy config/refresh_en.example.json). Settings
(EN_EMBED_MODEL, EN_* collections) come from .env, as for the English API.

Usage
  python scripts/en/refresh_weekly_en.py                 # normal weekly run (cron)
  python scripts/en/refresh_weekly_en.py --skip-fetch    # rebuild from data/en/raw/exports/*.csv
  python scripts/en/refresh_weekly_en.py --force         # rebuild even if the exports are unchanged
  python scripts/en/refresh_weekly_en.py --no-restart    # don't restart the API
"""

from __future__ import annotations

import argparse
import fcntl
import filecmp
import json
import os
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
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "en"
LOG_DIR = ROOT / "logs" / "refresh_en"
HINDI_LOCK = ROOT / "logs" / "refresh" / ".lock"
EXPORTS_DIR = ROOT / "data" / "en" / "raw" / "exports"
CANONICAL = ROOT / "data" / "en" / "final" / "articles_canonical.parquet"
BACKUP_ROOT = ROOT / "data" / "en" / "backups"

# Export names are the file names 01_prepare_english.py reads the post type from
EXPORT_NAMES = ["articles", "features", "ground_up", "posts", "podcasts"]

# Everything a rebuild overwrites; backed up before and restored on an offline failure
BACKUP_PATHS = [
    "data/en/raw/exports",
    "data/en/final/articles_canonical.parquet",
    "data/en/phase_3/chunks.parquet",
    "data/en/phase_3/article_vectors.parquet",
    "data/en/phase_3/chunk_vectors.parquet",
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
            "site": "en",
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
            text = line.rstrip("\n").split("\r")[-1]
            self._log.write("    " + text + "\n")
            self._log.flush()
            print("    " + text, flush=True)
        rc = proc.wait()
        secs = round(time.time() - t0, 1)
        self.status["steps"].append({"name": name, "rc": rc, "seconds": secs})
        self.log(f"=== {name}: {'ok' if rc == 0 else f'exit code {rc}'} ({secs}s)")
        if rc != 0 and fatal:
            raise StepFailed(name)
        return rc

    def py(self, script: str, *args: str) -> List[str]:
        # -u: unbuffered, so progress reaches the log as it happens, not when the script exits
        return [sys.executable, "-u", str(SCRIPTS / script), *args]

    # ---------- waiting for the Hindi refresh ----------

    def wait_for_hindi_refresh(self) -> None:
        """Both refreshes load an embedding model; on the shared server they must not overlap."""
        deadline = time.time() + 60 * int(self.cfg.get("hindi_wait_minutes", 240))
        waited = False
        while HINDI_LOCK.exists():
            with HINDI_LOCK.open("a") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    break
                except BlockingIOError:
                    pass
            if time.time() > deadline:
                raise StepFailed("wait: the Hindi refresh is still running")
            if not waited:
                self.log("The Hindi refresh is running; waiting for it to finish")
                waited = True
            time.sleep(60)
        if waited:
            self.log("The Hindi refresh has finished; starting")

    # ---------- download ----------

    @staticmethod
    def fetched_dates() -> Dict[str, str]:
        """When each export was last downloaded, from the last run that got that far."""
        try:
            return dict(json.loads((LOG_DIR / "last_status.json").read_text(encoding="utf-8")).get("fetched") or {})
        except (OSError, ValueError):
            return {}

    @staticmethod
    def fetch_due(exp: Dict[str, Any], last: Optional[str]) -> bool:
        """Exports with every_weeks (posts: none new since 2021) are downloaded only that often."""
        weeks = exp.get("every_weeks")
        if not weeks or not last or not (EXPORTS_DIR / f"{exp['name']}.csv").exists():
            return True
        age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).days
        return age_days >= 7 * int(weeks) - 1   # -1: a run an hour early still counts as N weeks

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
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[], low_memory=False)
        except Exception as e:
            raise StepFailed(f"validate:{name}: not a readable CSV ({type(e).__name__}: {e})")
        df.columns = [c.lstrip("﻿").strip() for c in df.columns]
        # The English exports name the id column "id" (01_prepare_english.py takes either)
        if "ID" not in df.columns and "id" in df.columns:
            df = df.rename(columns={"id": "ID"})
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise StepFailed(f"validate:{name}: missing columns {missing}")
        if len(df) == 0 or (df["ID"].str.strip() == "").all():
            raise StepFailed(f"validate:{name}: no rows")
        if "Status" not in df.columns:
            self.log(f"[{name}] WARNING: no 'Status' column; drafts and other unpublished posts will be indexed")

        prev = EXPORTS_DIR / f"{name}.csv"
        if prev.exists():
            prev_rows = len(pd.read_csv(prev, dtype=str, keep_default_na=False, na_values=[], low_memory=False))
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
        old_path = self.backup_dir / "data/en/final/articles_canonical.parquet"
        new = pd.read_parquet(CANONICAL, columns=["id", "title"])
        new_ids = set(new["id"].astype(str))
        self.status["articles"] = len(new_ids)
        if not old_path.exists():
            self.log(f"Canonical articles: {len(new_ids)} (no previous version)")
            return
        old_ids = set(pd.read_parquet(old_path, columns=["id"])["id"].astype(str))
        added, removed = sorted(new_ids - old_ids), sorted(old_ids - new_ids)
        self.status.update({"articles_added": len(added), "articles_removed": len(removed)})
        self.log(f"Canonical articles: {len(new_ids)} (was {len(old_ids)}; +{len(added)} / -{len(removed)})")
        titles = dict(zip(new["id"].astype(str), new["title"]))
        for a in added[:30]:
            self.log(f"  added {a}: {titles.get(a, '')}")
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
            if avail_gb < 2:
                self.log("WARNING: under 2 GB free; loading the embedding model may run out of memory")
        except Exception:
            pass

    def restart_and_check_api(self) -> None:
        cmd = self.cfg.get("restart_cmd")
        if not cmd:
            self.log("No restart_cmd configured; restart the English API manually to load the new data")
            return
        self.log(f"=== restart API: {cmd}")
        r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.log(r.stdout + r.stderr)
            raise StepFailed("restart_api")

        base = self.cfg.get("api_base", "http://localhost:8001").rstrip("/")
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
            names = [e["name"] for e in self.cfg["exports"]]
            unknown = sorted(set(names) - set(EXPORT_NAMES))
            if unknown or sorted(names) != sorted(set(names)):
                raise StepFailed(f"config: export names must be distinct and among {EXPORT_NAMES}; got {names}")
            if set(EXPORT_NAMES) - set(names):
                self.log(f"WARNING: no export configured for {sorted(set(EXPORT_NAMES) - set(names))}; "
                         "those items will drop out of the search")
            if not os.environ.get("EN_EMBED_MODEL"):
                raise StepFailed("config: EN_EMBED_MODEL is not set in .env")
            # Without lxml the HTML cleaner falls back to plain tag stripping: every list in every
            # article changes, and a third of all passages would be re-embedded (hours on 2 cores)
            try:
                import lxml  # noqa: F401
            except ImportError:
                raise StepFailed("environment: lxml is missing (pip install -r requirements.txt)")
            self.wait_for_hindi_refresh()

            # 1. Download + validate into a staging dir (nothing live is touched)
            incoming_root = ROOT / "data" / "en" / "raw" / "incoming"
            if incoming_root.exists():  # leftovers from earlier failed runs (kept for debugging until now)
                for p in incoming_root.iterdir():
                    shutil.rmtree(p, ignore_errors=True)
            incoming = incoming_root / self.run_id
            incoming.mkdir(parents=True, exist_ok=True)
            # Download dates: the previous ones stay on record until the new files replace the
            # saved copies, so a failed run never makes a stale copy look fresh
            fetched = self.fetched_dates()
            self.status["fetched"] = fetched
            new_dates = dict(fetched)
            if self.args.skip_fetch:
                self.log("Skipping download; using data/en/raw/exports/*.csv")
                for n in names:
                    shutil.copy2(EXPORTS_DIR / f"{n}.csv", incoming / f"{n}.csv")
            else:
                for exp in self.cfg["exports"]:
                    n = exp["name"]
                    if self.fetch_due(exp, fetched.get(n)):
                        self.fetch_export(exp, incoming / f"{n}.csv")
                        new_dates[n] = datetime.now(timezone.utc).isoformat()
                    else:
                        # Still rebuilt from the saved copy, so new translations in the other
                        # exports keep getting linked to these items
                        self.log(f"[{n}] reusing the saved export from {fetched[n][:10]} "
                                 f"(downloaded every {exp['every_weeks']} weeks)")
                        shutil.copy2(EXPORTS_DIR / f"{n}.csv", incoming / f"{n}.csv")
            for n in names:
                self.validate_export(n, incoming / f"{n}.csv")

            unchanged = all(
                (EXPORTS_DIR / f"{n}.csv").exists() and filecmp.cmp(incoming / f"{n}.csv", EXPORTS_DIR / f"{n}.csv", shallow=False)
                for n in names
            )
            if unchanged and not self.args.force and not self.args.skip_fetch:
                self.log("Exports unchanged since the last run; skipping the rebuild")
                self.status["fetched"] = new_dates
                shutil.rmtree(incoming, ignore_errors=True)
                return self.finish(True, "unchanged")

            # 2. Back up, then promote the downloads (only the configured ones: the prepare step
            # reads every CSV in the folder)
            self.backup()
            backed_up = True
            if EXPORTS_DIR.exists():
                shutil.rmtree(EXPORTS_DIR)
            EXPORTS_DIR.mkdir(parents=True)
            for n in names:
                shutil.copy2(incoming / f"{n}.csv", EXPORTS_DIR / f"{n}.csv")
            shutil.rmtree(incoming, ignore_errors=True)
            self.status["fetched"] = new_dates

            # 3. Offline rebuild
            self.step("prepare (clean + published only + translation links)",
                      self.py("01_prepare_english.py", "--input", str(EXPORTS_DIR)))
            self.check_canonical()
            self.step("chunk", self.py("02_chunk_english.py"))
            self.log_memory()
            self.step("embed (new and edited articles)", self.py("03_embed_english.py"))

            # 4. Live updates (in place: upsert + prune; synonyms reloaded)
            live_touched = True
            self.step("index (typesense + qdrant)", self.py("04_index_english.py", "--prune"))

            if self.args.no_restart:
                self.log("--no-restart: restart the English API to load the new data")
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
                    self.status["fetched"] = self.fetched_dates()   # the restored copies' dates
                except Exception as re:
                    self.log(f"Restore failed: {re}")
            elif live_touched:
                self.log(
                    "Live collections may be partially updated. Fix the cause and re-run with --skip-fetch; "
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
        text = (f"English search weekly refresh FAILED at step '{self.status['failed_step']}'. "
                f"Log: {self.log_path}")
        try:
            requests.post(url, json={"text": text}, timeout=15)
        except Exception as e:
            self.log(f"Alert webhook failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly data refresh for the English site search")
    ap.add_argument("--config", default=str(ROOT / "config" / "refresh_en.json"))
    ap.add_argument("--skip-fetch", action="store_true", help="Rebuild from data/en/raw/exports/*.csv")
    ap.add_argument("--force", action="store_true", help="Rebuild even if the exports are unchanged")
    ap.add_argument("--no-restart", action="store_true", help="Don't restart the English API at the end")
    args = ap.parse_args()

    # The pipeline scripts read EN_EMBED_MODEL and the EN_* collections from the environment
    load_dotenv(ROOT / ".env")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock = (LOG_DIR / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another English refresh is already running; exiting.", flush=True)
        sys.exit(0)

    sys.exit(RefreshRun(cfg, args).run())


if __name__ == "__main__":
    main()
