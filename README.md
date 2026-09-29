# Hindi Search Demo

End-to-end pipeline for Hindi search research:

1. Canonicalize the WordPress export into clean Hindi text with traceable metadata.
2. Build a lexical Typesense index plus facets for exploration.
3. Chunk + embed content, push vectors to Qdrant, and run hybrid retrieval (lexical + semantic).
4. Serve a FastAPI search API with logging + feedback UI for rapid labeling.

Everything runs from this repo; no hidden notebooks or cloud jobs.

---

## Repository layout

```
data/             # raw inputs, intermediate stages, final artifacts
  └─ raw/         # WordPress exports + combined articles.csv (with Image Featured column)
logs/             # JSON reports from each phase
scripts/          # numbered pipeline + CLI utilities
scripts/_phase4   # hybrid API, ranker, DB helpers
scripts/_phase5   # lightweight feedback UI
docker-compose.yml
requirements.txt
```

Key outputs:

- `data/raw/articles.csv` — concatenation of the three WordPress exports (run `scripts/00_concat_raw_exports.py` whenever the source CSVs change; includes `Image Featured` URLs).
- `data/final/articles_canonical.parquet` — canonical dataset from Phase 1.
- `data/phase_2/typesense_schema.json` — created collection schema.
- `data/phase_3/chunks.parquet`, `chunk_vectors.parquet`, `article_vectors.parquet`.
- `data/phase_45/gazetteer_v1.json` — metadata gazetteer for auto filters.

Each step also logs to `logs/*.json` so you can inspect data quality.

---

## Prerequisites

- Python 3.10+ (tested on 3.11).
- Local Docker for Typesense / Qdrant / Postgres (or point to existing services via `.env`).
- Optional GPU speeds up embedding, but CPU works.

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment variables

```
cp .env.example .env
# edit .env with Typesense, Qdrant, Postgres credentials + API knobs
# RAW_ARTICLES_CSV can override the path used to fetch Image Featured URLs
```

---

## Phase 1 — Canonicalize WordPress export

1. Drop the CSV at `data/raw/articles.csv`.
   - If you maintain separate export files (Articles, Features, Ground-Up Stories), regenerate `articles.csv` with:
     ```bash
     python scripts/00_concat_raw_exports.py \
       --inputs data/raw/Articles-Export-2024-January-25-0205.csv \
                data/raw/Features-Export-2024-January-25-0214.csv \
                data/raw/Ground-Up-Stories-Export-2024-January-25-0217.csv \
       --output data/raw/articles.csv
     ```
     (Defaults already point to these filenames, so `python scripts/00_concat_raw_exports.py` also works.)
2. Run the orchestrator (it skips stages whose outputs already exist):

```bash
python scripts/run_all.py --input data/raw/articles.csv
```

What happens:

- `scripts/01_load_and_validate.py` makes Stage 1 parquet with defensive CSV parsing, null-ID pruning, and schema logs (`logs/load_errors.json`).
- `scripts/02_clean_text_wp.py` strips HTML, normalizes Hindi text, and records script-contamination/short-content cases (`logs/wp_strip_logs.json`).
- `scripts/03_normalize_metadata.py` splits pipe-separated metadata, normalizes tokens, parses ISO dates, and saves `logs/multivalue_logs.json`.
- `scripts/04_quality_checks.py` summarizes nulls, content lengths, and sample script stats (`logs/quality_report.json`).
- Final output lives at `data/final/articles_canonical.parquet`.

If you only need a specific stage you can call the scripts directly with `--input` + `--root`.

---

## Phase 2 — Lexical Typesense index + facets

1. Start infra:

```bash
docker compose up -d   # spins up Typesense, Qdrant, Postgres with local volumes
```

2. Create/recreate the collection (reads `.env` values):

```bash
python scripts/05_typesense_create_collection.py
```

3. Ingest canonical data:

```bash
python scripts/06_typesense_ingest.py --input data/final/articles_canonical.parquet
```

4. Load the synonym list (`config/synonyms_v1.json`; re-run after editing it):

```bash
python scripts/24_typesense_synonyms.py
```

5. Quick smoke tests (auto-detects Hindi vs roman queries):

```bash
python scripts/07_typesense_search_cli.py --q "महिला"
python scripts/07_typesense_search_cli.py --q "bihar mahila yojana"
python scripts/07_typesense_search_cli.py --q "बिहार" --filter "locations_norm:=[बिहार]"
```

How roman queries match: Devanagari text is romanized the way people type Hindi (शिक्षा → shikshaa, सरकार → sarkaar) and both the index and the query are reduced to a phonetic *match key* (`utils.text_to_key`), so `shiksha`/`siksha` and `kisan`/`kisaan` hit the same documents. Metadata (locations, contributors, tags) is stored in Devanagari; `locations_key` / `contributors_key` hold its match keys.

Other helpers:

- `scripts/08_query_canonicalize.py --q "<query>"` shows the canonicalized query tokens.
- `scripts/09_export_typesense_schema.py` snapshots the live schema back into `data/phase_2`.

---

## Phase 3 — Chunking, embeddings, Qdrant ingest

1. Chunk title/summary/content with token-aware windowing (E5 tokenizer):

```bash
python scripts/10_chunk_articles.py \
  --input data/final/articles_canonical.parquet \
  --max-tokens 240 --overlap-tokens 40 --hard-max-tokens 480
```

2. Compute article + chunk embeddings (`intfloat/multilingual-e5-large`, 1024-dim):

```bash
python scripts/11_compute_embeddings.py \
  --articles data/final/articles_canonical.parquet \
  --chunks data/phase_3/chunks.parquet \
  --batch-size 32
```

3. Recreate Qdrant collections & ingest:

```bash
python scripts/12_qdrant_create_collections.py --dim 1024
python scripts/13_qdrant_ingest.py \
  --articles data/phase_3/article_vectors.parquet \
  --chunks data/phase_3/chunk_vectors.parquet \
  --batch-size 128
```

4. Semantic CLI demo (reads chunk parquet for snippets):

```bash
python scripts/14_semantic_search_cli.py --q "महिला सशक्तिकरण" --topk 10
```

---

## Phase 4 — Hybrid search API (Typesense + Qdrant + ranker)

Artifacts used:

- Canonical articles + chunk parquet
- Gazetteer (`python scripts/20_build_gazetteer.py`)
- Roman → Devanagari query vocabulary (`python scripts/21_build_translit_vocab.py`)
- Core QA query set (`python scripts/19_build_core_query_set.py`)
- Vector stores + Typesense index + Postgres
- Raw CSV (`data/raw/articles.csv`) for `Image Featured` links (override via `RAW_ARTICLES_CSV`)

Start the API (uvicorn example):

```bash
uvicorn scripts._phase4.hybrid_search_api:app --host 0.0.0.0 --port 8000
```

### API endpoints

#### `GET /health`
```json
{
  "ok": true,
  "ranker_version": "ranker_v2",
  "retrieval_version": "retrieval_v2"
}
```

#### `POST /search`
Request:
```json
{
  "query": "महिला सशक्तिकरण",
  "per_page": 10,
  "filter_by": "locations_norm:=[बिहार]",
  "explain": true,
  "log": true,
  "ranker": null
}
```
Response:
```json
{
  "query_id": 123,
  "mode": "dev",
  "query_used": "महिला सशक्तिकरण",
  "query_semantic": "महिला सशक्तिकरण",
  "results": [
    {
      "rank": 1,
      "id": "14521",
      "title": "महिला नेतृत्व कार्यक्रम",
      "date": "2023-09-12T05:30:00",
      "summary": "…",
      "url": "https://example.com/article",
      "image_url": "https://example.com/uploads/featured.jpg",
      "primary_category": "Gender",
      "categories": ["Gender"],
      "tags": ["training"],
      "location": ["bihar"],
      "partner_label": null,
      "contributors": ["IDR Staff"],
      "score": 0.92,
      "snippet": "…",
      "features": { "...": "..." },
      "explanation": [["lex", 0.5], ["sem_chunk", 0.3]]
    }
  ]
}
```
Notes:
- `mode` is `dev` (Devanagari), `roman`, or `mixed`.
- `log=false` skips Postgres logging (used by the evaluation script; `query_id` is then 0).
- `ranker` overrides `RANKER_VERSION` for one request (`ranker_v1` | `ranker_v2`).
- `image_url` comes from the `Image Featured` column in the raw CSV.
- Setting `explain=true` includes `features` and `explanation` arrays; omit to reduce payload size.

#### `POST /label`
```json
{
  "query_id": 123,
  "article_id": "14521",
  "label": 1,
  "note": "Perfect match"
}
```

#### `POST /label_query`
Used when *no* result is relevant:
```json
{
  "query_id": 123,
  "label": 0,
  "note": "Query misunderstood"
}
```

All writes land in Postgres via `scripts/_phase4/db.py`.

### How a search runs

1. **Canonicalize**: detect script (`dev` / `roman` / `mixed`), drop stopwords, turn Latin tokens into match keys.
2. **Entities** (`_phase4/query_entities.py`): match locations/contributors/categories/tags from the gazetteer as whole-word phrases, in Devanagari or romanized form.
   - Hard filter (Typesense `filter_by` + Qdrant restriction) only for specific locations and multi-word author names.
   - Locations tagged on >20% of articles (e.g. भारत) and single-word names only boost ranking.
   - Tags and categories always only boost.
3. **Semantic query**: raw query minus hard-matched entity words. Romanized Hindi is converted to Devanagari via the corpus vocabulary; English queries are embedded as typed. Embedded once for both Qdrant searches.
4. **Retrieve**: Typesense (top 80), Qdrant articles (top 40), Qdrant chunks (top 80).
5. **Rank** (`_phase4/ranker_v2.py`): reciprocal rank fusion of the three lists plus per-article entity-match, and recency boosts. Weights come from `data/phase_4/ranker_v2_weights.json` when present (see Phase 6), else defaults. `ranker_v1` (min-max score blend) is kept for comparison.

CLI client:

```bash
python scripts/18_hybrid_search_cli.py --q "महिला yojana" --k 15 --host http://localhost:8000 --explain
```

---

## Phase 5 — Feedback UI

Lightweight FastAPI frontend (`scripts/_phase5/feedback_ui.py`) that:

- Calls the hybrid API.
- Shows featured thumbnails when available, lets annotators sort (relevance/newest/oldest), and mark “Correct/Wrong/None”.
- Posts labels through `/label` and `/label_query`.

Run it alongside the API (default `SEARCH_API_BASE=http://localhost:8000`):

```bash
uvicorn scripts._phase5.feedback_ui:app --host 0.0.0.0 --port 8500
```

---

## Phase 6 — Measure and tune

Evaluate against the labels collected in the feedback UI (replays every labelled query with `log=false`; also snapshots the core query set from `scripts/19_build_core_query_set.py`):

```bash
python scripts/23_evaluate_search.py --save runs/baseline.json
# ...change something, restart the API...
python scripts/23_evaluate_search.py --save runs/new.json --compare runs/baseline.json
python scripts/23_evaluate_search.py --ranker ranker_v1   # A/B rankers on the same server
```

Reports nDCG@10, MRR, recall@50, known-wrong results in the top 10, and label coverage, split by query mode. Unlabelled results count as not relevant, so compare runs rather than reading absolute numbers.

Learn ranker weights once there are a few hundred labels on `ranker_v2` queries (writes the weights only if they beat the defaults on held-out queries):

```bash
python scripts/22_train_ranker.py
```

Faster CPU query encoding (optional). Export on a machine with ~6 GB free RAM, copy `models/` to the API host, then set `EMBED_BACKEND=onnx` in `.env`. The script prints how closely int8 query vectors match the original model; confirm with `23_evaluate_search.py` before switching.

```bash
python scripts/25_export_onnx_query_encoder.py --config avx2
```

### Upgrading an existing deployment to retrieval_v2

Vectors in Qdrant are unchanged; Typesense, the gazetteer and the new vocabulary need rebuilding:

```bash
python scripts/23_evaluate_search.py --save runs/before.json   # against the currently deployed API
python scripts/20_build_gazetteer.py
python scripts/21_build_translit_vocab.py
python scripts/05_typesense_create_collection.py                # drops + recreates the collection
python scripts/06_typesense_ingest.py --input data/final/articles_canonical.parquet
python scripts/24_typesense_synonyms.py
# restart the API (RANKER_VERSION=ranker_v2, RETRIEVAL_VERSION=retrieval_v2 in .env)
python scripts/23_evaluate_search.py --save runs/after.json --compare runs/before.json
```

---

## Weekly refresh (cron)

`scripts/refresh_weekly.py` keeps the index in sync with the site. Each run:

1. Triggers the three WP All Export jobs, polls until they finish, downloads the CSVs and validates them. A missing column, an unreadable file, or a drop below 90% of last week's rows fails the run before anything is touched.
2. If nothing changed since last week, skips the rebuild (it still retries ranker training).
3. Backs up current artifacts to `data/backups/<run_id>/` (keeps the last 4).
4. Rebuilds offline: concat (carrying `Image Featured` URLs over by ID when an export lacks them), Phase 1, chunking, **incremental** embeddings (only new/changed text is embedded), gazetteer, transliteration vocab. Any failure here restores the backup; live search is untouched.
5. Updates live indexes in place (upsert + delete removed articles; no collection drops), refreshes synonyms, retrains ranker weights if there are enough labels.
6. Restarts the API (`restart_cmd`), waits for `/health`, and runs smoke queries.

Setup on the server:

```bash
cp config/refresh.example.json config/refresh.json   # then fill in export keys/tokens (git-ignored)
sudo cp deploy/hindisearch-api.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now hindisearch-api
crontab -e    # paste the line from deploy/crontab.txt
```

Run it by hand the first time (in `tmux`): the first run embeds everything because earlier vector files have no text hashes.

```bash
python scripts/refresh_weekly.py            # --skip-fetch / --force / --no-restart
```

Logs: `logs/refresh/<run_id>.log` (full output), `logs/refresh/last_status.json` (outcome, failed step, article counts), `logs/refresh/history.jsonl`, and cron output in `logs/refresh_cron.log`. Set `alert_webhook_url` in the config (e.g. a Slack incoming webhook) to get a message when a run fails.

If a run fails after the live-update step, fix the cause and re-run with `--skip-fetch`.

---

## Retrieval v3 (search quality step 1)

- **Titles**: the site headline (`Title`) is the title; the Yoast SEO title is kept as `seo_title_hi` (searched, and included in the article vector). Section suffixes like `| हल्का-फुल्का | आईडीआर` are stripped.
- **Locations**: an article's locations are its tags plus any location named in its title/summary or mentioned 3+ times in the body (`locations_all`). Location filters use this set because tags alone miss many articles (e.g. असम: 20 tagged, 29 with mentions).
- **English place names**: `config/location_aliases.json` maps delhi/orissa/bengal/kashmir/... to the Hindi values.
- **Spelling folding**: Hindi fields and queries drop nukta and chandrabindu differences (ज़रूरत = जरूरत, गाँव = गांव).
- **Stemming**: Hindi queries also run against stemmed fields (`*_stem`), so बच्चा matches बच्चे/बच्चों; exact-form matches are fused ahead.

Deploying it changes the Typesense schema, so the first run must recreate the collection (about a minute of degraded search), and it re-embeds article vectors once because titles changed:

```bash
python scripts/refresh_weekly.py --skip-fetch --recreate-typesense
```

Set `RETRIEVAL_VERSION=retrieval_v3` in `.env` so query logs before/after are distinguishable.

---

## Troubleshooting & tips

- Logs (`logs/*.json`) are designed for quick sanity checks—skim them after each phase.
- `data/` is structured by stage; keep raw inputs immutable and rerun scripts when new CSVs arrive.
- Whenever you receive updated WordPress exports, run `python scripts/00_concat_raw_exports.py` before `scripts/run_all.py` so the new `Image Featured` URLs propagate all the way to the API/UI.
- The repo uses `scripts/utils.py` for reusable helpers. New scripts should import from there for consistent normalization/tokenization.
- Qdrant + Typesense data persists in `data/phase_2/typesense_data` and `data/phase_3/qdrant_storage`. Remove those directories if you need a clean rebuild.
- Postgres logs queries/candidates/labels. Update `DATABASE_URL` if you hook up a managed DB.

Happy searching! Let the team know if you add new stages so we can document them here.
