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

4. Quick smoke tests (auto-detects Hindi vs roman queries):

```bash
python scripts/07_typesense_search_cli.py --q "महिला"
python scripts/07_typesense_search_cli.py --q "bihar mahila yojana"
python scripts/07_typesense_search_cli.py --q "बिहार" --filter "locations_norm:=[bihar]"
```

Other helpers:

- `scripts/08_query_canonicalize.py --q "<query>"` shows the canonicalized query tokens.
- `scripts/09_export_typesense_schema.py` snapshots the live schema back into `data/phase_2`.

---

## Phase 3 — Chunking, embeddings, Qdrant ingest

1. Chunk title/summary/content with MPNet-aware windowing:

```bash
python scripts/10_chunk_articles.py \
  --input data/final/articles_canonical.parquet \
  --max-tokens 240 --overlap-tokens 40 --hard-max-tokens 480
```

2. Compute article + chunk embeddings (paraphrase-multilingual-mpnet-base-v2):

```bash
python scripts/11_compute_embeddings.py \
  --articles data/final/articles_canonical.parquet \
  --chunks data/phase_3/chunks.parquet \
  --batch-size 32
```

3. Recreate Qdrant collections & ingest:

```bash
python scripts/12_qdrant_create_collections.py --dim 768
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
  "ranker_version": "ranker_v1",
  "retrieval_version": "retrieval_v1"
}
```

#### `POST /search`
Request:
```json
{
  "query": "महिला सशक्तिकरण",
  "per_page": 10,
  "filter_by": "locations_norm:=[bihar]",
  "explain": true
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
- `mode` is `dev` (Devanagari) or `roman`.
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

## Troubleshooting & tips

- Logs (`logs/*.json`) are designed for quick sanity checks—skim them after each phase.
- `data/` is structured by stage; keep raw inputs immutable and rerun scripts when new CSVs arrive.
- Whenever you receive updated WordPress exports, run `python scripts/00_concat_raw_exports.py` before `scripts/run_all.py` so the new `Image Featured` URLs propagate all the way to the API/UI.
- The repo uses `scripts/utils.py` for reusable helpers. New scripts should import from there for consistent normalization/tokenization.
- Qdrant + Typesense data persists in `data/phase_2/typesense_data` and `data/phase_3/qdrant_storage`. Remove those directories if you need a clean rebuild.
- Postgres logs queries/candidates/labels. Update `DATABASE_URL` if you hook up a managed DB.

Happy searching! Let the team know if you add new stages so we can document them here.
