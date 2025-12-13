# Hindi Search Demo — Phase 1 Data Canonicalization (0–6 hours)

This repo builds a reproducible, resilient canonical dataset from a WordPress-export CSV.

## Input CSV columns expected
From your example:
- ID, Date, Title, Content, Permalink, Categories, Locations, Tags, Contributors,
  wph_summary, wph_article_type, wph_multimedia_type,
  _yoast_wpseo_title, _yoast_wpseo_metadesc, wph_partner_label

## Where to put the CSV
Copy your CSV to:
- data/raw/articles.csv

## Setup (server recommended)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_all.py --input data/raw/articles.csv


# Phase 2 — Typesense Lexical Search + Facets (6–14 hours)

Phase 2 builds a lexical search index in Typesense using the canonical dataset produced in Phase 1.

## Prerequisites
- Phase 1 output exists:
  - data/final/articles_canonical.parquet

## Start Typesense (server recommended)
Start Typesense:

cp .env.example .env
# edit .env, set TYPESENSE_API_KEY
docker compose up -d


Create collection:

python scripts/05_typesense_create_collection.py


Ingest from Phase 1 output:

python scripts/06_typesense_ingest.py --input data/final/articles_canonical.parquet


Search tests:

python scripts/07_typesense_search_cli.py --q "महिला"
python scripts/07_typesense_search_cli.py --q "bihar mahila yojana"
python scripts/07_typesense_search_cli.py --q "karoonga"
python scripts/07_typesense_search_cli.py --q "shiksha karyakram"
python scripts/07_typesense_search_cli.py --q "बिहार" --filter "locations_norm:=[bihar]"