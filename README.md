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
1) Copy env file:
```bash
cp .env.example .env

Edit .env (choose a strong API key).

Start Typesense:

docker compose up -d

Create collection (schema)
python scripts/10_typesense_create_collection.py --input data/final/articles_canonical.parquet

Ingest documents
python scripts/11_typesense_ingest.py --input data/final/articles_canonical.parquet

Quick search smoke test
python scripts/13_typesense_search_cli.py --q "महिला" --per-page 10
python scripts/13_typesense_search_cli.py --q "bihar mahila" --per-page 10
python scripts/13_typesense_search_cli.py --q "karoonga" --per-page 10

Query canonicalization (Roman→Devanagari v1)
python scripts/12_query_canonicalize.py --q "karoonga mahila yojana"
python scripts/12_query_canonicalize.py --q "महिला सशक्तिकरण"

Outputs/logs:

logs/phase2_ingest_report.json

logs/phase2_search_smoketest.json

logs/roman_canonicalize_report.json