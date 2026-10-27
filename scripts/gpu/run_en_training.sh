#!/usr/bin/env bash
# English: once the English questions are written (run_llm_jobs.sh has moved on), split them,
# score the base model, fine-tune it, score the result, and embed the site with it.
# Usage: bash scripts/gpu/run_en_training.sh intfloat/e5-base-v2 e5-base-v2-en-idr-v1
set -u
cd ~/hindisearch
PY=~/venv/bin/python
BASE=$1
NAME=$2
until grep -q "Judging real Hindi searches" ~/llm_jobs.log; do sleep 60; done
echo "== $(date -u +%H:%M) split"
$PY scripts/gpu/30_split_queries.py --queries data/synthetic/queries_en.jsonl --out data/synthetic/split_en.parquet
echo "== $(date -u +%H:%M) baseline $BASE"
$PY scripts/gpu/31_eval_vectors.py --lang en --model "$BASE" --out runs/gpu/en_$(basename $BASE)_base.json
echo "== $(date -u +%H:%M) fine-tune"
$PY scripts/gpu/32_finetune.py --lang en --base "$BASE" --out models/$NAME --epochs 2 --lr 2e-5
rm -rf models/$NAME/checkpoints
echo "== $(date -u +%H:%M) evaluate fine-tuned"
$PY scripts/gpu/31_eval_vectors.py --lang en --model models/$NAME --out runs/gpu/en_${NAME}.json
echo "== $(date -u +%H:%M) embed the English site with it"
$PY scripts/en/03_embed_english.py --model models/$NAME --out-dir data/en/phase_3_ft --batch-size 128 --no-reuse
echo "== $(date -u +%H:%M) done English training"
