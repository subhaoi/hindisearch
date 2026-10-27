#!/usr/bin/env bash
# Hindi: fine-tune one base model on the generated questions and score it beside the base model
# on the same test sets. Usage: bash scripts/gpu/run_hi_training.sh intfloat/multilingual-e5-base e5-base
set -u
cd ~/hindisearch
PY=~/venv/bin/python
BASE=$1
NAME=$2
echo "== $(date -u +%H:%M) baseline $BASE"
[ -f runs/gpu/hi_${NAME}_base.json ] || $PY scripts/gpu/31_eval_vectors.py --lang hi --model "$BASE" \
    --sets train,test_q,test_new --out runs/gpu/hi_${NAME}_base.json
echo "== $(date -u +%H:%M) fine-tune $BASE"
$PY scripts/gpu/32_finetune.py --lang hi --base "$BASE" --out models/${NAME}-idr-ft
echo "== $(date -u +%H:%M) evaluate fine-tuned"
$PY scripts/gpu/31_eval_vectors.py --lang hi --model models/${NAME}-idr-ft \
    --sets train,test_q,test_new --out runs/gpu/hi_${NAME}_ft.json
echo "== $(date -u +%H:%M) done $NAME"
