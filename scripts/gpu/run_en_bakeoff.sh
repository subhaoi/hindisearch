#!/usr/bin/env bash
# English embedding models side by side on the same generated questions, with query-encoding
# time on 2 CPU threads (as on the search server). Usage:
#   bash scripts/gpu/run_en_bakeoff.sh data/synthetic/split_en.parquet full
set -u
cd ~/hindisearch
PY=~/venv/bin/python
SPLIT=$1
TAG=$2
for M in BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5 intfloat/e5-base-v2 \
         nomic-ai/nomic-embed-text-v1.5 Snowflake/snowflake-arctic-embed-m-v2.0 intfloat/multilingual-e5-large; do
    echo "== $(date -u +%H:%M) $M"
    $PY scripts/gpu/31_eval_vectors.py --lang en --model "$M" --split "$SPLIT" --sets train,test_q,test_new \
        --cpu-latency --out "runs/gpu/en_${TAG}_$(basename $M).json" || echo "FAILED $M"
done
echo "== $(date -u +%H:%M) bake-off done"
