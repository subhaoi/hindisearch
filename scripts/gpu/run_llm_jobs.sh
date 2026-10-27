#!/usr/bin/env bash
# All language-model jobs, one after another (they share the GPU). Resumable: re-running
# skips finished articles and judged pairs. Log: ~/llm_jobs.log
cd "$(dirname "$0")/../.."
PY=~/venv/bin/python
M=${MODEL:-gemma3:12b}
echo "== $(date -u +%H:%M) Hindi queries"
$PY -u scripts/28_generate_queries.py --lang hi --model "$M" --workers 4 --newest-first \
    --out data/synthetic/queries_hi.jsonl
echo "== $(date -u +%H:%M) English queries"
$PY -u scripts/28_generate_queries.py --lang en --model "$M" --workers 4 --newest-first --max-chars 2000 \
    --articles data/en/final/articles_canonical.parquet --out data/synthetic/queries_en.jsonl
echo "== $(date -u +%H:%M) Judging real Hindi searches"
$PY -u scripts/gpu/33_judge.py --lang hi --from-logs --model "$M" --workers 4
echo "== $(date -u +%H:%M) LLM jobs done"
