#!/usr/bin/env bash
# Switch the Hindi search to the trained model (models/e5-base-hi-idr-v1, int8 query encoder,
# Qdrant collections *_v2), or back. The current model, its collections and its vector files
# are never deleted, so rollback is a settings change plus a restart.
#
#   bash deploy/hindi_model_switch.sh status
#   bash deploy/hindi_model_switch.sh apply      # backs up, builds *_v2, switches, smoke-tests;
#                                                # rolls back by itself if the smoke test fails
#   bash deploy/hindi_model_switch.sh rollback   # back to the state before `apply`
#
# Needs on the server (copied from the laptop): models/e5-base-hi-idr-v1/,
# models/e5-base-hi-idr-v1-onnx/ (model.onnx itself not needed), data/phase_3_ft/*_vectors.parquet.
# Don't run while the weekly refresh runs (Saturday 18:30 UTC; it holds logs/refresh/.lock).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SERVICE=hindisearch-api
API=http://localhost:8000
BACKUP=data/backups/pre-model-switch
NEW_MODEL=models/e5-base-hi-idr-v1
NEW_ONNX_DIR=models/e5-base-hi-idr-v1-onnx
NEW_ONNX_FILE=onnx/model_qint8_avx2.onnx
NEW_ART=idr_articles_vec_v2
NEW_CHK=idr_chunks_vec_v2
SMOKE=("महिला सशक्तिकरण" "mahila yojana bihar" "शिक्षा नीति")

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "ERROR: $*"; exit 1; }

set_env() {  # set_env KEY VALUE: replace the line in .env, or append it
    if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}

refresh_running() {
    [ -f logs/refresh/.lock ] && ! flock -n logs/refresh/.lock true 2>/dev/null
}

wait_healthy() {
    for _ in $(seq 1 60); do
        curl -sf "$API/health" | grep -q '"ok":true' && return 0
        sleep 5
    done
    return 1
}

smoke() {
    for q in "${SMOKE[@]}"; do
        n=$(curl -sf -X POST "$API/search" -H 'Content-Type: application/json' \
            -d "{\"query\": \"$q\", \"per_page\": 10, \"log\": false}" \
            | $PY -c 'import json,sys; print(len(json.load(sys.stdin).get("results", [])))') || n=0
        log "  smoke '$q': $n results"
        [ "${n:-0}" -ge 5 ] || return 1
    done
}

restart_and_check() {
    sudo systemctl restart "$SERVICE"
    wait_healthy || { log "API did not come up; journalctl -u $SERVICE -n 50"; return 1; }
    smoke
}

status() {
    log "Settings in .env:"
    grep -E "^(EMBED_MODEL|EMBED_BACKEND|EMBED_ONNX_DIR|EMBED_ONNX_FILE|QDRANT_COLLECTION_ARTICLES|QDRANT_COLLECTION_CHUNKS)=" .env || true
    log "Backup from apply: $([ -f "$BACKUP/.env" ] && echo "$BACKUP (present)" || echo none)"
    log "Qdrant collections:"
    curl -sf http://localhost:6333/collections | $PY -c 'import json,sys; print("  " + ", ".join(sorted(c["name"] for c in json.load(sys.stdin)["result"]["collections"])))'
    log "API: $(curl -sf "$API/health" || echo 'not answering')"
}

rollback() {
    [ -f "$BACKUP/.env" ] || die "no backup at $BACKUP: nothing to roll back to"
    log "Restoring .env and vector files from $BACKUP"
    cp "$BACKUP/.env" .env
    cp "$BACKUP"/*_vectors.parquet data/phase_3/
    restart_and_check || die "rollback restart failed: check journalctl -u $SERVICE"
    log "Rolled back. The *_v2 collections are left in place (unused)."
    log "If a weekly refresh ran on the trained model, run one now so the old collections get"
    log "the articles added since: .venv/bin/python scripts/refresh_weekly.py --skip-fetch --force"
}

apply() {
    refresh_running && die "the weekly refresh is running; try again when it has finished"
    grep -q 'os.environ.get("EMBED_MODEL"' scripts/_phase4/hybrid_search_api.py \
        || die "this code has no EMBED_MODEL setting: deploy the merged code first"
    for f in "$NEW_MODEL/config.json" "$NEW_ONNX_DIR/$NEW_ONNX_FILE" "$NEW_ONNX_DIR/tokenizer.json" \
             data/phase_3_ft/article_vectors.parquet data/phase_3_ft/chunk_vectors.parquet; do
        [ -e "$f" ] || die "missing $f (copy it from the laptop)"
    done
    if [ -f "$BACKUP/.env" ]; then
        log "Backup already exists at $BACKUP (from an earlier apply); keeping it"
    else
        log "Backing up .env and the current vector files to $BACKUP"
        mkdir -p "$BACKUP"
        cp .env "$BACKUP/.env"
        cp data/phase_3/article_vectors.parquet data/phase_3/chunk_vectors.parquet "$BACKUP/"
    fi

    log "Building $NEW_ART / $NEW_CHK beside the current collections"
    QDRANT_COLLECTION_ARTICLES=$NEW_ART QDRANT_COLLECTION_CHUNKS=$NEW_CHK \
        $PY scripts/12_qdrant_create_collections.py --dim 768
    QDRANT_COLLECTION_ARTICLES=$NEW_ART QDRANT_COLLECTION_CHUNKS=$NEW_CHK \
        $PY scripts/13_qdrant_ingest.py --articles data/phase_3_ft/article_vectors.parquet \
        --chunks data/phase_3_ft/chunk_vectors.parquet

    log "Switching settings"
    set_env EMBED_MODEL "$NEW_MODEL"
    set_env EMBED_BACKEND onnx
    set_env EMBED_ONNX_DIR "$NEW_ONNX_DIR"
    set_env EMBED_ONNX_FILE "$NEW_ONNX_FILE"
    set_env QDRANT_COLLECTION_ARTICLES "$NEW_ART"
    set_env QDRANT_COLLECTION_CHUNKS "$NEW_CHK"
    # The weekly refresh reuses these, embedding only new or edited articles with the new model
    cp data/phase_3_ft/article_vectors.parquet data/phase_3_ft/chunk_vectors.parquet data/phase_3/

    if restart_and_check; then
        log "Live on the trained model. Undo with: bash deploy/hindi_model_switch.sh rollback"
    else
        log "Smoke test failed: rolling back"
        rollback
        exit 1
    fi
}

case "${1:-status}" in
    status) status ;;
    apply) apply ;;
    rollback) rollback ;;
    *) die "usage: $0 status|apply|rollback" ;;
esac
