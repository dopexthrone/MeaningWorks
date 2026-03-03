#!/usr/bin/env bash
# Restore SQLite databases from Litestream (Cloudflare R2) backup.
# Run BEFORE starting the app on a fresh/recovered server.

set -euo pipefail

if [ -z "${LITESTREAM_ENDPOINT:-}" ] || [ -z "${LITESTREAM_BUCKET:-}" ]; then
    echo "Error: LITESTREAM_ENDPOINT and LITESTREAM_BUCKET must be set"
    exit 1
fi

DATA_DIR="${MOTHERLABS_DATA_DIR:-/data}"
mkdir -p "$DATA_DIR/corpus"

echo "=== Restoring databases from R2 ==="

for db in corpus/corpus.db auth.db huey.db outcomes.db; do
    target="$DATA_DIR/$db"
    echo "Restoring $db -> $target"
    litestream restore \
        -o "$target" \
        -config /etc/litestream.yml \
        "$target" 2>/dev/null || echo "  (no backup found for $db — starting fresh)"
done

echo "=== Restore complete ==="
