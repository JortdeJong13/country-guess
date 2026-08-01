#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEBUG="${DEBUG:-1}"
MODEL_NAME="${MODEL_NAME:-triplet_model}"
MLSERVER_URL="${MLSERVER_URL:-http://127.0.0.1:5001}"
DRAWING_STORE_URL="${DRAWING_STORE_URL:-http://127.0.0.1:8080}"
DATABASE_URL="${DATABASE_URL:-postgres://country_guess:country_guess_dev@127.0.0.1:5432/country_guess?sslmode=disable}"

if ! command -v pg_isready >/dev/null 2>&1 || ! pg_isready -h 127.0.0.1 -p 5432 >/dev/null 2>&1; then
    echo "Native PostgreSQL 16 is not ready. Start it and run 'just setup-local-db' first."
    exit 1
fi

uv sync --locked --only-group app

DRAWINGSTORE_PID=""
MLSERVER_PID=""

cleanup() {
    trap - EXIT INT TERM
    echo "Stopping local app services..."

    for pid in "$DRAWINGSTORE_PID" "$MLSERVER_PID"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    for pid in "$DRAWINGSTORE_PID" "$MLSERVER_PID"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}

trap cleanup EXIT INT TERM

echo "Starting drawingstore..."
DATABASE_URL="$DATABASE_URL" go -C drawingstore run . &
DRAWINGSTORE_PID=$!

echo "Starting ML server..."
DEBUG="$DEBUG" MODEL_NAME="$MODEL_NAME" uv run --locked --no-sync python -m mlserver.serve &
MLSERVER_PID=$!

echo "Starting web app..."
DEBUG="$DEBUG" MLSERVER_URL="$MLSERVER_URL" DRAWING_STORE_URL="$DRAWING_STORE_URL" uv run --locked --no-sync python -m webapp.app
