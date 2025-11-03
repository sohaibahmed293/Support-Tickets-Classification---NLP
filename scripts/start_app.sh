#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv/bin/activate"

if [ -f "$VENV_PATH" ]; then
    # shellcheck source=/dev/null
    source "$VENV_PATH"
else
    echo "warning: virtual environment not found at .venv/. Using system Python." >&2
fi

export FLASK_APP="${FLASK_APP:-webapp.app}"
export MODEL_ARTEFACT_DIR="${MODEL_ARTEFACT_DIR:-$ROOT_DIR/artifacts}"
export FLASK_RUN_PORT="${FLASK_RUN_PORT:-5000}"

echo "Starting Flask app..."
echo "    FLASK_APP=${FLASK_APP}"
echo "    MODEL_ARTEFACT_DIR=${MODEL_ARTEFACT_DIR}"
echo "    FLASK_RUN_PORT=${FLASK_RUN_PORT}"
echo

exec flask run "$@"
