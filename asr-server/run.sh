#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/.venv/bin/activate"

# Default model tuned for Spanish/English quality on DGX Spark CPU.
export ASR_MODEL="${ASR_MODEL:-large-v3}"
export ASR_DEVICE="${ASR_DEVICE:-cpu}"
export ASR_COMPUTE_TYPE="${ASR_COMPUTE_TYPE:-int8}"
export ASR_HOST="${ASR_HOST:-0.0.0.0}"
export ASR_PORT="${ASR_PORT:-8001}"

exec python "${SCRIPT_DIR}/server.py"
