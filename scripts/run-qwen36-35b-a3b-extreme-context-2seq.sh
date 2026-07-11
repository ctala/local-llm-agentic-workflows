#!/usr/bin/env bash
# EXTREME context config for Qwen 3.6 35B-A3B NVFP4 (nvidia checkpoint) on DGX Spark.
# Targets 262K context per sequence with 2 concurrent sequences.
# This is now equivalent to scripts/run-qwen36-35b-a3b.sh; kept for discoverability.
# Requiere: ~/vllm/qwen3.6-35b-a3b-nvfp4-nvidia

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPO_DIR}/scripts/run-qwen36-35b-a3b.sh" "$@"
