#!/usr/bin/env bash
# Qwen 3.8 27B NVFP4 + DSpark k=14 — fallback lite desde 2026-09-01.
# Reemplazado como default por Qwen3.8-Flash-Next NVFP4 hybrid, pero se conserva
# este script porque DSpark k=14 es muy competitivo en concurrencia y el modelo
# ocupa solo ~22 GB (cabe en pool junto con otras cosas).
#
# MEDIDO en este Spark (vLLM v0.27.1, 2026-08-18):
#   Fresh-code     29.67 tok/s
#   EDIT-heavy     73-76 tok/s (warm prefix cache)
#   c=1   70.66 tok/s  c=4 150.64  c=8 182.25  c=16 253.18 tok/s aggregate
#
# Recipe upstream: 0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark
# Adaptado a NVFP4 (unsloth/Qwen3.8-27B-NVFP4) y agregado DSpark (Doopeworld).
#
# Requiere:
#   - ~/vllm/qwen3.8-27b-nvfp4  (NVFP4 + MTP heads, ~22 GB)
#   - ~/vllm/qwen3.8-dspark     (DSpark drafter, ~2.7 GB)
#
# Uso interactivo: ./run-qwen38-27b-nvfp4-dspark.sh
# Uso como servicio (foreground, para systemd): SERVICE_MODE=1 ./run-qwen38-27b-nvfp4-dspark.sh

set -euo pipefail

MODEL_DIR="${HOME}/vllm/qwen3.8-27b-nvfp4"
DRAFTER_DIR="${HOME}/vllm/qwen3.8-dspark"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.27.1-aarch64}"
NAME="${NAME:-qwen38-nvfp4}"
PORT="${PORT:-8001}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b-nvfp4}"
CONTAINER_MODEL_DIR="/models/qwen3.8"
CONTAINER_DRAFTER_DIR="/models/qwen3.8-dspark"

TRITON_CACHE_DIR="${HOME}/vllm/triton-cache-qwen38-nvfp4"
VLLM_CACHE_DIR="${HOME}/vllm/vllm-cache-qwen38-nvfp4"

GMU="${GMU:-0.85}"
MAX_LEN="${MAX_LEN:-262144}"
MAX_BATCHED="${MAX_BATCHED:-16384}"
K="${K:-14}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: modelo NVFP4 no encontrado en ${MODEL_DIR}"
  echo "Descarga con: hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ${MODEL_DIR}"
  exit 1
fi
if [[ ! -d "${DRAFTER_DIR}" ]]; then
  echo "ERROR: DSpark drafter no encontrado en ${DRAFTER_DIR}"
  echo "Descarga con: hf download Doopeworld/Qwen3.8-27B-DSpark-vLLM --local-dir ${DRAFTER_DIR}"
  exit 1
fi

mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_DIR}"
docker rm -f "${NAME}-${PORT}" >/dev/null 2>&1 || true

SPEC_CFG="{\"method\":\"dspark\",\"model\":\"${CONTAINER_DRAFTER_DIR}\",\"num_speculative_tokens\":${K},\"draft_sample_method\":\"probabilistic\"}"

DOCKER_ARGS=(
  --rm --name "${NAME}-${PORT}"
  --gpus all
  --ipc host
  --network host
  --shm-size 64gb
  -v "${MODEL_DIR}:${CONTAINER_MODEL_DIR}:ro"
  -v "${DRAFTER_DIR}:${CONTAINER_DRAFTER_DIR}:ro"
  -v "${TRITON_CACHE_DIR}:/root/.triton"
  -v "${VLLM_CACHE_DIR}:/root/.cache/vllm"
  -e VLLM_TARGET_DEVICE=cuda
  -e VLLM_FLOAT32_MATMUL_PRECISION=high
  -e CUTE_DSL_ARCH=sm_121a
  -e TRITON_CACHE_DIR=/root/.triton
  -e HF_HUB_DISABLE_PROGRESS_BARS=1
)

# La imagen `vllm/vllm-openai:v0.27.1-aarch64` trae ENTRYPOINT `["vllm","serve"]`.
VLLM_ARGS=(
  "${CONTAINER_MODEL_DIR}"
  --served-model-name "${SERVED_NAME}"
  --host 0.0.0.0 --port "${PORT}"
  --max-model-len "${MAX_LEN}"
  --gpu-memory-utilization "${GMU}"
  --max-num-batched-tokens "${MAX_BATCHED}"
  --enable-prefix-caching
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --limit-mm-per-prompt.image 2
  --limit-mm-per-prompt.video 0
  --speculative-config "${SPEC_CFG}"
)

if [[ "${SERVICE_MODE}" == "1" ]]; then
  echo ">> ${NAME}-${PORT} en foreground: ${SERVED_NAME} (DSpark k=${K}, GMU=${GMU})..."
  exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${VLLM_ARGS[@]}"
fi

# Modo interactivo
echo ">> ${SERVED_NAME} en :${PORT} (DSpark k=${K} GMU=${GMU})..."
docker run -d "${DOCKER_ARGS[@]}" "${IMAGE}" "${VLLM_ARGS[@]}"
echo ">> Esperando healthcheck (~5-8 min primera vez por FP4 autotune + DSpark graphs)..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  sleep 10
done
echo ">> Listo en http://localhost:${PORT}/v1 (model: ${SERVED_NAME})"