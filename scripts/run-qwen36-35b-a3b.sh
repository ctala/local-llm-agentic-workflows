#!/usr/bin/env bash
# Lanza Qwen 3.6 35B-A3B NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4) en DGX Spark.
# Checkpoint W4A16 NVFP4 con vLLM nightly. Contexto máximo del modelo (262K),
# tool calling robusto vía qwen3_coder y dos secuencias en paralelo.
# Requiere: ~/vllm/qwen3.6-35b-a3b-nvfp4-nvidia
#
# Uso interactivo: ./scripts/run-qwen36-35b-a3b.sh
# Uso como servicio (foreground, para systemd): SERVICE_MODE=1 ./scripts/run-qwen36-35b-a3b.sh

set -euo pipefail

MODEL_DIR="${HOME}/vllm/qwen3.6-35b-a3b-nvfp4-nvidia"
IMAGE="vllm/vllm-openai:nightly@sha256:a671d5fcda70fe9ac6f245f9780821de459fb4ee22c018fd07a0f10a55279bf9"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.92}"
SERVICE_MODE="${SERVICE_MODE:-0}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: Modelo no encontrado en ${MODEL_DIR}"
  echo "Descárgalo con:"
  echo "  huggingface-cli download nvidia/Qwen3.6-35B-A3B-NVFP4 \\"
  echo "    --local-dir ${MODEL_DIR} --local-dir-use-symlinks False"
  exit 1
fi

# Ruta al chat-template relativa al repo (asumiendo ejecución desde la raíz)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAT_TEMPLATE_DIR="${REPO_DIR}/chat-templates"

DOCKER_ARGS=(
  --gpus all
  --ipc host
  -p "${PORT}:${PORT}"
  --shm-size 64gb
  -v "${MODEL_DIR}:/models/qwen3.6"
  -v "${CHAT_TEMPLATE_DIR}:/chat-templates"
  -e HF_HOME=/models
  -e VLLM_TARGET_DEVICE=cuda
)

VLLM_ARGS=(
  --model /models/qwen3.6
  --served-model-name qwen3.6-35b-a3b
  --host 0.0.0.0 --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size 1
  --attention-backend flashinfer
  --moe-backend marlin
  --kv-cache-dtype fp8
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len 262144
  --max-num-seqs 2
  --max-num-batched-tokens 32768
  --enable-chunked-prefill
  --async-scheduling
  --enable-prefix-caching
  --limit-mm-per-prompt '{"image":4}'
  --load-format fastsafetensors
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --enable-auto-tool-choice
  --chat-template /chat-templates/qwen3.6-miaai.jinja
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"auto_disable_thinking_with_tools":true}'
)

if [[ "${SERVICE_MODE}" == "1" ]]; then
  DOCKER_ARGS+=(--rm --name "qwen36-vllm-${PORT}")
  echo "Iniciando Qwen 3.6 35B-A3B NVFP4 en foreground (puerto ${PORT})..."
  exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${VLLM_ARGS[@]}"
fi

echo "Lanzando Qwen 3.6 35B-A3B NVFP4 (nvidia, 262K context) en puerto ${PORT}..."

docker run -d --name "qwen36-35b-a3b-${PORT}" \
  "${DOCKER_ARGS[@]}" \
  "${IMAGE}" \
  "${VLLM_ARGS[@]}"

echo "Esperando healthcheck..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  sleep 5
done
echo "Listo en http://localhost:${PORT}/v1 (modelo: qwen3.6-35b-a3b)"
