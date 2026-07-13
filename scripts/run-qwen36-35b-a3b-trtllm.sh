#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${HOME}/vllm/qwen3.6-35b-a3b-nvfp4-mlponly-user"
IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"
PORT="${PORT:-8000}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: Checkpoint no encontrado en ${MODEL_DIR}"
  echo "Generalo con scripts/quantize-qwen36-nvfp4.sh o descargalo."
  exit 1
fi

# Ruta al chat-template relativa al repo (asumiendo ejecución desde la raíz)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAT_TEMPLATE_DIR="${REPO_DIR}/chat-templates"

echo "Lanzando Qwen 3.6 35B-A3B MLP-only NVFP4 en TensorRT-LLM (puerto ${PORT})..."

docker run -d --name "qwen36-35b-a3b-trtllm-${PORT}" \
  --gpus all --ipc host --network host --shm-size 64gb \
  -v "${MODEL_DIR}:/models/qwen3.6" \
  -v "${CHAT_TEMPLATE_DIR}:/chat-templates" \
  -e HF_HOME=/models "${IMAGE}" \
    trtllm-serve /models/qwen3.6 --host 0.0.0.0 --port "${PORT}" --backend pytorch \
    --max_seq_len 262144 --max_batch_size 2 --kv_cache_dtype fp8 \
    --chat_template /chat-templates/qwen3.6-miaai.jinja

echo "Esperando healthcheck..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  sleep 5
done
echo "Listo en http://localhost:${PORT}/v1 (modelo: qwen3.6-35b-a3b)"
