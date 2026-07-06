#!/usr/bin/env bash
# Lanza Gemma 4 31B IT NVFP4 (denso) en DGX Spark.
# Requiere: ~/vllm/gemma4-31b-it-nvfp4

set -euo pipefail

MODEL_DIR="${HOME}/vllm/gemma4-31b-it-nvfp4"
IMAGE="vllm/vllm-openai:gemma4-0505-cu130"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"

echo "Lanzando Gemma 4 31B IT en puerto ${PORT}..."

docker run -d --name "gemma4-31b-${PORT}" \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v "${MODEL_DIR}:/models/gemma31" \
  -e HF_HOME=/models \
  "${IMAGE}" \
    --model /models/gemma31 \
    --served-model-name gemma-4-31b-it \
    --host 0.0.0.0 --port "${PORT}" \
    --quantization modelopt \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser pythonic

echo "Esperando healthcheck..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  sleep 5
done
echo "Listo en http://localhost:${PORT}/v1 (modelo: gemma-4-31b-it)"
