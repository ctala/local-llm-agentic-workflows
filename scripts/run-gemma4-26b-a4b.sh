#!/usr/bin/env bash
# Lanza Gemma 4 26B-A4B IT NVFP4 (community parcheado) en DGX Spark.
# Requiere: ~/vllm/gemma4-26b-a4b-nvfp4-community descargado con gemma4_patched.py

set -euo pipefail

MODEL_DIR="${HOME}/vllm/gemma4-26b-a4b-nvfp4-community"
PATCH_FILE="${MODEL_DIR}/gemma4_patched.py"
IMAGE="vllm/vllm-openai:gemma4-cu130"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"

echo "Lanzando Gemma 4 26B-A4B en puerto ${PORT}..."

docker run -d --name "gemma4-26b-a4b-${PORT}" \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v "${MODEL_DIR}:/models/gemma4" \
  -v "${PATCH_FILE}:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py" \
  -e HF_HOME=/models \
  "${IMAGE}" \
    --model /models/gemma4 \
    --served-model-name gemma-4-26b-a4b \
    --host 0.0.0.0 --port "${PORT}" \
    --quantization modelopt \
    --moe-backend marlin \
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
echo "Listo en http://localhost:${PORT}/v1 (modelo: gemma-4-26b-a4b)"
