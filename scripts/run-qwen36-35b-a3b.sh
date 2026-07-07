#!/usr/bin/env bash
# Lanza Qwen 3.6 35B-A3B NVFP4 (RedHatAI / compressed-tensors) en DGX Spark.
# Requiere: ~/vllm/qwen3.6-35b-a3b-nvfp4-redhat

set -euo pipefail

MODEL_DIR="${HOME}/vllm/qwen3.6-35b-a3b-nvfp4-redhat"
IMAGE="vllm/vllm-openai:gemma4-0505-cu130"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"

echo "Lanzando Qwen 3.6 35B-A3B en puerto ${PORT}..."

docker run -d --name "qwen36-35b-a3b-${PORT}" \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v "${MODEL_DIR}:/models/qwen3.6" \
  -e HF_HOME=/models \
  "${IMAGE}" \
    --model /models/qwen3.6 \
    --served-model-name qwen3.6-35b-a3b \
    --host 0.0.0.0 --port "${PORT}" \
    --quantization compressed-tensors \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3

echo "Esperando healthcheck..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  sleep 5
done
echo "Listo en http://localhost:${PORT}/v1 (modelo: qwen3.6-35b-a3b)"
