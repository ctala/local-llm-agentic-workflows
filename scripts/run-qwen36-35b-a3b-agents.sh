#!/usr/bin/env bash
# Optimized Qwen 3.6 35B-A3B for agentic workflows on DGX Spark / 128 GB unified memory.
# Prioritizes: long context, moderate parallelism, tool calling, image/video support.

set -euo pipefail

MODEL_DIR="${HOME}/vllm/qwen3.6-35b-a3b-nvfp4-redhat"
IMAGE="vllm/vllm-openai:gemma4-0505-cu130"
PORT="${PORT:-8000}"

echo "Launching Qwen 3.6 35B-A3B (agents config) on port ${PORT}..."

docker run -d --name "qwen36-35b-a3b-agents-${PORT}" \
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
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser pythonic \
    --trust-remote-code

echo "Waiting for healthcheck..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  sleep 5
done
echo "Ready at http://localhost:${PORT}/v1 (model: qwen3.6-35b-a3b)"
