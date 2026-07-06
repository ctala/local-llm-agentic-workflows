#!/usr/bin/env bash
# Qwen 3.6 35B-A3B optimized for maximum context per session on DGX Spark.
# Uses ~90 GB of unified memory headroom for KV cache.
# Recommended for long-context agents (Hermes/OpenClaw with large workspaces).

set -euo pipefail

MODEL_DIR="${HOME}/vllm/qwen3.6-35b-a3b-nvfp4-redhat"
IMAGE="vllm/vllm-openai:gemma4-0505-cu130"
PORT="${PORT:-8000}"

# Tune these two variables to trade context length vs parallel sessions.
# The Spark can fit ~2.2M tokens of FP8 KV cache after loading Qwen.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"   # 128k context per session
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"          # 2 parallel sessions

echo "Launching Qwen 3.6 35B-A3B (long-context config) on port ${PORT}..."
echo "  max_model_len = ${MAX_MODEL_LEN} tokens per sequence"
echo "  max_num_seqs  = ${MAX_NUM_SEQS} parallel sequences"

docker run -d --name "qwen36-35b-a3b-longctx-${PORT}" \
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
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens 16384 \
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
