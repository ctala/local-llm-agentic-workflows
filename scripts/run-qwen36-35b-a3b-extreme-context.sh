: '#'
#!/usr/bin/env bash
# EXTREME context config for Qwen 3.6 35B-A3B on DGX Spark.
# Targets 260k context per sequence with 4 concurrent sequences.
# WARNING: this pushes the 128 GB unified-memory pool close to its limit.
# Start sessions with small prompts and let context grow; avoid 4x 260k-token prefills at once.

set -euo pipefail

MODEL_DIR="${HOME}/vllm/qwen3.6-35b-a3b-nvfp4-redhat"
IMAGE="vllm/vllm-openai:gemma4-0505-cu130"
PORT="${PORT:-8000}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"   # model's hard limit
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

echo "Launching Qwen 3.6 35B-A3B (extreme-context config) on port ${PORT}..."
echo "  max_model_len = ${MAX_MODEL_LEN} tokens per sequence"
echo "  max_num_seqs  = ${MAX_NUM_SEQS} parallel sequences"
echo "  Estimated KV cache at full capacity: ~40 GB"

docker run -d --name "qwen36-35b-a3b-extreme-${PORT}" \
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
    --max-num-batched-tokens 32768 \
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
