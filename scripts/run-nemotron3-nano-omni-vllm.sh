#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="~/vllm/nemotron3-nano-omni-30b-a3b-reasoning-nvfp4"
IMAGE="vllm/vllm-openai:gemma4-0505-cu130"
docker run -d --name "nemotron3-nano-omni-vllm-8000" \
  --gpus all --ipc host --network host --shm-size 64gb \
  -v "${MODEL_DIR}:/models/nemotron" -e HF_HOME=/models "${IMAGE}" \
    --model /models/nemotron --served-model-name nemotron3-nano-omni \
    --host 0.0.0.0 --port 8000 --quantization modelopt_fp4 \
    --moe-backend marlin --kv-cache-dtype fp8 --max-model-len 8192 \
    --max-num-seqs 1 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 --trust-remote-code
