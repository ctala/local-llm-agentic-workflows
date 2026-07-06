#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="~/vllm/qwen3.6-35b-a3b-nvfp4-mlponly-user"
IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"
docker run -d --name "qwen36-35b-a3b-trtllm-8000" \
  --gpus all --ipc host --network host --shm-size 64gb \
  -v "${MODEL_DIR}:/models/qwen3.6" -e HF_HOME=/models "${IMAGE}" \
    trtllm-serve /models/qwen3.6 --host 0.0.0.0 --port 8000 --backend pytorch \
    --max_seq_len 32768 --max_batch_size 2 --kv_cache_dtype fp8
