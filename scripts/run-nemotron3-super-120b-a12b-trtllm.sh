#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="~/vllm/nemotron3-super-120b-a12b-nvfp4"
IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"
docker run -d --name "nemotron3-super-120b-a12b-trtllm-8000" \
  --gpus all --ipc host --network host --shm-size 64gb \
  -v "${MODEL_DIR}:/models/nemotron" -e HF_HOME=/models "${IMAGE}" \
    trtllm-serve /models/nemotron --host 0.0.0.0 --port 8000 --backend pytorch \
    --max_seq_len 8192 --max_batch_size 1 --kv_cache_dtype fp8
