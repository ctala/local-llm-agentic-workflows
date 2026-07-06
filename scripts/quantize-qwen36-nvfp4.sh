#!/usr/bin/env bash
set -euo pipefail

# 1. Convert the VLM checkpoint to text-only (required by Model Optimizer)
source ~/venvs/modelopt440/bin/activate
python3 "$(dirname "$0")/convert-qwen36-vlm-to-text.py"

# 2. Quantize to NVFP4 MLP-only (compatible with TRT-LLM 1.3.0rc13)
export PYTHONPATH=~/model-optimizer-0.44.0/examples/llm_ptq${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
python3 ~/model-optimizer-0.44.0/examples/llm_ptq/hf_ptq.py \
  --pyt_ckpt_path ~/vllm/qwen3.6-35b-a3b-bf16-textonly \
  --qformat nvfp4_mlp_only \
  --kv_cache_qformat fp8_cast \
  --dataset wikitext \
  --calib_size 128 \
  --batch_size 1 \
  --calib_seq 512 \
  --export_path ~/vllm/qwen3.6-35b-a3b-nvfp4-mlponly-user \
  --trust_remote_code \
  --use_seq_device_map \
  --gpu_max_mem_percentage 0.45 \
  --skip_generate \
  --verbose
