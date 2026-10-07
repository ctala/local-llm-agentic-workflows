---
title: 'vLLM and NIM Recipes for DGX Spark: Tested docker run Commands at 128K × 4 Users'
description: 23 copy-paste recipes (Qwen3.6, Qwen3.8, Nemotron 3, GPT-OSS, Gemma 4) with weight download, docker run command and measured speed on a DGX Spark.
keywords:
- DGX Spark vLLM recipe
- docker run vLLM GB10
- NVFP4 vLLM command
- speculative decoding recipe
- NIM DGX Spark
---


# vLLM and NIM Recipes for DGX Spark

Each recipe is a YAML file in [`recipes/`](https://github.com/ctala/local-llm-agentic-workflows/tree/main/recipes) with the exact flags we used in the [October 2026 round](../benchmarks/). The `docker run` command is produced by `scripts/recipe_to_docker.py`, which applies the measured context (128K), users (4), KV cache (FP8) and `--gpu-memory-utilization`.

Want an agent to do it for you? Use the [agent deployment prompt](../deploy-with-agent/).

```bash
git clone https://github.com/ctala/local-llm-agentic-workflows.git && cd local-llm-agentic-workflows
pip install pyyaml "huggingface_hub[cli]"
python3 scripts/recipe_to_docker.py recipes/<id>.yaml            # prints the docker run
python3 scripts/recipe_to_docker.py recipes/<id>.yaml --ctx 262144 --users 2   # another configuration
```

| Recipe | 1 user | 4 users (total) | Memory | Capabilities |
|---|---:|---:|---:|---|
| [`qwen36-35b-a3b-nvfp4-rapida-v031-mtp`](#qwen36-35b-a3b-nvfp4-rapida-v031-mtp) | 102.3 tok/s | 240.0 tok/s | 55.5 GB | text, image, video, tools, reasoning |
| [`gpt-oss-20b-v031`](#gpt-oss-20b-v031) | 96.2 tok/s | 150.6 tok/s | 25.7 GB | text, tools, reasoning |
| [`nemotron35-lightning-nvfp4-dspark-v031`](#nemotron35-lightning-nvfp4-dspark-v031) | 94.4 tok/s | 213.2 tok/s | 43.5 GB | text, tools, reasoning |
| [`qwen36-35b-a3b-nvfp4-rapida-v031`](#qwen36-35b-a3b-nvfp4-rapida-v031) | 78.1 tok/s | 196.6 tok/s | 47.0 GB | text, image, video, tools, reasoning |
| [`nemotron35-lightning-nvfp4-v031`](#nemotron35-lightning-nvfp4-v031) | 70.3 tok/s | 170.7 tok/s | 45.0 GB | text, tools, reasoning |
| [`qwen36-35b-a3b-nvfp4-rapida-v031-dflash`](#qwen36-35b-a3b-nvfp4-rapida-v031-dflash) | 67.8 tok/s | 148.6 tok/s | 71.8 GB | text, image, video, tools, reasoning |
| [`nim-gpt-oss-20b`](#nim-gpt-oss-20b) | 62.8 tok/s | 151.4 tok/s | 33.0 GB | text, tools, reasoning |
| [`gpt-oss-20b-eagle3-v031`](#gpt-oss-20b-eagle3-v031) | 59.4 tok/s | 144.7 tok/s | 38.4 GB | text, tools, reasoning |
| [`nemotron3-nano-omni-nvfp4-v031`](#nemotron3-nano-omni-nvfp4-v031) | 58.8 tok/s | 184.6 tok/s | 43.6 GB | text, image, audio, video, tools, reasoning |
| [`gemma4-26b-a4b-nvfp4-mtp-v031`](#gemma4-26b-a4b-nvfp4-mtp-v031) | 56.5 tok/s | 186.6 tok/s | 45.1 GB | text, image, audio, video |
| [`nim-nemotron3-nano-omni-30b`](#nim-nemotron3-nano-omni-30b) | 56.0 tok/s | 161.3 tok/s | 62.7 GB | text, image, audio, video, tools, reasoning |
| [`gpt-oss-120b-v031`](#gpt-oss-120b-v031) | 44.4 tok/s | 124.9 tok/s | 80.5 GB | text, tools, reasoning |
| [`gpt-oss-120b-eagle3-v031`](#gpt-oss-120b-eagle3-v031) | 40.5 tok/s | 91.8 tok/s | 87.5 GB | text, tools, reasoning |
| [`gemma4-26b-a4b-nvfp4-v031`](#gemma4-26b-a4b-nvfp4-v031) | 30.8 tok/s | 114.0 tok/s | 39.9 GB | text, image, audio, video |
| [`gemma4-26b-a4b-nvfp4`](#gemma4-26b-a4b-nvfp4) | 30.7 tok/s | 108.5 tok/s | 39.4 GB | text, image, audio, video |
| [`qwen38-flash-next`](#qwen38-flash-next) | 26.8 tok/s | 52.5 tok/s | 95.7 GB | text, image, video, tools, reasoning |
| [`nim-llama31-8b`](#nim-llama31-8b) | 26.4 tok/s | 102.3 tok/s | 33.6 GB | text, tools |
| [`qwen3-8b-fp8-v031`](#qwen3-8b-fp8-v031) | 23.2 tok/s | 95.7 tok/s | 55.5 GB | text, tools, reasoning |
| [`nemotron3-super-120b-nvfp4-mtp-v031`](#nemotron3-super-120b-nvfp4-mtp-v031) | 23.0 tok/s | 57.0 tok/s | 95.2 GB | text, tools, reasoning |
| [`qwen38-27b-nvfp4-mtp-v031`](#qwen38-27b-nvfp4-mtp-v031) | 19.0 tok/s | 67.0 tok/s | 59.5 GB | text, image, tools, reasoning |
| [`nemotron3-super-120b-nvfp4-v031`](#nemotron3-super-120b-nvfp4-v031) | 16.1 tok/s | 42.8 tok/s | 94.3 GB | text, tools, reasoning |
| [`qwen38-27b-nvfp4-dspark-v031`](#qwen38-27b-nvfp4-dspark-v031) | 12.5 tok/s | 33.9 tok/s | 63.6 GB | text, image, tools, reasoning |
| [`qwen38-27b-nvfp4-base-v031`](#qwen38-27b-nvfp4-base-v031) | 11.0 tok/s | 39.6 tok/s | 47.0 GB | text, image, tools, reasoning |

## qwen36-35b-a3b-nvfp4-rapida-v031-mtp

**Qwen3.6-35B-A3B NVFP4 (NVIDIA) + MTP k=2 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 102.3 tok/s for 1 user, 240.0 tok/s total with 4, loads in 2.7 min, ≈55.5 GB of unified memory.

Download:

```bash
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir ~/vllm/qwen3.6-35b-a3b-nvfp4-nvidia
```

Run:

```bash
docker run -d \
  --name qwen36-35b-a3b-nvfp4-rapida-v031-mtp \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/qwen3.6-35b-a3b-nvfp4-nvidia \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b-a3b \
  --trust-remote-code \
  --attention-backend=flashinfer \
  --moe-backend=marlin \
  --max-num-batched-tokens=32768 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  '--limit-mm-per-prompt={"image":4}' \
  --load-format=fastsafetensors \
  --enable-auto-tool-choice \
  --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 \
  '--speculative-config={"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.5
```

## gpt-oss-20b-v031

**GPT-OSS-20B (MXFP4) · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 96.2 tok/s for 1 user, 150.6 tok/s total with 4, loads in 2.9 min, ≈25.7 GB of unified memory.

Download:

```bash
hf download openai/gpt-oss-20b --local-dir ~/vllm/gpt-oss-20b-v031
```

Run:

```bash
docker run -d \
  --name gpt-oss-20b-v031 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/gpt-oss-20b-v031 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name gpt-oss-20b \
  --enable-prefix-caching \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.25
```

## nemotron35-lightning-nvfp4-dspark-v031

**Nemotron 3.5 Lightning 30B-A3B NVFP4 + DSpark k=3 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 94.4 tok/s for 1 user, 213.2 tok/s total with 4, loads in 4.5 min, ≈43.5 GB of unified memory.

Download:

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --local-dir ~/vllm/nemotron-3.5-lightning-nvfp4
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark --local-dir ~/vllm/nemotron-3.5-lightning-dspark
```

Run:

```bash
docker run -d \
  --name nemotron35-lightning-nvfp4-dspark-v031 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/nemotron-3.5-lightning-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name nemotron-3.5-lightning \
  --enable-prefix-caching \
  --trust-remote-code \
  '--speculative-config={"method":"dspark","model":"/vllm/nemotron-3.5-lightning-dspark","num_speculative_tokens":3}' \
  --chat-template=/vllm/nemotron-3.5-lightning-nvfp4/chat_template.jinja \
  --reasoning-parser=nemotron_v3 \
  --tool-call-parser=qwen3_coder \
  --enable-auto-tool-choice \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.4
```

## qwen36-35b-a3b-nvfp4-rapida-v031

**Qwen3.6-35B-A3B NVFP4 (NVIDIA) · vLLM 0.31.0 estable + flags rápidos** · status: verified

Measured 2026-10-06: 78.1 tok/s for 1 user, 196.6 tok/s total with 4, loads in 2.1 min, ≈47.0 GB of unified memory.

Download:

```bash
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir ~/vllm/qwen3.6-35b-a3b-nvfp4-nvidia
```

Run:

```bash
docker run -d \
  --name qwen36-35b-a3b-nvfp4-rapida-v031 \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/qwen3.6-35b-a3b-nvfp4-nvidia \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b-a3b \
  --trust-remote-code \
  --attention-backend=flashinfer \
  --moe-backend=marlin \
  --max-num-batched-tokens=32768 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  '--limit-mm-per-prompt={"image":4}' \
  --load-format=fastsafetensors \
  --enable-auto-tool-choice \
  --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.4
```

## nemotron35-lightning-nvfp4-v031

**Nemotron 3.5 Lightning 30B-A3B NVFP4 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 70.3 tok/s for 1 user, 170.7 tok/s total with 4, loads in 3.9 min, ≈45.0 GB of unified memory.

Download:

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --local-dir ~/vllm/nemotron-3.5-lightning-nvfp4
```

Run:

```bash
docker run -d \
  --name nemotron35-lightning-nvfp4-v031 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/nemotron-3.5-lightning-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name nemotron-3.5-lightning \
  --enable-prefix-caching \
  --trust-remote-code \
  --chat-template=/vllm/nemotron-3.5-lightning-nvfp4/chat_template.jinja \
  --reasoning-parser=nemotron_v3 \
  --tool-call-parser=qwen3_coder \
  --enable-auto-tool-choice \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.4
```

## qwen36-35b-a3b-nvfp4-rapida-v031-dflash

**Qwen3.6-35B-A3B NVFP4 (NVIDIA) + DFlash k=11 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 67.8 tok/s for 1 user, 148.6 tok/s total with 4, loads in 3.4 min, ≈71.8 GB of unified memory.

Download:

```bash
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir ~/vllm/qwen3.6-35b-a3b-nvfp4-nvidia
hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir ~/vllm/qwen36-dflash
```

Run:

```bash
docker run -d \
  --name qwen36-35b-a3b-nvfp4-rapida-v031-dflash \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/qwen3.6-35b-a3b-nvfp4-nvidia \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b-a3b \
  --trust-remote-code \
  --attention-backend=flashinfer \
  --moe-backend=marlin \
  --max-num-batched-tokens=32768 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  '--limit-mm-per-prompt={"image":4}' \
  --load-format=fastsafetensors \
  --enable-auto-tool-choice \
  --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 \
  '--speculative-config={"method":"dflash","model":"/vllm/qwen36-dflash","num_speculative_tokens":11}' \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.6
```

## nim-gpt-oss-20b

**GPT-OSS-20B · NIM oficial** · status: verified

Measured 2026-10-06: 62.8 tok/s for 1 user, 151.4 tok/s total with 4, loads in 4.8 min, ≈33.0 GB of unified memory.

Download:

```bash
docker pull nvcr.io/nim/openai/gpt-oss-20b:2.0.9  # NIM: downloads weights on first start (needs NGC_API_KEY)
```

Run:

```bash
docker run -d \
  --name nim-gpt-oss-20b \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e NIM_SERVED_MODEL_NAME=gpt-oss-20b \
  -e NGC_API_KEY \
  -v "$HOME/.cache/nim:/opt/nim/.cache" \
  nvcr.io/nim/openai/gpt-oss-20b:2.0.9 \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.3
```

## gpt-oss-20b-eagle3-v031

**GPT-OSS-20B (MXFP4) + Eagle3 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 59.4 tok/s for 1 user, 144.7 tok/s total with 4, loads in 2.7 min, ≈38.4 GB of unified memory.

Download:

```bash
hf download RedHatAI/gpt-oss-20b-speculator.eagle3 --local-dir ~/vllm/gpt-oss-20b-eagle3
```

Run:

```bash
docker run -d \
  --name gpt-oss-20b-eagle3-v031 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/gpt-oss-20b-eagle3-v031 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name gpt-oss-20b \
  --attention-backend=TRITON_ATTN \
  --enable-prefix-caching \
  '--speculative-config={"method":"eagle3","model":"/vllm/gpt-oss-20b-eagle3","num_speculative_tokens":3}' \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.35
```

## nemotron3-nano-omni-nvfp4-v031

**Nemotron 3 Nano Omni 30B-A3B NVFP4 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 58.8 tok/s for 1 user, 184.6 tok/s total with 4, loads in 4.1 min, ≈43.6 GB of unified memory.

Download:

```bash
hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 --local-dir ~/vllm/nemotron3-nano-omni-30b-a3b-reasoning-nvfp4
```

Run:

```bash
docker run -d \
  --name nemotron3-nano-omni-nvfp4-v031 \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/nemotron3-nano-omni-30b-a3b-reasoning-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name nemotron-3-nano-omni \
  --chat-template=/vllm/nemotron3-nano-omni-30b-a3b-reasoning-nvfp4/chat_template.jinja \
  --tool-call-parser=qwen3_coder \
  --enable-auto-tool-choice \
  --reasoning-parser=nemotron_v3 \
  --quantization=modelopt_fp4 \
  --moe-backend=marlin \
  --max-num-batched-tokens=8192 \
  --enable-prefix-caching \
  --mamba-cache-mode=align \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.4
```

## gemma4-26b-a4b-nvfp4-mtp-v031

**Gemma 4 26B-A4B NVFP4 + drafter MTP oficial · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 56.5 tok/s for 1 user, 186.6 tok/s total with 4, loads in 4.8 min, ≈45.1 GB of unified memory.

Download:

```bash
hf download nvidia/Gemma-4-26B-A4B-NVFP4 --local-dir ~/vllm/gemma4-26b-a4b-nvfp4
hf download google/gemma-4-26B-A4B-it-assistant --local-dir ~/vllm/gemma4-26b-a4b-assistant
```

Run:

```bash
docker run -d \
  --name gemma4-26b-a4b-nvfp4-mtp-v031 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/gemma4-26b-a4b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name gemma-4-26b-a4b \
  --enable-prefix-caching \
  --max-num-batched-tokens=8192 \
  '--speculative-config={"method":"mtp","model":"/vllm/gemma4-26b-a4b-assistant","num_speculative_tokens":3}' \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.4
```

## nim-nemotron3-nano-omni-30b

**Nemotron 3 Nano Omni 30B-A3B · NIM oficial** · status: verified

Measured 2026-10-06: 56.0 tok/s for 1 user, 161.3 tok/s total with 4, loads in 6.7 min, ≈62.7 GB of unified memory.

Download:

```bash
docker pull nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:2.0.13  # NIM: downloads weights on first start (needs NGC_API_KEY)
```

Run:

```bash
docker run -d \
  --name nim-nemotron3-nano-omni-30b \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e NIM_SERVED_MODEL_NAME=nemotron-3-nano-omni \
  -e NGC_API_KEY \
  -v "$HOME/.cache/nim:/opt/nim/.cache" \
  nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:2.0.13 \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.55
```

## gpt-oss-120b-v031

**GPT-OSS-120B (MXFP4) · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 44.4 tok/s for 1 user, 124.9 tok/s total with 4, loads in 9.3 min, ≈80.5 GB of unified memory.

Download:

```bash
hf download openai/gpt-oss-120b --local-dir ~/vllm/gpt-oss-120b-v031
```

Run:

```bash
docker run -d \
  --name gpt-oss-120b-v031 \
  --gpus all \
  --ipc host \
  --shm-size 32g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/gpt-oss-120b-v031 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name gpt-oss-120b \
  --enable-prefix-caching \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.7
```

## gpt-oss-120b-eagle3-v031

**GPT-OSS-120B (MXFP4) + Eagle3 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 40.5 tok/s for 1 user, 91.8 tok/s total with 4, loads in 9.2 min, ≈87.5 GB of unified memory.

Download:

```bash
hf download nvidia/gpt-oss-120b-Eagle3-long-context --local-dir ~/vllm/gpt-oss-120b-eagle3-long
```

Run:

```bash
docker run -d \
  --name gpt-oss-120b-eagle3-v031 \
  --gpus all \
  --ipc host \
  --shm-size 32g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/gpt-oss-120b-eagle3-v031 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name gpt-oss-120b \
  --attention-backend=TRITON_ATTN \
  --enable-prefix-caching \
  '--speculative-config={"method":"eagle3","model":"/vllm/gpt-oss-120b-eagle3-long","num_speculative_tokens":3}' \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.75
```

## gemma4-26b-a4b-nvfp4-v031

**Gemma 4 26B-A4B NVFP4 (NVIDIA) · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 30.8 tok/s for 1 user, 114.0 tok/s total with 4, loads in 4.3 min, ≈39.9 GB of unified memory.

Download:

```bash
hf download nvidia/Gemma-4-26B-A4B-NVFP4 --local-dir ~/vllm/gemma4-26b-a4b-nvfp4
```

Run:

```bash
docker run -d \
  --name gemma4-26b-a4b-nvfp4-v031 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/gemma4-26b-a4b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name gemma-4-26b-a4b \
  --enable-prefix-caching \
  --max-num-batched-tokens=8192 \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.35
```

## gemma4-26b-a4b-nvfp4

**Gemma 4 26B-A4B NVFP4 (NVIDIA) · vLLM** · status: verified

Measured 2026-10-06: 30.7 tok/s for 1 user, 108.5 tok/s total with 4, loads in 3.8 min, ≈39.4 GB of unified memory.

Download:

```bash
hf download nvidia/Gemma-4-26B-A4B-NVFP4 --local-dir ~/vllm/gemma4-26b-a4b-nvfp4
```

Run:

```bash
docker run -d \
  --name gemma4-26b-a4b-nvfp4 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  vllm/vllm-openai:gemma4-0505-cu130 \
  /vllm/gemma4-26b-a4b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name gemma-4-26b-a4b \
  --enable-prefix-caching \
  --max-num-batched-tokens=8192 \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.35
```

## qwen38-flash-next

**Qwen3.8-Flash-Next NVFP4 (RadixArk) en vLLM con MTP — contexto largo 256K** · status: verified

Measured 2026-10-06: 26.8 tok/s for 1 user, 52.5 tok/s total with 4, loads in 13.8 min, ≈95.7 GB of unified memory.

Download:

```bash
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4  # served from the Hugging Face cache
```

Run:

```bash
docker run -d \
  --name qwen38-flash-next \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=0 \
  -e VLLM_PLE_MMAP=1 \
  -e VLLM_PLE_MMAP_WORKERS=32 \
  -e VLLM_PLE_MMAP_PREWARM=0 \
  -e VLLM_QSA_EXACT_TOPK=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  qwen38-flash-dgx:latest \
  /root/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.8-flash-next \
  --load-format=safetensors \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens=8192 \
  -cc.cudagraph_mode=PIECEWISE \
  '-cc.splitting_ops=["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]' \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 \
  '--speculative-config={"method":"mtp","num_speculative_tokens":2}' \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.8
```

## nim-llama31-8b

**Llama 3.1 8B Instruct · NIM oficial** · status: verified

Measured 2026-10-06: 26.4 tok/s for 1 user, 102.3 tok/s total with 4, loads in 5.1 min, ≈33.6 GB of unified memory.

Download:

```bash
docker pull nvcr.io/nim/meta/llama-3.1-8b-instruct:2.0.9  # NIM: downloads weights on first start (needs NGC_API_KEY)
```

Run:

```bash
docker run -d \
  --name nim-llama31-8b \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e NIM_SERVED_MODEL_NAME=llama-3.1-8b \
  -e NGC_API_KEY \
  -v "$HOME/.cache/nim:/opt/nim/.cache" \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:2.0.9 \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.3
```

## qwen3-8b-fp8-v031

**Qwen3-8B FP8 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 23.2 tok/s for 1 user, 95.7 tok/s total with 4, loads in 2.8 min, ≈55.5 GB of unified memory.

Download:

```bash
hf download Qwen/Qwen3-8B-FP8 --local-dir ~/vllm/qwen3-8b-fp8-v031
```

Run:

```bash
docker run -d \
  --name qwen3-8b-fp8-v031 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/qwen3-8b-fp8-v031 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-8b \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser=hermes \
  --reasoning-parser=qwen3 \
  '--hf-overrides={"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000},"max_position_embeddings":131072}' \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.5
```

## nemotron3-super-120b-nvfp4-mtp-v031

**Nemotron 3 Super 120B-A12B NVFP4 + MTP k=3 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 23.0 tok/s for 1 user, 57.0 tok/s total with 4, loads in 13.4 min, ≈95.2 GB of unified memory.

Download:

```bash
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --local-dir ~/vllm/nemotron3-super-120b-a12b-nvfp4
```

Run:

```bash
docker run -d \
  --name nemotron3-super-120b-nvfp4-mtp-v031 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/nemotron3-super-120b-a12b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name nemotron-3-super \
  --enable-prefix-caching \
  --trust-remote-code \
  '--speculative-config={"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.8
```

## qwen38-27b-nvfp4-mtp-v031

**Qwen3.8-27B NVFP4 + MTP k=2 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 19.0 tok/s for 1 user, 67.0 tok/s total with 4, loads in 7.7 min, ≈59.5 GB of unified memory.

Download:

```bash
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/vllm/qwen3.8-27b-nvfp4
```

Run:

```bash
docker run -d \
  --name qwen38-27b-nvfp4-mtp-v031 \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -e CUTE_DSL_ARCH=sm_121a \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/qwen3.8-27b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.8-27b \
  --max-num-batched-tokens=16384 \
  --enable-prefix-caching \
  --reasoning-parser=qwen3 \
  --tool-call-parser=qwen3_xml \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt.image=2 \
  --limit-mm-per-prompt.video=0 \
  '--speculative-config={"method":"mtp","num_speculative_tokens":2}' \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.5
```

## nemotron3-super-120b-nvfp4-v031

**Nemotron 3 Super 120B-A12B NVFP4 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 16.1 tok/s for 1 user, 42.8 tok/s total with 4, loads in 10.3 min, ≈94.3 GB of unified memory.

Download:

```bash
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --local-dir ~/vllm/nemotron3-super-120b-a12b-nvfp4
```

Run:

```bash
docker run -d \
  --name nemotron3-super-120b-nvfp4-v031 \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/nemotron3-super-120b-a12b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name nemotron-3-super \
  --enable-prefix-caching \
  --trust-remote-code \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.8
```

## qwen38-27b-nvfp4-dspark-v031

**Qwen3.8-27B NVFP4 + DSpark k=14 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 12.5 tok/s for 1 user, 33.9 tok/s total with 4, loads in 8.8 min, ≈63.6 GB of unified memory.

Download:

```bash
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/vllm/qwen3.8-27b-nvfp4
hf download Doopeworld/Qwen3.8-27B-DSpark-vLLM --local-dir ~/vllm/qwen3.8-dspark
```

Run:

```bash
docker run -d \
  --name qwen38-27b-nvfp4-dspark-v031 \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -e CUTE_DSL_ARCH=sm_121a \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/qwen3.8-27b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.8-27b \
  --max-num-batched-tokens=16384 \
  --enable-prefix-caching \
  '--speculative-config={"method":"dspark","model":"/vllm/qwen3.8-dspark","num_speculative_tokens":14,"draft_sample_method":"probabilistic"}' \
  --reasoning-parser=qwen3 \
  --tool-call-parser=qwen3_xml \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt.image=2 \
  --limit-mm-per-prompt.video=0 \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.55
```

## qwen38-27b-nvfp4-base-v031

**Qwen3.8-27B NVFP4 · vLLM 0.31.0** · status: verified

Measured 2026-10-06: 11.0 tok/s for 1 user, 39.6 tok/s total with 4, loads in 6.6 min, ≈47.0 GB of unified memory.

Download:

```bash
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/vllm/qwen3.8-27b-nvfp4
```

Run:

```bash
docker run -d \
  --name qwen38-27b-nvfp4-base-v031 \
  --gpus all \
  --ipc host \
  --shm-size 64g \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm:/vllm:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=cuda \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -e CUTE_DSL_ARCH=sm_121a \
  vllm/vllm-openai:v0.31.0-aarch64 \
  /vllm/qwen3.8-27b-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.8-27b \
  --max-num-batched-tokens=16384 \
  --enable-prefix-caching \
  --reasoning-parser=qwen3 \
  --tool-call-parser=qwen3_xml \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt.image=2 \
  --limit-mm-per-prompt.video=0 \
  --max-model-len=131072 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8 \
  --gpu-memory-utilization=0.4
```
