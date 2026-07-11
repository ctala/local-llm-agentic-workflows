# Local LLM Benchmark Results on DGX Spark & 96–128 GB Edge AI Workstations

> For the quick start guide see [Home](/local-llm-agentic-workflows/). For the original Spanish log see [Results in Spanish](/local-llm-agentic-workflows/results.es/).

This document contains the full benchmark results and technical notes for running **Gemma 4**, **Qwen 3.6** and **NVIDIA Nemotron 3** locally on the NVIDIA DGX Spark and equivalent high-memory edge AI hardware.

---

## Hardware / software base

| Component | Value |
|-----------|-------|
| Hardware | NVIDIA DGX Spark (GB10 Grace Blackwell) |
| CPU | 20 cores ARM64 (aarch64) |
| GPU | NVIDIA GB10 (sm_121) |
| Memory | 128 GB LPDDR5x unified (~121 GB usable) |
| NVIDIA driver | 580.142 |
| CUDA | 13.0 |
| Docker | 29.2.1 |
| NVIDIA Container Toolkit | 1.19.0 |

**Critical GB10 limitation**: no native FP4 compute. NVFP4 checkpoints run through the **Marlin** backend (`--moe-backend marlin`) in vLLM or the PyTorch backend in TensorRT-LLM, decompressing FP4 → BF16 at runtime. This limits throughput compared to FP4-native GPUs such as the B200.

---

## Benchmark methodology

Text benchmarks used a ~120-token Spanish prompt, `max_tokens=512`, temperature 0.7 and streaming enabled. We report **decode tok/s** (excluding cold-start TTFT) and hot TTFT. Memory numbers reflect peak unified memory usage observed during the run.

## Full results table

| Model | Checkpoint | Framework / container | Decode tok/s | Hot TTFT | Memory | Notes |
|-------|------------|----------------------|--------------|----------|--------|-------|
| **Qwen 3.6 35B-A3B** | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:nightly` | **~75–77** | ~0.10 s | ~22 GB | **Current recommended.** W4A16 NVFP4 (`modelopt`), `qwen3_coder` parser, 262K context. |
| **Gemma 4 26B-A4B IT** | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` + `gemma4_patched.py` | `vllm/vllm-openai:gemma4-cu130` | **~49.5** | ~0.08 s | ~22 GB | Best raw speed for agents. Requires community patch. |
| Qwen 3.6 35B-A3B | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~42.2 | ~0.10 s | ~22 GB | `compressed-tensors` format. Stable previous checkpoint. |
| **Nemotron-3-Nano-Omni-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~40.0** | ~0.10 s | **~40 GB** | **Official multimodal**: text + image work. Audio decoding still unresolved in this container. |
| Qwen 3.6 35B-A3B (n-gram speculative) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~34–37 | ~0.10 s | ~22 GB | Worse for non-repetitive text. |
| **Qwen 3.6 35B-A3B TRT-LLM (MLP-only NVFP4)** | Quantized with Model Optimizer 0.44.0 from `Qwen/Qwen3.6-35B-A3B` BF16 | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~34.4** | ~0.09 s | ~41 GB | `modelopt` NVFP4 MLP-only + FP8 KV. Works with TRT-LLM PyTorch backend. |
| Gemma 4 26B-A4B IT (official) | `nvidia/Gemma-4-26B-A4B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~30.1 | ~0.20 s | ~21 GB | Works without patch, but ~20 tok/s slower. |
| **Nemotron-3-Nano-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~28.8** | ~0.22 s | **~118 GB** | Dense BF16. Uses almost all unified memory. |
| Nemotron-3-Nano-30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `vllm/vllm-openai:gemma4-0505-cu130` | ~28.3 | ~0.20 s | ~72 GB | vLLM alternative. Stop other GPU services first. |
| **Nemotron-3-Super-120B-A12B** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~14.7** | ~0.29 s | **~110 GB** | Official NVFP4 checkpoint. Higher quality, lower speed due to more active experts. |
| **Gemma 4 31B IT** | `nvidia/Gemma-4-31B-IT-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~6.7** | ~1.8 s | ~31 GB | Dense, memory-bandwidth limited. Not recommended for fast interaction. |
| Qwen 3.6 35B-A3B (MTP) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | Load error | – | – | `moe_backend='marlin'` not supported by the non-quantized drafter. |
| Gemma 4 26B-A4B on TRT-LLM 1.3.0rc13 | `nvidia/Gemma-4-26B-A4B-NVFP4` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | Load error | – | – | Transformers in container does not recognize `model_type: gemma4`. |
| Qwen 3.6 35B-A3B on TRT-LLM 1.3.0rc13 | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | Load error | – | – | `AssertionError` in `quant_algo` for modelopt NVFP4 checkpoint. |
| Nemotron-3-Super-120B-A12B | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | — | — | — | **Not viable**: `CUDA OOM` on engine init; first attempt hung the Spark due to memory exhaustion. |

### Key takeaways

- **For best quality/speed balance (~75–77 tok/s)**: use **Qwen 3.6 35B-A3B nvidia NVFP4 + vLLM nightly**. Also supports the model's full 262K context window and robust tool calling.
- **For maximum speed (~50 tok/s)**: use **Gemma 4 26B-A4B community + patch**.
- **Qwen 3.6 35B-A3B RedHatAI** (~42 tok/s) remains a stable fallback if the nvidia checkpoint or nightly image are unavailable.
- **Gemma 4 31B dense** is not viable for fast interactive use on GB10 (~7 tok/s).
- **TensorRT-LLM** works for Qwen 3.6 if you manually quantize the base BF16 model to **NVFP4 MLP-only** (~34 tok/s), but it is slower than vLLM with the nvidia checkpoint.
- **Speculative decoding** did not help on general prompts; it may only help for very repetitive text or with a compatible MTP drafter.
- **Nemotron-3-Nano-Omni** runs at ~40 tok/s on vLLM and **processes images**. Audio failed due to container decoding issues, not the model itself.
- **Nemotron-3-Super-120B-A12B** is only stable with **TensorRT-LLM**. vLLM fails with `CUDA OOM` or hangs the system. See detailed analysis below.

---

## Why Nemotron-3 Super 120B-A12B fails on vLLM

We attempted twice to serve `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` with `vllm/vllm-openai:gemma4-0505-cu130`.

### First attempt
The container loaded all 17 checkpoint shards (~75 GB on disk) in ~8 minutes. The last logs showed:

```
Loading weights took 489.67 seconds
WARNING ... Your GPU does not have native support for FP4 computation ...
Using MoEPrepareAndFinalizeNoDPEPModular
Using uncalibrated q_scale 1.0 ...
```

Before the HTTP server came up, the Spark hung and had to be rebooted. Root cause: two `llama-server` processes running Qwen GGUF models were consuming ~76 GB of unified memory, leaving insufficient room for vLLM.

### Second attempt (after freeing memory)
After stopping `qwen27-local.service` and `qwen35-local.service`, the system had ~117 GB free. However, vLLM failed immediately while initializing the `EngineCore`:

```
torch.AcceleratorError: CUDA error: out of memory
  File ".../vllm/utils/mem_utils.py", line 108, in measure
    self.free_memory, self.total_memory = current_platform.mem_get_info(device)
```

vLLM could not even query available memory without hitting an OOM.

### Diagnosis

The model uses a hybrid Mamba-MoE architecture (`NemotronHForCausalLM`) with:

- 88 layers
- 512 routed experts
- 22 active experts per token
- 1 shared expert
- Hidden size 4096, head dim 128, GQA with 2 KV heads

Although the checkpoint is ~75 GB on disk, decompressing FP4 → BF16 via Marlin, allocating KV cache under `--gpu-memory-utilization 0.90`, and compiling kernels with the vLLM V1 engine (`torch.compile`/`inductor`) exceeds the 128 GB unified-memory budget.

### Why TensorRT-LLM works

The same checkpoint serves stably at **~14.7 tok/s** and ~110 GB with:

```bash
trtllm-serve /models/nemotron --backend pytorch --max_seq_len 8192 --max_batch_size 1 --kv_cache_dtype fp8
```

TRT-LLM reserves a fixed memory budget (model + KV cache for the given seq_len/batch + activations) instead of vLLM's percentage-based dynamic PagedAttention allocation.

**Conclusion**: on DGX Spark and equivalent 128 GB unified-memory hardware, **Nemotron-3 Super 120B-A12B should only be used with TensorRT-LLM**.

---

## Extreme context scaling: Qwen 3.6 35B-A3B

For agentic workflows that ingest very long contexts (codebases, conversation history, RAG dumps, multi-turn agent traces) we tested how far `nvidia/Qwen3.6-35B-A3B-NVFP4` can scale on the DGX Spark. We used vLLM nightly with:

```bash
--model /models/qwen3.6 \
--trust-remote-code \
--tensor-parallel-size 1 \
--attention-backend flashinfer \
--moe-backend marlin \
--kv-cache-dtype fp8 \
--gpu-memory-utilization 0.92 \
--max-model-len 262144 \
--max-num-seqs 2 \
--max-num-batched-tokens 32768 \
--enable-chunked-prefill \
--async-scheduling \
--enable-prefix-caching \
--load-format fastsafetensors \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--reasoning-parser qwen3
```

The **2-session configuration is the recommended default** because it leaves headroom for LiteLLM, ASR and other auxiliary services while still supporting two 262K sessions in parallel. We previously tested a 3-session config with the older RedHatAI checkpoint, but it left only ~1–2 GB of free unified memory and made the system vulnerable to memory spikes; it has not been re-validated with the newer nvidia checkpoint + vLLM nightly.

### Single-sequence context scaling

| Input tokens | Output tokens | TTFT | Decode tok/s | Notes |
|--------------|---------------|------|--------------|-------|
| 1,000 | 32 | 0.28 s | 45.57 | Warm baseline. |
| 50,000 | 64 | 27.1 s | 45.66 | First large-context call; includes some JIT warmup. |
| 100,000 | 64 | 22.87 s | 39.80 | Faster TTFT than 50K because kernels are warm. |
| 200,000 | 64 | 65.45 s | 33.17 | Stable, memory ~120 GB. |
| 262,000 | 64 | 56.49 s | 30.22 | Near the model's hard limit (262,144 tokens). |

### Concurrent-session context scaling (3 sessions)

| Input tokens/session | Wall time | Session TTFTs | Session decode tok/s | Notes |
|----------------------|-----------|---------------|----------------------|-------|
| 50,000 | 4.28 s | 1.28–2.09 s | 20.72–28.33 | Excellent interactivity. |
| 100,000 | 4.59 s | 1.88–2.81 s | 21.54–32.69 | Still very responsive. |
| 200,000 | 4.87 s | 1.26–3.06 s | 14.39–28.23 | Chunked prefill keeps wall time low. |
| 262,000 | 92.32 s | 41.93–89.87 s | 1.13–23.44 | Works, but TTFT becomes noticeable. |

### Memory behavior

- **At rest after loading**: ~119 GB used / ~121 GB total, ~2 GB available.
- **During 3×262K prefill**: ~120 GB used, ~1.5 GB available, ~6 GB swap in use.
- **Stable**: no OOM, no hang, no reboot required during these tests.

### Practical guidance for agents

- **Average agent turn**: OpenClaw / Hermes-style agents typically use **8K–32K tokens** of active context per session.
- **Conservative production setting**: **2 parallel sessions × 64K context** runs with sub-second TTFT and leaves comfortable headroom for LiteLLM/ASR.
- **Maximum context per session**: **~262K tokens** is achievable with 2 concurrent sessions by default; 3 concurrent sessions are possible but leave little memory headroom.
- **Do not use 4 sessions at 262K** unless the machine is dedicated to a single model and you can tolerate hangs from memory spikes.

---

## Launch scripts

Ready-to-run recipes are in [`scripts/`](scripts/):

| Script | Model / framework |
|--------|-------------------|
| `scripts/run-gemma4-26b-a4b.sh` | Gemma 4 26B-A4B IT NVFP4 community patch on vLLM |
| `scripts/run-qwen36-35b-a3b.sh` | **Qwen 3.6 35B-A3B nvidia NVFP4 on vLLM nightly (recommended, 262K context)** |
| `scripts/run-qwen36-35b-a3b-extreme-context-2seq.sh` | Alias to `run-qwen36-35b-a3b.sh` (262K × 2 sessions) |
| `scripts/run-qwen36-35b-a3b-trtllm.sh` | Qwen 3.6 35B-A3B custom MLP-only NVFP4 on TRT-LLM |
| `scripts/run-gemma4-31b.sh` | Gemma 4 31B IT NVFP4 on vLLM |
| `scripts/run-nemotron3-nano-30b-a3b-trtllm.sh` | Nemotron-3 Nano BF16 on TRT-LLM |
| `scripts/run-nemotron3-nano-30b-a3b-vllm.sh` | Nemotron-3 Nano BF16 on vLLM |
| `scripts/run-nemotron3-super-120b-a12b-trtllm.sh` | Nemotron-3 Super NVFP4 on TRT-LLM |
| `scripts/run-nemotron3-nano-omni-vllm.sh` | Nemotron-3 Nano Omni NVFP4 multimodal on vLLM |

Run a benchmark after the container reports healthy:

```bash
python3 benchmarks/bench_model.py gemma-4-26b-a4b 512
```

---

## Important technical notes

1. **ARM64**: always verify the Docker image has an `arm64` manifest.
2. **Marlin is required for MoE NVFP4 on GB10**: use `--moe-backend marlin`. Native FP4 backends (CUTLASS/FlashInfer) may fail or produce NaN on sm_121.
3. **FP8 KV cache** saves memory but relies on checkpoint scaling factors; vLLM warns about accuracy if they are missing.
4. **Prefix caching** speeds up requests with shared context or repeated long prompts.
5. **Tool calling**: Qwen 3.6 requires `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3` (the `qwen3_coder` parser is more robust for multi-turn than the older `qwen3_xml`); without a parser, vLLM returns XML in `content` and leaves the native `tool_calls` array empty.
6. **max-num-batched-tokens**: for multimodal input (Gemma 4), must be >= `max_tokens_per_mm_item` (e.g. 2496 for Gemma 4, 4096 by default).
7. **TensorRT Model Optimizer on GB10**: quantizing Qwen 3.6 from BF16 requires converting the VLM checkpoint to text-only and using total GPU memory (not free memory) because `accelerate` does not understand the 128 GB unified pool.
8. **TRT-LLM PyTorch backend** reads `hf_quant_config.json`; use `--backend pytorch` and `--kv_cache_dtype fp8` for HF NVFP4 checkpoints.
9. **Background services**: stop VRAM consumers such as `llama-server` before launching large models. Two Qwen GGUF servers used ~76 GB in our tests and caused OOM/hangs.

---

## Final recommendation

For agentic workflows on DGX Spark and similar 96–128 GB edge AI workstations:

- **Qwen 3.6 35B-A3B nvidia NVFP4 + vLLM nightly** → **~75–77 tok/s**, tool calling with `qwen3_coder`, image/video support, and the model's full **262K context window** with 2 parallel sessions. **This is the current default recommendation.**
- **Gemma 4 26B-A4B community + patch** → ~49.5 tok/s, tool calling, low VRAM.
- **Qwen 3.6 35B-A3B RedHatAI** → ~42.2 tok/s, stable fallback if the nvidia checkpoint or nightly image are unavailable.
- **Qwen 3.6 35B-A3B MLP-only NVFP4** (custom TRT-LLM) → ~34.4 tok/s if you prefer the official NVIDIA stack.
- **Nemotron-3-Nano-Omni** → ~40.0 tok/s, text + image, best official multimodal option.
- **Nemotron-3-Super-120B-A12B** → ~14.7 tok/s with TRT-LLM only, for quality-first workloads.

Gemma 4 31B dense should be reserved only for tasks where the dense model quality justifies ~7 tok/s.

**If your agent framework (OpenClaw, Hermes, etc.) needs the largest possible context window on a single local GPU**, Qwen 3.6 35B-A3B on vLLM is the clear choice: it delivers the model's full 262K context length across 2 parallel sessions by default (3 sessions are possible but leave little memory headroom).
