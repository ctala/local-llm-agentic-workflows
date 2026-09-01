# Setup Log: Running Local LLMs on DGX Spark & 96–128 GB Edge AI Workstations

> For the quick start guide see [Home](/local-llm-agentic-workflows/). For the original Spanish log see [Setup in Spanish](/local-llm-agentic-workflows/setup.es/).

Per-model setup guides (faster than this log for actual installation):

- **[`docs/setup-qwen38-flash-next.md`](https://github.com/ctala/local-llm-agentic-workflows/blob/main/docs/setup-qwen38-flash-next.md)** — **Qwen3.8-Flash-Next NVFP4 hybrid** (default 2026-09-01, recipe [`blazux/qwen3.8-Flash-DGX`](https://github.com/blazux/qwen3.8-Flash-DGX), 14 min cold start, ~139 GB disk)
- **[`docs/setup-qwen38-27b-nvfp4-dspark.md`](https://github.com/ctala/local-llm-agentic-workflows/blob/main/docs/setup-qwen38-27b-nvfp4-dspark.md)** — **Qwen 3.8 27B NVFP4 + DSpark k=14** (fallback lite, 5-8 min cold start, ~25 GB disk)

This document is a detailed work log of the attempts, errors and fixes encountered while optimizing **Gemma 4**, **Qwen 3.6**, **Qwen 3.8 (27B + Flash-Next)** and **NVIDIA Nemotron 3** for local agentic workflows on the NVIDIA DGX Spark and equivalent high-memory edge AI hardware.

---

## Goal

Find the best way to run large local LLMs on the **NVIDIA DGX Spark** (GB10 Grace Blackwell, ARM64/aarch64, 128 GB unified memory, CUDA 13.0, sm_121), focused on use with agents such as **Hermes**, **OpenClaw**, n8n and Open WebUI.

Target models:
- Google Gemma 4 31B IT (dense)
- Google Gemma 4 26B-A4B IT (MoE)
- Qwen 3.6 35B-A3B (MoE)
- **Qwen 3.8 27B NVFP4 + DSpark k=14** (dense hybrid, current fallback lite since 2026-09-01)
- **Qwen3.8-Flash-Next NVFP4 hybrid** (MoE 176B with 6B active, Mamba + QSA + PLE, current default since 2026-09-01)
- NVIDIA Nemotron-3 Nano 30B-A3B (MoE, BF16)
- NVIDIA Nemotron-3 Super 120B-A12B (MoE, NVFP4)
- NVIDIA Nemotron-3 Nano Omni 30B-A3B (multimodal, NVFP4)

---

## Base system

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
| Disk | 3.7 TB, 2.7 TB free |

Previously running services included Open WebUI, n8n, qdrant, searxng, browserless, several NIMs, and two `llama-server` instances with Qwen GGUF models.

---

## Key lessons about the DGX Spark

1. **ARM64/aarch64 matters**: many x86_64 containers fail with `exec format error`. Always check for an `arm64` manifest.
2. **GB10 (sm_121) has no native FP4 compute**: unlike B200. NVFP4 weights run through the **Marlin** backend (`--moe-backend marlin`), which decompresses FP4 → BF16 at runtime. This caps throughput.
3. **Ollama/llama.cpp do not extract maximum performance** on Spark: they use generic backends without Blackwell-specific kernels or native NVFP4.
4. **For agents, model architecture matters more than framework alone**: Gemma 4 31B dense is memory-bandwidth limited (~6–7 tok/s), while Gemma 4 26B-A4B MoE (~3.8B active params) reaches ~50 tok/s.
5. **Background GPU services can hang the system**: two Qwen GGUF `llama-server` processes consumed ~76 GB of unified memory and caused vLLM to OOM/hang when loading Nemotron Super.

---

## Test results by model

### Gemma 4 26B-A4B NVFP4 on vLLM

Models tested:
- `nvidia/Gemma-4-26B-A4B-NVFP4` (~16.5 GB)
- `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` community + `gemma4_patched.py`

Working containers:
- Official: `vllm/vllm-openai:gemma4-0505-cu130` (vLLM 0.20.2rc1)
- Community patched: `vllm/vllm-openai:gemma4-cu130`

| Configuration | Decode tok/s | Hot TTFT | Notes |
|---------------|--------------|----------|-------|
| Official base (`gemma4-0505-cu130`) | ~30.1 | ~0.20 s | Stable, tool calling enabled |
| Community + patch (`gemma4-cu130`) | **~49.5** | ~0.08 s | Best option for agents |
| Community, gpu_util 0.92, max-seqs 4, batched 8192 | ~49.3 | ~1.9 s | No real improvement |
| Community, gpu_util 0.90, batched 2048 | ~49.3 | ~2.4 s | No real improvement |
| n-gram speculative decoding | ~24.6 | ~2.5 s | Worse on non-repetitive prompt |
| MTP (`google/gemma-4-26B-A4B-it-assistant`) | Error | – | `AssertionError` on drafter shape |

Issues:
- Without `--max-num-batched-tokens 4096` it fails due to chunked multimodal input.
- `vllm/vllm-openai:gemma4-cu130` does not load the official checkpoint: `KeyError: 'layers.0.experts.0.down_proj.input_scale'`.
- `vllm/vllm-openai:gemma4-0505-cu130` loads the official checkpoint but is ~20 tok/s slower than the community patch.

### Gemma 4 31B IT (dense)

Model: `nvidia/Gemma-4-31B-IT-NVFP4` (~31 GB)  
Container: `vllm/vllm-openai:gemma4-0505-cu130`

| Configuration | Decode tok/s | Hot TTFT | Notes |
|---------------|--------------|----------|-------|
| Base NVFP4 | **~6.7** | ~1.8 s | Memory-bandwidth limited |

Not a candidate for 50 tok/s due to being dense and memory-bandwidth bound.

### Qwen 3.6 35B-A3B

Models tested:
- `nvidia/Qwen3.6-35B-A3B-NVFP4` → **works with vLLM nightly** (current recommendation).
- `RedHatAI/Qwen3.6-35B-A3B-NVFP4` → works with `vllm/vllm-openai:gemma4-0505-cu130` (stable fallback).

Containers:
- `vllm/vllm-openai:nightly@sha256:a671d5fcda70fe9ac6f245f9780821de459fb4ee22c018fd07a0f10a55279bf9` for the nvidia checkpoint.
- `vllm/vllm-openai:gemma4-0505-cu130` for the RedHatAI checkpoint.

| Configuration | Checkpoint | Container | Decode tok/s | Hot TTFT | Notes |
|---------------|------------|-----------|--------------|----------|-------|
| **NVIDIA NVFP4 W4A16 + marlin + flashinfer** | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:nightly` | **~75–77** | ~0.10 s | **Current recommendation.** `modelopt` W4A16, `qwen3_coder` parser, `fastsafetensors`, `async-scheduling`, 262K context. |
| Base (`compressed-tensors` + `marlin`) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~42.2 | ~0.10 s | Stable fallback, tool calling enabled |
| max-seqs 4, batched 8192, gpu_util 0.92 | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~42.2 | ~1.4 s | No real improvement |
| n-gram speculative (`num_spec_tokens=5`) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~34–37 | ~0.10 s | Worse for non-repetitive text |
| MTP (`qwen3_5_mtp`) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | Error | – | Non-quantized drafter does not support `moe_backend='marlin'` |
| TRT-LLM 1.3.0rc13 (custom MLP-only NVFP4) | Quantized from BF16 | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~34.4** | ~0.09 s | Quantized from BF16 with Model Optimizer 0.44.0 |

### NVIDIA Nemotron 3

Models tested:
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (~89 GB)
- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (~75 GB)

#### TensorRT-LLM 1.3.0rc13

| Configuration | Decode tok/s | Hot TTFT | Peak memory | Notes |
|---------------|--------------|----------|-------------|-------|
| Nano-30B-A3B BF16 | **~28.8** | ~0.22 s | **~118 GB** | Direct load; almost fills unified pool |
| Super-120B-A12B NVFP4 | **~14.7** | ~0.29 s | **~110 GB** | Official NVFP4; slower due to more active experts |

#### vLLM `gemma4-0505-cu130`

| Configuration | Decode tok/s | Hot TTFT | Peak memory | Notes |
|---------------|--------------|----------|-------------|-------|
| Nano-30B-A3B BF16 | ~28.3 | ~0.20 s | ~72 GB | Works; stop other large GPU services first |
| Super-120B-A12B NVFP4 | — | — | — | **Not viable**: `CUDA OOM` on engine init; Spark hung on first attempt |

See `RESULTS.md` for the detailed analysis of why Nemotron Super fails on vLLM.

### NVIDIA Nemotron 3 Nano Omni (multimodal)

Model: `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (~21 GB)  
Container: `vllm/vllm-openai:gemma4-0505-cu130`

| Configuration | Decode tok/s | Hot TTFT | Peak memory | Multimodal | Notes |
|---------------|--------------|----------|-------------|------------|-------|
| vLLM, `modelopt_fp4`, `marlin` | **~40.0** | ~0.10 s | **~40 GB** | Image ✅ | Fast text; image answers correctly |
| Audio via OpenAI `input_audio` | – | – | – | Audio ❌ | Failed with `Invalid or unsupported audio file` in container decoding |

TRT-LLM 1.3.0rc13 was also tested but failed to parse multimodal messages:
> `AttributeError: 'NoneType' object has no attribute 'model_type'` in `parse_chat_messages_coroutines`.

---

## Official TensorRT-LLM attempts

We tested NVIDIA's official TensorRT-LLM containers for Spark:

| Container | TRT-LLM | Gemma 4 26B | Qwen 3.6 35B | Nemotron Nano | Nemotron Super |
|-----------|---------|-------------|--------------|---------------|----------------|
| `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` | 1.1.0rc3 | No `gemma4` support | Argument/quant error | – | – |
| `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10` | 1.3.0rc10 | No `gemma4` support | `AssertionError` quant_algo | – | – |
| `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | 1.3.0rc13 | No `gemma4` support | `AssertionError` quant_algo (pre-quantized), OK custom MLP-only | **OK** BF16 | **OK** NVFP4 | Omni: multimodal parse error |

### Key errors

- **Gemma 4**: `ValueError: model type 'gemma4' but Transformers does not recognize this architecture`. The bundled `transformers` (4.55–4.57) lacks Gemma 4 support.
- **Qwen 3.6 35B-A3B NVFP4 (NVIDIA)**: `AssertionError` in `QuantMode.from_quant_algo`; the TRT-LLM PyTorch backend does not recognize the `modelopt` NVFP4 algorithm of NVIDIA's pre-quantized checkpoint.

### Custom Qwen 3.6 quantization

We followed NVIDIA's flow starting from the BF16 base `Qwen/Qwen3.6-35B-A3B` (~70 GB) using **TensorRT Model Optimizer 0.44.0**:

1. Download BF16 base.
2. Convert VLM checkpoint to text-only (`qwen3_5_moe_text`) because Model Optimizer cannot load `Qwen3_5MoeForConditionalGeneration` directly.
3. Patch `modelopt` quantizers to support per-expert quantization of Qwen3.5/3.6.
4. Patch `example_utils.py` to use **total** GPU memory instead of free memory reported by `accelerate` (required for the GB10 unified pool).
5. Quantize with `--qformat nvfp4` and `--qformat nvfp4_mlp_only`.

| Quant config | Size | Result on TRT-LLM 1.3.0rc13 |
|--------------|------|------------------------------|
| Full NVFP4 | ~20 GB | `NotImplementedError`: split linear-attention packing does not support quantized linear-attention `input_scale`/`weight_scale` tensors. |
| **MLP-only NVFP4** | ~22 GB | **Serves correctly** with `trtllm-serve --backend pytorch`. |

Benchmark: ~34.4 decode tok/s, hot TTFT ~0.09 s, ~41 GB unified memory.

---

## Final recommended configurations

| Model | Checkpoint | Container | Decode tok/s | Recommended use |
|-------|------------|-----------|--------------|-----------------|
| **Qwen3.8-Flash-Next NVFP4 hybrid** | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `qwen38-flash-dgx` (vLLM `release/qwen38next` + 7 parches GB10) | **~37 warm / 117 @c=8** | **Default (2026-09-01).** Best quality (GSM8K 97.27%), MoE 176B (6B activos), multimodal texto+imagen+video, 262K contexto. |
| **Qwen 3.8 27B NVFP4 + DSpark k=14** | `unsloth/Qwen3.8-27B-NVFP4` | `vllm/vllm-openai:v0.27.1-aarch64` | 30 fresh / 70-76 warm / 253 @c=16 | **Fallback lite.** Mejor concurrencia, multimodal, 262K contexto. |
| **Qwen 3.6 35B-A3B** | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:nightly` | **~75–77** | Single-stream long-context 262K champion; 1 secuencia por sesión. |
| **Gemma 4 26B-A4B** | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` + patch | `vllm/vllm-openai:gemma4-cu130` | **~49.5** | Maximum speed for agents |
| **Qwen 3.6 35B-A3B** | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~42.2** | Stable fallback |
| **Gemma 4 31B** | `nvidia/Gemma-4-31B-IT-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~6.7** | Only if dense model is needed |
| **Qwen 3.6 35B-A3B** | Custom MLP-only NVFP4 | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~34.4** | Official NVIDIA stack alternative |
| **Nemotron-3-Nano-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~28.8** | Official NVIDIA dense model; uses almost all memory |
| **Nemotron-3-Super-120B-A12B** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~14.7** | Large official model; quality-first |
| **Nemotron-3-Nano-Omni-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~40.0** | Best official multimodal: text + image |

---

## Technical notes

- **HF_TOKEN required** to download Gemma/Qwen checkpoints from HuggingFace.
- **Memory**: Gemma 4 26B-A4B NVFP4 uses ~18 GB at load; FP8 KV cache leaves ~82 GB available.
- **Marlin backend** is mandatory on GB10 for MoE NVFP4. Native FP4 backends may fail or produce NaN on sm_121.
- **Tool calling**: Qwen 3.6 requires `--enable-auto-tool-choice --tool-call-parser qwen3_coder`; the `qwen3_coder` parser is more robust for multi-turn than the older `qwen3_xml`. We intentionally omit `--reasoning-parser` because the `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint does not emit `<think></think>` tags; with the parser enabled, reasoning leaks into the `reasoning` field and `content` appears empty in agents. With thinking enabled but no parser, reasoning and answer both land in `content`, which agents display correctly. `auto_disable_thinking_with_tools` disables thinking automatically when tools are present.
- **TRT-LLM with Nemotron 3**: official checkpoints load directly with `trtllm-serve --backend pytorch --kv_cache_dtype fp8`.
- **vLLM with Nemotron 3**: Nano BF16 and Omni NVFP4 work. Super 120B-A12B does not due to aggressive memory reservation in the V1 engine.
- **Background services**: stop `llama-server` and other GPU consumers before launching large models.

---

## Deliverables in this repository

- `README.md`: executive summary and quick-start guide.
- `RESULTS.md`: full benchmark tables and technical analysis.
- `SETUP.md`: this detailed work log.
- `RESULTS.es.md` / `SETUP.es.md`: Spanish versions of the above.
- `scripts/run-*.sh`: Docker launch recipes.
- `scripts/quantize-qwen36-nvfp4.sh`: Qwen 3.6 BF16 → NVFP4 MLP-only quantization helper.
- `scripts/convert-qwen36-vlm-to-text.py`: extracts the text-only part of the Qwen 3.6 VLM checkpoint.
- `benchmarks/bench_model.py`: reproducible text benchmark.
- `benchmarks/test_multimodal.py`: image/audio tests for multimodal models.

---

## Next steps

1. Validate MTP speculative decoding with the nvidia checkpoint + vLLM nightly (earlier tests on RedHatAI slowed decode single-user).
2. Test `--tool-call-parser gemma4` for native Gemma 4 tool calling.
3. Evaluate agentic quality with `tool-eval-bench` for Hermes/OpenClaw.
4. Test full NVFP4 Qwen 3.6 on TRT-LLM when linear-attention scales are supported.
5. Compare Qwen 3.6 nvidia W4A16 quality against RedHatAI `compressed-tensors` and custom MLP-only NVFP4.
6. Test GPT-OSS with TRT-LLM.
7. Test GGUF and MTP variants of Qwen 3.6 to compare quality vs speed trade-offs.

---

## Qwen 3.8 family (2026-08-15 → 2026-09-01)

Two new Qwen 3.8 checkpoints were validated on the Spark:

### Qwen 3.8 27B NVFP4 + DSpark k=14 (fallback lite since 2026-09-01)

**Setup** (~5 min from scratch):

```bash
# 1. Download checkpoint + drafter (~22 GB + 2.7 GB)
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/vllm/qwen3.8-27b-nvfp4
hf download Doopeworld/Qwen3.8-27B-DSpark-vLLM --local-dir ~/vllm/qwen3.8-dspark

# 2. Pull image
docker pull vllm/vllm-openai:v0.27.1-aarch64

# 3. Launch (foreground, port 8001)
./scripts/run-qwen38-27b-nvfp4-dspark.sh
```

Recipe upstream: [`0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark`](https://github.com/0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark) — adapted from FP8 to NVFP4 + added systemd service for production. Key flags:

- `--speculative-config '{"method":"dspark","model":"/models/qwen3.8-dspark","num_speculative_tokens":14,"draft_sample_method":"probabilistic"}'` — DSpark k=14 drafter gives **3× speedup** on warm prefix cache.
- `--enable-prefix-caching` — **mandatory**; without it, EDIT-heavy workloads drop from 73-76 tok/s to ~30 tok/s.
- `--tool-call-parser qwen3_xml --reasoning-parser qwen3` — XML parser (not `qwen3_coder`).
- `--limit-mm-per-prompt.image 2 --limit-mm-per-prompt.video 0` — multimodal (text + image + video).

**Measured** (Spark, page cache warm):

| Workload | Tokens/s | Notes |
|----------|---------:|-------|
| Fresh-code single-stream | **29.67** | No cache hits |
| EDIT-heavy (warm prefix cache) | **73–76** | Cache hit on system prompt |
| Aggregate c=1 | 70.66 | Single-session warm |
| Aggregate c=4 | 150.64 | |
| Aggregate c=8 | 182.25 | |
| **Aggregate c=16** | **253.18** | |
| Determinism (20/20 byte-identical) | ✓ | |

### Qwen3.8-Flash-Next NVFP4 hybrid (default since 2026-09-01)

A hybrid-architecture MoE model (Mamba + QSA sparse attention + PLE n-gram injection, 48 decoder layers × 512 routed experts, top-10 routing, 1 shared expert + 1 MTP layer). Quantized with NVIDIA Model Optimizer (`v0.46.0` snapshot `87c9f8cf`) using **NVFP4 W4A4**, scoped to the routed experts only. The dense side layers (GDN, QSA, shared experts) are kept in bf16 in the published checkpoint; the recipe [`blazux/qwen3.8-Flash-DGX`](https://github.com/blazux/qwen3.8-Flash-DGX) optionally converts them to **blockwise fp8-e4m3** (128×128 blocks, DeepSeek layout) for an additional ~20% decode speedup (`MODE=hybrid`).

**Setup** (~50 min from scratch):

```bash
# 1. Clone the recipe repo (provides Dockerfile + patches + scripts)
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git upstream

# 2. Build the patched Docker image (~3 min)
cd upstream && docker build -t qwen38-flash-dgx .
# The Dockerfile applies 7 GB10-targeted patches on top of
# vllm/vllm-openai:qwen38-flash-next (release/qwen38next recipe / PR #53896):
#   1. PLE n-gram table served from NVMe via mmap (VLLM_PLE_MMAP=1)
#   2. GB10 FLA shared-memory gate (sm_121 reports 99 KiB, not 100)
#   3. Mamba state-copy race fix (vllm#50729 + bounds guard)
#   4. Prefix-caching block_size fix (was silently restoring all-zero Mamba state)
#   5. Exact deterministic QSA top-k (VLLM_QSA_EXACT_TOPK=1)
#   6. NVFP4 experts + fp8 side layers hybrid dispatch (VLLM_FP8_HYBRID=1)
#   7. fp8_e4m3 KV cache on the QSA path

# 3. Download checkpoint (~20 min, 126 GiB → ~/.cache/huggingface/)
cd .. && bash upstream/scripts/download-weights.sh
# Requires HF_TOKEN or accepts unauthenticated downloads (model is public).

# 4. (Optional) Prepare the hybrid snapshot, +13 GB, ~10 min
bash upstream/scripts/prepare-hybrid.sh
# Touches 4 shards / 300 tensors, max relative error 3.54%.

# 5. Launch (foreground, port 8001)
./scripts/run-qwen38-flash-next.sh
# Or set MODE=hybrid after running prepare-hybrid.sh.

# 6. Enable systemd for auto-start on boot
cp systemd/qwen38-flash-next-vllm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qwen38-flash-next-vllm.service
```

**Measured** (Spark, page cache warm ~25 min, MODE=hybrid):

| Workload | Tokens/s | Notes |
|----------|---------:|-------|
| chat_short fresh (warm) | **36.66** | Median of 5 runs, stddev ~0.5 tok/s |
| chat_short fresh (cold first request) | ~22 | ~2-3 s TTFT warmup |
| code_quicksort fresh | 28.52 | max_tokens=512 |
| agent_with_tools (single) | 23.59 | Tool calling with `qwen3_coder` parser |
| Tool loop multi-step (4 steps, 3 tools) | 28.65 median | 14.5 s wall clock per loop |
| Aggregate c=1 | 30.29 | |
| Aggregate c=4 | 45.09 | |
| **Aggregate c=8** | **116.75** | +30% vs NVFP4 mode |
| **Aggregate c=16** | **129.65** | +24% vs the 27B default |
| Aggregate c=32 | 129.69 | Plateau (SEQS=8 cap) |
| Prefill cold (8K prompt) | 1018 tok/s | |
| **Prefix-hit TTFT** (same 20K prompt) | **1.46 s** | 7.2× speedup vs 10.47 s cold |
| Determinism (EXACT_TOPK=1) | ✓ | First-token logprobs byte-identical |
| Memory | ~98 GiB / 121 GiB | PLE table mmap'd off-pool |
| GSM8K (RadixArk eval) | **97.27%** | t=0.6, top-p=0.95, max=8192 |
| AIME26 (RadixArk eval) | **98.75% pass@1** | 30 problems × 8, t=1.0, max=130k |
| Cold start | | ~14 min first time; ~1-2 min cached |
| Disk | | ~139 GiB (126 GB NVFP4 + 13 GB hybrid) |

**Tuning notes**: the upstream defaults are optimal. We benchmarked three variants and all were worse:

| Variant | Δ vs upstream defaults |
|---------|------------------------|
| `SEQS=32, max-num-batched-tokens=16384` | @c=4-8 +16%, **@c=16 −23%** (scheduler context switching) |
| `MTP=4` (instead of `MTP=2`) | -7 to -34% across the board (acceptance rate drops) |
| `SEQS=16` | @c=8 +5%, **@c=16 −23%** (same scheduler issue) |

Counterintuitive finding: keeping `SEQS=8` below the desired concurrency level gives better aggregate throughput because the first N sequences are processed without context switching; the rest wait briefly.

### Why Qwen3.8-Flash-Next replaced Qwen 3.8 27B as default

- **Quality**: GSM8K 97.27% / AIME26 98.75% vs ~95% for the 27B dense hybrid.
- **Fresh single-stream**: 36.66 tok/s vs 13.93 tok/s for the 27B fresh (+163%).
- **Concurrency @c=8**: 117 tok/s aggregate vs 40.40 for the 27B fresh (+189%).
- **Multimodal**: text + image + video out of the box (vs the 27B also has it but with limited recipes).
- **Context**: 262K native, 500K YaRN (vs 262K / 1M YaRN on the 27B).
- **Trade-off**: the 27B + DSpark k=14 still wins @c=16 (253 tok/s aggregate) because DSpark k=14 is a more aggressive drafter than MTP=2 in this MoE setup. Keep the 27B as a high-concurrency fallback.

### Critical GB10 caveats solved by the recipe

The vanilla vLLM `release/qwen38next` recipe fails on the Spark GB10 (sm_121). The [`blazux/qwen3.8-Flash-DGX`](https://github.com/blazux/qwen3.8-Flash-DGX) image bundles **7 patches**:

1. **PLE mmap** (most important): the checkpoint is 126 GiB but only ~76 GiB are loaded into GPU memory; the 48 GiB PLE n-gram embedding table is served from NVMe via `mmap` and the lookup goes through a `vllm::ple_mmap_lookup` splitting op outside CUDA graphs. **Without this patch, the model does not fit in 128 GB**.
2. **GB10 FLA fixes**: sm_121 reports 99 KiB of shared memory per block, but the flash-linear-attention gate asked for 100 KiB, so all 36 GDN layers silently ran on small tiles; a `sed` lowers the gate to 99 KiB.
3. **Mamba state-copy race fix** (vllm#50729 + bounds guard by `@Saren-Arterius`): turns an out-of-range block id into a skipped copy plus a log counter instead of a dead CUDA context.
4. **Prefix-caching block_size fix**: vLLM's `cache_config.block_size` was being overwritten with the smallest KV-group block size (8 tokens here, the QSA raw-key ring), but the Mamba block is 1600 tokens; two consumers used the former as the latter, so every prefix hit restored an **all-zero Mamba state** — silently wrong answers. With this fix, cold and cache-hit outputs are bit-identical.
5. **Exact deterministic QSA top-k** (`VLLM_QSA_EXACT_TOPK=1`): the stock `persistent_topk` kernel is non-deterministic on GB10 and can drop legitimate top-k candidates ([vllm#51782](https://github.com/vllm-project/vllm/issues/51782)). The patched path uses `torch.topk` over the visible columns — identical greedy outputs, costs ~10% on long prefills.
6. **NVFP4 + blockwise fp8 hybrid dispatch** (`VLLM_FP8_HYBRID=1`): routed experts stay NVFP4 (where quality lives); GDN/QSA/shared-expert side layers go through blockwise fp8 GEMM (where bandwidth dominates). Same quality tournament score, +20% decode.
7. **fp8_e4m3 KV cache** (optional, `--kv-cache-dtype fp8_e4m3`): 1.9× KV pool, 1M context on one box — at the cost of -10% decode, -30% prefill, and a measurable quality dip in long-context. Not enabled by default.
