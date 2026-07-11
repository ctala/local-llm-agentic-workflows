# Setup Log: Running Local LLMs on DGX Spark & 96–128 GB Edge AI Workstations

> For the quick start guide see [Home](/local-llm-agentic-workflows/). For the original Spanish log see [Setup in Spanish](/local-llm-agentic-workflows/setup.es/).

This document is a detailed work log of the attempts, errors and fixes encountered while optimizing **Gemma 4**, **Qwen 3.6** and **NVIDIA Nemotron 3** for local agentic workflows on the NVIDIA DGX Spark and equivalent high-memory edge AI hardware.

---

## Goal

Find the best way to run large local LLMs on the **NVIDIA DGX Spark** (GB10 Grace Blackwell, ARM64/aarch64, 128 GB unified memory, CUDA 13.0, sm_121), focused on use with agents such as **Hermes**, **OpenClaw**, n8n and Open WebUI.

Target models:
- Google Gemma 4 31B IT (dense)
- Google Gemma 4 26B-A4B IT (MoE)
- Qwen 3.6 35B-A3B (MoE)
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
| **Qwen 3.6 35B-A3B** | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:nightly` | **~75–77** | **Best quality/speed balance; full 262K context; tool calling.** |
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
- **Tool calling**: Qwen 3.6 requires `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`; the `qwen3_coder` parser is more robust for multi-turn than the older `qwen3_xml`. Without a parser, vLLM returns XML in `content` and agent frameworks receive an empty `tool_calls` array.
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
