---
title: "Best Local LLM Deployment Guide for DGX Spark, GB10 & 96–128 GB Edge AI Workstations"
description: "Practical guide, benchmarks and copy-paste Docker recipes for running Gemma 4, Qwen 3.6 and NVIDIA Nemotron 3 locally on DGX Spark, GB10 and other high-memory edge AI workstations using vLLM and TensorRT-LLM."
keywords:
  - local LLM
  - DGX Spark
  - NVIDIA GB10
  - edge AI
  - vLLM
  - TensorRT-LLM
  - Gemma 4
  - Qwen 3.6
  - Nemotron 3
  - Hermes agent
  - OpenClaw
  - Open WebUI
  - 128 GB unified memory
---

# Best Local LLM Deployment Guide for DGX Spark, GB10 & 96–128 GB Edge AI Workstations

A practical guide, reproducible benchmark results and ready-to-run Docker recipes for running the best local LLMs on high-memory edge AI hardware such as the **NVIDIA DGX Spark** (GB10 Grace Blackwell, ARM64/aarch64, 128 GB unified memory, sm_121) and other single-GPU workstations with **96–128 GB of unified or pooled memory**.

This repository answers one question: **what is the best way to run local large language models on memory-rich edge devices?** It compares production inference engines (**vLLM** and **TensorRT-LLM**), quantization formats (**NVFP4/Marlin**, **FP8 KV cache**, **BF16**) and model families (**Gemma 4**, **Qwen 3.6**, **NVIDIA Nemotron 3**), measuring real decode throughput, memory footprint and stability for local AI deployment.

Use cases covered include local chatbots, coding assistants, agentic workflows with **Hermes** and **OpenClaw**, Open WebUI, n8n, multi-turn tool calling, multimodal agents and local speech-to-text (ASR) on the DGX Spark and similar edge AI workstations.

If you are looking for broader cloud-vs-local model comparisons, see the related benchmark projects by [Cristian Tala](https://github.com/ctala) listed at the end of this guide.

> **Critical edge-AI limitation**: the NVIDIA GB10 has no native FP4 compute. NVFP4 checkpoints run through the **Marlin** backend (vLLM) or the PyTorch backend of TensorRT-LLM, decompressing FP4 → BF16 at runtime. This limits throughput compared to FP4-native datacenter GPUs such as the B200, but still makes 30B–120B parameter models runnable on a single edge device.

---

## Table of contents

- [Quick answer: best models for DGX Spark](#quick-answer)
- [Target hardware](#target-hardware)
- [Repository layout](#repository-layout)
- [Usage](#usage)
- [Results summary](#results-summary-local-llm-throughput-on-dgx-spark)
- [Recommended models by use case](#recommended-models-by-use-case)
- [Parallel long-context sessions with Qwen 3.6](#parallel-long-context-sessions-with-qwen-3-6-35b-a3b)
- [Connecting to agent frameworks](#connecting-to-agent-frameworks)
- [Full local stack](#full-local-stack)
- [Important memory lesson](#important-memory-lesson-for-128-gb-unified-memory-systems)
- [Author and related benchmarks](#author-and-related-benchmarks)
- [License](#license)

---

## Quick answer

For local LLM deployment on DGX Spark or equivalent 96–128 GB edge hardware, start here:

| Priority | Model | Framework | Decode tok/s | Memory | Tool calling | Multimodal | Max context tested |
|----------|-------|-----------|--------------|--------|--------------|------------|--------------------|
| **Quality/speed balance** | Qwen 3.6 35B-A3B (nvidia NVFP4) | vLLM nightly | **~75–77** | ~22 GB | ✅ | ✅ image/video | **262K** |
| **Speed** | Gemma 4 26B-A4B IT (community patch) | vLLM | **~49.5** | ~22 GB | ✅ | ✅ image/video | 128K |
| **Official NVIDIA multimodal** | Nemotron-3 Nano Omni 30B-A3B | vLLM | **~40.0** | ~40 GB | ✅ | ✅ image | 128K |

For quality-first workloads where speed is less important:

| Model | Framework | Decode tok/s | Memory | Notes |
|-------|-----------|--------------|--------|-------|
| Qwen 3.6 35B-A3B (custom MLP-only NVFP4) | TensorRT-LLM | ~34.4 | ~41 GB | Official NVIDIA stack; requires manual quantization |
| Nemotron-3 Super 120B-A12B | TensorRT-LLM | ~14.7 | ~110 GB | Best official quality; use only with TRT-LLM |

---

## Target hardware

The recipes were tested on the **NVIDIA DGX Spark** and should work on any ARM64 or x86_64 single-GPU system with similar memory characteristics:

| Component | DGX Spark value | Typical compatible edge AI workstation |
|-----------|-----------------|----------------------------------------|
| CPU | 20 cores ARM64 (aarch64) | ARM64 or x86_64, 16+ cores |
| GPU | NVIDIA GB10 (sm_121) | Single GPU with 96–128 GB accessible memory |
| Memory | 128 GB LPDDR5x unified (~121 GB usable) | 96–128 GB unified, pooled or host memory |
| NVIDIA driver | 580.142 | Recent 570+ series |
| CUDA | 13.0 | CUDA 12.6+ / 13.0 |
| Docker | 29.2.1 | Docker 24+ with NVIDIA Container Toolkit |
| NVIDIA Container Toolkit | 1.19.0 | 1.14+ |

> Most Docker images used here are ARM64 manifests. If you run x86_64 hardware, replace the image tags with the equivalent x86_64 builds from NVIDIA and vLLM.

---

## Repository layout

```
.
├── README.md                     # This file
├── Results                       # Full benchmark tables and technical notes
├── Setup                         # Detailed setup log, errors and fixes
├── Agents                        # Hermes / OpenClaw / Opencode / LiteLLM integration guide
├── LICENSE                       # MIT
├── scripts/                      # Docker launch recipes and helpers
│   ├── run-gemma4-26b-a4b.sh
│   ├── run-qwen36-35b-a3b.sh
│   ├── run-qwen36-35b-a3b-extreme-context-2seq.sh
│   ├── run-qwen36-35b-a3b-trtllm.sh
│   ├── run-gemma4-31b.sh
│   ├── run-nemotron3-nano-30b-a3b-trtllm.sh
│   ├── run-nemotron3-nano-30b-a3b-vllm.sh
│   ├── run-nemotron3-super-120b-a12b-trtllm.sh
│   ├── run-nemotron3-nano-omni-vllm.sh
│   ├── quantize-qwen36-nvfp4.sh
│   └── convert-qwen36-vlm-to-text.py
└── benchmarks/
    ├── bench_model.py            # Text throughput benchmark (OpenAI-compatible API)
    └── test_multimodal.py        # Image/audio tests for multimodal models
```

---

## Usage

1. Download the desired checkpoint under `~/vllm/`.
2. Run the matching script from `scripts/`.
3. Wait for the container to load the model.
4. Benchmark text generation:
   ```bash
   python3 benchmarks/bench_model.py gemma-4-26b-a4b 512
   ```
5. For multimodal models:
   ```bash
   python3 benchmarks/test_multimodal.py nemotron3-nano-omni image /path/to/image.png
   ```

> Only one container can bind port 8000 at a time. To run several models concurrently, change the port in the script or use a proxy such as LiteLLM.

---

## Results summary: local LLM throughput on DGX Spark

Benchmarks use a ~120-token prompt, `max_tokens=512`, temperature 0.7 and streaming, reporting decode tok/s and hot TTFT.

| Model | Checkpoint | Framework | Decode tok/s | Memory | Tool calling | Multimodal | Max context | Recommendation |
|-------|------------|-----------|--------------|--------|--------------|------------|-------------|----------------|
| **Qwen 3.6 35B-A3B** | `nvidia/Qwen3.6-35B-A3B-NVFP4` | vLLM nightly | **~75–77** | ~22 GB | ✅ | ✅ image/video | **262K** | **Current recommendation: best quality/speed balance and longest context** |
| **Gemma 4 26B-A4B** | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` + patch | vLLM | **~49.5** | ~22 GB | ✅ | ✅ image/video | 128K | **Fastest raw speed for text agents** |
| **Qwen 3.6 35B-A3B** | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | vLLM | **~42.2** | ~22 GB | ✅ | ✅ image/video | **262K** | Stable fallback |
| **Nemotron-3 Nano Omni 30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | vLLM | **~40.0** | ~40 GB | ✅ | ✅ image | 128K | **Best official multimodal option** |
| Qwen 3.6 35B-A3B | Custom MLP-only NVFP4 | TRT-LLM | ~34.4 | ~41 GB | ✅ | ❌ text only | 32K | Official NVIDIA stack |
| Nemotron-3 Nano 30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | TRT-LLM | ~28.8 | ~118 GB | ✅ | ❌ text only | 8K | Official dense model; leaves little free memory |
| Nemotron-3 Nano 30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | vLLM | ~28.3 | ~72 GB | ✅ | ❌ text only | 8K | vLLM alternative; stop other GPU services first |
| **Nemotron-3 Super 120B-A12B** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | TRT-LLM | **~14.7** | ~110 GB | ✅ | ❌ text only | 8K | Best official quality; **not viable with vLLM on GB10** |
| Gemma 4 31B | `nvidia/Gemma-4-31B-IT-NVFP4` | vLLM | ~6.7 | ~31 GB | ✅ | ✅ image/video | 128K | Use only if you need the dense variant |

See [Results](/local-llm-agentic-workflows/results/) for full tables, launch commands and error analysis.

---

## What makes this different from other local LLM comparisons

Most public benchmarks focus on **raw capability** using llama.cpp GGUF quantizations. This repository focuses on **production local LLM deployment** on high-memory edge AI workstations such as the NVIDIA DGX Spark and GB10:

- **Inference engines**: vLLM vs TensorRT-LLM on ARM64/aarch64 edge devices.
- **Quantization formats**: NVFP4/Marlin, FP8 KV cache, BF16 and compressed-tensors.
- **Memory management**: how to fit 30B–120B parameter models into 128 GB unified memory.
- **Stability**: which model + engine combinations actually load and serve reliably on GB10-class hardware.
- **Throughput**: real decode tok/s measured with a reproducible OpenAI-compatible benchmark.
- **Agent integration**: tested connectivity with Hermes, OpenClaw, Opencode, Open WebUI, n8n and LiteLLM, plus a local faster-whisper ASR server for voice messages.

The recipes are designed to be copy-pasteable into a DGX Spark, GB10 or any 96–128 GB edge AI workstation.

---

## Full local stack

For a complete self-hosted agent setup on the DGX Spark — inference, proxy, agent, search and speech-to-text — see the [Stack](/local-llm-agentic-workflows/stack/) page. It documents every service, port, systemd unit and the typical workflow to keep everything local.

---

## Important memory lesson for 128 GB unified-memory systems

Before launching large models, stop background services that consume VRAM. During our tests, two `llama-server` instances running Qwen GGUF models were using ~76 GB of unified memory and caused vLLM to OOM/hang the system. We disabled the systemd units to prevent automatic restart:

```bash
systemctl --user stop qwen27-local.service qwen35-local.service
systemctl --user disable qwen27-local.service qwen35-local.service
```

Re-enable them later if you need those servers back:

```bash
systemctl --user enable --now qwen27-local.service qwen35-local.service
```

---

## Recommended models by use case

Based on throughput, memory headroom and tool-calling support on DGX Spark and equivalent 96–128 GB edge AI hardware:

1. **Best quality/speed balance** → **Qwen 3.6 35B-A3B** (`nvidia/Qwen3.6-35B-A3B-NVFP4`) on vLLM nightly.
   - **~75–77 tok/s**, ~22 GB memory, full **262K context**, excellent tool calling with `qwen3_coder`, image/video support.
2. **Fastest raw speed for agents and chatbots** → **Gemma 4 26B-A4B IT** (community patch) on vLLM.
   - ~49.5 tok/s, ~22 GB memory. Requires the community patch.
3. **Stable fallback** → **Qwen 3.6 35B-A3B** (`RedHatAI/Qwen3.6-35B-A3B-NVFP4`) on vLLM.
   - ~42 tok/s, ~22 GB memory, tool calling, image/video support.
4. **Official NVIDIA multimodal** → **Nemotron-3 Nano Omni 30B-A3B** on vLLM.
   - ~40 tok/s, ~40 GB memory, image support. Audio decoding needs more work.
5. **Official NVIDIA stack** → **Qwen 3.6 35B-A3B MLP-only NVFP4** on TensorRT-LLM.
   - ~34.4 tok/s, ~41 GB memory. Requires manual quantization from BF16.
6. **Quality over speed** → **Nemotron-3 Super 120B-A12B** on TensorRT-LLM.
   - ~14.7 tok/s, ~110 GB memory. Use only with TRT-LLM; vLLM is unstable on GB10.

---

## Parallel long-context sessions with Qwen 3.6 35B-A3B

Agentic frameworks such as **Hermes** and **OpenClaw** can run multiple conversations or subagents at the same time. We validated how many parallel long-context sessions the DGX Spark can sustain with `nvidia/Qwen3.6-35B-A3B-NVFP4` served by vLLM nightly.

### Configuration

Use the recipe in `scripts/run-qwen36-35b-a3b.sh` (or its alias `scripts/run-qwen36-35b-a3b-extreme-context-2seq.sh`):

```bash
--model /models/qwen3.6 \
--trust-remote-code \
--tensor-parallel-size 1 \
--attention-backend flashinfer \
--moe-backend marlin \
--kv-cache-dtype fp8 \
--gpu-memory-utilization 0.92 \
--max-model-len 262144 \
--max-num-seqs 1 \
--max-num-batched-tokens 32768 \
--enable-chunked-prefill \
--async-scheduling \
--enable-prefix-caching \
--load-format fastsafetensors \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder
```

This is the recommended default for agentic workloads. It gives **one 262K-token session** while leaving headroom for LiteLLM, ASR or other auxiliary services. We omit `--reasoning-parser` because the `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint does not emit `<think></think>` tags; with the parser, agents see empty `content`.

### Verified single-session context scaling

| Input tokens | Output tokens | Notes |
|--------------|---------------|-------|
| 50K | 25K | Sub-second TTFT, very responsive. |
| 100K | 25K | Sub-second TTFT. |
| 180K | 25K | Stable within the 262K limit. |
| 237K | 25K | Near the current operational limit; use for extreme long-context sessions. |

### Practical guidance

- **Typical agent turn**: Hermes-style agents use **8K–32K tokens** of active context per session.
- **Comfortable production default**: **1 session × 64K–128K context** runs with sub-second TTFT and leaves memory headroom for LiteLLM/ASR.
- **Maximum input per session**: **~237K tokens** with 1 concurrent session at the 262K config (reserving 25K for output).
- **Memory at rest after loading**: ~117–119 GB used / ~121 GB total. Keep other GPU/VRAM consumers stopped.

See [Results](/local-llm-agentic-workflows/results/) for the older 3-session measurements and full context-scaling tables.

See [Results](/local-llm-agentic-workflows/results/) for the full context-scaling tables.

---

## Connecting to agent frameworks

All recipes expose an **OpenAI-compatible API** on `http://localhost:8000/v1`. You can connect Hermes, OpenClaw, Opencode, n8n, Open WebUI, or any other OpenAI-compatible client directly, or route everything through a **LiteLLM proxy** for a single endpoint that can be shared across the network.

See [Agents](/local-llm-agentic-workflows/agents/) for the full integration guide covering:

- **Hermes** — direct and via LiteLLM
- **OpenClaw** — direct, LiteLLM, and migration to Hermes
- **Opencode** — direct and via LiteLLM
- **LiteLLM proxy** — unified endpoint on `http://0.0.0.0:4000/v1`
- **Network access** — using the Spark from other machines
- **Local ASR** — faster-whisper server for voice message transcription
- **Local web search** — SearXNG self-hosted meta-search
- **Troubleshooting**

Quick start for Hermes (via LiteLLM proxy):

```yaml
providers:
  local-qwen-35b-vllm-fast:
    base_url: http://localhost:4000/v1
    name: Spark Qwen 35B-A3B (vLLM fast)
    api_key: local-no-key-needed
    model: qwen3.6-35b-a3b-vllm-fast
    context_length: 237000
    max_tokens: 25000
```

```bash
hermes chat --provider local-qwen-35b-vllm-fast -m qwen3.6-35b-a3b-vllm-fast
```

---

## Next steps

- Add quality measurements with agentic benchmarks on top of throughput numbers for `nvidia/Qwen3.6-35B-A3B-NVFP4`.
- Evaluate Qwen 3.6 MTP variants and Unsloth-optimized builds if they become available for GB10, always checking quality parity.
- Resolve audio decoding for Nemotron-3 Nano Omni.
- Keep LiteLLM proxy recipes current as the single endpoint for Hermes, OpenClaw and Open WebUI.
- Expand coverage to other high-memory edge devices beyond DGX Spark.
- Validate 1×262K context under real Hermes multi-turn traces.

---

## Frequently asked questions

### What is the best local LLM for the NVIDIA DGX Spark?
**Qwen 3.6 35B-A3B** (`nvidia/Qwen3.6-35B-A3B-NVFP4`) served by vLLM nightly is the current sweet spot: **~75–77 tok/s**, full **262K context**, strong tool calling and image/video support, at ~22 GB of memory.

### Can the DGX Spark run 70B+ parameter models locally?
Yes. With 128 GB of unified memory and NVFP4 quantization, the Spark can run the **Nemotron-3 Super 120B-A12B** model at ~14.7 tok/s, but only with TensorRT-LLM. vLLM reserves memory too aggressively for this model on GB10.

### Is vLLM or TensorRT-LLM better on DGX Spark?
It depends on the model. **vLLM** is faster and more flexible for Gemma 4, Qwen 3.6 and Nemotron-3 Nano Omni. **TensorRT-LLM** is more memory-efficient and the only stable option for Nemotron-3 Super 120B-A12B.

### Which models support tool calling and agents?
All models in the results table support tool calling when served with the correct parser. Qwen 3.6 specifically requires `--tool-call-parser qwen3_coder` for Hermes/OpenClaw-style agents.

### Can I use these recipes on non-NVIDIA hardware?
The Docker images and Marlin backend are NVIDIA-specific. For x86_64 workstations with 96–128 GB of GPU memory, use the equivalent x86_64 vLLM/TRT-LLM images; the launch flags remain largely the same.

---

## Author and related benchmarks

This repository is maintained by **Cristian Tala** as part of a broader set of AI benchmark and optimization projects:

- [`ctala/ai-benchmarks-alternativos`](https://github.com/ctala/ai-benchmarks-alternativos) — Comparative AI benchmarks covering cloud, local and edge deployment scenarios, focused on choosing the right model and infrastructure for each case.
- [benchmarks.cristiantala.com](https://benchmarks.cristiantala.com/) — Published benchmark reports and recommendations.

The work in this repo focuses specifically on local, high-memory edge AI hardware such as the NVIDIA DGX Spark.

---

## License

MIT. See [LICENSE](/local-llm-agentic-workflows/LICENSE).
