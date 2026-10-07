---
title: 'DGX Spark LLM Benchmark (October 2026): 23 Models at 128K Context × 4 Users'
description: 'Same-hardware benchmark of 23 local LLM configurations on the NVIDIA DGX Spark (GB10, 128 GB): Qwen3.6, Qwen3.8, Nemotron 3 (Lightning, Omni, Super), GPT-OSS, Gemma 4, Llama 3.1 and Qwen3
  with vLLM 0.31, NIM, MTP, DSpark, DFlash and Eagle3.'
keywords:
- DGX Spark benchmark
- best LLM for DGX Spark
- fastest model DGX Spark
- GB10 vLLM
- Qwen3.6 MTP
- NVFP4
- NIM vs vLLM
- speculative decoding DGX Spark
- 128K context
- Nemotron Super DGX Spark
- GPT-OSS DGX Spark
- Gemma 4 DGX Spark
---

# DGX Spark LLM Benchmark — October 2026 (128K context × 4 users)

> **Short answer:** the fastest well-rounded model on a single NVIDIA DGX Spark is **Qwen3.6-35B-A3B NVFP4 on vLLM 0.31.0 with MTP speculative decoding**: **102.3 tok/s** for one user and **240.0 tok/s** total with 4 users, at 128K context, using ~56 GB. It sustained 218.0 tok/s for 15 minutes with zero failed requests.
>
> Español: [Benchmark DGX Spark (octubre 2026)](/local-llm-agentic-workflows/benchmarks.es/) · Ready-to-run configs: [Recipes](/local-llm-agentic-workflows/recipes/) · Let an AI agent deploy one: [Agent prompt](/local-llm-agentic-workflows/deploy-with-agent/)

Last updated: 2026-10-06. Hardware: one NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory, ~121 GB usable, aarch64). Every model ran with **the same configuration**: 128K tokens of context per conversation, 4 concurrent users, FP8 KV cache, and a memory budget computed for that size.

## How we measured

1. **1 user:** 3 streamed requests with a fixed ~200-word prompt; median time to first token (TTFT) and decode tokens/s.
2. **4 users:** 4 simultaneous requests; total tokens/s across users.
3. **Long prompt:** one ~46K-token prompt (prefill speed).
4. **Stability:** 3 rounds of 4 concurrent users plus the long prompt; any server error, failed request or container restart marks the model as unstable.
5. **Extended check** for the top pick: varied prompts (code, JSON/tool calls, long summary, Spanish writing) at temperature 0 and 0.7, then 15 minutes of sustained load.

The benchmark script is in [`benchmarks/standard-round/standard_round.py`](https://github.com/ctala/local-llm-agentic-workflows/blob/main/benchmarks/standard-round/standard_round.py) (standard library only) and works against any OpenAI-compatible server.

## Results (sorted by single-user speed)

| Model | Params (total / active) | Engine | Speculative | 1 user tok/s | 4 users tok/s (total) | TTFT | 46K-token prompt | Load | Memory | Stable |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **Qwen3.6-35B-A3B** | 35B / 3B MoE | vLLM 0.31.0 | MTP k=2 | **102.3** | 240.0 | 106 ms | 9.5 s | 2.7 min | 55.5 GB | yes |
| **GPT-OSS-20B** | 21B / 3.6B MoE | vLLM 0.31.0 | — | **96.2** | 150.6 | 3319 ms | 8.9 s | 2.9 min | 25.7 GB | yes |
| **Nemotron 3.5 Lightning 30B-A3B** | 30B / 3B MoE (hybrid Mamba) | vLLM 0.31.0 | DSpark k=3 | **94.4** | 213.2 | 95 ms | 9.5 s | 4.5 min | 43.5 GB | yes |
| **Qwen3.6-35B-A3B** | 35B / 3B MoE | vLLM 0.31.0 | — | **78.1** | 196.6 | 63 ms | 9.6 s | 2.1 min | 47.0 GB | yes |
| **Nemotron 3.5 Lightning 30B-A3B** | 30B / 3B MoE (hybrid Mamba) | vLLM 0.31.0 | — | **70.3** | 170.7 | 100 ms | 10.2 s | 3.9 min | 45.0 GB | yes |
| **Qwen3.6-35B-A3B** | 35B / 3B MoE | vLLM 0.31.0 | DFlash k=11 | **67.8** | 148.6 | 95 ms | 9.5 s | 3.4 min | 71.8 GB | yes |
| **GPT-OSS-20B** | 21B / 3.6B MoE | NIM 2.0.9 | — | **62.8** | 151.4 | 1650 ms | 13.8 s | 4.8 min | 33.0 GB | yes |
| **GPT-OSS-20B** | 21B / 3.6B MoE | vLLM 0.31.0 | Eagle3 k=3 | **59.4** | 144.7 | 1314 ms | 16.4 s | 2.7 min | 38.4 GB | yes |
| **Nemotron 3 Nano Omni 30B-A3B** | 30B / 3B MoE (hybrid Mamba) | vLLM 0.31.0 | — | **58.8** | 184.6 | 123 ms | 8.6 s | 4.1 min | 43.6 GB | yes |
| **Gemma 4 26B-A4B** | 26B / 4B MoE | vLLM 0.31.0 | MTP drafter k=3 | **56.5** | 186.6 | 119 ms | 29.7 s | 4.8 min | 45.1 GB | yes |
| **Nemotron 3 Nano Omni 30B-A3B** | 30B / 3B MoE (hybrid Mamba) | NIM 2.0.13 | — | **56.0** | 161.3 | 148 ms | 11.4 s | 6.7 min | 62.7 GB | yes |
| **GPT-OSS-120B** | 117B / 5.1B MoE | vLLM 0.31.0 | — | **44.4** | 124.9 | 1921 ms | 14.5 s | 9.3 min | 80.5 GB | yes |
| **GPT-OSS-120B** | 117B / 5.1B MoE | vLLM 0.31.0 | Eagle3 k=3 | **40.5** | 91.8 | 2143 ms | 26.1 s | 9.2 min | 87.5 GB | yes |
| **Gemma 4 26B-A4B** | 26B / 4B MoE | vLLM 0.31.0 | — | **30.8** | 114.0 | 91 ms | 26.0 s | 4.3 min | 39.9 GB | yes |
| **Gemma 4 26B-A4B** | 26B / 4B MoE | vLLM gemma4 image | — | **30.7** | 108.5 | 82 ms | 25.4 s | 3.8 min | 39.4 GB | yes |
| **Qwen3.8-Flash-Next** | 176B / 6B MoE | patched vLLM (qwen38-flash-dgx) | MTP k=2 | **26.8** | 52.5 | 259 ms | 30.2 s | 13.8 min | 95.7 GB | yes |
| **Llama 3.1 8B Instruct** | 8B dense | NIM 2.0.9 | — | **26.4** | 102.3 | 81 ms | 15.5 s | 5.1 min | 33.6 GB | yes |
| **Qwen3-8B** | 8B dense | vLLM 0.31.0 (YaRN x4) | — | **23.2** | 95.7 | 84 ms | 21.7 s | 2.8 min | 55.5 GB | yes |
| **Nemotron 3 Super 120B-A12B** | 120B / 12B MoE (hybrid Mamba) | vLLM 0.31.0 | MTP k=3 | **23.0** | 57.0 | 424 ms | 25.8 s | 13.4 min | 95.2 GB | yes |
| **Qwen3.8-27B** | 27B dense | vLLM 0.31.0 | MTP k=2 | **19.0** | 67.0 | 233 ms | 41.9 s | 7.7 min | 59.5 GB | yes |
| **Nemotron 3 Super 120B-A12B** | 120B / 12B MoE (hybrid Mamba) | vLLM 0.31.0 | — | **16.1** | 42.8 | 326 ms | 22.8 s | 10.3 min | 94.3 GB | yes |
| **Qwen3.8-27B** | 27B dense | vLLM 0.31.0 | DSpark k=14 | **12.5** | 33.9 | 256 ms | 43.6 s | 8.8 min | 63.6 GB | yes |
| **Qwen3.8-27B** | 27B dense | vLLM 0.31.0 | — | **11.0** | 39.6 | 202 ms | 47.7 s | 6.6 min | 47.0 GB | yes |

TTFT for GPT-OSS includes its reasoning phase. "Memory" is how much free memory the Spark lost when the model loaded.

## Speculative decoding: real gains by workload (Qwen3.6-35B + MTP)

MTP never lowers quality (the main model verifies every drafted token); what changes is how many drafts are accepted, and that depends on the content.

| Workload | Without MTP | MTP, temp 0 | MTP, temp 0.7 | MTP acceptance |
|---|---:|---:|---:|---:|
| Code | 78.6 | **125.2** | 117.1 | 84.2–77.4 % |
| JSON / tool calls | 79.2 | **129.3** | 120.3 | 90.2–82.6 % |
| Long summary | 75.0 | **106.2** | 102.8 | 70.4–67.6 % |
| Spanish writing | 78.4 | **95.1** | 92.4 | 51.3–49.3 % |
| **15 min sustained, 4 users** | 170.9 | **218.0** (mixed) | | 69.3 % |

- **Agents and code benefit most** (+55–65%). Free-form writing gains least (+20%).
- **Gemma 4 + its official MTP drafter** (`google/gemma-4-26B-A4B-it` assistant model) nearly doubled: 30.8 → 56.5 tok/s.
- **DSpark** helped Nemotron 3.5 Lightning (70.3 → 94.4) but barely helped Qwen3.8-27B on fresh text (11.0 → 12.5); its big wins show up in repetitive code editing with a warm prefix cache.
- **Eagle3** slowed GPT-OSS down on the Spark (96.2 → 59.4 on 20B, 44.4 → 40.5 on 120B): the drafter forces the Triton attention backend instead of FlashInfer and costs more than it saves. Run GPT-OSS without it.
- **DFlash** on top of the fast Qwen3.6 recipe was *slower* than plain decoding (67.8 vs 78.1) and needed much more memory.

## NIM vs your own vLLM container

| Model | NIM (1 user / 4 users) | vLLM 0.31.0 (1 user / 4 users) |
|---|---|---|
| GPT-OSS-20B | 62.8 / 151.4 | **96.2** / 150.6 |
| Nemotron 3 Nano Omni | 56.0 / 161.3 | **58.8** / 184.6 |

NIMs are convenient and pass vLLM flags through, but on the Spark a plain, recent vLLM image was faster. Check the image architecture before pulling: `nvcr.io/nim/qwen/qwen3.8-27b` (`latest` and `2.1.1-variant`) and `nvcr.io/nim/qwen/qwen3-32b` were **amd64-only** on 2026-10-06.

## What did not fit or did not run

- **Llama 3.3 70B NVFP4:** does not fit at 128K × 4 (dense, ~160 KB of KV cache per token → ~133 GB). Run it with 1–2 users.
- **Gemma 4 MTP on the older `gemma4` vLLM image:** unsupported; works on vLLM 0.31.0.
- **Qwen3.8-Flash-Next:** needs a patched vLLM image; it runs (26.8 tok/s) but uses ~91 GB and loads in ~14 minutes.

## Recommendations by use case

| Use case | Pick | Why |
|---|---|---|
| Agents / tool calling (e.g. Hermes, OpenClaw) | **Qwen3.6-35B + MTP** | Fastest overall, 80–90% MTP acceptance on JSON/tool calls, image + video input |
| Coding assistant | **Qwen3.6-35B + MTP**; second opinion with **GPT-OSS-120B** | Fast iteration plus an independent model family to review |
| Multimodal with audio/video | **Nemotron 3 Nano Omni (vLLM)** | Text, image, audio and video in one model; fastest long-prompt prefill |
| Fast general model, low memory | **Nemotron 3.5 Lightning + DSpark** or **GPT-OSS-20B** | 94.4 / 96.2 tok/s single user |
| Deep reasoning, long documents, fewer users | **Nemotron 3 Super 120B + MTP** | Largest model that fits with 128K × 4; tiny KV cache thanks to its hybrid Mamba design |

## FAQ

### What is the fastest LLM on the NVIDIA DGX Spark?

In our October 2026 round at 128K context and 4 concurrent users, Qwen3.6-35B-A3B NVFP4 on vLLM 0.31.0 with MTP speculative decoding was the fastest: 102.3 tok/s for one user and 240.0 tok/s total with 4 users, using about 56 GB of the 128 GB unified memory.

### Is NVIDIA NIM faster than vLLM on the DGX Spark?

Not in our tests. With the same model, a plain vLLM 0.31.0 container was faster than the official NIM: GPT-OSS-20B ran at 96.2 vs 62.8 tok/s for one user, and Nemotron 3 Nano Omni at 58.8 vs 56.0 tok/s. Some NIMs (Qwen3-32B, Qwen3.8-27B) are only published for amd64 and do not run on the Spark.

### Does speculative decoding (MTP, DSpark, Eagle3) help on the DGX Spark?

Yes, mostly for agent and code workloads. Qwen3.6 with MTP went from ~78 to ~125 tok/s on code and JSON/tool calls (80-90% acceptance) but only to ~95 tok/s on free-form writing (~50% acceptance). Gemma 4 with its official MTP drafter nearly doubled (31 to 57 tok/s). DSpark helped Nemotron 3.5 Lightning (+34%) but barely helped Qwen3.8-27B on fresh text. Eagle3 made GPT-OSS slower (96 to 59 tok/s on 20B), so skip it there.

### Can I run Llama 3.3 70B with 128K context and 4 users on a DGX Spark?

No. Llama 3.3 70B is dense with 80 full-attention layers, so its KV cache needs about 160 KB per token: four 128K conversations plus the weights need roughly 133 GB, more than the Spark's 121 GB of usable unified memory. Use fewer users or a shorter context.

### Which model should I use for agents, coding, content or research on the DGX Spark?

Agents (tool calling, low latency): Qwen3.6-35B + MTP. Coding: Qwen3.6-35B + MTP, with GPT-OSS-120B as an independent second opinion. Multimodal with audio/video: Nemotron 3 Nano Omni (vLLM). Long, careful reasoning with fewer users: Nemotron 3 Super 120B + MTP.
