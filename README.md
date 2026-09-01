# Local LLM Agentic Workflows

A practical guide, reproducible benchmarks and ready-to-run Docker recipes for running the best local LLMs on high-memory edge AI hardware such as the **NVIDIA DGX Spark** (GB10 Grace Blackwell, 128 GB unified memory) and other 96–128 GB edge AI workstations.

**Live site:** [ctala.github.io/local-llm-agentic-workflows](https://ctala.github.io/local-llm-agentic-workflows/)

## Quick links

- [Results](https://ctala.github.io/local-llm-agentic-workflows/results/)
- [Setup guide](https://ctala.github.io/local-llm-agentic-workflows/setup/)
- [Agent integration (Hermes / OpenClaw / Opencode)](https://ctala.github.io/local-llm-agentic-workflows/agents/)
- [Full local stack](https://ctala.github.io/local-llm-agentic-workflows/stack/)

## Quick answers

| Model | Decode speed | Best for |
|-------|--------------|----------|
| **Qwen3.8-Flash-Next NVFP4 hybrid** (vLLM `release/qwen38next` + 7 parches GB10) | **~37 tok/s** warm / 117 @c=8 aggregate | **Current default (2026-09-01).** Best quality (GSM8K 97.27%, AIME26 98.75%), MoE 176B (6B activos), multimodal texto+imagen+video, 262K contexto nativo (500K YaRN), tool calling `qwen3_coder`. |
| **Qwen 3.8 27B NVFP4 + DSpark k=14** (vLLM 0.27.1) | ~30 tok/s fresh / 70-76 warm / **253 @c=16** | Fallback lite. Excellent concurrencia con prefix caching caliente, multimodal texto+imagen+video. |
| **Qwen 3.6 35B-A3B** (nvidia NVFP4, vLLM nightly, `flashinfer` + Marlin) | **~76 tok/s** | Single-stream más rápido con long-context 262K (1-seq), 1 secuencia por sesión. |
| **Gemma 4 26B-A4B IT** (community patch) | **~49.5 tok/s** | Maximum raw speed for agents. |
| **Nemotron-3-Nano-Omni-30B-A3B** | **~40.0 tok/s** | Official NVIDIA multimodal (text + image). |
| **Qwen 3.6 35B-A3B** (RedHatAI) | **~42.2 tok/s** | Stable fallback if nvidia checkpoint is unavailable. |
| **Nemotron-3-Super-120B-A12B** | **~14.7 tok/s** | Quality-first, TRT-LLM only. |
| **Gemma 4 31B IT** | **~6.7 tok/s** | Dense model, only when quality justifies the speed cost. |

> Full benchmark tables and launch scripts are in the [Results](/local-llm-agentic-workflows/results/) page.

## What this covers

- **Models**: Gemma 4, **Qwen 3.8 (27B + Flash-Next)**, Qwen 3.6, NVIDIA Nemotron 3 (Nano, Super, Omni).
- **Engines**: vLLM and TensorRT-LLM.
- **Quantization**: NVFP4/Marlin, FP8 KV cache, BF16, **fp8-e4m3 hybrid** (NVFP4 experts + fp8 side layers).
- **Agent frameworks**: Hermes, OpenClaw, Opencode, LiteLLM, Open WebUI.
- **Use cases**: chatbots, coding assistants, multi-turn tool calling and multimodal agents.
- **Local ASR**: faster-whisper server for transcribing voice messages on the Spark.
- **Local web extraction**: fastCRW (Firecrawl-compatible) for scraping URLs without cloud APIs.

See the live site for the full guide, benchmark tables and copy-paste launch scripts.

## Related work

- [`ctala/ai-benchmarks-alternativos`](https://github.com/ctala/ai-benchmarks-alternativos) — Comparative AI benchmarks covering cloud, local and edge deployment.
- [benchmarks.cristiantala.com](https://benchmarks.cristiantala.com/) — Published benchmark reports and recommendations.

## License

MIT. See [LICENSE](./LICENSE).
