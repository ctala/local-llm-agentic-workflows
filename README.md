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
| **Qwen 3.6 35B-A3B** (nvidia NVFP4, vLLM nightly, `flashinfer` + Marlin) | **~76 tok/s** | Best quality/speed balance, robust tool calling, 196K context. Marlin NVFP4 backend for GB10/SM121, FP8 KV cache, 1 sequence for long-context headroom. |
| **Gemma 4 26B-A4B IT** (community patch) | **~49.5 tok/s** | Maximum raw speed for agents. |
| **Nemotron-3-Nano-Omni-30B-A3B** | **~40.0 tok/s** | Official NVIDIA multimodal (text + image). |
| **Qwen 3.6 35B-A3B** (RedHatAI) | **~42.2 tok/s** | Stable fallback if nvidia checkpoint is unavailable. |
| **Nemotron-3-Super-120B-A12B** | **~14.7 tok/s** | Quality-first, TRT-LLM only. |
| **Gemma 4 31B IT** | **~6.7 tok/s** | Dense model, only when quality justifies the speed cost. |

> Full benchmark tables and launch scripts are in the [Results](/local-llm-agentic-workflows/results/) page.

## What this covers

- **Models**: Gemma 4, Qwen 3.6, NVIDIA Nemotron 3 (Nano, Super, Omni).
- **Engines**: vLLM and TensorRT-LLM.
- **Quantization**: NVFP4/Marlin, FP8 KV cache, BF16.
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
