# Local LLM Agentic Workflows

A practical guide, reproducible benchmarks and ready-to-run Docker recipes for running the best local LLMs on high-memory edge AI hardware such as the **NVIDIA DGX Spark** (GB10 Grace Blackwell, 128 GB unified memory) and other 96–128 GB edge AI workstations.

**Live site:** [ctala.github.io/local-llm-agentic-workflows](https://ctala.github.io/local-llm-agentic-workflows/)

## Quick links

- [Results](https://ctala.github.io/local-llm-agentic-workflows/results/)
- [Setup guide](https://ctala.github.io/local-llm-agentic-workflows/setup/)
- [Agent integration (Hermes / OpenClaw / Opencode)](https://ctala.github.io/local-llm-agentic-workflows/agents/)
- [Full local stack](https://ctala.github.io/local-llm-agentic-workflows/stack/)

## Quick answers

> ⚠️ **Comparación apples-to-apples.** Todos los números son del mismo Spark, mismos prompts, mismo método de medición. La columna **single-stream** mide 1 usuario con 1 request; la columna **@c=8 aggregate** mide 8 requests concurrentes. Single-stream warm = primer request después del cold start (TTFT ~0.24 s, prefix cache poblado).
>
> "70-76 tok/s warm cache" del 27B con DSpark que aparece en otros benchmarks mide un escenario de edición muy específico con prefix cache al 100% caliente — NO comparable con uso real interactivo.
>
> **Total = parámetros totales del modelo · Activos = los que se ejecutan por token (importante para el bandwidth del decode).**

| Modelo | **Params (total / activos)** | **Single-stream warm (1 user)** | **Single-stream fresh (1 user, primer request)** | **Aggregate @c=8 (8 users)** | Trade-off principal |
|---|---|---:|---:|---:|---|
| **Qwen3.8-Flash-Next NVFP4 hybrid** | **176B / 6B** (MoE 512 routed top-10 + 1 shared + PLE 51B) | **36.66 tok/s** chat | ~22 tok/s (cold TTFT 2.7s) | **116.75 tok/s** | **Default 2026-09-01.** Mejor calidad (GSM8K 97.27%), multimodal completo, 262K contexto. Solo 6B activos = muy eficiente en bandwidth. |
| **Qwen 3.8 27B NVFP4 + DSpark k=14** | 27B / 27B (dense) | 13.93 tok/s chat (fresh) | 13.93 tok/s | 40.40 tok/s fresh / 182 warm | **Fallback lite.** Mejor concurrencia con prefix caching (253 @c=16 warm), pero pierde single-stream fresh. 27B activos pesan en bandwidth. |
| **Qwen 3.6 35B-A3B** (nvidia NVFP4, 1-seq/262K) | 35B / 3B (MoE top-K small) | **~76 tok/s** | ~76 tok/s | ~76 tok/s (limitado por max-num-seqs=1) | **Single-stream long-context champion** — sacrifica concurrencia (1-seq/262K, sin batching). Solo 3B activos, banda ultra-baja. |
| **Gemma 4 26B-A4B IT** (community patch) | 26B / 4B (MoE) | **~49.5 tok/s** | ~49.5 tok/s | n/a medido | Velocidad pura para agentes cortos. |
| **Qwen 3.6 35B-A3B** (RedHatAI) | 35B / 3B (MoE) | ~42.2 tok/s | ~42.2 tok/s | n/a medido | Stable fallback. |
| **Nemotron-3-Nano-Omni-30B-A3B** | 30B / 3B (MoE) | ~40.0 tok/s | ~40.0 tok/s | n/a medido | Multimodal oficial NVIDIA (text+image). |
| **Nemotron-3-Super-120B-A12B** | 120B / 12B (MoE) | ~14.7 tok/s | ~14.7 tok/s | n/a medido | Calidad máxima oficial, TRT-LLM only. |
| **Gemma 4 31B IT** | 31B / 31B (dense) | ~6.7 tok/s | ~6.7 tok/s | n/a medido | Solo si se necesita el dense. Banda saturada por 31B activos. |

**Resumen rápido**: para uso agentico en el Spark hoy (2026-09-01), **Qwen3.8-Flash-Next hybrid** es el mejor balance entre velocidad, calidad y multimodal. **Qwen 3.6 35B-A3B** sigue siendo el más rápido en single-stream long-context (3B activos es la clave), pero no aguanta concurrencia. **Qwen 3.8 27B + DSpark** gana en concurrencia pero pierde single-stream fresh.

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
