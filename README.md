# Local LLM Agentic Workflows

A practical guide, reproducible benchmarks and ready-to-run Docker recipes for running the best local LLMs on high-memory edge AI hardware such as the **NVIDIA DGX Spark** (GB10 Grace Blackwell, 128 GB unified memory) and other 96–128 GB edge AI workstations.

**Live site:** [ctala.github.io/local-llm-agentic-workflows](https://ctala.github.io/local-llm-agentic-workflows/)

## Quick links

- [Results](https://ctala.github.io/local-llm-agentic-workflows/results/)
- [Setup guide](https://ctala.github.io/local-llm-agentic-workflows/setup/)
- [Agent integration (Hermes / OpenClaw / Opencode)](https://ctala.github.io/local-llm-agentic-workflows/agents/)

## What this covers

- **Models**: Gemma 4, Qwen 3.6, NVIDIA Nemotron 3 (Nano, Super, Omni).
- **Engines**: vLLM and TensorRT-LLM.
- **Quantization**: NVFP4/Marlin, FP8 KV cache, BF16.
- **Agent frameworks**: Hermes, OpenClaw, Opencode, LiteLLM, Open WebUI.
- **Use cases**: chatbots, coding assistants, multi-turn tool calling and multimodal agents.

See the live site for the full guide, benchmark tables and copy-paste launch scripts.

## Related work

- [`ctala/ai-benchmarks-alternativos`](https://github.com/ctala/ai-benchmarks-alternativos) — Comparative AI benchmarks covering cloud, local and edge deployment.
- [benchmarks.cristiantala.com](https://benchmarks.cristiantala.com/) — Published benchmark reports and recommendations.

## License

MIT. See [LICENSE](./LICENSE).
