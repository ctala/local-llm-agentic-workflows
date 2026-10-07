# Local LLM Agentic Workflows on the NVIDIA DGX Spark

Same-hardware benchmarks, tested `docker run` recipes and an agent deployment prompt for running local LLMs on the **NVIDIA DGX Spark** (GB10 Grace Blackwell, 128 GB unified memory, aarch64) and other 96–128 GB edge AI workstations.

**Live site:** [ctala.github.io/local-llm-agentic-workflows](https://ctala.github.io/local-llm-agentic-workflows/) · [Leer en español](README.es.md)

## Start here

- **[October 2026 benchmark](https://ctala.github.io/local-llm-agentic-workflows/benchmarks/)**: 23 configurations at 128K context × 4 concurrent users on vLLM 0.31.0 and NVIDIA NIM.
- **[Recipes](https://ctala.github.io/local-llm-agentic-workflows/recipes/)**: weight download and `docker run` for each configuration. Source YAML in [`recipes/`](recipes/).
- **[Agent deployment prompt](https://ctala.github.io/local-llm-agentic-workflows/deploy-with-agent/)**: paste it into Claude Code, Codex or OpenCode on the Spark and it deploys a recipe for you. Also in [`recipes/AGENT-PROMPT.md`](recipes/AGENT-PROMPT.md).

## Quick answers (October 2026, vLLM 0.31.0, 128K context × 4 users, FP8 KV cache)

| Need | Pick | 1 user | 4 users (total) | Memory |
|---|---|---:|---:|---:|
| Fastest overall, multimodal, tools | Qwen3.6-35B-A3B NVFP4 + MTP | 102 tok/s | 240 tok/s | ~56 GB |
| Small and fast | GPT-OSS-20B | 96 tok/s | 151 tok/s | ~26 GB |
| Agents, long context | Nemotron 3.5 Lightning 30B-A3B + DSpark | 94 tok/s | 213 tok/s | ~44 GB |
| Image, audio and video input | Nemotron 3 Nano Omni 30B-A3B | 59 tok/s | 185 tok/s | ~44 GB |
| Image and audio, Google model | Gemma 4 26B-A4B + MTP drafter | 57 tok/s | 187 tok/s | ~45 GB |
| Large reasoning model | GPT-OSS-120B | 44 tok/s | 125 tok/s | ~81 GB |
| Largest that fits | Nemotron 3 Super 120B-A12B + MTP | 23 tok/s | 57 tok/s | ~95 GB |

What we learned:

- **Speculative decoding pays off for agents and code.** Qwen3.6 + MTP goes from ~79 to ~125–129 tok/s on code and tool-call JSON (84–90% acceptance), but only to ~95 tok/s on free-form writing.
- **Not every drafter helps.** Eagle3 made GPT-OSS slower (20B: 96 → 59 tok/s), DFlash was slower than plain Qwen3.6, and DSpark barely helped Qwen3.8-27B on fresh text.
- **NIM vs vLLM:** with the same model, a plain vLLM 0.31.0 container matched or beat the official NIM. Some NIMs are amd64-only and do not run on the Spark.
- **Nemotron 3 Super 120B now runs on vLLM 0.31.0** (earlier images failed; see the earlier results).
- **Always set `--gpu-memory-utilization`.** On unified memory, vLLM's 0.9 default reserves ~110 GB even for a 20 GB model.
- **Llama 3.3 70B does not fit at 128K × 4** (dense KV cache, ~133 GB). Use 1–2 users.

## Run a recipe

```bash
git clone https://github.com/ctala/local-llm-agentic-workflows.git && cd local-llm-agentic-workflows
pip install pyyaml "huggingface_hub[cli]"
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir ~/vllm/qwen3.6-35b-a3b-nvfp4-nvidia
python3 scripts/recipe_to_docker.py recipes/qwen36-35b-a3b-nvfp4-rapida-v031-mtp.yaml   # prints the docker run
```

Each recipe lists its `download` commands, the `measured` configuration (context, users, KV cache, `gpu_util`, speed) and the full `entry` (image, env, vLLM args). Pass `--ctx`, `--users`, `--kv` or `--gpu-util` to `recipe_to_docker.py` for other configurations.

Measure your own endpoint with the same method:

```bash
python3 benchmarks/standard-round/standard_round.py --help   # Python stdlib only, any OpenAI-compatible endpoint
```

## Repository layout

| Path | What it is |
|---|---|
| [`recipes/`](recipes/) | Tested recipes (YAML) from the October 2026 round |
| [`scripts/recipe_to_docker.py`](scripts/recipe_to_docker.py) | Turns a recipe into a `docker run` command |
| [`benchmarks/standard-round/`](benchmarks/standard-round/) | Benchmark script: single user, 4 users, 46K prompt, stability, sustained load, speculative acceptance |
| [`scripts/run-*.sh`](scripts/) | Earlier launch scripts (vLLM and TensorRT-LLM, April–September 2026) |
| [`chat-templates/`](chat-templates/) | Chat templates needed by some checkpoints |
| [`hermes-plugins/`](hermes-plugins/), [`asr-server/`](asr-server/), [`web-extractor/`](web-extractor/) | Local agent stack: Hermes plugins, faster-whisper ASR, Firecrawl-compatible extraction |
| [`src/`](src/) | The Astro site published on GitHub Pages |

## Earlier results (April–September 2026)

The previous rounds compared vLLM and TensorRT-LLM with older images, different context sizes and 8 concurrent requests, so their numbers are not directly comparable with the October round. They are kept in the [Results](https://ctala.github.io/local-llm-agentic-workflows/results/) page and the [Setup log](https://ctala.github.io/local-llm-agentic-workflows/setup/). Agent integration (Hermes, OpenClaw, Opencode, LiteLLM) is in [Agents](https://ctala.github.io/local-llm-agentic-workflows/agents/) and the full self-hosted stack in [Stack](https://ctala.github.io/local-llm-agentic-workflows/stack/).

## Related work

- [`ctala/ai-benchmarks-alternativos`](https://github.com/ctala/ai-benchmarks-alternativos): comparative AI benchmarks covering cloud, local and edge deployment.
- [benchmarks.cristiantala.com](https://benchmarks.cristiantala.com/): published benchmark reports and recommendations.

## License

MIT. See [LICENSE](./LICENSE).
