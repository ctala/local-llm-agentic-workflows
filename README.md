# Best Local LLMs for Agentic Workflows on DGX Spark & 96–128 GB Edge AI Workstations

Benchmarks and ready-to-run Docker recipes for deploying **local LLMs for agentic workflows** on high-memory edge AI hardware such as the **NVIDIA DGX Spark** (GB10 Grace Blackwell, ARM64/aarch64, 128 GB unified memory, sm_121) and other single-GPU workstations with **96–128 GB of unified or pooled memory**.

This repo focuses on **Hermes Agent**, **OpenClaw**, Open WebUI and n8n use cases: multi-turn tool calling, function calling, autonomous planning and multimodal agents. It compares **vLLM** and **TensorRT-LLM** deployments with **NVFP4/Marlin** and **FP8 KV cache**, measuring real decode throughput, memory usage and stability.

> **Critical edge-AI limitation**: the NVIDIA GB10 has no native FP4 compute. NVFP4 checkpoints run through the **Marlin** backend (vLLM) or the PyTorch backend of TensorRT-LLM, decompressing FP4 → BF16 at runtime. This limits throughput compared to FP4-native datacenter GPUs (e.g. B200), but still makes 30B–120B parameter models runnable on a single edge device.

---

## Quick answer

For agents on DGX Spark or equivalent 96–128 GB edge hardware, start with one of these three depending on your priority:

| Priority | Model | Framework | Decode tok/s | Memory | Tool calling | Multimodal |
|----------|-------|-----------|--------------|--------|--------------|------------|
| **Speed** | Gemma 4 26B-A4B IT (community patch) | vLLM | **~49.5** | ~22 GB | ✅ | ✅ image/video |
| **Quality/speed balance** | Qwen 3.6 35B-A3B | vLLM | **~42.2** | ~22 GB | ✅ | ✅ image/video |
| **Official NVIDIA multimodal** | Nemotron-3 Nano Omni 30B-A3B | vLLM | **~40.0** | ~40 GB | ✅ | ✅ image |

For pure quality when speed is less important:

| Model | Framework | Decode tok/s | Memory | Notes |
|-------|-----------|--------------|--------|-------|
| Nemotron-3 Super 120B-A12B | TensorRT-LLM | ~14.7 | ~110 GB | Best official quality, but slow |
| Qwen 3.6 35B-A3B (custom MLP-only NVFP4) | TensorRT-LLM | ~34.4 | ~41 GB | Official stack, requires manual quantization |

---

## Target hardware

The recipes were tested on the **NVIDIA DGX Spark** and should work on any ARM64 or x86_64 single-GPU system with similar memory characteristics:

| Component | DGX Spark value | Typical compatible edge AI workstation |
|-----------|-----------------|----------------------------------------|
| CPU | 20 cores ARM64 (aarch64) | ARM64 or x86_64, 16+ cores |
| GPU | NVIDIA GB10 (sm_121) | Single GPU with 96–128 GB accessible memory |
| Memory | 128 GB LPDDR5x unified (~121 GB usable) | 96–128 GB unified, pooled or host memory |
| Driver NVIDIA | 580.142 | Recent 570+ series |
| CUDA | 13.0 | CUDA 12.6+ / 13.0 |
| Docker | 29.2.1 | Docker 24+ with NVIDIA Container Toolkit |
| NVIDIA Container Toolkit | 1.19.0 | 1.14+ |

> Most Docker images used here are ARM64 manifests. If you run x86_64 hardware, replace the image tags with the equivalent x86_64 builds from NVIDIA and vLLM.

---

## Repository layout

```
.
├── README.md                     # This file
├── RESULTS.md                    # Full benchmark tables and technical notes (English)
├── RESULTS.es.md                 # Spanish version of RESULTS.md
├── SETUP.md                      # Detailed log of attempts, errors and fixes (English)
├── SETUP.es.md                   # Spanish version of SETUP.md
├── LICENSE                       # MIT
├── scripts/                      # Docker launch recipes and helpers
│   ├── run-gemma4-26b-a4b.sh
│   ├── run-qwen36-35b-a3b.sh
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
3. Wait for the model to load.
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

| Model | Checkpoint | Framework | Decode tok/s | Memory | Tool calling | Multimodal | Recommendation |
|-------|------------|-----------|--------------|--------|--------------|------------|----------------|
| **Gemma 4 26B-A4B** | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` + patch | vLLM | **~49.5** | ~22 GB | ✅ | ✅ image/video | **Fastest for text agents** |
| **Qwen 3.6 35B-A3B** | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | vLLM | **~42.2** | ~22 GB | ✅ | ✅ image/video | **Best quality/speed trade-off** |
| Qwen 3.6 35B-A3B | Custom MLP-only NVFP4 | TRT-LLM | ~34.4 | ~41 GB | ✅ | ❌ text only | Official NVIDIA stack |
| Nemotron-3 Nano 30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | TRT-LLM | ~28.8 | ~118 GB | ✅ | ❌ text only | Official dense model; leaves little free memory |
| Nemotron-3 Nano 30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | vLLM | ~28.3 | ~72 GB | ✅ | ❌ text only | vLLM alternative; stop other GPU services first |
| **Nemotron-3 Super 120B-A12B** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | TRT-LLM | **~14.7** | ~110 GB | ✅ | ❌ text only | Best official quality; **not viable with vLLM on GB10** |
| **Nemotron-3 Nano Omni 30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | vLLM | **~40.0** | ~40 GB | ✅ | ✅ image | **Best official multimodal option** |
| Gemma 4 31B | `nvidia/Gemma-4-31B-IT-NVFP4` | vLLM | ~6.7 | ~31 GB | ✅ | ✅ image/video | Use only if you need the dense variant |

See [`RESULTS.md`](RESULTS.md) for full tables, command recipes and error analysis.

---

## How this compares to other agentic LLM benchmarks

[MiaAI-Lab/Best-Local-Model_Agentic-Workflows_2026](https://github.com/MiaAI-Lab/Best-Local-Model_Agentic-Workflows_2026) evaluates agentic capability using `tool-eval-bench` (84 scenarios, 8 trials) on llama.cpp GGUF quantizations for DGX Spark and similar 96–128 GB units. Their top pick for Hermes Agent is **Qwen 3.6 35B A3B UD Q8_K_XL** (score 91.0), followed by **Qwen 3.6 27B NVFP4** (89.0) and **Qwopus 3.6 27B Coder MTP** (85.2).

This repository focuses on **throughput-optimized local deployment** using vLLM and TensorRT-LLM with FP8 KV cache and NVFP4/Marlin. We have not yet run `tool-eval-bench` quality scores, so the two benchmarks are complementary:

- Use MiaAI's work to pick the most capable model for your agent task.
- Use our recipes to deploy that model at the highest sustainable tok/s on GB10-class hardware.

### Candidates worth testing next for agentic quality

| Model | Why test | Expected trade-off |
|-------|----------|--------------------|
| **Qwen 3.6 35B A3B Q8_K_XL (GGUF)** | Highest agentic score in their benchmark | Likely slower than NVFP4/vLLM, but may improve reliability |
| **Qwopus 3.6 27B Coder MTP** | Fastest reliable tier in their ranking (~2.2 s/turn) | Could be a strong coding/agent model on Spark-class hardware |
| **Agents-A1 Q8_0** | Purpose-built agent model | Unknown quality/speed on GB10 |

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

## Recommended models for Hermes / OpenClaw

Based on throughput, memory headroom and tool-calling support on DGX Spark and equivalent 96–128 GB edge AI hardware:

1. **Default production agent** → **Qwen 3.6 35B-A3B** (`RedHatAI/Qwen3.6-35B-A3B-NVFP4`) on vLLM.
   - ~42 tok/s, ~22 GB memory, excellent tool-calling, image/video support.
2. **Maximum speed** → **Gemma 4 26B-A4B IT** (community patch) on vLLM.
   - ~49.5 tok/s, ~22 GB memory. Requires the community patch.
3. **Official NVIDIA multimodal** → **Nemotron-3 Nano Omni 30B-A3B** on vLLM.
   - ~40 tok/s, ~40 GB memory, image support. Audio decoding needs more work.
4. **Quality over speed** → **Nemotron-3 Super 120B-A12B** on TensorRT-LLM.
   - ~14.7 tok/s, ~110 GB memory. Use only with TRT-LLM; vLLM is unstable on GB10.

---

## Next steps

- Run `tool-eval-bench` on our fastest recipes to measure agentic quality, not just speed.
- Test Qwen 3.6 35B A3B Q8_K_XL (GGUF) and Qwopus 3.6 27B Coder MTP from the MiaAI ranking on DGX Spark.
- Fix audio decoding for Nemotron-3 Nano Omni.
- Add LiteLLM proxy recipes to expose multiple models on different ports.

---

## Author and related benchmarks

This repository is maintained by **Cristian Tala** as part of a broader set of AI benchmark and optimization projects:

- [`ctala/ai-benchmarks-alternativos`](https://github.com/ctala/ai-benchmarks-alternativos) — Comparative AI benchmarks covering cloud, local and edge deployment scenarios.
- [benchmarks.cristiantala.com](https://benchmarks.cristiantala.com/) — Published benchmark reports and recommendations for choosing the right model and infrastructure in each case.

The work here focuses specifically on local, high-memory edge AI hardware such as the NVIDIA DGX Spark, while the projects above cover a wider range of cloud and on-premise alternatives.

---

## License

MIT. See [`LICENSE`](LICENSE).
