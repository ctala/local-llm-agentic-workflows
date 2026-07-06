# Local LLMs for Agentic Workflows on NVIDIA DGX Spark

Benchmarks and deployment recipes for running local LLMs on the **NVIDIA DGX Spark** (GB10 Grace Blackwell, ARM64/aarch64, 128 GB unified memory, sm_121), focused on agentic workflows with **Hermes**, **OpenClaw**, Open WebUI and n8n.

> **Critical GB10 limitation**: it has no native FP4 compute. NVFP4 checkpoints run through the **Marlin** backend (vLLM) or the PyTorch backend of TensorRT-LLM, decompressing FP4 → BF16 at runtime. This limits throughput compared to FP4-native GPUs (e.g. B200).

---

## Quick answer

For agents on DGX Spark, start with one of these three depending on your priority:

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

## Hardware / software base

| Component | Value |
|-----------|-------|
| Hardware | NVIDIA DGX Spark (GB10 Grace Blackwell) |
| CPU | 20 cores ARM64 (aarch64) |
| GPU | NVIDIA GB10 (sm_121) |
| Memory | 128 GB LPDDR5x unified (~121 GB usable) |
| Driver NVIDIA | 580.142 |
| CUDA | 13.0 |
| Docker | 29.2.1 |
| NVIDIA Container Toolkit | 1.19.0 |

---

## Repository layout

```
.
├── README.md                     # This file
├── RESULTS.md                    # Full benchmark tables and technical notes
├── SETUP.md                      # Detailed log of attempts, errors and fixes
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

## Results summary

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

## How this compares to other benchmarks

[MiaAI-Lab/Best-Local-Model_Agentic-Workflows_2026](https://github.com/MiaAI-Lab/Best-Local-Model_Agentic-Workflows_2026) evaluates agentic capability using `tool-eval-bench` (84 scenarios, 8 trials) on llama.cpp GGUF quantizations. Their top pick for Hermes Agent is **Qwen 3.6 35B A3B UD Q8_K_XL** (score 91.0), followed by **Qwen 3.6 27B NVFP4** (89.0) and **Qwopus 3.6 27B Coder MTP** (85.2).

This repository focuses on **throughput-optimized deployment** on DGX Spark using vLLM and TensorRT-LLM with FP8 KV cache and NVFP4/Marlin. We have not yet run `tool-eval-bench` quality scores, so the two benchmarks are complementary:

- Use MiaAI's work to pick the most capable model for your agent task.
- Use our recipes to deploy that model at the highest sustainable tok/s on GB10.

Candidates worth testing next from their ranking:

| Model | Why test | Expected trade-off |
|-------|----------|--------------------|
| Qwen 3.6 35B A3B Q8_K_XL (GGUF) | Highest agentic score in their benchmark | Likely slower than NVFP4/vLLM, but may improve reliability |
| Qwopus 3.6 27B Coder MTP | Fastest reliable tier in their ranking (~2.2 s/turn) | Could be a strong coding/agent model on Spark |
| Agents-A1 Q8_0 | Purpose-built agent model | Unknown quality/speed on GB10 |

---

## Important memory lesson

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

Based on throughput, memory headroom and tool-calling support:

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
- Test Qwen 3.6 35B A3B Q8_K_XL (GGUF) and Qwopus 3.6 27B Coder MTP from the MiaAI ranking.
- Fix audio decoding for Nemotron-3 Nano Omni.
- Add LiteLLM proxy recipes to expose multiple models on different ports.

---

## License

MIT. See [`LICENSE`](LICENSE).
