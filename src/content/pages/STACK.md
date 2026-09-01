---
title: "Fully Local Agent Stack for DGX Spark"
description: "Complete self-hosted stack for running autonomous agents locally on NVIDIA DGX Spark: vLLM, LiteLLM, Hermes, SearXNG and faster-whisper ASR."
keywords:
  - DGX Spark
  - local agent stack
  - self-hosted AI
  - vLLM
  - LiteLLM
  - Hermes
  - SearXNG
  - faster-whisper
  - local ASR
  - local web search
  - Qwen3.8-Flash-Next
  - MoE
---

# Fully Local Agent Stack for DGX Spark

This is the complete self-hosted stack we run on the **NVIDIA DGX Spark** to power local autonomous agents without relying on cloud APIs for inference, search or speech recognition.

Every service below runs on the Spark itself or is reachable from other machines on the LAN through LiteLLM.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Client layer                            │
│  Hermes (CLI/TUI) · Opencode · Open WebUI · n8n             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     LiteLLM proxy                            │
│     OpenAI-compatible gateway · port 4000 · 0.0.0.0         │
│  Routes to vLLM, exposes models, handles thinking toggle     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      vLLM inference                          │
│    Qwen3.8-Flash-Next (default) · Qwen 3.8 27B · Qwen 3.6   │
│    Gemma 4 · Nemotron 3 · port 8001                         │
└─────────────────────────────────────────────────────────────┘

Auxiliary services:
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  SearXNG         │  │  faster-whisper  │  │  fastCRW         │  │  Browser / file  │
│  local search    │  │  ASR server      │  │  web extract     │  │  tools           │
│  port 8080       │  │  port 8001       │  │  port 3000       │  │  (optional)      │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Services

### 1. vLLM inference server

Serves the active LLM with an OpenAI-compatible API. Multiple model stacks can be installed; only one runs at a time because they share the unified memory pool.

| Property | Value (Qwen3.8-Flash-Next default) | Value (Qwen 3.8 27B fallback) | Value (Qwen 3.6 35B-A3B legacy) |
|----------|-----------------------------------|-------------------------------|--------------------------------|
| **URL** | `http://localhost:8001/v1` | `http://localhost:8001/v1` | `http://localhost:8001/v1` |
| **Served name** | `qwen3.8-flash-next` | `qwen3.8-27b-nvfp4` | `qwen3.6-35b-a3b` |
| **Checkpoint** | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `unsloth/Qwen3.8-27B-NVFP4` | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| **Container** | `qwen38-flash-dgx` (vLLM `release/qwen38next` + 7 parches GB10) | `vllm/vllm-openai:v0.27.1-aarch64` + DSpark k=14 | `vllm/vllm-openai:nightly` |
| **Launch script** | `scripts/run-qwen38-flash-next.sh` | `scripts/run-qwen38-27b-nvfp4-dspark.sh` | `scripts/run-qwen36-35b-a3b.sh` |
| **Systemd unit** | `qwen38-flash-next-vllm.service` (enabled by default) | `qwen38-nvfp4-dspark-vllm.service` (disabled, fallback) | `qwen36-vllm.service` (disabled, legacy) |
| **Max context** | 262,144 tokens (500K YaRN) | 262,144 tokens (1M YaRN) | 262,144 tokens |
| **Tool calling** | `--tool-call-parser qwen3_coder` | `--tool-call-parser qwen3_xml` | `--tool-call-parser qwen3_coder` |
| **Reasoning parser** | `qwen3` | `qwen3` | (omit: nvidia checkpoint does not emit `<think>` tags) |
| **Multimodal** | Text + image + video | Text + image + video | Text + image + video |
| **Memory** | ~98 GB | ~105 GB | ~22 GB (1-seq/262K config) |
| **Cold start** | ~14 min | ~5-8 min | ~5-8 min |

**Switch models by stopping one systemd unit and starting another** (one model at a time — they share the unified pool).

```bash
# Stop current default (Qwen3.8-Flash-Next) and start the 27B fallback
systemctl --user stop qwen38-flash-next-vllm.service
systemctl --user start qwen38-nvfp4-dspark-vllm.service
```

---

### 2. LiteLLM proxy

Unified gateway that exposes vLLM (and optional cloud fallbacks) under one OpenAI-compatible endpoint. Required when multiple clients share the backend or when accessing the Spark from another machine.

| Property | Value |
|----------|-------|
| **URL** | `http://0.0.0.0:4000/v1` |
| **Config** | `~/litellm/config.yaml` |
| **Service** | `litellm-proxy.service` |
| **Models exposed (active in 2026-09)** | `qwen3.8-flash-next-vllm` (default), 6 Ollama models (`qwen3.5-9b`, `phi4`, `llama3.1-8b`, `qwen3.5-35b`, `gemma4-31b`, `nemotron3-33b`), `ollama/*` wildcard, `minimax-m3`, `minimax-m2.7`, `nomic-embed-text` |
| **Legacy models (in config for reference, require stopping default)** | `qwen3.8-27b-nvfp4-vllm`, `qwen3.6-35b-heretic-vllm` — both point to the same `:8001` after the default swap |
| **Auth** | Disabled (`disable_user_auth: true`); any non-empty `Authorization` header works |

Restart after editing the config:

```bash
systemctl --user restart litellm-proxy.service
```

---

### 3. Hermes agent

Primary agent framework. Connected to LiteLLM so it can switch models and toggle thinking.

| Property | Value |
|----------|-------|
| **Config** | `~/.hermes/config.yaml` |
| **Provider** | `litellm-local` |
| **Default model** | `qwen3.8-flash-next-vllm` (since 2026-09-01) |
| **Reasoning overrides** | `qwen3.8-flash-next-vllm: low`; `qwen3.8-27b-nvfp4-vllm: low` (fallback) |
| **Context length** | 237,000 (forced in config) |
| **Service** | `hermes-gateway.service` |
| **Hermes plugin** | `~/.hermes/plugins/model-providers/litellm-local/` (maps Hermes reasoning levels to Qwen `enable_thinking` + `reasoning_effort`; advertises `none`, `low`, `medium`, `xhigh`; quietly maps `minimal`, `high`, `max`, `ultra` to the nearest Qwen native level) |

See [Agents](/local-llm-agentic-workflows/agents/) for the full config and tool-calling notes.

---

### 4. Opencode

Editor-native agent client (the one driving this repo's CI). Configured to point at the same LiteLLM proxy.

| Property | Value |
|----------|-------|
| **Config** | `~/.config/opencode/opencode.json` |
| **Default model** | `spark-litellm/qwen3.8-flash-next-vllm` (since 2026-09-01) |
| **Fallback** | `spark-litellm/qwen3.8-27b-nvfp4-vllm` (same container after stopping the default) |

---

### 5. SearXNG local search

Self-hosted meta-search engine. Hermes uses it as `web.search_backend`, removing the need for Firecrawl/Tavily/cloud APIs.

| Property | Value |
|----------|-------|
| **URL** | `http://localhost:8080` |
| **JSON API** | `http://localhost:8080/search?q=<query>&format=json` |
| **Config** | `~/searxng/settings.yml` |
| **Compose** | `~/searxng/docker-compose.yml` |
| **Service** | `searxng.service` |
| **Engines used** | DuckDuckGo, Brave, Bing, Google (aggregated) |

---

### 6. faster-whisper ASR server

Local speech-to-text for voice messages received via Telegram/WhatsApp/etc.

| Property | Value |
|----------|-------|
| **URL** | `http://localhost:8001/v1` |
| **Model** | `Systran/faster-whisper-large-v3` |
| **Framework** | faster-whisper 1.2.1 + FastAPI |
| **Device** | CPU (int8) |
| **Config in Hermes** | `stt.provider: openai`, `base_url: http://localhost:8001/v1` |
| **Service** | `local-asr-server.service` |

> ⚠️ The ASR server also binds port **8001** (different scheme from the vLLM inference server on the same port). If you start the vLLM container on the default port the ASR container must move to a different port (or stop) because Docker cannot bind two processes to the same host port.

---

### 7. fastCRW local web extractor

Firecrawl-compatible web scraper and crawler. It gives Hermes the ability to extract clean markdown from URLs without calling cloud APIs.

| Property | Value |
|----------|-------|
| **URL** | `http://localhost:3000` |
| **Image** | `ghcr.io/us/crw:latest` |
| **API** | Firecrawl-compatible (`/v1/scrape`, `/v1/crawl`, `/v1/search`) |
| **RAM** | ~15–50 MB idle; ~200 MB peak |
| **JS rendering** | LightPanda fallback (no Chromium baseline) |
| **Search backend** | Existing SearXNG container (`http://searxng:8080`) |
| **Config in Hermes** | `web.extract_backend: firecrawl`, `FIRECRAWL_API_URL=http://localhost:3000` |
| **Compose** | `web-extractor/docker-compose.yml` |
| **Service** | `fastcrw.service` |

Start it:

```bash
cd web-extractor
./run.sh
```

Or with systemd:

```bash
systemctl --user enable --now fastcrw.service
```

Test scrape:

```bash
curl -X POST http://localhost:3000/v1/scrape \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com","formats":["markdown"]}'
```

---

## Ports summary

| Service | Port | Reachable from LAN |
|---------|------|--------------------|
| vLLM (inference) | 8001 | No (localhost only) |
| LiteLLM | 4000 | Yes |
| Hermes gateway | varies | Via Telegram/Discord/etc. |
| SearXNG | 8080 | No (localhost only) |
| ASR server | 8001 | No (localhost only) — **collides with vLLM port** |
| fastCRW | 3000 | No (localhost only) |

Only LiteLLM is exposed to the LAN by design. The other services are consumed locally by Hermes or through LiteLLM.

---

## Systemd services

Enable/start everything after boot:

```bash
# Inference — default Qwen3.8-Flash-Next + fallback lite (27B) + legacy 3.6 heretic
systemctl --user enable --now qwen38-flash-next-vllm.service
# fallback lite, disabled by default:
systemctl --user enable --now qwen38-nvfp4-dspark-vllm.service
# legacy, disabled by default:
systemctl --user enable --now qwen36-heretic-vllm.service

systemctl --user enable --now litellm-proxy.service
systemctl --user enable --now hermes-gateway.service
systemctl --user enable --now local-asr-server.service
systemctl --user enable --now searxng.service
systemctl --user enable --now fastcrw.service
```

Check status:

```bash
systemctl --user status qwen38-flash-next-vllm.service
systemctl --user status qwen38-nvfp4-dspark-vllm.service
systemctl --user status qwen36-heretic-vllm.service
systemctl --user status litellm-proxy.service
systemctl --user status hermes-gateway.service
systemctl --user status local-asr-server.service
systemctl --user status searxng.service
systemctl --user status fastcrw.service
```

> **Mutually exclusive**: `qwen38-flash-next-vllm`, `qwen38-nvfp4-dspark-vllm`, `qwen36-heretic-vllm` and `qwen36-vllm` all bind port 8001 and consume the entire GPU unified pool. Stop one before starting another.

---

## Typical workflow

1. Start the LLM:
   ```bash
   # Default (Qwen3.8-Flash-Next NVFP4 hybrid):
   systemctl --user start qwen38-flash-next-vllm.service
   # Or fallback lite (Qwen 3.8 27B + DSpark k=14):
   systemctl --user stop qwen38-flash-next-vllm.service
   systemctl --user start qwen38-nvfp4-dspark-vllm.service
   ```
2. Verify LiteLLM and SearXNG are running.
3. Launch Hermes:
   ```bash
   hermes chat
   ```
4. Ask Hermes to search the web or transcribe a voice message — everything stays local.

---

## Cloud dependencies remaining

With this stack, the only external calls are:

- SearXNG querying public search engines.
- Telegram/Discord/WhatsApp message servers (if you use those platforms).
- Optional cloud fallbacks configured in LiteLLM (MiniMax, etc.).

Inference, search, speech-to-text and web extraction are fully local.
