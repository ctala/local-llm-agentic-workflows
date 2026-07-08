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
---

# Fully Local Agent Stack for DGX Spark

This is the complete self-hosted stack we run on the **NVIDIA DGX Spark** to power local autonomous agents without relying on cloud APIs for inference, search or speech recognition.

Every service below runs on the Spark itself or is reachable from other machines on the LAN through LiteLLM.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Client layer                            │
│  Hermes (CLI/TUI) · OpenClaw · Opencode · Open WebUI · n8n  │
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
│    Qwen 3.6 35B-A3B · Gemma 4 · Nemotron 3 · port 8000      │
└─────────────────────────────────────────────────────────────┘

Auxiliary services:
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  SearXNG         │  │  faster-whisper  │  │  Browser / file  │
│  local search    │  │  ASR server      │  │  tools           │
│  port 8080       │  │  port 8001       │  │  (optional)      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Services

### 1. vLLM inference server

Serves the active LLM with an OpenAI-compatible API.

| Property | Value |
|----------|-------|
| **URL** | `http://localhost:8000/v1` |
| **Default model** | `qwen3.6-35b-a3b` |
| **Container** | `vllm/vllm-openai` (cu130, ARM64) |
| **Launch script** | `scripts/run-qwen36-35b-a3b-extreme-context-2seq.sh` |
| **Max context** | 262,144 tokens |
| **Tool calling** | `--tool-call-parser qwen3_xml` |

Switch models by running a different launch script and updating LiteLLM/Hermes.

---

### 2. LiteLLM proxy

Unified gateway that exposes vLLM (and optional cloud fallbacks) under one OpenAI-compatible endpoint. Required when multiple clients share the backend or when accessing the Spark from another machine.

| Property | Value |
|----------|-------|
| **URL** | `http://0.0.0.0:4000/v1` |
| **Config** | `~/litellm/config.yaml` |
| **Service** | `litellm-proxy.service` |
| **Models exposed** | `qwen3.6-35b-a3b-vllm`, `qwen3.6-35b-a3b-vllm-fast`, Ollama models, MiniMax |
| **Auth** | Disabled (`disable_user_auth: true`); any non-empty `Authorization` header works |

---

### 3. Hermes agent

Primary agent framework. Connected to LiteLLM so it can switch between thinking and non-thinking Qwen aliases.

| Property | Value |
|----------|-------|
| **Config** | `~/.hermes/config.yaml` |
| **Provider** | `litellm-local` |
| **Default model** | `qwen3.6-35b-a3b-vllm` |
| **Context length** | 262,144 tokens (forced in config) |
| **Service** | `hermes-gateway.service` |

See [Agents](/local-llm-agentic-workflows/agents/) for the full config.

---

### 4. SearXNG local search

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

### 5. faster-whisper ASR server

Local speech-to-text for voice messages received via Telegram/WhatsApp/etc.

| Property | Value |
|----------|-------|
| **URL** | `http://localhost:8001/v1` |
| **Model** | `Systran/faster-whisper-large-v3` |
| **Framework** | faster-whisper 1.2.1 + FastAPI |
| **Device** | CPU (int8) |
| **Config in Hermes** | `stt.provider: openai`, `base_url: http://localhost:8001/v1` |
| **Service** | `local-asr-server.service` |

---

## Ports summary

| Service | Port | Reachable from LAN |
|---------|------|--------------------|
| vLLM | 8000 | No (localhost only) |
| LiteLLM | 4000 | Yes |
| Hermes gateway | varies | Via Telegram/Discord/etc. |
| SearXNG | 8080 | No (localhost only) |
| ASR server | 8001 | No (localhost only) |

Only LiteLLM is exposed to the LAN by design. The other services are consumed locally by Hermes or through LiteLLM.

---

## Systemd services

Enable/start everything after boot:

```bash
systemctl --user enable --now litellm-proxy.service
systemctl --user enable --now hermes-gateway.service
systemctl --user enable --now local-asr-server.service
systemctl --user enable --now searxng.service
```

Check status:

```bash
systemctl --user status litellm-proxy.service
systemctl --user status hermes-gateway.service
systemctl --user status local-asr-server.service
systemctl --user status searxng.service
```

---

## Typical workflow

1. Start the LLM:
   ```bash
   ./scripts/run-qwen36-35b-a3b-extreme-context-2seq.sh
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

Inference, search and speech-to-text are fully local.
