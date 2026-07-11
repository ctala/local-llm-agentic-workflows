# Agent Framework Integration Guide

This guide explains how to connect the local models running on the DGX Spark to three agent frameworks:

- **Hermes** (Nous Research)
- **OpenClaw** (predecessor to Hermes / OpenClaw-compatible configs)
- **Opencode**

All inference servers in this repository expose an **OpenAI-compatible API** on `http://localhost:8000/v1`, so any framework that supports custom OpenAI endpoints can connect directly.

---

## Two connection patterns

### 1. Direct to vLLM / TRT-LLM (localhost:8000)

Use this when:

- You are running the agent **on the same machine** as the model.
- You want the **lowest latency**.
- You only need one backend model at a time.

### 2. Through LiteLLM proxy (localhost:4000 or network)

Use this when:

- You want a **single endpoint** for multiple tools (Hermes, Opencode, Open WebUI, n8n).
- You need to access the model **from another computer on the network**.
- You want auth, rate limiting, or fallback routing between models.

The LiteLLM proxy binds to `0.0.0.0:4000` and exposes all configured backends under one OpenAI-compatible URL.

---

## Hermes

Hermes supports custom OpenAI-compatible endpoints. The cleanest setup for the Spark is to use **one provider that points to the LiteLLM proxy**; LiteLLM then exposes whichever model is currently loaded (Qwen, Gemma, Nemotron, etc.) under a single URL.

> **Important:** LiteLLM's `/v1/models` endpoint does not report `max_tokens`/`context_length` for custom OpenAI-compatible backends. If you do not set `model.context_length` (and the per-model `context_length` below), Hermes falls back to its built-in family defaults and shows **131,072 tokens** instead of the real **262,144**.

Add this to `~/.hermes/config.yaml`:

```yaml
model:
  default: qwen3.6-35b-a3b-vllm
  provider: litellm-local
  context_length: 262144

providers:
  litellm-local:
    name: LiteLLM local
    base_url: http://localhost:4000/v1
    api_key: sk-spark-local
    discover_models: false
    models:
      qwen3.6-35b-a3b-vllm:
        context_length: 262144
      qwen3.6-35b-a3b-vllm-fast:
        context_length: 262144
```

Because LiteLLM auth is disabled, any non-empty `api_key` works (`sk-spark-local` is just a placeholder).

`discover_models: false` keeps the picker limited to the two Qwen aliases. Without it, Hermes lists every model LiteLLM knows about (Ollama, MiniMax, embeddings, etc.).

Run Hermes with the default (thinking) model:

```bash
hermes chat
```

Or explicitly pick the non-thinking model:

```bash
hermes chat -m qwen3.6-35b-a3b-vllm-fast
```

### Migrating from OpenClaw to Hermes

If you are coming from OpenClaw, run:

```bash
hermes claw migrate
```

This imports OpenClaw-compatible provider definitions into Hermes.

---

## OpenClaw

OpenClaw has been superseded by Hermes and has no dedicated reasoning/no-reasoning switch for local vLLM endpoints. Migrate with:

```bash
hermes claw migrate
```

Then use the Hermes configuration above.

---

## Opencode

Opencode uses a JSON config at `~/.config/opencode/opencode.json`.

Add a single LiteLLM provider and select the active model:

```json
{
  "provider": {
    "spark-litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Spark LiteLLM",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "headers": {
          "Authorization": "Bearer sk-spark-local"
        }
      },
      "models": {
        "qwen3.6-35b-a3b-vllm": { "name": "Qwen3.6 35B-A3B vLLM (think, 262K)" },
        "qwen3.6-35b-a3b-vllm-fast": { "name": "Qwen3.6 35B-A3B vLLM (no think, 262K)" }
      }
    }
  },
  "model": "spark-litellm/qwen3.6-35b-a3b-vllm-fast"
}
```

Switch at runtime:

```bash
# thinking mode
opencode run -m spark-litellm/qwen3.6-35b-a3b-vllm "explain this bug"

# non-thinking mode
opencode run -m spark-litellm/qwen3.6-35b-a3b-vllm-fast "summarize this file"
```

---

## LiteLLM proxy setup

LiteLLM gives you one OpenAI-compatible endpoint that can route to multiple local (or cloud) backends and can be exposed to other machines on the network.

### Important: auth mode

This setup runs LiteLLM **without DB-backed user authentication**. It is intended for a local machine or a trusted LAN. Any client that sends a non-empty `Authorization: Bearer <key>` header can use the proxy. This avoids needing a Postgres/Prisma database on the Spark, which would consume extra memory.

If you need per-key access control, install Prisma, configure a database, and set `master_key: os.environ/LITELLM_MASTER_KEY` instead of `disable_user_auth: true`.

### Config file

Example `~/litellm/config.yaml`:

```yaml
model_list:
  # vLLM Qwen 3.6 35B-A3B (262K context)
  - model_name: qwen3.6-35b-a3b-vllm
    litellm_params:
      model: openai/qwen3.6-35b-a3b
      api_base: http://localhost:8000/v1
      api_key: local
    model_info:
      mode: chat
      supports_vision: true
      max_input_tokens: 262144
      max_tokens: 262144

  - model_name: qwen3.6-35b-a3b-vllm-fast
    litellm_params:
      model: openai/qwen3.6-35b-a3b
      api_base: http://localhost:8000/v1
      api_key: local
      extra_body:
        chat_template_kwargs:
          enable_thinking: false
    model_info:
      mode: chat
      supports_vision: true
      max_input_tokens: 262144
      max_tokens: 262144

litellm_settings:
  drop_params: true
  request_timeout: 600
  num_retries: 1

router_settings:
  routing_strategy: simple-shuffle

general_settings:
  # No master_key / DB auth. See note above.
  disable_user_auth: true
```

### Wrapper script for systemd

Create `~/litellm/run-no-auth.sh` so the systemd service can load `.env` (needed for cloud keys such as `MINIMAX_API_KEY`) while keeping `LITELLM_MASTER_KEY` unset:

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"
unset LITELLM_MASTER_KEY
exec "${SCRIPT_DIR}/.venv/bin/litellm" \
  --config "${SCRIPT_DIR}/config.yaml" \
  --host 0.0.0.0 --port 4000 --num_workers 2
```

Make it executable:

```bash
chmod +x ~/litellm/run-no-auth.sh
```

Then update `~/.config/systemd/user/litellm-proxy.service`:

```ini
[Service]
Type=simple
ExecStart=~/litellm/run-no-auth.sh
WorkingDirectory=~/litellm
Restart=always
RestartSec=5
```

### Start LiteLLM

With systemd:

```bash
systemctl --user daemon-reload
systemctl --user enable --now litellm-proxy.service
```

Or manually:

```bash
cd ~/litellm
./run-no-auth.sh
```

### Expose to the network

Because LiteLLM is started with `--host 0.0.0.0`, it listens on all interfaces. From another machine on the same network, use:

```
http://<spark-ip>:4000/v1
```

Make sure your firewall allows port 4000 on the Spark if you want remote access. Remember that in this configuration the proxy has no authentication, so only expose it inside a trusted network.

On the Spark, find its LAN IP with:

```bash
hostname -I
```

If the other machine cannot reach port 4000, open it. Examples:

```bash
# Ubuntu / ufw
sudo ufw allow 4000/tcp

# Or generic iptables
sudo iptables -I INPUT -p tcp --dport 4000 -j ACCEPT
```

> **Why LiteLLM and not vLLM directly?**  
> The vLLM container on the Spark usually binds `localhost:8000`. Even when it binds `0.0.0.0:8000`, it is a single model and has no auth. LiteLLM gives you one stable, network-exposed endpoint that can route to whichever model is currently loaded, and lets you keep the vLLM port closed to the outside.

### Test the proxy from the Spark

Because auth is disabled, any non-empty key works:

```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer sk-spark-local"

curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-spark-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b-vllm",
    "messages": [{"role": "user", "content": "hola"}],
    "max_tokens": 64
  }'
```

### Test the proxy from another machine on the network

Replace `<spark-ip>` with the LAN IP of the Spark (e.g. `192.168.1.100`):

```bash
# List available models
curl http://<spark-ip>:4000/v1/models \
  -H "Authorization: Bearer sk-spark-local"

# Quick chat completion with thinking disabled (fast alias)
curl http://<spark-ip>:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-spark-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b-vllm-fast",
    "messages": [{"role": "user", "content": "hola"}],
    "max_tokens": 64,
    "temperature": 0.7
  }'

# Same with reasoning enabled
curl http://<spark-ip>:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-spark-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b-vllm",
    "messages": [{"role": "user", "content": "explain this bug"}],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Use the remote endpoint in a client

Any OpenAI-compatible client works. Example with the official Python SDK from the remote machine:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<spark-ip>:4000/v1",
    api_key="sk-spark-local",  # any non-empty string; auth is disabled
)

response = client.chat.completions.create(
    model="qwen3.6-35b-a3b-vllm-fast",
    messages=[{"role": "user", "content": "hola"}],
    max_tokens=64,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

### Connect Hermes from another machine

On the remote PC, use a single LiteLLM provider and switch models with `-m`:

```yaml
# ~/.hermes/config.yaml on the remote machine
model:
  default: qwen3.6-35b-a3b-vllm
  provider: litellm-remote
  context_length: 262144

providers:
  litellm-remote:
    name: Spark LiteLLM (remote)
    base_url: http://<spark-ip>:4000/v1
    api_key: sk-spark-local
    discover_models: false
    models:
      qwen3.6-35b-a3b-vllm:
        context_length: 262144
      qwen3.6-35b-a3b-vllm-fast:
        context_length: 262144
```

Run with the default (thinking) model:

```bash
hermes chat
```

Or explicitly switch to the non-thinking alias:

```bash
hermes chat -m qwen3.6-35b-a3b-vllm-fast
```

### Connect Opencode from another machine

On the remote PC, add a LiteLLM provider in `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "litellm-remote": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Spark LiteLLM (remote LAN)",
      "options": {
        "baseURL": "http://<spark-ip>:4000/v1",
        "headers": {
          "Authorization": "Bearer sk-spark-local"
        }
      },
      "models": {
        "qwen3.6-35b-a3b-vllm": { "name": "Qwen3.6 35B-A3B (think, remote)" },
        "qwen3.6-35b-a3b-vllm-fast": { "name": "Qwen3.6 35B-A3B (no think, remote)" }
      }
    }
  },
  "model": "litellm-remote/qwen3.6-35b-a3b-vllm"
}
```

Run:

```bash
opencode run -m litellm-remote/qwen3.6-35b-a3b-vllm-fast "hola"
```

---

## Switching between thinking and no-thinking modes

Qwen 3.6 is a hybrid reasoning model: by default it emits a long internal chain-of-thought (`<think>...</think>`) before the final answer. vLLM exposes this through the chat-template flag `enable_thinking` inside `chat_template_kwargs`:

- `enable_thinking: true`  → reasoning mode (higher quality for hard tasks, slower).
- `enable_thinking: false` → non-reasoning mode (faster, lower latency, good for routine agent turns).

The agent frameworks themselves do **not** send this flag automatically today. Because LiteLLM exposes both aliases, the cleanest way to choose a mode is to pick the model alias you want for each task or session.

### Hermes

Hermes has a `/reasoning` command (`/reasoning none`, `/reasoning medium`, etc.) and an `agent.reasoning_effort` setting. Those map to provider-specific formats such as OpenRouter's `extra_body.reasoning`, Kimi's `extra_body.thinking`, or LM Studio's top-level `reasoning_effort`.

For a **local vLLM/Qwen** endpoint, however, Hermes' generic reasoning controls do not translate into `chat_template_kwargs.enable_thinking`. Therefore `/reasoning none` will **not** disable Qwen 3.6 thinking. Use the two LiteLLM aliases instead:

```bash
hermes chat -m qwen3.6-35b-a3b-vllm      # reasoning mode
hermes chat -m qwen3.6-35b-a3b-vllm-fast # non-reasoning mode
```

### OpenClaw

OpenClaw has been superseded by Hermes and has no dedicated reasoning/no-reasoning switch for local vLLM endpoints. Migrate with:

```bash
hermes claw migrate
```

Then use the two-model pattern above.

### Opencode

Opencode supports `--variant <effort>` and `--thinking` flags, but they control **display** of reasoning blocks or provider-specific reasoning effort, not Qwen's `enable_thinking` chat-template flag. There is no documented way to send `chat_template_kwargs` per model in `opencode.json`.

Use two model entries and select the active one:

```bash
# thinking mode
opencode run -m spark-litellm/qwen3.6-35b-a3b-vllm "explain this bug"

# non-thinking mode
opencode run -m spark-litellm/qwen3.6-35b-a3b-vllm-fast "summarize this file"
```

### Recommended practice for agents

| Pattern | When to use |
|---------|-------------|
| **Two aliases** (`*-vllm` / `*-vllm-fast`) | Default. Pick the mode per task or per agent. |
| LiteLLM proxy | Required when multiple tools (Hermes, Opencode, Open WebUI, n8n) share the same backend. |
| Direct vLLM | Only for a single tool on the same machine; you still need two model aliases to toggle thinking. |

For routine agent turns (file edits, web searches, small code generation), the no-thinking alias is usually faster and wastes fewer tokens. Reserve the thinking alias for architecture decisions, complex debugging, or multi-step planning.

---

## Tool calling with Qwen 3.6

Hermes (and most OpenAI-compatible agents) expect tool calls in the native `tool_calls` array of the chat-completion response. Qwen 3.6, however, emits tool calls as XML inside the message `content` unless vLLM is told to parse them.

When serving Qwen 3.6 with vLLM, add these flags:

```bash
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--reasoning-parser qwen3
```

All Qwen 3.6 launch scripts in this repo already include them. The `qwen3_coder` parser is more robust for multi-turn tool calling than the older `qwen3_xml` parser. Without a parser, vLLM returns XML such as:

```xml
<tool_call>\n<name>get_weather</name>\n<arguments>{"location":"Paris"}</arguments>\n</tool_call>
```

Hermes sees the XML text but receives an empty `tool_calls` array, so it prints `<tool_call>` instead of executing the tool.

### Verify tool calls at the vLLM level

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "weather in Paris"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
    "tool_choice": "auto"
  }' | jq '.choices[0].message.tool_calls'
```

A working response contains a populated `tool_calls` array:

```json
[
  {
    "id": "chatcmpl-tool-...",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\": \"Paris\"}"
    }
  }
]
```

If the result is `null`, check that the parser flags are present in the launch script and restart the container.

## Long-context launch scripts

For agentic workloads on the Spark, Qwen 3.6 35B-A3B is served at its maximum context length (262,144 tokens) using the nvidia checkpoint and vLLM nightly. The recommended default uses 2 parallel sequences:

| Script | `max_num_seqs` | `gpu-memory-utilization` | KV cache (full) | Use case |
|--------|---------------|--------------------------|-----------------|----------|
| `run-qwen36-35b-a3b.sh` | 2 | 0.92 | ~82 GB | **Recommended.** Two 262K sessions in parallel with headroom for LiteLLM/ASR. |
| `run-qwen36-35b-a3b-extreme-context-2seq.sh` | 2 | 0.92 | ~82 GB | Alias to the script above for discoverability. |

The 2-sequence config is the best default for Hermes/OpenClaw because it leaves breathing room for auxiliary services without sacrificing much total context (524K tokens across both sessions).

Start the recommended config:

```bash
./scripts/run-qwen36-35b-a3b.sh
```

## Choosing the right model alias

With the current vLLM-only setup on the Spark, the usable Qwen 3.6 35B-A3B aliases are the two LiteLLM-routed `*-vllm` models. The non-`-vllm` aliases (e.g. `qwen3.6-35b-a3b`, `qwen3.6-35b-a3b-fast`) previously routed to local llama.cpp/Ollama endpoints and are no longer active.

| Alias | Backend | Context | Thinking | Use case |
|-------|---------|---------|----------|----------|
| `qwen3.6-35b-a3b-vllm` | LiteLLM → vLLM | 262K | enabled | Reasoning mode (hard tasks, planning) |
| `qwen3.6-35b-a3b-vllm-fast` | LiteLLM → vLLM | 262K | disabled | Non-reasoning mode (fast routine turns) |

For agentic work that benefits from reasoning, use the `*-vllm` variant. For faster, non-reasoning turns, use `*-vllm-fast`.

---

## Local speech-to-text (ASR)

Hermes can transcribe incoming voice messages locally using the OpenAI-compatible ASR server in [`asr-server/`](https://github.com/ctala/local-llm-agentic-workflows/tree/main/asr-server). It runs **faster-whisper** on the Spark's CPU and exposes `/v1/audio/transcriptions` on port `8001`.

### Start the ASR server

```bash
cd asr-server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Or run it as a systemd user service:

```bash
cp asr-server/local-asr-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now local-asr-server.service
```

### Configure Hermes STT

Add this to `~/.hermes/config.yaml`:

```yaml
stt:
  enabled: true
  provider: openai
  openai:
    model: whisper-1
    base_url: http://localhost:8001/v1
    api_key: sk-local-asr
```

When a voice message arrives through a connected platform (Telegram, WhatsApp, etc.), Hermes sends it to this endpoint and receives the transcript without using a cloud STT service.

### Test the endpoint

```bash
curl -s -X POST http://localhost:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-local-asr" \
  -F file=@/path/to/audio.wav \
  -F model=whisper-1 \
  -F response_format=text
```

---

## Self-hosted web search with SearXNG

Hermes can use a local **[SearXNG](https://github.com/searxng/searxng)** instance as its web search backend, so you don't need Firecrawl, Tavily or any cloud search API to browse the web.

### Install SearXNG

Create `~/searxng/docker-compose.yml`:

```yaml
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    restart: unless-stopped
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./settings.yml:/etc/searxng/settings.yml:ro
    environment:
      - SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
```

Create `~/searxng/settings.yml`:

```yaml
use_default_settings: true
server:
  secret_key: "54986978f2ef1e54e3486ac7011af8eb"
  bind_address: "0.0.0.0"
search:
  formats:
    - html
    - json
```

> Generate a fresh secret key with `openssl rand -hex 32`.

Start it:

```bash
cd ~/searxng
docker compose up -d
```

Or use the systemd user service:

```bash
cp /home/ctala/.config/systemd/user/searxng.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now searxng.service
```

### Configure Hermes

Add to `~/.hermes/.env`:

```bash
SEARXNG_URL=http://localhost:8080
```

And set the search backend in `~/.hermes/config.yaml`:

```yaml
web:
  backend: firecrawl
  search_backend: searxng
  extract_backend: ''
  use_gateway: false
```

Restart the gateway:

```bash
systemctl --user restart hermes-gateway.service
```

### Test the search endpoint

```bash
curl -s 'http://localhost:8080/search?q=NVIDIA+DGX+Spark&format=json' | jq '.results[:3]'
```

From Hermes, just ask it to search the web:

```bash
hermes chat -z "busca en la web: NVIDIA DGX Spark precio"
```

SearXNG aggregates results from multiple engines (DuckDuckGo, Brave, etc.) without sending your queries to a single commercial provider.

---

## Self-hosted web extraction with fastCRW

Hermes can extract clean markdown from URLs using a local **[fastCRW](https://github.com/us/crw)** instance. fastCRW exposes a Firecrawl-compatible API (`/v1/scrape`, `/v1/crawl`, `/v1/search`) and uses only ~15–50 MB of RAM, making it ideal for the Spark.

### Why fastCRW instead of Firecrawl self-hosted?

| | fastCRW | Firecrawl self-hosted |
|---|---|---|
| **Idle RAM** | ~15–50 MB | ~1–2 GB |
| **Peak RAM** | ~200 MB | **8–14 GB** |
| **Extra infra** | None | Redis + Postgres + RabbitMQ + Playwright |
| **API** | Firecrawl-compatible | Native Firecrawl |
| **JS rendering** | LightPanda fallback | Playwright/Chromium |

Firecrawl self-hosted is a better fit for a dedicated scraping server. On the Spark, where every gigabyte counts for LLM KV cache, fastCRW is the pragmatic choice.

### Install fastCRW

The service is in the repo:

```bash
cd web-extractor
./run.sh
```

Or install the systemd user service:

```bash
cp web-extractor/fastcrw.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fastcrw.service
```

This connects fastCRW to the existing SearXNG container on its Docker network (`searxng_default`) so `/v1/search` works through SearXNG.

### Configure Hermes

Add to `~/.hermes/.env`:

```bash
FIRECRAWL_API_URL=http://localhost:3000
FIRECRAWL_API_KEY=local
```

And set the extraction backend in `~/.hermes/config.yaml`:

```yaml
web:
  backend: firecrawl
  search_backend: searxng
  extract_backend: firecrawl
  use_gateway: false
```

Restart the gateway:

```bash
systemctl --user restart hermes-gateway.service
```

### Test the extraction endpoint

```bash
curl -s -X POST http://localhost:3000/v1/scrape \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com","formats":["markdown"]}' | jq '.data.markdown'
```

### Use from Hermes

Once configured, Hermes' `web_search` and `web_extract` tools will route through local services:

- Search → SearXNG (`http://localhost:8080`)
- Extract / crawl → fastCRW (`http://localhost:3000`)

Example:

```bash
hermes chat -z "busca en la web últimas noticias de NVIDIA y resume la primera página"
```

Everything stays on the Spark.

---

## Troubleshooting

### "Connection refused" from the agent

- Verify the vLLM container is healthy: `curl http://localhost:8000/health`
- Verify LiteLLM is running (if using the proxy): `curl http://localhost:4000/v1/models`
- Verify SearXNG is running: `curl http://localhost:8080/healthz`
- Verify fastCRW is running: `curl http://localhost:3000/health`
- Check that nothing else is bound to port 8000, 4000, 8080 or 3000.

### Context length shows 131,072 instead of 262,144 in Hermes

This happens because LiteLLM's `/v1/models` response does not include `max_tokens` for OpenAI-compatible backends, so Hermes falls back to its generic `qwen` family default (131K).

Fix:

1. Set `model.context_length: 262144` in `~/.hermes/config.yaml`.
2. Also set `context_length: 262144` under each model in `providers.litellm-local.models`.
3. Restart the gateway: `systemctl --user restart hermes-gateway.service`.

### Context length errors

- Make sure `context_length` in Hermes matches the model's real limit (262144 for Qwen 3.6 35B-A3B).
- In Opencode, set `max_input_tokens` in the LiteLLM `model_info` if needed.

### Out of memory when using multiple frameworks

- The vLLM Qwen 3.6 35B-A3B container uses ~119–120 GB of the 121 GB unified memory.
- LiteLLM itself is lightweight, but running additional local services (llama-server, Ollama, etc.) can push the system over the edge.
- Stop unused model services before starting another large model.

### Hermes prints `<tool_call>` but does not execute the tool

This happens when vLLM returns the tool call as XML text in `message.content` instead of the native `message.tool_calls` array. Hermes only executes tools from the native array.

Fix:

1. Stop the vLLM container.
2. Confirm the launch script includes:
   ```bash
   --enable-auto-tool-choice \
   --tool-call-parser qwen3_xml \
   --reasoning-parser qwen3
   ```
3. Restart the container and verify with the curl test in the "Tool calling with Qwen 3.6" section above.

### Network access not working

- Confirm LiteLLM started with `--host 0.0.0.0`.
- Check `ufw`/firewall rules on the Spark.
- Verify the client machine can reach the Spark IP on port 4000: `curl http://<spark-ip>:4000/v1/models`.
