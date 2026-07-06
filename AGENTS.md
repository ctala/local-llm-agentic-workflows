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

### Direct connection to vLLM

Add a provider block to `~/.hermes/config.yaml` under `providers:`:

```yaml
providers:
  local-qwen-35b-vllm-262k:
    base_url: http://localhost:8000/v1
    name: Spark Qwen 35B-A3B (vLLM 262K)
    api_key: local-no-key-needed
    model: qwen3.6-35b-a3b
    context_length: 262144
```

Run Hermes with that provider:

```bash
hermes chat --provider local-qwen-35b-vllm-262k -m qwen3.6-35b-a3b
```

You can also set it as the default model/provider in the top-level `model:` block:

```yaml
model:
  default: qwen3.6-35b-a3b
  provider: custom
  base_url: http://localhost:8000/v1
  context_length: 262144
```

### Through LiteLLM proxy

If you are using the LiteLLM proxy at `http://localhost:4000/v1`, add a provider that points to it. Because the proxy runs in no-auth mode (see LiteLLM setup below), any non-empty API key works:

```yaml
providers:
  litellm-qwen-vllm-262k:
    base_url: http://localhost:4000/v1
    name: Qwen 3.6 35B-A3B vLLM via LiteLLM
    api_key: sk-spark-local
    model: qwen3.6-35b-a3b-vllm
    context_length: 262144
```

Run Hermes with that provider:

```bash
hermes chat --provider litellm-qwen-vllm-262k -m qwen3.6-35b-a3b-vllm
```

### Migrating from OpenClaw to Hermes

If you are coming from OpenClaw, run:

```bash
hermes claw migrate
```

This imports OpenClaw-compatible provider definitions into Hermes.

---

## OpenClaw

OpenClaw uses the same provider shape as early Hermes configs. Add a block like this to your OpenClaw configuration file (typically `~/.openclaw/config.yaml` or the equivalent path used by your OpenClaw build):

```yaml
providers:
  spark-qwen-vllm:
    base_url: http://localhost:8000/v1
    name: Spark Qwen 35B-A3B (vLLM)
    api_key: local-no-key-needed
    model: qwen3.6-35b-a3b
    context_length: 262144
```

Or via LiteLLM (any non-empty `api_key` works because the proxy runs in no-auth mode):

```yaml
providers:
  spark-litellm-qwen-vllm:
    base_url: http://localhost:4000/v1
    name: Spark Qwen 35B-A3B via LiteLLM
    api_key: sk-spark-local
    model: qwen3.6-35b-a3b-vllm
    context_length: 262144
```

> **Note**: OpenClaw has been superseded by Hermes. If you are still on OpenClaw, consider migrating with `hermes claw migrate` and using the Hermes instructions above.

---

## Opencode

Opencode uses a JSON config at `~/.config/opencode/opencode.json`.

### Direct connection to vLLM

Add a provider block:

```json
{
  "provider": {
    "spark-vllm-direct": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Spark vLLM (directo, 262K)",
      "options": {
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
        "qwen3.6-35b-a3b": { "name": "Qwen3.6 35B-A3B (think, 262K)" }
      }
    }
  }
}
```

Set it as the active model:

```json
{
  "model": "spark-vllm-direct/qwen3.6-35b-a3b"
}
```

### Through LiteLLM proxy

Add the model to your existing LiteLLM provider:

```json
{
  "provider": {
    "spark-litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Spark LiteLLM (local, vía proxy con auth)",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "headers": {
          "Authorization": "Bearer sk-spark-local"
        }
      },
      "models": {
        "qwen3.6-35b-a3b-vllm": { "name": "Qwen3.6 35B-A3B vLLM (think, 262K)" },
        "qwen3.6-35b-a3b-vllm-fast": { "name": "Qwen3.6 35B-A3B vLLM (sin thinking, 262K)" }
      }
    }
  }
}
```

Set the active model:

```json
{
  "model": "spark-litellm/qwen3.6-35b-a3b-vllm"
}
```

---

## LiteLLM proxy setup

LiteLLM gives you one OpenAI-compatible endpoint that can route to multiple local (or cloud) backends and can be exposed to other machines on the network.

### Important: auth mode

This setup runs LiteLLM **without DB-backed user authentication**. It is intended for a local machine or a trusted LAN. Any client that sends a non-empty `Authorization: Bearer <key>` header can use the proxy. This avoids needing a Postgres/Prisma database on the Spark, which would consume extra memory.

If you need per-key access control, install Prisma, configure a database, and set `master_key: os.environ/LITELLM_MASTER_KEY` instead of `disable_user_auth: true`.

### Config file

Example `/home/ctala/litellm/config.yaml`:

```yaml
disable_user_auth: true

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

  - model_name: qwen3.6-35b-a3b-vllm-fast
    litellm_params:
      model: openai/qwen3.6-35b-a3b
      api_base: http://localhost:8000/v1
      api_key: local
      extra_body: {chat_template_kwargs: {enable_thinking: false}}
    model_info:
      mode: chat
      supports_vision: true
      max_input_tokens: 262144

litellm_settings:
  drop_params: true
  request_timeout: 600
  num_retries: 1

general_settings:
  # No master_key / DB auth. See note above.
  disable_user_auth: true
```

### Wrapper script for systemd

Create `/home/ctala/litellm/run-no-auth.sh` so the systemd service can load `.env` (needed for cloud keys such as `MINIMAX_API_KEY`) while keeping `LITELLM_MASTER_KEY` unset:

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
chmod +x /home/ctala/litellm/run-no-auth.sh
```

Then update `~/.config/systemd/user/litellm-proxy.service`:

```ini
[Service]
Type=simple
ExecStart=/home/ctala/litellm/run-no-auth.sh
WorkingDirectory=/home/ctala/litellm
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
cd /home/ctala/litellm
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

Replace `<spark-ip>` with the LAN IP of the Spark (e.g. `192.168.88.190`):

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

On the remote PC, add a provider that points to the Spark's LiteLLM proxy:

```yaml
# ~/.hermes/config.yaml on the remote machine
providers:
  spark-qwen-fast:
    base_url: http://<spark-ip>:4000/v1
    name: Spark Qwen 35B-A3B (no thinking, remote)
    api_key: sk-spark-local
    model: qwen3.6-35b-a3b-vllm-fast
    context_length: 262144

  spark-qwen-think:
    base_url: http://<spark-ip>:4000/v1
    name: Spark Qwen 35B-A3B (thinking, remote)
    api_key: sk-spark-local
    model: qwen3.6-35b-a3b-vllm
    context_length: 262144
```

Run:

```bash
hermes chat --provider spark-qwen-fast
```

### Connect Opencode from another machine

On the remote PC, add a LiteLLM provider in `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "spark-litellm-remote": {
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
  "model": "spark-litellm-remote/qwen3.6-35b-a3b-vllm-fast"
}
```

Run:

```bash
opencode run -m spark-litellm-remote/qwen3.6-35b-a3b-vllm-fast "hola"
```

---

## Switching between thinking and no-thinking modes

Qwen 3.6 is a hybrid reasoning model: by default it emits a long internal chain-of-thought (`<think>...</think>`) before the final answer. vLLM exposes this through the chat-template flag `enable_thinking` inside `chat_template_kwargs`:

- `enable_thinking: true`  → reasoning mode (higher quality for hard tasks, slower).
- `enable_thinking: false` → non-reasoning mode (faster, lower latency, good for routine agent turns).

The agent frameworks themselves do **not** send this flag automatically today, so the cleanest way to choose a mode is to expose **two separate model aliases** and pick the one you want for each task or session.

### Hermes

Hermes has a `/reasoning` command (`/reasoning none`, `/reasoning medium`, etc.) and an `agent.reasoning_effort` setting. Those map to provider-specific formats such as OpenRouter's `extra_body.reasoning`, Kimi's `extra_body.thinking`, or LM Studio's top-level `reasoning_effort`.

For a **local vLLM/Qwen** endpoint, however, Hermes' generic `custom` provider does not translate `/reasoning` into `chat_template_kwargs.enable_thinking`. Therefore:

- `/reasoning none` will **not** disable Qwen 3.6 thinking when talking directly to vLLM.
- Use **two providers or two LiteLLM aliases** instead:

```yaml
providers:
  spark-qwen-think:
    base_url: http://localhost:4000/v1
    name: Spark Qwen 35B-A3B (thinking)
    api_key: sk-spark-local
    model: qwen3.6-35b-a3b-vllm
    context_length: 262144

  spark-qwen-fast:
    base_url: http://localhost:4000/v1
    name: Spark Qwen 35B-A3B (no thinking)
    api_key: sk-spark-local
    model: qwen3.6-35b-a3b-vllm-fast
    context_length: 262144
```

Switch at runtime:

```bash
hermes chat --provider spark-qwen-think  # reasoning mode
hermes chat --provider spark-qwen-fast   # non-reasoning mode
```

If you want a single default and only occasionally change mode, set `spark-qwen-think` as default and create a short `/fast` slash command or shell alias that launches Hermes with `--provider spark-qwen-fast`.

### OpenClaw

OpenClaw has been superseded by Hermes and has no dedicated reasoning/no-reasoning switch for local vLLM endpoints. Migrate with:

```bash
hermes claw migrate
```

Then use the two-provider pattern above.

### Opencode

Opencode supports `--variant <effort>` and `--thinking` flags, but they control **display** of reasoning blocks or provider-specific reasoning effort, not Qwen's `enable_thinking` chat-template flag. There is no documented way to send `chat_template_kwargs` per model in `opencode.json`.

Use two model entries and select the active one:

```json
{
  "provider": {
    "spark-litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Spark LiteLLM",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "headers": { "Authorization": "Bearer sk-spark-local" }
      },
      "models": {
        "qwen3.6-35b-a3b-vllm":      { "name": "Qwen3.6 35B-A3B (think)" },
        "qwen3.6-35b-a3b-vllm-fast": { "name": "Qwen3.6 35B-A3B (no think)" }
      }
    }
  },
  "model": "spark-litellm/qwen3.6-35b-a3b-vllm"
}
```

Switch at runtime:

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
| Direct vLLM | Fine for a single tool on the same machine, but you still need two provider blocks to toggle thinking. |

For routine agent turns (file edits, web searches, small code generation), the no-thinking alias is usually faster and wastes fewer tokens. Reserve the thinking alias for architecture decisions, complex debugging, or multi-step planning.

---

## Choosing the right model alias

With the current vLLM-only setup on the Spark, the usable Qwen 3.6 35B-A3B aliases are the two LiteLLM-routed `*-vllm` models. The non-`-vllm` aliases (e.g. `qwen3.6-35b-a3b`, `qwen3.6-35b-a3b-fast`) previously routed to local llama.cpp/Ollama endpoints and are no longer active.

| Alias | Backend | Context | Thinking | Use case |
|-------|---------|---------|----------|----------|
| `qwen3.6-35b-a3b-vllm` | LiteLLM → vLLM | 262K | enabled | Reasoning mode (hard tasks, planning) |
| `qwen3.6-35b-a3b-vllm-fast` | LiteLLM → vLLM | 262K | disabled | Non-reasoning mode (fast routine turns) |

For agentic work that benefits from reasoning, use the `*-vllm` variant. For faster, non-reasoning turns, use `*-vllm-fast`.

---

## Troubleshooting

### "Connection refused" from the agent

- Verify the vLLM container is healthy: `curl http://localhost:8000/health`
- Verify LiteLLM is running (if using the proxy): `curl http://localhost:4000/v1/models`
- Check that nothing else is bound to port 8000 or 4000.

### Context length errors

- Make sure `context_length` in Hermes/OpenClaw matches the model's real limit (262144 for Qwen 3.6 35B-A3B).
- In Opencode, set `max_input_tokens` in the LiteLLM `model_info` if needed.

### Out of memory when using multiple frameworks

- The vLLM Qwen 3.6 35B-A3B container uses ~119–120 GB of the 121 GB unified memory.
- LiteLLM itself is lightweight, but running additional local services (llama-server, Ollama, etc.) can push the system over the edge.
- Stop unused model services before starting another large model.

### Network access not working

- Confirm LiteLLM started with `--host 0.0.0.0`.
- Check `ufw`/firewall rules on the Spark.
- Verify the client machine can reach the Spark IP on port 4000: `curl http://<spark-ip>:4000/v1/models`.
