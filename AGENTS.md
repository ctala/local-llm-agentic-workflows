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

If you are using the LiteLLM proxy at `http://localhost:4000/v1`, add a provider that points to it. Because the proxy runs in no-auth mode (see LiteLLM setup below), any non-empty API key works; the example uses `key_env` so Hermes reads the key from your shell:

```yaml
providers:
  litellm-qwen-vllm-262k:
    base_url: http://localhost:4000/v1
    name: Qwen 3.6 35B-A3B vLLM via LiteLLM
    key_env: LITELLM_MASTER_KEY
    model: qwen3.6-35b-a3b-vllm
    context_length: 262144
```

Export the key in your shell (the value does not need to match anything in LiteLLM; it just must be non-empty):

```bash
export LITELLM_MASTER_KEY=sk-spark-local
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

### Test the proxy

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

---

## Choosing the right model alias

| Alias | Backend | Context | Use case |
|-------|---------|---------|----------|
| `qwen3.6-35b-a3b` | Direct vLLM | 262K | Direct connection from Hermes/Opencode |
| `qwen3.6-35b-a3b-vllm` | LiteLLM → vLLM | 262K | Via LiteLLM, with thinking enabled |
| `qwen3.6-35b-a3b-vllm-fast` | LiteLLM → vLLM | 262K | Via LiteLLM, thinking disabled |

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
