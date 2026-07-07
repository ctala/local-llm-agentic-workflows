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

Add this to `~/.hermes/config.yaml`:

```yaml
model:
  default: qwen3.6-35b-a3b-vllm-fast
  provider: spark-litellm-local
  base_url: http://localhost:4000/v1
  context_length: 262144

providers:
  spark-litellm-local:
    base_url: http://localhost:4000/v1
    name: LiteLLM local
    api_key: sk-spark-local
    model: qwen3.6-35b-a3b-vllm-fast
    context_length: 262144
    extra_body:
      chat_template_kwargs:
        enable_thinking: false
```

Because LiteLLM auth is disabled, any non-empty `api_key` works (`sk-spark-local` is just a placeholder).

Run Hermes with the default model:

```bash
hermes chat
```

Or explicitly pick the thinking model:

```bash
hermes chat -m qwen3.6-35b-a3b-vllm
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

On the remote PC, use a single LiteLLM provider and switch models with `-m`:

```yaml
# ~/.hermes/config.yaml on the remote machine
model:
  default: qwen3.6-35b-a3b-vllm-fast
  provider: spark-litellm-remote
  base_url: http://<spark-ip>:4000/v1
  context_length: 262144

providers:
  spark-litellm-remote:
    base_url: http://<spark-ip>:4000/v1
    name: Spark LiteLLM (remote)
    api_key: sk-spark-local
    model: qwen3.6-35b-a3b-vllm-fast
    context_length: 262144
```

Run with the default (fast) model:

```bash
hermes chat
```

Or explicitly switch to thinking mode:

```bash
hermes chat -m qwen3.6-35b-a3b-vllm
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
--tool-call-parser qwen3_xml \
--reasoning-parser qwen3
```

All Qwen 3.6 launch scripts in this repo already include them. Without `--tool-call-parser qwen3_xml`, vLLM returns XML such as:

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
