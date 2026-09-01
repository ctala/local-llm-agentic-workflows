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

> **Important:** LiteLLM's `/v1/models` endpoint does not report `max_tokens`/`context_length` for custom OpenAI-compatible backends. If you do not set `model.context_length` (and the per-model `context_length` below), Hermes falls back to its built-in family defaults and shows **131,072 tokens** instead of the real **262,144** (or **237,000** if you reserve 25K for output).

Add this to `~/.hermes/config.yaml`:

```yaml
model:
  default: qwen3.8-flash-next-vllm   # current default (2026-09-01)
  provider: litellm-local
  context_length: 237000
  max_tokens: 16000

providers:
  litellm-local:
    name: LiteLLM local
    base_url: http://localhost:4000/v1
    api_key: sk-spark-local
    discover_models: false
    models:
      qwen3.8-flash-next-vllm:
        context_length: 237000
        max_tokens: 16000
      qwen3.8-27b-nvfp4-vllm:        # fallback lite, same container after stopping default
        context_length: 237000
        max_tokens: 16000
      qwen3.6-35b-a3b-vllm:          # legacy single-stream champion, requires swap
        context_length: 237000
        max_tokens: 16000
    default_model: qwen3.8-flash-next-vllm

agent:
  reasoning_overrides:
    qwen3.8-flash-next-vllm: low
    qwen3.8-27b-nvfp4-vllm: low

code_execution:
  mode: project
  timeout: 300
  max_tool_calls: 50

compression:
  enabled: true
  threshold: 0.85
  target_ratio: 0.2
  protect_last_n: 20
  hygiene_hard_message_limit: 400
  protect_first_n: 3
  abort_on_summary_failure: false
  codex_gpt55_autoraise: true
```

### Hermes plugin `litellm-local` (recommended for Qwen 3.x reasoning)

Without this plugin, `get_provider_profile("litellm-local")` returns `None` and Hermes falls back to the gate at `run_agent.py:_supports_reasoning_extra_body`, which only allows OpenRouter/Nous/GitHub Models/LMStudio/Ollama Cloud to receive `reasoning` / `reasoning_effort` in their request body. As a result `/reasoning <level>` and `--reasoning <level>` are silently dropped for our stack.

Install the plugin at `~/.hermes/plugins/model-providers/litellm-local/__init__.py`:

```python
"""litellm-local provider profile.

Translates Hermes' `reasoning_config` to a LiteLLM-proxy-backed vLLM endpoint
(this Spark's setup). Registers a NEW canonical name so it does not interfere
with the bundled `custom` profile.

Wire shape (matches Qwen 3.x + LiteLLM proxy + vLLM):

  - reasoning_config = {"enabled": False} (or effort="none")
      → top-level reasoning_effort: "none"
      → extra_body.chat_template_kwargs.enable_thinking: false
        (the Qwen template flag that actually suppresses <think>...</think>)
  - reasoning_config = {"enabled": True, "effort": "<level>"}
      → top-level reasoning_effort: "<level>" for vLLM
      → server picks the Qwen thinking depth

Qwen 3.8 chat template accepts ONLY low/medium/xhigh. The plugin maps
Hermes' universal level names to the Qwen native set:

  none → none     minimal → low     low → low
  medium → medium  high → xhigh    xhigh → xhigh
  max → xhigh     ultra → xhigh

Levels advertised to the Hermes UI: low/medium/xhigh (plus 'none').
"""
# See gemma4-optimizado for the full implementation reference.
```

The plugin advertises `low`, `medium`, `xhigh` (plus `none`) in the `/reasoning` slash command and CLI help, and silently maps the hidden levels so a user typing `/reasoning minimal` sees no error.

Because LiteLLM auth is disabled, any non-empty `api_key` works (`sk-spark-local` is just a placeholder).

`discover_models: false` keeps the picker limited to the three Qwen aliases. Without it, Hermes lists every model LiteLLM knows about (Ollama, MiniMax, embeddings, etc.).

Run Hermes with the default (Qwen3.8-Flash-Next):

```bash
hermes chat
```

Or explicitly pick the fallback:

```bash
hermes chat -m qwen3.8-27b-nvfp4-vllm
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
        "qwen3.8-flash-next-vllm":     { "name": "Qwen3.8-Flash-Next NVFP4 hybrid (DEFAULT 2026-09-01)" },
        "qwen3.8-27b-nvfp4-vllm":      { "name": "Qwen3.8 27B NVFP4 + DSpark k=14 (FALLBACK lite)" },
        "qwen3.6-35b-a3b-vllm":        { "name": "Qwen3.6 35B-A3B vLLM (LEGACY single-stream champion)" },
        "qwen3.6-35b-a3b-vllm-fast":   { "name": "Qwen3.6 35B-A3B vLLM (no think, 262K)" }
      }
    }
  },
  "model": "spark-litellm/qwen3.8-flash-next-vllm"
}
```

Switch at runtime:

```bash
# default (Qwen3.8-Flash-Next, hybrid, reasoning low by default)
opencode run -m spark-litellm/qwen3.8-flash-next-vllm "explain this bug"

# fallback lite (Qwen 3.8 27B + DSpark, higher concurrency)
opencode run -m spark-litellm/qwen3.8-27b-nvfp4-vllm "summarize this file"

# legacy single-stream (Qwen 3.6 35B-A3B, requires manual swap of vLLM container)
opencode run -m spark-litellm/qwen3.6-35b-a3b-vllm-fast "non-thinking chat"
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
  # vLLM Qwen 3.6 35B-A3B (262144 tokens; 1 sequence)
  # max_input_tokens=237000 + max_tokens=25000 usan casi toda la ventana.
  - model_name: qwen3.6-35b-a3b-vllm
    litellm_params:
      model: openai/qwen3.6-35b-a3b
      api_base: http://localhost:8000/v1
      api_key: local
    model_info:
      mode: chat
      supports_vision: true
      max_input_tokens: 237000
      max_tokens: 25000

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
      max_input_tokens: 237000
      max_tokens: 25000

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
  context_length: 237000
  max_tokens: 25000

providers:
  litellm-remote:
    name: Spark LiteLLM (remote)
    base_url: http://<spark-ip>:4000/v1
    api_key: sk-spark-local
    discover_models: false
    models:
      qwen3.6-35b-a3b-vllm:
        context_length: 237000
        max_tokens: 25000
      qwen3.6-35b-a3b-vllm-fast:
        context_length: 237000
        max_tokens: 25000
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

Qwen 3.6, 3.8 and Qwen3.8-Flash-Next are hybrid reasoning models: they emit a long internal chain-of-thought (`<think>...</think>`) before the final answer when `enable_thinking` is on. vLLM exposes this through the chat-template flag `enable_thinking` inside `chat_template_kwargs`:

- `enable_thinking: true`  → reasoning mode (higher quality for hard tasks, slower).
- `enable_thinking: false` → non-reasoning mode (faster, lower latency, good for routine agent turns).

The agent frameworks themselves do **not** send this flag automatically today. Use the model aliases exposed by LiteLLM, or — better — install the [Hermes plugin `litellm-local`](#hermes) so that `/reasoning <level>` and `--reasoning <level>` translate to the right wire shape automatically.

### Hermes

With the `litellm-local` plugin installed (recommended), use the built-in `/reasoning` slash command and CLI flag:

```bash
hermes chat --reasoning low              # reasoning mode
hermes chat --reasoning none             # non-reasoning mode
# Or in-session:
#   /reasoning xhigh
```

The plugin advertises `none`, `low`, `medium`, `xhigh` in the UI and quietly maps `minimal`/`high`/`max`/`ultra` to the closest Qwen native level. `agent.reasoning_overrides` in `~/.hermes/config.yaml` sets the per-model default:

```yaml
agent:
  reasoning_overrides:
    qwen3.8-flash-next-vllm: low     # default: low reasoning for fast agentic work
    qwen3.8-27b-nvfp4-vllm: low      # fallback lite
```

Without the plugin, `/reasoning none` does **not** disable Qwen thinking — use a non-reasoning alias instead:

```bash
# Manual aliases (still works, less convenient)
hermes chat -m qwen3.8-flash-next-vllm        # reasoning mode (default)
hermes chat -m qwen3.6-35b-a3b-vllm-fast     # non-reasoning mode
```

### Opencode

Opencode supports `--variant <effort>` and `--thinking` flags, but they control **display** of reasoning blocks or provider-specific reasoning effort, not Qwen's `enable_thinking` chat-template flag. There is no documented way to send `chat_template_kwargs` per model in `opencode.json`.

Use one model entry per reasoning mode and select at runtime:

```bash
# reasoning mode
opencode run -m spark-litellm/qwen3.8-flash-next-vllm "explain this bug"

# non-reasoning mode (Qwen 3.6 legacy alias)
opencode run -m spark-litellm/qwen3.6-35b-a3b-vllm-fast "summarize this file"
```

### Recommended practice for agents

| Pattern | When to use |
|---------|-------------|
| **Hermes plugin `litellm-local`** + `/reasoning <level>` | **Recommended.** Single model + variable thinking mode. |
| Two model aliases (`*-vllm` / `*-vllm-fast`) | Manual fallback. Pick the mode per task or per agent. |
| LiteLLM proxy | Required when multiple tools (Hermes, Opencode, Open WebUI, n8n) share the same backend. |
| Direct vLLM | Only for a single tool on the same machine; you still need two model aliases to toggle thinking. |

For routine agent turns (file edits, web searches, small code generation), the no-thinking alias is usually faster and wastes fewer tokens. Reserve the thinking alias for architecture decisions, complex debugging, or multi-step planning.

---

## Tool calling with Qwen 3.x

Hermes (and most OpenAI-compatible agents) expect tool calls in the native `tool_calls` array of the chat-completion response. Qwen 3.6 and 3.8 emit tool calls as XML inside the message `content` unless vLLM is told to parse them.

### Qwen 3.8 (default since 2026-09-01)

For both **Qwen3.8-Flash-Next** (`qwen38-flash-dgx` image) and **Qwen 3.8 27B** (`vllm/vllm-openai:v0.27.1-aarch64`), add these flags:

```bash
# Qwen3.8-Flash-Next (default):
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--reasoning-parser qwen3

# Qwen 3.8 27B (fallback lite, uses DSpark k=14 drafter):
--enable-auto-tool-choice \
--tool-call-parser qwen3_xml \
--reasoning-parser qwen3
```

The `qwen3_coder` parser is more robust for multi-turn tool calling than the older `qwen3_xml` parser. Without a parser, vLLM returns XML such as:

```xml
<tool_call>\n<name>get_weather</name>\n<arguments>{"location":"Paris"}</arguments>\n</tool_call>
```

Hermes sees the XML text but receives an empty `tool_calls` array, so it prints `<tool_call>` instead of executing the tool.

### Qwen 3.6 (legacy)

For **Qwen 3.6 35B-A3B** (`vllm/vllm-openai:nightly`, single-stream long-context config), the recipe is the same but we intentionally **omit `--reasoning-parser`** for the `nvidia/Qwen3.6-35B-A3B-NVFP4` checkpoint because it does not emit `<think></think>` tags; with the parser enabled, reasoning leaks into the `reasoning` field and `content` appears empty in agents. With thinking enabled but no parser, reasoning and answer both land in `content`, which agents display correctly. The `auto_disable_thinking_with_tools` chat-template flag disables thinking automatically when tools are present.

### Verify tool calls at the vLLM level

```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "weather in Tokyo"}],
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
      "arguments": "{\"location\": \"Tokyo\"}"
    }
  }
]
```

If the result is `null`, check that the parser flags are present in the launch script and restart the container.

## Long-context launch scripts

The Spark runs **one model at a time** because each stack consumes the entire unified-memory pool. Switch by stopping one systemd unit and starting another.

### Current default (2026-09-01): Qwen3.8-Flash-Next NVFP4 hybrid

The recipe [`blazux/qwen3.8-Flash-DGX`](https://github.com/blazux/qwen3.8-Flash-DGX) bundles 7 GB10-targeted patches on top of `vllm/vllm-openai:qwen38-flash-next` (release/qwen38next recipe). The script (`scripts/run-qwen38-flash-next.sh`) uses defaults validated in this repo's tuning study:

| Variable | Value | Rationale |
|----------|-------|-----------|
| `--max-num-seqs` | 8 | Below concurrency targets — keeps @c=16 above 100 tok/s aggregate |
| `--max-num-batched-tokens` | 8192 | 16384 fragmenta prefills in long contexts |
| `--max-model-len` | 262144 | Native; use `YARN=1` for up to 500k |
| `--gpu-memory-utilization` | 0.80 | 0.85 OOM-kills on 300k prefill with MTP |
| `MTP` (env) | 2 | MTP=4 reduces decode by 7-34% |
| `--enable-chunked-prefill` | ✓ | Required for long contexts |
| `--enable-prefix-caching` | ✓ | 7.2× speedup on repeated system prompts |
| `--tool-call-parser` | `qwen3_coder` | Robust multi-turn tool calling |
| `--reasoning-parser` | `qwen3` | Native `<think>` block support |

Environment variables set inside the container: `VLLM_PLE_MMAP=1` (serves the 48 GiB PLE n-gram table from NVMe via `mmap`), `VLLM_QSA_EXACT_TOPK=1` (deterministic QSA top-k), `VLLM_PLE_MMAP_WORKERS=32`.

**Performance (DGX Spark, page cache warm ~25 min):**

| Workload | Tokens/s |
|----------|---------:|
| chat_short warm | **36.66** |
| code_quicksort fresh | 28.52 |
| agent_with_tools | 23.59 |
| tool loop multi-step (4 steps, 3 tools) | 28.65 median |
| Aggregate c=1 | 30.29 |
| Aggregate c=8 | **116.75** |
| Aggregate c=16 | **129.65** |
| Aggregate c=32 | 129.69 (plateau) |
| Prefill cold (8K) | 1018 tok/s |
| Prefix-hit TTFT (same 20K prompt) | **1.46 s** |
| Determinism @ T=0 (EXACT_TOPK=1) | ✓ |

### Fallback lite: Qwen 3.8 27B NVFP4 + DSpark k=14

For workloads that need higher aggregate throughput (e.g. many parallel auxiliary agents in Hermes/Opencode), the 27B + DSpark k=14 keeps `@c=16` at **253 tok/s aggregate** thanks to its external drafter. Cold start is faster (~5-8 min vs 14 min).

### Single-stream long-context champion: Qwen 3.6 35B-A3B

If you need ~76 tok/s sustained on a single long-context session (~237K input + 25K output), Qwen 3.6 with the nvidia NVFP4 checkpoint + Marlin backend + `flashinfer` + FP8 KV is still the fastest. It sacrifices concurrency (1-seq/262K) but recovers the original 75-77 tok/s while using the model's full context window. Recipe: `--moe-backend marlin --attention-backend flashinfer --kv-cache-dtype fp8 --max-num-seqs 1 --max-model-len 262144`, with `VLLM_TEST_FORCE_FP8_MARLIN=1` and `VLLM_MARLIN_USE_ATOMIC_ADD=1` in the container env.

### Choosing the right model alias

Each model is exposed under one or more LiteLLM aliases. Pick the one matching the current vLLM container.

| Alias | Backend | Context | Thinking | Use case |
|-------|---------|---------|----------|----------|
| `qwen3.8-flash-next-vllm` | LiteLLM → vLLM `release/qwen38next` + 7 parches GB10 | 262K (500K YaRN) | enabled (low by default) | **Default 2026-09-01.** Agentic workflows, código, razonamiento, multimodal |
| `qwen3.8-27b-nvfp4-vllm` | LiteLLM → vLLM 0.27.1 + DSpark k=14 | 262K (1M YaRN) | enabled (low by default) | **Fallback lite.** Concurrencia @c=16 con DSpark |
| `qwen3.6-35b-a3b-vllm` | LiteLLM → vLLM nightly + Marlin | 262K | enabled | **Single-session long-context champion** (~76 tok/s) |
| `qwen3.6-35b-a3b-vllm-fast` | LiteLLM → vLLM nightly + Marlin | 262K | disabled | Non-reasoning mode for fast routine turns |

For agentic work that benefits from reasoning, use the `*-vllm` variants. For faster, non-reasoning turns, use `*-vllm-fast` or set `reasoning_overrides: none` in Hermes.

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

1. Set `model.context_length: 237000` in `~/.hermes/config.yaml`.
2. Also set `context_length: 237000` under each model in `providers.litellm-local.models`.
3. Restart the gateway: `systemctl --user restart hermes-gateway.service`.

### Context length errors

- Make sure `context_length` in Hermes matches the input budget configured in LiteLLM (`237000` for Qwen 3.6 35B-A3B with 25K output).
- In Opencode, set `max_input_tokens` in the LiteLLM `model_info` if needed.

### Out of memory when using multiple frameworks

- The vLLM Qwen 3.6 35B-A3B container uses ~119–120 GB of the 121 GB unified memory.
- LiteLLM itself is lightweight, but running additional local services (llama-server, Ollama, etc.) can push the system over the edge.
- Stop unused model services before starting another large model.

### LiteLLM / Hermes times out after ~600 s on long contexts

Symptom: LiteLLM returns `HTTP 500` with `Timeout on reading data from socket` or `MidStreamFallbackError` after ~600 s, while vLLM eventually completes the request.

What we observed with Qwen 3.6 35B-A3B and a 154K-token context:

- vLLM did **not** crash; it returned `200 OK` several minutes after LiteLLM gave up.
- The Spark's unified memory was saturated (~120 GiB used / 121 GiB total, ~12 GiB swap in use), so vLLM slowed down enough to exceed LiteLLM's `request_timeout: 600`.
- With KV cache in BF16 the system had almost no headroom; switching to FP8 KV cache roughly halves KV-cache memory use.

Mitigations:

1. Reduce KV-cache memory pressure: use `--attention-backend flashinfer --kv-cache-dtype fp8` without `--enforce-eager` so torch.compile/CUDAGraph stay active.
2. Increase LiteLLM's `request_timeout` if you regularly hit 600 s, but fix the memory/swap cause first.
3. Keep auxiliary services (Ollama, llama-server, etc.) stopped when running large-context workloads.

### Hermes prints `<tool_call>` but does not execute the tool

This happens when vLLM returns the tool call as XML text in `message.content` instead of the native `message.tool_calls` array. Hermes only executes tools from the native array.

Fix:

1. Stop the vLLM container.
2. Confirm the launch script includes:
   ```bash
   --enable-auto-tool-choice \
   --tool-call-parser qwen3_coder
   ```
   Note: we omit `--reasoning-parser` for the nvidia checkpoint because it does not produce `<think></think>` tags; with the parser, `content` becomes empty in agents.
3. Restart the container and verify with the curl test in the "Tool calling with Qwen 3.6" section above.

### Network access not working

- Confirm LiteLLM started with `--host 0.0.0.0`.
- Check `ufw`/firewall rules on the Spark.
- Verify the client machine can reach the Spark IP on port 4000: `curl http://<spark-ip>:4000/v1/models`.

## Common Hermes configuration pitfall: "Context: 1,000,000 tokens"

When running a custom OpenAI-compatible endpoint (LiteLLM → vLLM), Hermes' `/model` switch and startup banner may report a wildly wrong context length (e.g. **1,000,000 tokens** for a model that actually supports 262K). The cause and fix:

1. **Cause**: Hermes resolves the context window through `agent/model_metadata.py:get_model_context_length`, which checks the `models.dev` registry first. Custom aliases like `qwen3.8-flash-next-vllm` are not in models.dev, so the resolver falls through to hardcoded-c family defaults — for Qwen 3.x that returns **1,000,000**.

2. **Why `model.context_length` doesn't fix it**: Hermes' runtime helper `agent/agent_runtime_helpers.py` intentionally clears `agent._config_context_length` on every `/model` switch so the new model can resolve its own window. Top-level `model.context_length: 237000` therefore only applies on **startup**, not after `/model`.

3. **The fix — `model_overrides`** (persistent path #0b in `get_model_context_length`):

   ```yaml
   # ~/.hermes/config.yaml
   model_overrides:
     litellm-local:
       qwen3.8-flash-next-vllm:
         context_window: 237000
         max_output_tokens: 16000
         supports_vision: true
         supports_tools: true
       qwen3.8-27b-nvfp4-vllm:
         context_window: 262144
         max_output_tokens: 16000
         supports_vision: true
         supports_tools: true
       qwen3.6-35b-a3b-vllm:
         context_window: 237000
         max_output_tokens: 25000
         supports_tools: true
   ```

4. **Verify** (via tmux, the `/model` command only works in interactive sessions):

   ```text
   ✓ Model switched: qwen3.8-flash-next-vllm
     Provider: litellm-local
     Context: 237,000 tokens
     Max output: 16,000 tokens
     Capabilities: tools, vision
   ```

This is independent of the `[plugin litellm-local](#hermes)` reasoning-level translation; both pieces are required for the full Hermes + LiteLLM + custom-local-vLLM stack to behave correctly.

## Why `/reasoning <level>` was being silently dropped (and the fix)

Even with the plugin installed at `~/.hermes/plugins/model-providers/litellm-local/`, the reasoning-level translation did NOT fire for our local Qwen 3.x models — the `/model` output looked like:

```
┌─ Reasoning ──────────────────────────────────────────────────────────────────┐
The user just said "decí hola"...
└──────────────────────────────────────────────────────────────────────────────
```

…and `--reasoning none` still produced a `<think>` block.

### Root cause

`hermes_cli/runtime_provider.py:_resolve_named_custom_runtime` always canonicalizes every named custom provider to `{"provider": "custom", ...}` before returning to the agent. That overwrites `agent.provider` with `"custom"` (see `hermes_cli/cli_agent_setup_mixin.py:_ensure_runtime_credentials`, ~line 165, where `self.provider = resolved_provider`).

The plugin profile was originally registered as:

```python
LitellmLocalProfile(name="litellm-local", aliases=("litellm",))
```

So `get_provider_profile("custom")` returned the stock `CustomProfile` (no `build_api_kwargs_extras`), and our profile was never looked up for our own provider. `agent.reasoning_config` never reached the plugin, and the Qwen template kept emitting thinking tokens.

### Fix

Rename the plugin profile to register under `"custom"` (the canonical name every custom provider is collapsed to) with the original name as an alias:

```python
litellm_local = LitellmLocalProfile(
    name="custom",
    aliases=("custom", "litellm-local", "litellm"),
    ...
)
register_provider(litellm_local)
```

That makes `get_provider_profile("custom")` return **our** profile (overriding the bundled one), so `build_api_kwargs_extras(reasoning_config=...)` finally fires for our request.

### Scope guard

We don't want our profile rewriting `reasoning_effort`/`enable_thinking` for every model that happens to pass through the proxy. The hook short-circuits unless the model name starts with `qwen` (case-insensitive):

```python
def build_api_kwargs_extras(self, *, reasoning_config=None, model="", **ctx):
    if model and not model.lower().startswith("qwen"):
        return {}, {}
    ...
```

Ollama, llama.cpp and other non-Qwen models served by the same proxy get an empty payload and the host does its own thinking handling.

### Verified end-to-end

| `--reasoning` | Reasoning tokens | Sample content |
|---|---|---|
| `none` | **0** (no `<think>` block) | `"Hola"` |
| `low` | ~50 | `"The user is asking a simple arithmetic question. 12 × 15 = 180"` |
| `xhigh` | ~150 | multi-step reasoning + tool call + `"101, 103, 107 y 109"` |

Wire shape sent to vLLM (verified via curl + log inspection):

| `--reasoning` | top-level `reasoning_effort` | `chat_template_kwargs.enable_thinking` |
|---|---|---|
| `none` | `"none"` | `false` |
| `low` | `"low"` | `true` |
| `xhigh` | `"xhigh"` | `true` |

## Comprehensive reasoning-level reference

The plugin translates every Hermes reasoning level to the closest Qwen 3.x native level. Verified end-to-end on 2026-09-01 against `qwen3.8-flash-next-vllm` (vLLM release/qwen38next):

| Hermes level | → Qwen native | top-level `reasoning_effort` | `chat_template_kwargs.enable_thinking` | Observed reasoning tokens (`7 × 8 = ?`) |
|---|---|---|---|---|
| `none` | `none` | `"none"` | `false` | **0** (no `<think>` block) |
| `minimal` | `low` | `"low"` | `true` | ~30 |
| `low` | `low` | `"low"` | `true` | 35 |
| `medium` | `medium` | `"medium"` | `true` | ~80 |
| `high` | `xhigh` | `"xhigh"` | `true` | ~120 |
| `xhigh` | `xhigh` | `"xhigh"` | `true` | 35 (clamped by max_tokens) |
| `max` | `xhigh` | `"xhigh"` | `true` | ~120 |
| `ultra` | `xhigh` | `"xhigh"` | `true` | ~120 |

Notes:

- The **advertised levels** in `/reasoning` slash command and CLI help are only `none`, `low`, `medium`, `xhigh`. The other four (`minimal`, `high`, `max`, `ultra`) are accepted by the plugin and quietly mapped to the nearest Qwen native level — so a user typing `/reasoning minimal` never sees an error, just gets `low` behavior on the wire.
- `none` is special: it sets BOTH `enable_thinking: false` AND top-level `reasoning_effort: "none"`. Qwen 3.x ignores the top-level flag without the template kwarg, so both are needed (belt-and-suspenders).
- The plugin scope-guards by model name — non-Qwen models served by the same proxy (Ollama, llama.cpp) get an empty payload and the host handles its own thinking format.
- `xhigh` produces the most thinking tokens per prompt but is bounded by `max_tokens` in the request. For reasoning-heavy workloads, raise `max_tokens` or pre-allocate via `agent.reasoning_effort` config.
