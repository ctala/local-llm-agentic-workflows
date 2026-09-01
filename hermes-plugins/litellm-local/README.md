# Hermes user plugins — litellm-local

This directory contains the user-side **Hermes provider plugin** that lets `/reasoning <level>` and `--reasoning <level>` actually translate to the Qwen 3.x wire shape when the model is served via a local `LiteLLM → vLLM` proxy on the DGX Spark.

Without this plugin, `agent.reasoning_config` never reaches the Qwen template and the model emits `<think>` blocks unconditionally (regardless of `--reasoning none`).

## Why this plugin exists

Hermes' runtime resolver (`hermes_cli/runtime_provider.py:_resolve_named_custom_runtime`) always canonicalizes every named custom provider to `{"provider": "custom", ...}` before returning to the agent. That overwrites `agent.provider` with `"custom"` (`hermes_cli/cli_agent_setup_mixin.py:_ensure_runtime_credentials`, ~line 165).

The plugin registers itself under `name="custom"` and `aliases=("custom", "litellm-local", "litellm")` so `get_provider_profile("custom")` returns **this** profile (overriding the bundled `CustomProfile`). Scope is gated by `model.lower().startswith("qwen")` so non-Qwen models served by the same proxy are untouched.

The hook short-circuits when no `reasoning_config` is set or when the model name doesn't start with `qwen`, so installing this plugin doesn't change behavior for other providers.

## Install

```bash
# From a fresh Hermes install:
mkdir -p ~/.hermes/plugins/model-providers/litellm-local
cp __init__.py plugin.yaml ~/.hermes/plugins/model-providers/litellm-local/
rm -rf ~/.hermes/plugins/model-providers/litellm-local/__pycache__/
```

The plugin is picked up lazily on the first `get_provider_profile()` call (typically at chat startup). No restart needed beyond the next `hermes` invocation.

## What it does

Maps Hermes' 8 universal reasoning levels (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra`) to the 3 Qwen 3.x native levels (`low`, `medium`, `xhigh`, `none`):

| Hermes level | → Qwen native | `reasoning_effort` | `enable_thinking` |
|---|---|---|---|
| `none` | `none` | `"none"` | `false` |
| `minimal` | `low` | `"low"` | `true` |
| `low` | `low` | `"low"` | `true` |
| `medium` | `medium` | `"medium"` | `true` |
| `high` | `xhigh` | `"xhigh"` | `true` |
| `xhigh` | `xhigh` | `"xhigh"` | `true` |
| `max` | `xhigh` | `"xhigh"` | `true` |
| `ultra` | `xhigh` | `"xhigh"` | `true` |

`none` is special: it sends BOTH `enable_thinking: false` AND top-level `reasoning_effort: "none"`. Qwen 3.x ignores the top-level flag without the template kwarg, so both are needed (belt-and-suspenders).

The 4 "hidden" levels (`minimal`, `high`, `max`, `ultra`) are not advertised in `/reasoning` slash command help but are accepted and quietly mapped to the closest Qwen native level — useful for scripting.

## How to use it

Anywhere `hermes chat` runs against the local LiteLLM proxy:

```bash
# CLI flag (one-shot, no persist)
hermes chat --reasoning none   # chat rápido, sin thinking
hermes chat --reasoning low    # razonamiento breve
hermes chat --reasoning xhigh  # razonamiento máximo

# Slash command (interactive session)
/reasoning none
/reasoning low
/reasoning xhigh
```

## Verify

```bash
# Should NOT show a Reasoning block:
hermes chat --reasoning none --toolsets '' --no-restore-cwd \
  -m qwen3.8-flash-next-vllm -q "decí hola en una palabra"
```

If a `┌─ Reasoning ─...┐` block appears under `--reasoning none`, the plugin is not being invoked. Check:

1. The file is at `~/.hermes/plugins/model-providers/litellm-local/__init__.py`
2. The plugin is registered: `python3 ~/.hermes/hermes-agent/.venv/bin/python3 -c "from providers import get_provider_profile; print(get_provider_profile('custom'))"` should return a `LitellmLocalProfile`, not `CustomProfile`.
3. No `__pycache__/` stale module — `rm -rf ~/.hermes/plugins/model-providers/litellm-local/__pycache__/`.

## Bug history

- **2026-08-15**: original plugin installed at `~/.hermes/plugins/model-providers/litellm-local/`, registered as `name="litellm-local"`.
- **2026-09-01**: discovered the plugin was never actually invoked — `_resolve_named_custom_runtime` rewrote `agent.provider` to `"custom"`, but the plugin was registered under the original name, so `get_provider_profile("custom")` returned the bundled `CustomProfile` and the reasoning translation never fired. `--reasoning none` was silently dropped.
- **2026-09-01 (fix)**: renamed plugin profile to `name="custom"` with `aliases=("custom", "litellm-local", "litellm")`. Now `get_provider_profile("custom")` returns this profile. Scope-gated by model name so non-Qwen custom providers are untouched.

## Files

- `__init__.py` — the plugin code (imports + `LitellmLocalProfile` class + `register_provider(litellm_local)`).
- `plugin.yaml` — manifest required by the plugin discovery (`name`, `kind: model-provider`, `version`, `description`).