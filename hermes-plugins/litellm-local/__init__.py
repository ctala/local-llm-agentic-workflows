"""litellm-local provider profile.

User plugin: exposes Hermes' ``reasoning_config`` translation to a
LiteLLM-proxy-backed vLLM endpoint (this Spark's setup).

This profile is registered with ``name="custom"`` and
``aliases=("custom", "litellm-local", "litellm")``. That last bit is
load-bearing: ``hermes_cli/runtime_provider.py:_resolve_named_custom_runtime``
always canonicalizes every named custom provider to
``provider="custom"`` before reaching the chat transport. Without
registering under the name ``"custom"`` we'd never be picked by
``get_provider_profile(agent.provider)`` and Hermes would fall back to
the bundled ``CustomProfile``, which has no
``build_api_kwargs_extras`` for the Qwen thinking template — silently
dropping ``/reasoning <level>`` for our stack.

Scope is gated by ``model.lower().startswith("qwen")`` so we don't
silently rewrite thinking flags for non-Qwen custom endpoints
(ollama, llama.cpp, etc.). For those, we return empty kwargs and let
the host do its thing.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Qwen 3.x native reasoning_effort levels accepted by vLLM (Qwen3 reasoning parser).
# Anything outside this set is clamped to "medium" by vLLM; we forward as-is
# and let the server decide.
#
# Qwen 3.x chat template (chat_template.jinja) accepts ONLY these
# reasoning_effort values when enable_thinking is on:
#   low, medium, xhigh
# `high` is a template alias auto-converted to xhigh, but vLLM's Qwen3
# reasoning parser treats it inconsistently (sometimes 0 reasoning chars).
# Anything else (minimal, max, ultra) raises a template exception.
_QWEN_EFFORT_LEVELS = {"low", "medium", "xhigh"}

# Map Hermes' universal level names to the Qwen 3.x native set so the user
# can type any of the 8 Hermes levels and the plugin translates quietly.
# Hermes levels: none, minimal, low, medium, high, xhigh, max, ultra.
_QWEN_LEVEL_MAPPING = {
    "none": "none",          # handled separately (enable_thinking=false)
    "minimal": "low",        # below the lowest Qwen level
    "low": "low",
    "medium": "medium",
    "high": "xhigh",         # template alias, but ambiguous in vLLM → map up
    "xhigh": "xhigh",
    "max": "xhigh",          # vLLM/Qwen clamp to xhigh anyway
    "ultra": "xhigh",        # vLLM rejects before template; treat as xhigh
}

# Levels advertised to the Hermes UI (/reasoning slash command, CLI help).
# These are the only values that map cleanly to the wire without surprises.
_QWEN_ADVERTISED_LEVELS = ["none", "low", "medium", "xhigh"]


class LitellmLocalProfile(ProviderProfile):
    """LiteLLM-proxy → vLLM/llama.cpp endpoint: thinking on/off + effort.

    Registers under ``name="custom"`` so Hermes' runtime resolver
    (which canonicalizes every named custom provider to ``provider="custom"``)
    routes the request through us instead of the bundled
    ``CustomProfile``. The hook short-circuits for non-Qwen models so
    we don't accidentally rewrite thinking flags for Ollama/llama.cpp
    endpoints served by the same proxy.
    """

    def supported_reasoning_levels(
        self, model: str | None = None
    ) -> list[str] | None:
        """Qwen 3.x native: low/medium/xhigh + none for off.

        Levels hidden from the UI (minimal, high, max, ultra) are still
        accepted by the plugin and quietly mapped to the closest native
        equivalent — so a user typing `/reasoning minimal` sees no error,
        just gets ``low`` behavior on the wire.
        """
        if model and not model.lower().startswith("qwen"):
            return None  # let the host deal with non-Qwen models
        return _QWEN_ADVERTISED_LEVELS

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str = "",
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Only translate for Qwen-family models. Non-Qwen custom endpoints
        # (Ollama, llama.cpp, etc.) get an empty payload — the host handles
        # its own thinking format.
        if model and not model.lower().startswith("qwen"):
            return extra_body, top_level

        if not (reasoning_config and isinstance(reasoning_config, dict)):
            return extra_body, top_level

        effort = (reasoning_config.get("effort") or "").strip().lower()
        enabled = reasoning_config.get("enabled", True)

        # Disabled: stop Qwen from emitting the <think>...</think> block AND
        # tell vLLM top-level reasoning_effort="none" (belt-and-suspenders —
        # Qwen 3.x ignores the top-level flag without the template kwarg).
        if effort == "none" or enabled is False:
            top_level["reasoning_effort"] = "none"
            chat_template = extra_body.get("chat_template_kwargs", {})
            chat_template["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = chat_template
            return extra_body, top_level

        # Enabled with explicit effort: translate to the closest Qwen 3.x
        # native level and forward as top-level reasoning_effort. The mapping
        # covers all 8 Hermes levels (none/minimal/low/medium/high/xhigh/max/
        # ultra) so the user can type any of them and never get a template
        # error from the upstream.
        if effort and effort in _QWEN_LEVEL_MAPPING:
            mapped = _QWEN_LEVEL_MAPPING[effort]
            if mapped != "none":
                top_level["reasoning_effort"] = mapped
                # Make sure the Qwen template actually emits the  block
                # (some levels like "high" are ambiguous in vLLM; explicit
                # enable_thinking=true ensures the parser sees a thinking run).
                chat_template = extra_body.get("chat_template_kwargs", {})
                chat_template["enable_thinking"] = True
                extra_body["chat_template_kwargs"] = chat_template

        return extra_body, top_level


# Register under name="custom" so we intercept the canonicalized "custom"
# provider that _resolve_named_custom_runtime always returns. Aliases
# preserve the original provider name (so /model switcher can match
# "litellm-local" in dropdowns if needed).
litellm_local = LitellmLocalProfile(
    name="custom",
    aliases=("custom", "litellm-local", "litellm"),
    env_vars=(),
    base_url="http://localhost:4000/v1",  # default LiteLLM proxy
    # Generous floor — user can override per-model. Hermes' LiteLLM provider
    # entries already cap at 16000; this only kicks in when the model entry
    # has no max_tokens set.
    default_max_tokens=16000,
)
register_provider(litellm_local)