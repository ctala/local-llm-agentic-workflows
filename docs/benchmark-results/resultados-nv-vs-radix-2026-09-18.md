# Experimento NVIDIA oficial vs Radix Ark — 2026-09-18

## TL;DR
**El checkpoint oficial `nvidia/Qwen3.8-Flash-Next-NVFP4` NO funciona en este Spark con el recipe `release/qwen38next` + parches GB10 actuales.** Dos causas distintas bloquean su adopción. **Nos quedamos con Radix Ark.**

## Diagnóstico: ¿por qué no hay audio?
El modelo **base** `Qwen/Qwen3.8-Flash-Next` (BF16 sin cuantizar, publicado por el equipo Qwen) tiene en su `config.json`:
- `image_token_id`, `video_token_id`, `vision_config` ✓
- `audio_token_id`, `audio_config`, `audio_tower` ✗ (no existen)

**El audio nunca estuvo en el Flash-Next.** Es decisión arquitectónica upstream. La versión que sí incluye audio es `Qwen/Qwen3-Omni-30B-A3B-Instruct` (model_type `qwen3_omni_moe` con `talker_config` + `code2wav` + `enable_audio_output`). El cuantizado NVFP4 de NVIDIA y Radix Ark heredan esa decisión — no es bug del recipe de cuantización.

Para audio en Hermes hoy y mañana: `faster-whisper-server` en :8090 ya transcribe OGG/Opus → texto → modelo.

## Setup experimental
- Hardware: NVIDIA DGX Spark (GB10), 128 GB memoria unificada
- Servicio en `:8001` (vLLM) y `:4000` (LiteLLM proxy) durante baseline
- Bench: warm-up 2× + chat_short/code/agent/vision con `temperature=1.0, top_p=0.95, max_tokens=400`
- Concurrencia: 1, 4, 8 con ThreadPoolExecutor

## Resultados

### Variante C — Radix Ark baseline (lo que está corriendo)
Imagen: `qwen38-flash-dgx:latest` (vllm 0.1.dev20073+g8e685d198) · Checkpoint: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (ModelOpt 0.46.0 estable) · Parches GB10 activos.

| Métrica | Resultado |
|---|---|
| chat_short warm | 23.5 tok/s (181 tok, 148 reasoning + 145 content) |
| code_quicksort | 24.6 tok/s (400 tok, todo reasoning) |
| agent_with_tools | 21.0 tok/s (400 tok, 368 reasoning + 60 content) |
| vision (VoxCPM PNG) | 19.6 tok/s (200 tok, todo reasoning) |
| c=1 aggregate | 22.9 |
| c=4 aggregate | 39.4 (median per-req 11.1) |
| **c=8 aggregate** | **74.6** (median per-req 10.0) — coincide con AGENTS.md |

Servicio OK post-experimento (sanity check post-restore: 20-22 tok/s warm, vision 17.2 tok/s).

### Variante A — NVIDIA oficial SIN parches GB10
Imagen: `nvcr.io/nvidia/vllm:26.04-py3` (vLLM 0.19.0+6bc3197f upstream) · Checkpoint: `nvidia/Qwen3.8-Flash-Next-NVFP4`.

**Resultado: FAIL.** vLLM 0.19 upstream no soporta `model_type='qwen4_exp'`:
```
ERROR: Failed to import AutoConfig: model_type='qwen4_exp' is not supported
You can update Transformers... or get the most up-to-date code by installing Transformers from source
```

Necesita el recipe `release/qwen38next` con commit `g8e685d198` o superior, **idéntico al que usa la imagen `qwen38-flash-dgx`**.

### Variante B — NVIDIA oficial CON parches GB10
Imagen: `qwen38-flash-dgx:latest` (vllm 0.1.dev20073+g8e685d198, 7 parches GB10) · Checkpoint: `nvidia/Qwen3.8-Flash-Next-NVFP4`.

**Resultado: FAIL durante carga de shards.** Cargó 10/11 shards y luego:
```
ERROR: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'
       for checkpoint weight 'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'
RuntimeError: Engine core initialization failed.
```

El checkpoint NV oficial usa `ModelOpt 0.46.0.dev281+g73d778422` (versión dev con fixes nuevos) y cambió el formato de los scales NVFP4 (`weight_scale_inv`) en los experts MoE. El commit `g8e685d198` no reconoce ese formato.

## Conclusiones

1. **vLLM upstream 0.19** no soporta `qwen4_exp` → necesitamos el recipe release/qwen38next.
2. **release/qwen38next commit g8e685d198** sí carga el modelo base Qwen3.8-Flash-Next, pero solo entiende los scales NVFP4 de ModelOpt `0.46.0` estable (formato Radix Ark).
3. **NVIDIA oficial ModelOpt 0.46.0.dev281** trae un formato de scale más nuevo que el commit `g8e685d198` no parsea. Es la misma incompatibilidad que está parcheada vía `#55513` para MTP upstream (mencionado en el README NVIDIA).

## Recomendación operativa

**Quedarse con Radix Ark como default.** El checkpoint NVIDIA oficial necesita uno de:
- **Opción A**: esperar a que `blazux/qwen3.8-Flash-DGX` suba un commit vllm más reciente que parsee ModelOpt `0.46.0.dev281`.
- **Opción B**: NVIDIA sube el recipe `vllm-openai:qwen38-flash-next` con vLLM `>=0.20` que incluya vllm#55513 (merge pendiente).
- **Opción C**: parchar manualmente el commit `g8e685d198` para registrar el nuevo parámetro `w2_weight_scale_inv` (estimación: ~10 líneas en `vllm/models/qwen3_8_flash_next/nvidia/mtp.py:343`).

Si querés publicar nuestra versión propia (`ctala/Qwen3.8-Flash-Next-GB10-NVFP4`):
- Mantener Radix Ark + parches como base (funciona y es estable)
- Aplicar el parche de la opción C encima del recipe `release/qwen38next`
- Attribution: NVIDIA Open Model License (base) + Apache-2.0 blazux (parches)
- Valor agregado: documentación explícita para DGX Spark + smoke tests + benchmarks publicados

## Trabajo realizado (no perdido)

- `~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/` (124 GB) → mantener o borrar para liberar disco
- Scripts en `qwen38-flash-next/exp-nv/`: `run-variant-A-nvidia-upstream.sh`, `run-variant-B-nvidia-patched.sh`, `bench.py` → conservar como referencia
- Logs: `/tmp/nv-exp/{variant-A.log, variant-B.log, bench-RadixArk-baseline.json, resultados.txt}`
