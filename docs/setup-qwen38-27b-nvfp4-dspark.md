# Setup Qwen 3.8 27B NVFP4 + DSpark k=14 en DGX Spark (GB10)

> **Fallback lite desde 2026-09-01.** Default operativo desde 2026-08-18 → 2026-09-01 (reemplazado por Qwen3.8-Flash-Next NVFP4 hybrid).
>
> Datos: 27B densos, MoE 0/dense hybrid 27B, GSM8K ~95%, multimodal texto+imagen+video, 262K contexto nativo (1M YaRN). Decode fresh ~14-26 tok/s, **con prefix cache caliente 70-76 tok/s single-stream / 253 @c=16 aggregate**.

## TL;DR — Setup en 5 pasos (~10 min total)

```bash
# 0. Pre-flight: GPU vacía + ~50 GB libres en disco
df -h /home/ctala | head -2
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 1. Descargar modelos (~25 GB, 4 min en red normal)
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir ~/vllm/qwen3.8-27b-nvfp4
hf download Doopeworld/Qwen3.8-27B-DSpark-vLLM --local-dir ~/vllm/qwen3.8-dspark

# 2. Pull imagen vLLM con soporte DSpark (~22 GB, 2-4 min)
docker pull vllm/vllm-openai:v0.27.1-aarch64

# 3. Copiar script de lanzamiento
cp /home/ctala/Playground/local-llm-agentic-workflows/scripts/run-qwen38-27b-nvfp4-dspark.sh /home/ctala/bin/
chmod +x /home/ctala/bin/run-qwen38-27b-nvfp4-dspark.sh

# 4. Instalar servicio systemd (auto-arranque en boot)
cat > ~/.config/systemd/user/qwen38-nvfp4-dspark-vllm.service <<'EOF'
[Unit]
Description=Qwen3.8 27B NVFP4 vLLM + DSpark k=14 (fallback lite since 2026-09-01)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/ctala
ExecStartPre=/usr/bin/docker rm -f qwen38-nvfp4
ExecStart=/home/ctala/bin/run-qwen38-27b-nvfp4-dspark.sh
ExecStop=/usr/bin/docker stop qwen38-nvfp4
Restart=on-failure
RestartSec=60
Environment=SERVICE_MODE=1

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable qwen38-nvfp4-dspark-vllm.service

# 5. Arrancar y verificar
systemctl --user start qwen38-nvfp4-dspark-vllm.service
# Esperar ~6 min la primera vez (autotune flashinfer + DSpark graphs + CUDA graphs)
until [ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8001/health)" = "200" ]; do
  sleep 15
done
curl -sS http://127.0.0.1:8001/v1/models
# → {"data":[{"id":"qwen3.8-27b-nvfp4", ...}]}
```

## Rendimiento esperado (medido en este Spark)

| Métrica | Valor | Notas |
|---|---:|---|
| Fresh-code single-stream | **29.67 tok/s** | Sin cache hits |
| EDIT-heavy (warm prefix cache) | **73-76 tok/s** | Cache hit en system prompt |
| Aggregate c=1 | 70.66 tok/s | Single-session warm |
| Aggregate c=4 | 150.64 tok/s | |
| Aggregate c=8 | 182.25 tok/s | |
| **Aggregate c=16** | **253.18 tok/s** | Mejor concurrencia de todos los Qwen |
| Determinism 20/20 byte-identical | ✓ | |
| Cold start | ~5-8 min | vs ~14 min del Flash-Next |
| Disco | ~25 GB | vs ~139 GB del Flash-Next |
| Memory | ~105 GiB | deja 16-22 GiB libres para Hermes/auxiliares |

## Estructura de archivos relevantes

```
gemma4-optimizado/
├── run-qwen38-27b-nvfp4-dspark.sh            # script de lanzamiento (también en scripts/ del repo público)
├── systemd/
│   └── qwen38-nvfp4-dspark-vllm.service       # unidad systemd user-level (auto-arranca)
├── bench-qwen38-nvfp4/                       # scripts de benchmark edit/conc/prefill/determinism
│   ├── serve.sh
│   ├── edit_bench.py
│   ├── conc_bench.py
│   ├── prefill_bench.py
│   └── determinism.py
├── report/                                    # Reportes del benchmark + documentación
│   ├── compare-configs.md                    # FP8+DSpark k=7 vs NVFP4+DSpark k=14 vs MTP k=8
│   └── benchmark-results.md                   # tablas detalladas del bench del 27B
└── resultados-qwen38-2026-08-15.md            # raw benchmark del 27B (recipe original)
```

Para Qwen3.8-Flash-Next (default 2026-09-01) ver [`docs/setup-qwen38-flash-next.md`](setup-qwen38-flash-next.md).

## Flags clave del servidor (de `run-qwen38-27b-nvfp4-dspark.sh`)

```bash
serve "${MODEL_DIR}"                              # path al NVFP4
--served-model-name qwen3.8-27b-nvfp4
--host 0.0.0.0 --port 8001
--max-model-len 262144                           # nativo (YaRN a 1M opcional)
--gpu-memory-utilization 0.85                    # 0.85 deja ~16-22 GiB libres (vs 0.80 del Flash-Next)
--max-num-batched-tokens 16384                  # alto para batches grandes con DSpark k=14
--enable-prefix-caching                          # CRÍTICO para edit-heavy / 70-76 tok/s warm
--reasoning-parser qwen3                         # Qwen3 thinking (para ambos Qwen 3.x)
--tool-call-parser qwen3_xml                     # NO confundir con qwen3_coder (ese es Flash-Next y Qwen 3.6)
--enable-auto-tool-choice
--limit-mm-per-prompt.image 2
--limit-mm-per-prompt.video 0
--speculative-config '{"method":"dspark","model":"/models/qwen3.8-dspark","num_speculative_tokens":14,"draft_sample_method":"probabilistic"}'
```

### Por qué estos flags (resumen)

| Flag | Por qué |
|---|---|
| `--speculative-config` con `num_speculative_tokens:14` | DSpark k=14 drafter da **+28% en EDIT-heavy** (76 vs 59 tok/s con k=7). Sin sacrificar batch. |
| `--enable-prefix-caching` | **Mandatory**. Sin esto, EDIT-heavy cae de 76 a ~30 tok/s. |
| `--tool-call-parser qwen3_xml` | XML parser (NO `qwen3_coder` que usa Flash-Next). Específico de Qwen 3.5/3.6/3.8. |
| `--gpu-memory-utilization 0.85` | Deja ~16-22 GiB libres (más que el 0.80 del Flash-Next porque el 27B es más liviano). |
| `--max-num-batched-tokens 16384` | Alto para aprovechar DSpark k=14 (cada batch = 14 drafts × N seqs). |
| `--reasoning-parser qwen3` | Para `<think>...</think>` blocks (válido para todos los Qwen 3.x). |

## Integración con Hermes

```yaml
# ~/.hermes/config.yaml — añadir esta entrada:
providers:
  litellm-local:
    models:
      qwen3.8-27b-nvfp4-vllm:
        context_length: 237000
        max_tokens: 16000

model:
  default: qwen3.8-flash-next-vllm  # sigue siendo el default

agent:
  reasoning_overrides:
    qwen3.8-27b-nvfp4-vllm: low    # o none para chat rápido
```

```bash
hermes chat                                  # usa el default (Flash-Next)
hermes chat -m qwen3.8-27b-nvfp4-vllm       # usa el 27B explícitamente (después de parar el Flash-Next)
```

## Integración con LiteLLM

```yaml
# ~/litellm/config.yaml — ya está integrado:
- model_name: qwen3.8-27b-nvfp4-vllm
  litellm_params:
    model: openai/qwen3.8-27b-nvfp4
    api_base: http://localhost:8001/v1
    api_key: local-vllm
  model_info:
    mode: chat
    supports_vision: true
    max_input_tokens: 237000
    max_tokens: 16000
```

Después de editar, reiniciar: `systemctl --user restart litellm-proxy.service`

## Integración con OpenCode

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
        "qwen3.8-27b-nvfp4-vllm": { "name": "Qwen3.8 27B NVFP4 + DSpark k=14 (FALLBACK lite)" }
      }
    }
  },
  "model": "spark-litellm/qwen3.8-flash-next-vllm"  # default, pero se puede cambiar
}
```

## Rollback (volver al Flash-Next default)

```bash
systemctl --user stop qwen38-nvfp4-dspark-vllm.service
systemctl --user start qwen38-flash-next-vllm.service
```

O usar el helper:
```bash
bash /home/ctala/Playground/gemma4-optimizado/qwen38-flash-next/rollback-to-default.sh
```

**Mutuamente excluyente en puerto 8001**: el 27B y el Flash-Next NO pueden correr al mismo tiempo — cada uno consume toda la memoria unificada del Spark.

## Warmup y costos de primer arranque

| Etapa | Duración | Notas |
|---|---:|---|
| torch.compile + autotune | 1-2 min | Compila kernels específicos para el modelo |
| FlashInfer fp4_gemm autotune | 2-3 min | **Primera vez solamente** — cache persiste en `~/vllm/vllm-cache-qwen38-nvfp4/` y `/root/.cache/vllm/flashinfer_autotune_cache/` |
| CUDA graph capture (48 sizes) | 1-2 min | Para batch sizes 1-512 |
| DSpark CUDA graphs | 30s | Específico del drafter |
| **Total primera vez** | **~6 min** | Después: **~1-2 min** (caches hot) |

## Troubleshooting

| Síntoma | Causa probable | Fix |
|---|---|---|
| Health 200 pero `Connection reset` | Falta `--host 0.0.0.0` o usar `--network host` | Ver línea `--host 0.0.0.0 --port 8001` |
| `vllm: error: unrecognized arguments: serve` | ENTRYPOINT de la imagen incluye `serve` ya | Quitar `serve` de `VLLM_ARGS` en el script |
| `KV cache size: 0` | `--gpu-memory-utilization` demasiado bajo | Subir a 0.85 |
| OOM al arrancar | Otro modelo consumiendo GPU | Detenerlo primero (`systemctl --user stop qwen38-flash-next-vllm.service`) |
| Edit-heavy no mejora sobre fresh | Prefix caching desactivado o prompt muy distinto cada vez | Verificar `--enable-prefix-caching` y revisar los prompts |
| Spec decoding no aplica | DSpark drafter no accesible | Verificar el mount `-v ~/vllm/qwen3.8-dspark:/models/qwen3.8-dspark:ro` |
| `min_p` o `logit_bias` rechazado | vLLM 0.27.1 + spec decoding incompatibles | Quitar `min_p`/`logit_bias` del request o desactivar spec decoding con `SPEC=off` |
| `flashinfer autotune warning` | Esperado la primera vez | Espera 2-3 min mientras autotunea |

## Benchmarks (cómo correrlos)

```bash
# Edit bench (single-stream fresh vs edit-heavy)
cd /home/ctala/Playground/gemma4-optimizado/bench-qwen38-nvfp4
python3 edit_bench.py http://127.0.0.1:8001/v1 qwen3.8-27b-nvfp4

# Concurrency sweep (c=1,4,8,16)
python3 conc_bench.py http://127.0.0.1:8001/v1 qwen3.8-27b-nvfp4 --levels=1,4,8,16

# Prefill sweep (8K, 32K, 100K)
python3 prefill_bench.py http://127.0.0.1:8001/v1 qwen3.8-27b-nvfp4

# Determinism (20 prompts × 2 runs)
python3 determinism.py /tmp/det1.json http://127.0.0.1:8001/v1 qwen3.8-27b-nvfp4
python3 determinism.py /tmp/det2.json http://127.0.0.1:8001/v1 qwen3.8-27b-nvfp4
python3 determinism.py --compare /tmp/det1.json /tmp/det2.json
```

---

## Referencias

### Upstream (referencia usada)
- **0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark** — recipe original en FP8, base de esta adaptación
  - Repo: <https://github.com/0xBakeer/Qwen3.8-27B-FP8-on-a-single-DGX-Spark>
  - Adaptamos: FP8 → NVFP4 (`unsloth/Qwen3.8-27B-NVFP4`), agregamos servicio systemd para producción, integración con LiteLLM/Hermes

### Modelos
- `unsloth/Qwen3.8-27B-NVFP4` — checkpoint NVFP4 (~22 GB, incluye MTP heads en `model_mtp.safetensors`)
- `Doopeworld/Qwen3.8-27B-DSpark-vLLM` — DSpark drafter externo (~2.7 GB)
- `Qwen/Qwen3.8-27B-FP8` — el checkpoint FP8 original (referencia upstream)

### Frameworks y deps
- **vLLM 0.27.1-aarch64** (con soporte DSpark)
- **hermes-cli** (LiteLLM proxy) — `~/litellm/config.yaml`
- **hermes-agent** (agent CLI) — `~/.hermes/config.yaml` + plugin `~/.hermes/plugins/`

### Documentación interna relacionada
- **Setup del default actual (Qwen3.8-Flash-Next NVFP4 hybrid)**: [`docs/setup-qwen38-flash-next.md`](setup-qwen38-flash-next.md)
- **Por qué llegamos a 70 tok/s**: `report/why-fast.md`
- **DSpark en detalle**: `report/dspark-explained.md`
- **Comparativa de configs**: `report/compare-configs.md`
- **Resultados detallados del 27B**: `report/benchmark-results.md`, `resultados-qwen38-2026-08-15.md`
