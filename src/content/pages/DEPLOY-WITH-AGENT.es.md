---
title: Prompt para que un agente despliegue un LLM en el DGX Spark
description: 'Prompt listo para Claude Code, Codex u OpenCode: descarga los pesos, arranca la receta vLLM o NIM probada y verifica la velocidad en un DGX Spark.'
keywords:
- DGX Spark agent
- Claude Code DGX Spark
- deploy vLLM with agent
- AI agent model deployment
---


# Prompt para que un agente despliegue un LLM en el DGX Spark

Pega este prompt en un agente de código que corra **en el Spark** (por SSH) y reemplaza `<RECIPE_ID>` por una de las [recetas](../recipes.es/). El agente descarga, arranca, verifica y te informa la velocidad frente a la que medimos.

Si no sabes cuál elegir: `qwen36-35b-a3b-nvfp4-rapida-v031-mtp` es la más rápida para uso general; mira las [recomendaciones por caso de uso](../benchmarks.es/).

```text
Vas a desplegar un LLM local en un NVIDIA DGX Spark (GB10, 128 GB de memoria unificada, aarch64, Ubuntu).
Usa la receta probada `<RECIPE_ID>` de https://github.com/ctala/local-llm-agentic-workflows y sigue estos pasos. Detente y pregúntame antes de cualquier acción destructiva.

1. Revisa el equipo: `nvidia-smi`, `free -g`, `docker info`. Si otro contenedor de modelo usa más de 40 GB, avísame y pregunta antes de detenerlo.
2. Clona el repo (o haz `git pull`) y lee `recipes/<RECIPE_ID>.yaml`: `measured` tiene el contexto, los usuarios, la caché KV, el gpu_util y la velocidad esperada; `entry` tiene la imagen, las variables y los argumentos de vLLM.
3. Descarga los pesos con cada comando de `download` (requiere `pip install "huggingface_hub[cli]"` y, en modelos con acceso restringido, `HF_TOKEN`). Las recetas NIM necesitan `NGC_API_KEY` en el entorno; nunca la muestres.
4. Genera el comando con `python3 scripts/recipe_to_docker.py recipes/<RECIPE_ID>.yaml --port 8000` y ejecútalo. Respeta el `--gpu-memory-utilization` de la receta: con memoria unificada, el 0.9 por defecto de vLLM ocupa ~110 GB.
5. Espera a que `curl -s localhost:8000/v1/models` liste el modelo (sigue `docker logs -f <RECIPE_ID>`; los modelos grandes tardan de 3 a 12 minutos). Si falla, lee las últimas 80 líneas del log e informa la causa antes de cambiar flags.
6. Prueba rápida: una respuesta de chat con un prompt corto y `max_tokens: 256`; informa los tokens/s. Si la receta incluye `tools`, envía también una llamada a herramienta.
7. Informa: la URL del endpoint, el nombre del modelo servido, la memoria usada (`nvidia-smi`), la velocidad medida frente a `measured.single_user_tok_s` y el comando exacto que ejecutaste.

No cambies puertos de otros servicios, no ejecutes `docker system prune` y no expongas el puerto fuera de localhost salvo que te lo pida.
```

## Recetas más rápidas

- `qwen36-35b-a3b-nvfp4-rapida-v031-mtp`: 102.3 tok/s, ≈55.5 GB
- `gpt-oss-20b-v031`: 96.2 tok/s, ≈25.7 GB
- `nemotron35-lightning-nvfp4-dspark-v031`: 94.4 tok/s, ≈43.5 GB
- `qwen36-35b-a3b-nvfp4-rapida-v031`: 78.1 tok/s, ≈47.0 GB
- `nemotron35-lightning-nvfp4-v031`: 70.3 tok/s, ≈45.0 GB
- `qwen36-35b-a3b-nvfp4-rapida-v031-dflash`: 67.8 tok/s, ≈71.8 GB
