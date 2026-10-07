# LLMs locales y flujos agénticos en el NVIDIA DGX Spark

Benchmarks en el mismo equipo, recetas `docker run` probadas y un prompt para que un agente despliegue modelos LLM locales en el **NVIDIA DGX Spark** (GB10 Grace Blackwell, 128 GB de memoria unificada, aarch64) y otras estaciones edge AI de 96 a 128 GB.

**Sitio:** [ctala.github.io/local-llm-agentic-workflows](https://ctala.github.io/local-llm-agentic-workflows/benchmarks.es/) · [Read in English](README.md)

## Por dónde empezar

- **[Benchmark de octubre 2026](https://ctala.github.io/local-llm-agentic-workflows/benchmarks.es/)**: 23 configuraciones con 128K de contexto y 4 usuarios simultáneos, en vLLM 0.31.0 y NVIDIA NIM.
- **[Recetas](https://ctala.github.io/local-llm-agentic-workflows/recipes.es/)**: descarga de pesos y `docker run` de cada configuración. Los YAML están en [`recipes/`](recipes/).
- **[Prompt para agentes](https://ctala.github.io/local-llm-agentic-workflows/deploy-with-agent.es/)**: pégalo en Claude Code, Codex u OpenCode dentro del Spark y el agente despliega la receta por ti.

## Respuestas rápidas (octubre 2026, vLLM 0.31.0, 128K de contexto × 4 usuarios, caché KV FP8)

| Necesidad | Modelo | 1 usuario | 4 usuarios (total) | Memoria |
|---|---|---:|---:|---:|
| El más rápido, multimodal, herramientas | Qwen3.6-35B-A3B NVFP4 + MTP | 102 tok/s | 240 tok/s | ~56 GB |
| Pequeño y rápido | GPT-OSS-20B | 96 tok/s | 151 tok/s | ~26 GB |
| Agentes, contexto largo | Nemotron 3.5 Lightning 30B-A3B + DSpark | 94 tok/s | 213 tok/s | ~44 GB |
| Entrada de imagen, audio y video | Nemotron 3 Nano Omni 30B-A3B | 59 tok/s | 185 tok/s | ~44 GB |
| Imagen y audio, modelo de Google | Gemma 4 26B-A4B + drafter MTP | 57 tok/s | 187 tok/s | ~45 GB |
| Modelo grande de razonamiento | GPT-OSS-120B | 44 tok/s | 125 tok/s | ~81 GB |
| El más grande que cabe | Nemotron 3 Super 120B-A12B + MTP | 23 tok/s | 57 tok/s | ~95 GB |

Lo que aprendimos:

- **La decodificación especulativa rinde en agentes y código.** Qwen3.6 + MTP pasa de ~79 a ~125–129 tok/s en código y JSON de herramientas (84–90 % de aceptación), pero solo a ~95 tok/s en redacción libre.
- **No todos los drafters ayudan.** Eagle3 hizo más lento a GPT-OSS (20B: 96 → 59 tok/s), DFlash fue más lento que Qwen3.6 sin él, y DSpark casi no ayudó a Qwen3.8-27B con texto nuevo.
- **NIM frente a vLLM:** con el mismo modelo, un contenedor propio con vLLM 0.31.0 igualó o superó al NIM oficial. Algunos NIM solo existen para amd64 y no corren en el Spark.
- **Nemotron 3 Super 120B ya funciona con vLLM 0.31.0** (con imágenes anteriores fallaba).
- **Define siempre `--gpu-memory-utilization`.** Con memoria unificada, el 0.9 por defecto de vLLM reserva ~110 GB incluso para un modelo de 20 GB.
- **Llama 3.3 70B no cabe con 128K × 4** (caché KV densa, ~133 GB). Úsalo con 1 o 2 usuarios.

## Usar una receta

```bash
git clone https://github.com/ctala/local-llm-agentic-workflows.git && cd local-llm-agentic-workflows
pip install pyyaml "huggingface_hub[cli]"
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir ~/vllm/qwen3.6-35b-a3b-nvfp4-nvidia
python3 scripts/recipe_to_docker.py recipes/qwen36-35b-a3b-nvfp4-rapida-v031-mtp.yaml   # imprime el docker run
```

Cada receta trae sus comandos de descarga (`download`), la configuración medida (`measured`: contexto, usuarios, caché KV, `gpu_util`, velocidad) y la entrada completa (`entry`: imagen, variables y argumentos de vLLM). Para otra configuración, pasa `--ctx`, `--users`, `--kv` o `--gpu-util` a `recipe_to_docker.py`.

Para medir tu propio endpoint con el mismo método:

```bash
python3 benchmarks/standard-round/standard_round.py --help   # solo biblioteca estándar de Python, cualquier endpoint compatible con OpenAI
```

## Resultados anteriores (abril a septiembre de 2026)

Las rondas previas compararon vLLM y TensorRT-LLM con imágenes antiguas, otros tamaños de contexto y 8 peticiones simultáneas, así que sus cifras no se comparan directamente con la ronda de octubre. Están en [Resultados](https://ctala.github.io/local-llm-agentic-workflows/results.es/) y en el [log de configuración](https://ctala.github.io/local-llm-agentic-workflows/setup.es/).

## Licencia

MIT. Revisa [LICENSE](./LICENSE).
