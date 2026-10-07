---
title: 'Benchmark de LLM en DGX Spark (octubre 2026): 23 modelos a 128K × 4 usuarios'
description: 'Benchmark en el mismo equipo de 23 configuraciones de LLM locales en el NVIDIA DGX Spark (GB10, 128 GB): Qwen3.6, Qwen3.8, Nemotron 3, GPT-OSS, Gemma 4, Llama 3.1 y Qwen3 con vLLM 0.31, NIM,
  MTP, DSpark, DFlash y Eagle3.'
keywords:
- benchmark DGX Spark
- mejor modelo DGX Spark
- modelo más rápido DGX Spark
- vLLM GB10
- Qwen3.6 MTP
- NIM vs vLLM
- decodificación especulativa
---

# Benchmark de LLM en DGX Spark — octubre 2026 (128K de contexto × 4 usuarios)

> **Respuesta corta:** el modelo más rápido y completo en un solo NVIDIA DGX Spark es **Qwen3.6-35B-A3B NVFP4 en vLLM 0.31.0 con decodificación especulativa MTP**: **102.3 tok/s** con un usuario y **240.0 tok/s** en total con 4, a 128K de contexto y usando ~47 GB. Sostuvo 218.0 tok/s durante 15 minutos sin peticiones fallidas.
>
> English: [DGX Spark benchmark (October 2026)](/local-llm-agentic-workflows/benchmarks/) · Configuraciones listas: [Recetas](/local-llm-agentic-workflows/recipes.es/) · Que un agente de IA la instale: [Prompt para agentes](/local-llm-agentic-workflows/deploy-with-agent.es/)

Actualizado: 2026-10-06. Equipo: un NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB de memoria unificada, ~121 GB utilizables, aarch64). Todos los modelos corrieron con **la misma configuración**: 128K tokens de contexto por conversación, 4 usuarios simultáneos, caché KV en FP8 y memoria calculada para ese tamaño.

## Cómo medimos

1. **1 usuario:** 3 peticiones con un prompt fijo de ~200 palabras; mediana del tiempo al primer token y de tokens por segundo.
2. **4 usuarios:** 4 peticiones a la vez; tokens por segundo totales.
3. **Prompt largo:** un prompt de ~46K tokens (velocidad de lectura).
4. **Estabilidad:** 3 rondas de 4 usuarios más el prompt largo; cualquier error, petición fallida o reinicio marca el modelo como inestable.
5. **Verificación extendida** del recomendado: prompts variados (código, JSON con herramientas, resumen largo, redacción en español) a temperatura 0 y 0,7, y 15 minutos de carga sostenida.

El script está en [`benchmarks/standard-round/standard_round.py`](https://github.com/ctala/local-llm-agentic-workflows/blob/main/benchmarks/standard-round/standard_round.py) (solo biblioteca estándar) y sirve con cualquier servidor compatible con OpenAI.

## Resultados (ordenados por velocidad con un usuario)

| Modelo | Parámetros (total / activos) | Motor | Especulativa | 1 usuario tok/s | 4 usuarios tok/s (total) | Primer token | Prompt de 46K tokens | Carga | Memoria | Estable |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **Qwen3.6-35B-A3B** | 35B / 3B MoE | vLLM 0.31.0 | MTP k=2 | **102.3** | 240.0 | 106 ms | 9.5 s | 2.7 min | 55.5 GB | sí |
| **GPT-OSS-20B** | 21B / 3.6B MoE | vLLM 0.31.0 | — | **96.2** | 150.6 | 3319 ms | 8.9 s | 2.9 min | 25.7 GB | sí |
| **Nemotron 3.5 Lightning 30B-A3B** | 30B / 3B MoE (hybrid Mamba) | vLLM 0.31.0 | DSpark k=3 | **94.4** | 213.2 | 95 ms | 9.5 s | 4.5 min | 43.5 GB | sí |
| **Qwen3.6-35B-A3B** | 35B / 3B MoE | vLLM 0.31.0 | — | **78.1** | 196.6 | 63 ms | 9.6 s | 2.1 min | 47.0 GB | sí |
| **Nemotron 3.5 Lightning 30B-A3B** | 30B / 3B MoE (hybrid Mamba) | vLLM 0.31.0 | — | **70.3** | 170.7 | 100 ms | 10.2 s | 3.9 min | 45.0 GB | sí |
| **Qwen3.6-35B-A3B** | 35B / 3B MoE | vLLM 0.31.0 | DFlash k=11 | **67.8** | 148.6 | 95 ms | 9.5 s | 3.4 min | 71.8 GB | sí |
| **GPT-OSS-20B** | 21B / 3.6B MoE | NIM 2.0.9 | — | **62.8** | 151.4 | 1650 ms | 13.8 s | 4.8 min | 33.0 GB | sí |
| **GPT-OSS-20B** | 21B / 3.6B MoE | vLLM 0.31.0 | Eagle3 k=3 | **59.4** | 144.7 | 1314 ms | 16.4 s | 2.7 min | 38.4 GB | sí |
| **Nemotron 3 Nano Omni 30B-A3B** | 30B / 3B MoE (hybrid Mamba) | vLLM 0.31.0 | — | **58.8** | 184.6 | 123 ms | 8.6 s | 4.1 min | 43.6 GB | sí |
| **Gemma 4 26B-A4B** | 26B / 4B MoE | vLLM 0.31.0 | MTP drafter k=3 | **56.5** | 186.6 | 119 ms | 29.7 s | 4.8 min | 45.1 GB | sí |
| **Nemotron 3 Nano Omni 30B-A3B** | 30B / 3B MoE (hybrid Mamba) | NIM 2.0.13 | — | **56.0** | 161.3 | 148 ms | 11.4 s | 6.7 min | 62.7 GB | sí |
| **GPT-OSS-120B** | 117B / 5.1B MoE | vLLM 0.31.0 | — | **44.4** | 124.9 | 1921 ms | 14.5 s | 9.3 min | 80.5 GB | sí |
| **GPT-OSS-120B** | 117B / 5.1B MoE | vLLM 0.31.0 | Eagle3 k=3 | **40.5** | 91.8 | 2143 ms | 26.1 s | 9.2 min | 87.5 GB | sí |
| **Gemma 4 26B-A4B** | 26B / 4B MoE | vLLM 0.31.0 | — | **30.8** | 114.0 | 91 ms | 26.0 s | 4.3 min | 39.9 GB | sí |
| **Gemma 4 26B-A4B** | 26B / 4B MoE | vLLM gemma4 image | — | **30.7** | 108.5 | 82 ms | 25.4 s | 3.8 min | 39.4 GB | sí |
| **Qwen3.8-Flash-Next** | 176B / 6B MoE | patched vLLM (qwen38-flash-dgx) | MTP k=2 | **26.8** | 52.5 | 259 ms | 30.2 s | 13.8 min | 95.7 GB | sí |
| **Llama 3.1 8B Instruct** | 8B dense | NIM 2.0.9 | — | **26.4** | 102.3 | 81 ms | 15.5 s | 5.1 min | 33.6 GB | sí |
| **Qwen3-8B** | 8B dense | vLLM 0.31.0 (YaRN x4) | — | **23.2** | 95.7 | 84 ms | 21.7 s | 2.8 min | 55.5 GB | sí |
| **Nemotron 3 Super 120B-A12B** | 120B / 12B MoE (hybrid Mamba) | vLLM 0.31.0 | MTP k=3 | **23.0** | 57.0 | 424 ms | 25.8 s | 13.4 min | 95.2 GB | sí |
| **Qwen3.8-27B** | 27B dense | vLLM 0.31.0 | MTP k=2 | **19.0** | 67.0 | 233 ms | 41.9 s | 7.7 min | 59.5 GB | sí |
| **Nemotron 3 Super 120B-A12B** | 120B / 12B MoE (hybrid Mamba) | vLLM 0.31.0 | — | **16.1** | 42.8 | 326 ms | 22.8 s | 10.3 min | 94.3 GB | sí |
| **Qwen3.8-27B** | 27B dense | vLLM 0.31.0 | DSpark k=14 | **12.5** | 33.9 | 256 ms | 43.6 s | 8.8 min | 63.6 GB | sí |
| **Qwen3.8-27B** | 27B dense | vLLM 0.31.0 | — | **11.0** | 39.6 | 202 ms | 47.7 s | 6.6 min | 47.0 GB | sí |

El primer token de GPT-OSS incluye su fase de razonamiento. «Memoria» es cuánto bajó la memoria libre del Spark al cargar el modelo.

## Decodificación especulativa: ganancia real según el trabajo (Qwen3.6-35B + MTP)

MTP no baja la calidad (el modelo principal verifica cada token propuesto); lo que cambia es cuántas propuestas acierta, y eso depende del contenido.

| Tipo de trabajo | Sin MTP | MTP, temp 0 | MTP, temp 0,7 | Aceptación MTP |
|---|---:|---:|---:|---:|
| Código | – | **–** | – | ––– % |
| JSON con herramientas | – | **–** | – | ––– % |
| Resumen largo | – | **–** | – | ––– % |
| Redacción en español | – | **–** | – | ––– % |
| **15 min sostenidos, 4 usuarios** | 170.9 | **218.0** (mixto) | | 69.3 % |

- **Agentes y código ganan más** (+55–65 %); la redacción libre, menos (+20 %).
- **Gemma 4 con su drafter MTP oficial** casi duplicó: 30.8 → 56.5 tok/s.
- **DSpark** ayudó a Nemotron 3.5 Lightning (70.3 → 94.4) pero casi nada a Qwen3.8-27B con texto nuevo (11.0 → 12.5); rinde en edición repetitiva de código con caché caliente.
- **Eagle3** hizo más lento a GPT-OSS en el Spark (96.2 → 59.4 en 20B, 44.4 → 40.5 en 120B): el drafter obliga a usar atención Triton en vez de FlashInfer y cuesta más de lo que ahorra. Usa GPT-OSS sin él.
- **DFlash** sobre la receta rápida de Qwen3.6 fue *más lento* que sin él (67.8 contra 78.1) y pidió mucha más memoria.

## NIM contra contenedor propio con vLLM

| Modelo | NIM (1 usuario / 4) | vLLM 0.31.0 (1 usuario / 4) |
|---|---|---|
| GPT-OSS-20B | 62.8 / 151.4 | **96.2** / 150.6 |
| Nemotron 3 Nano Omni | 56.0 / 161.3 | **58.8** / 184.6 |

Los NIM son cómodos y aceptan flags de vLLM, pero en el Spark una imagen de vLLM reciente fue más rápida. Revisa la arquitectura antes de descargar: `nvcr.io/nim/qwen/qwen3.8-27b` y `nvcr.io/nim/qwen/qwen3-32b` estaban **solo para amd64** el 2026-10-06.

## Lo que no cupo o no corrió

- **Llama 3.3 70B NVFP4:** no cabe a 128K × 4 (denso, ~160 KB de caché por token → ~133 GB). Úsalo con 1–2 usuarios.
- **Gemma 4 con MTP en la imagen `gemma4` antigua:** no soportado; funciona en vLLM 0.31.0.
- **Qwen3.8-Flash-Next:** necesita una imagen de vLLM parchada; corre (26.8 tok/s) pero usa ~91 GB y tarda ~14 minutos en cargar.

## Recomendación por caso de uso

| Caso de uso | Elección | Por qué |
|---|---|---|
| Agentes / herramientas (Hermes, OpenClaw) | **Qwen3.6-35B + MTP** | El más rápido, 80–90 % de aceptación MTP en JSON y herramientas, entiende imagen y video |
| Asistente de código | **Qwen3.6-35B + MTP**; segunda opinión con **GPT-OSS-120B** | Iteración rápida y otra familia de modelos para revisar |
| Multimodal con audio y video | **Nemotron 3 Nano Omni (vLLM)** | Texto, imagen, audio y video en un modelo; la lectura de prompts largos más rápida |
| Modelo general rápido y liviano | **Nemotron 3.5 Lightning + DSpark** o **GPT-OSS-20B** | 94.4 / 96.2 tok/s con un usuario |
| Razonamiento profundo, documentos largos, menos usuarios | **Nemotron 3 Super 120B + MTP** | El modelo más grande que cabe a 128K × 4; caché mínima por su diseño híbrido Mamba |

## Preguntas frecuentes

### ¿Cuál es el modelo más rápido en el NVIDIA DGX Spark?

En nuestra ronda de octubre de 2026 con 128K de contexto y 4 usuarios simultáneos, Qwen3.6-35B-A3B NVFP4 on vLLM 0.31.0 with MTP speculative decoding fue el más rápido: 102.3 tok/s con un usuario y 240.0 tok/s en total con 4 usuarios, usando unos 47 GB de los 128 GB de memoria unificada.

### ¿Un NIM de NVIDIA es más rápido que vLLM en el DGX Spark?

No en nuestras pruebas. Con el mismo modelo, un contenedor propio con vLLM 0.31.0 fue más rápido que el NIM oficial: GPT-OSS-20B dio 96.2 contra 62.8 tok/s con un usuario, y Nemotron 3 Nano Omni 58.8 contra 56.0. Algunos NIM (Qwen3-32B, Qwen3.8-27B) solo existen para amd64 y no corren en el Spark.

### ¿Sirve la decodificación especulativa (MTP, DSpark, Eagle3) en el DGX Spark?

Sí, sobre todo para agentes y código. Qwen3.6 con MTP pasó de ~78 a ~125 tok/s en código y JSON con herramientas (80-90 % de aceptación), pero solo a ~95 tok/s en redacción libre (~50 %). Gemma 4 con su drafter MTP oficial casi duplicó (31 a 57 tok/s). DSpark ayudó a Nemotron 3.5 Lightning (+34 %) pero casi nada a Qwen3.8-27B con texto nuevo. Eagle3 hizo más lento a GPT-OSS (96 a 59 tok/s en 20B), así que conviene no usarlo ahí.

### ¿Puedo correr Llama 3.3 70B con 128K de contexto y 4 usuarios en un DGX Spark?

No. Llama 3.3 70B es denso, con 80 capas de atención completa: su caché KV pide unos 160 KB por token, y cuatro conversaciones de 128K más los pesos suman ~133 GB, más que los 121 GB utilizables del Spark. Usa menos usuarios o menos contexto.

### ¿Qué modelo usar para agentes, código, contenido o investigación en el DGX Spark?

Agentes (herramientas, baja latencia): Qwen3.6-35B + MTP. Código: Qwen3.6-35B + MTP, con GPT-OSS-120B como segunda opinión independiente. Multimodal con audio y video: Nemotron 3 Nano Omni (vLLM). Razonamiento largo y cuidadoso con menos usuarios: Nemotron 3 Super 120B + MTP.
