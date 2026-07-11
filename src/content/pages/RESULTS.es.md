# Resultados finales: Gemma 4 / Qwen 3.6 / Nemotron 3 en NVIDIA DGX Spark

> Para una guía rápida de uso, véase [Inicio](/local-llm-agentic-workflows/). Este documento conserva el detalle completo de resultados y notas técnicas.

## Hardware / software base

| Componente | Valor |
|------------|-------|
| Hardware | NVIDIA DGX Spark (GB10 Grace Blackwell) |
| CPU | 20 cores ARM64 (aarch64) |
| GPU | NVIDIA GB10 (sm_121) |
| Memoria | 128 GB LPDDR5x unificada (~121 GB usable) |
| Driver NVIDIA | 580.142 |
| CUDA | 13.0 |
| Docker | 29.2.1 |
| NVIDIA Container Toolkit | 1.19.0 |

**Restricción crítica del GB10**: no tiene compute FP4 nativo. Los checkpoints NVFP4 se ejecutan vía el backend **Marlin** (`--moe-backend marlin`), que descomprime FP4 → BF16 en runtime. Esto explica por qué los tok/s están por debajo de lo que se vería en un B200 con FP4 nativo.

---

## Resumen de resultados

Benchmark realizado con prompt de ~120 tokens en español, `max_tokens=512`, temperatura 0.7, midiendo **decode tok/s** (descartando TTFT de la primera corrida fría).

| Modelo | Checkpoint usado | Contenedor vLLM | Mejor decode tok/s | TTFT caliente | VRAM usada | Notas |
|--------|------------------|-----------------|--------------------|---------------|------------|-------|
| **Gemma 4 26B-A4B IT** | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` + `gemma4_patched.py` | `vllm/vllm-openai:gemma4-cu130` | **~49.5 tok/s** | ~0.08 s | ~22 GB | Mejor opción para agentes. Requiere parche community. |
| Gemma 4 26B-A4B IT (oficial) | `nvidia/Gemma-4-26B-A4B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~30.1 tok/s | ~0.20 s | ~21 GB | Funciona sin parche, pero es ~20 tok/s más lento. |
| **Gemma 4 31B IT** | `nvidia/Gemma-4-31B-IT-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~6.7 tok/s** | ~1.8 s | ~31 GB | Denso, limitado por ancho de banda. No recomendable si se busca fluidez. |
| **Qwen 3.6 35B-A3B** | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:nightly` | **~75–77 tok/s** | ~0.10 s | ~22 GB | **Recomendado actual.** W4A16 NVFP4 (`modelopt`), parser `qwen3_coder`, 262K contexto. |
| Qwen 3.6 35B-A3B | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~42.2 tok/s | ~0.10 s | ~22 GB | Formato `compressed-tensors`. Fallback estable. |
| Qwen 3.6 35B-A3B (n-gram speculative) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~34–37 tok/s | ~0.10 s | ~22 GB | Empeora para texto no repetitivo. |
| Qwen 3.6 35B-A3B (MTP) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | Error de carga | – | – | `moe_backend='marlin'` no es soportado por el drafter no cuantizado. Requiere más investigación. |
| Gemma 4 26B-A4B en TensorRT-LLM 1.3.0rc13 | `nvidia/Gemma-4-26B-A4B-NVFP4` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | Error de carga | – | – | Transformers en el contenedor no reconoce `model_type: gemma4`. |
| Qwen 3.6 35B-A3B en TensorRT-LLM 1.3.0rc13 | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | Error de carga | – | – | `AssertionError` en `quant_algo` del checkpoint modelopt NVFP4. |
| **Qwen 3.6 35B-A3B TRT-LLM (MLP-only NVFP4)** | Cuantiizado con Model Optimizer 0.44.0 desde `Qwen/Qwen3.6-35B-A3B` BF16 | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~34.4 tok/s** | ~0.09 s | ~41 GB | Formato `modelopt` NVFP4 MLP-only + KV FP8. Funciona con TRT-LLM PyTorch backend. |
| **Nemotron-3-Nano-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~28.8 tok/s** | ~0.22 s | **~118 GB** | Modelo denso BF16. Ocupa casi toda la memoria unificada. |
| Nemotron-3-Nano-30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `vllm/vllm-openai:gemma4-0505-cu130` | ~28.3 tok/s | ~0.20 s | ~72 GB | Alternativa vLLM. Requiere liberar VRAM de otros servicios. |
| **Nemotron-3-Super-120B-A12B** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~14.7 tok/s** | ~0.29 s | **~110 GB** | Checkpoint NVFP4 oficial. Mayor calidad, menor velocidad por más expertos activos. |
| Nemotron-3-Super-120B-A12B | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | — | — | — | **No viable**: `CUDA OOM` al inicializar el engine; el Spark se colgó en el primer intento por falta de memoria. |
| **Nemotron-3-Nano-Omni-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~40.0 tok/s** | ~0.10 s | **~40 GB** | **Multimodal oficial**: texto + imagen funcionan. Audio aún no probado funcional en este contenedor. |

### Conclusiones rápidas

- **Para mejor calidad/velocidad (~75–77 tok/s)**: usar **Qwen 3.6 35B-A3B nvidia NVFP4 + vLLM nightly**. También soporta los 262K de contexto del modelo y tool calling robusto.
- **Para máxima velocidad (~50 tok/s)**: usar **Gemma 4 26B-A4B community + parche**.
- **Qwen 3.6 35B-A3B RedHatAI** (~42 tok/s) sigue siendo un fallback estable si el checkpoint nvidia o la imagen nightly no están disponibles.
- **Gemma 4 31B** no es viable para uso interactivo rápido en GB10 (~7 tok/s).
- **TensorRT-LLM** es funcional para Qwen 3.6 si se cuantiza manualmente el modelo base BF16 a **NVFP4 MLP-only** (~34 tok/s), pero por ahora es más lento que vLLM con el checkpoint nvidia.
- **Speculative decoding** no ayudó en estos prompts generales; puede ser útil solo para texto muy repetitivo o si se configura un drafter MTP compatible.
- **Nemotron-3-Nano-Omni** con vLLM da ~40 tok/s en texto e **sí procesa imágenes**. El audio falló en las pruebas por problemas de decodificación en el contenedor, no por el modelo en sí.
- **Nemotron-3-Super-120B-A12B** solo es estable con **TensorRT-LLM**. Con vLLM el engine falla por `CUDA OOM` o colgaba el Spark; ver sección detallada más abajo.

---

## Scripts recomendados

### 1. Gemma 4 26B-A4B (mejor velocidad)

Requiere descargar `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` y tener `gemma4_patched.py` en el mismo directorio.

```bash
#!/usr/bin/env bash
# run-gemma4-26b-a4b.sh

docker run -d --name gemma4-26b-a4b \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v ~/vllm/gemma4-26b-a4b-nvfp4-community:/models/gemma4 \
  -v ~/vllm/gemma4-26b-a4b-nvfp4-community/gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py \
  -e HF_HOME=/models \
  vllm/vllm-openai:gemma4-cu130 \
    --model /models/gemma4 \
    --served-model-name gemma-4-26b-a4b \
    --host 0.0.0.0 --port 8000 \
    --quantization modelopt \
    --moe-backend marlin \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser pythonic
```

### 2. Qwen 3.6 35B-A3B (mejor calidad-velocidad)

Requiere descargar `nvidia/Qwen3.6-35B-A3B-NVFP4`.

```bash
#!/usr/bin/env bash
# run-qwen36-35b-a3b.sh

docker run -d --name qwen36-35b-a3b \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v ~/vllm/qwen3.6-35b-a3b-nvfp4-nvidia:/models/qwen3.6 \
  -v $(pwd)/chat-templates:/chat-templates \
  -e HF_HOME=/models \
  -e VLLM_TARGET_DEVICE=cuda \
  vllm/vllm-openai:nightly@sha256:a671d5fcda70fe9ac6f245f9780821de459fb4ee22c018fd07a0f10a55279bf9 \
    --model /models/qwen3.6 \
    --served-model-name qwen3.6-35b-a3b \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --attention-backend flashinfer \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 262144 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":4}' \
    --load-format fastsafetensors \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --chat-template /chat-templates/qwen3.6-miaai.jinja \
    --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"auto_disable_thinking_with_tools":true}'
```

### 3. Qwen 3.6 35B-A3B con TensorRT-LLM (cuantizado MLP-only NVFP4)

Requiere generar el checkpoint con TensorRT Model Optimizer (ver registro de trabajo). El checkpoint resultante ocupa ~22 GB.

```bash
#!/usr/bin/env bash
# run-qwen36-35b-a3b-trtllm.sh

docker run -d --name qwen36-35b-a3b-trtllm \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v ~/vllm/qwen3.6-35b-a3b-nvfp4-mlponly-user:/models/qwen3.6 \
  -e HF_HOME=/models \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 \
    trtllm-serve /models/qwen3.6 \
    --host 0.0.0.0 --port 8000 \
    --backend pytorch \
    --max_seq_len 32768 \
    --max_batch_size 2 \
    --kv_cache_dtype fp8
```

### 4. Nemotron-3-Nano-30B-A3B con TensorRT-LLM (BF16)

Requiere descargar `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (~89 GB).

```bash
#!/usr/bin/env bash
# run-nemotron3-nano-30b-a3b-trtllm.sh

docker run -d --name nemotron3-nano-30b-a3b-trtllm \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v ~/vllm/nemotron3-nano-30b-a3b-bf16:/models/nemotron \
  -e HF_HOME=/models \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 \
    trtllm-serve /models/nemotron \
    --host 0.0.0.0 --port 8000 \
    --backend pytorch \
    --max_seq_len 8192 \
    --max_batch_size 1 \
    --kv_cache_dtype fp8
```

### 5. Nemotron-3-Super-120B-A12B con TensorRT-LLM (NVFP4)

Requiere descargar `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (~75 GB).

```bash
#!/usr/bin/env bash
# run-nemotron3-super-120b-a12b-trtllm.sh

docker run -d --name nemotron3-super-120b-a12b-trtllm \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v ~/vllm/nemotron3-super-120b-a12b-nvfp4:/models/nemotron \
  -e HF_HOME=/models \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 \
    trtllm-serve /models/nemotron \
    --host 0.0.0.0 --port 8000 \
    --backend pytorch \
    --max_seq_len 8192 \
    --max_batch_size 1 \
    --kv_cache_dtype fp8
```

### 6. Nemotron-3-Nano-30B-A3B con vLLM (BF16)

Requiere descargar `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (~89 GB). Es una alternativa vLLM al script de TRT-LLM; úsala si prefieres el stack vLLM y has liberado suficiente VRAM.

```bash
#!/usr/bin/env bash
# run-nemotron3-nano-30b-a3b-vllm.sh

docker run -d --name nemotron3-nano-30b-a3b-vllm \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v ~/vllm/nemotron3-nano-30b-a3b-bf16:/models/nemotron \
  -e HF_HOME=/models \
  vllm/vllm-openai:gemma4-0505-cu130 \
    --model /models/nemotron \
    --served-model-name nemotron3-nano-30b-a3b \
    --host 0.0.0.0 --port 8000 \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

### 7. Nemotron-3-Nano-Omni-30B-A3B con vLLM (multimodal NVFP4)

Requiere descargar `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (~21 GB). Es la opción oficial de NVIDIA para texto + imagen + video + audio.

```bash
#!/usr/bin/env bash
# run-nemotron3-nano-omni-vllm.sh

docker run -d --name nemotron3-nano-omni-vllm \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v ~/vllm/nemotron3-nano-omni-30b-a3b-reasoning-nvfp4:/models/nemotron \
  -e HF_HOME=/models \
  vllm/vllm-openai:gemma4-0505-cu130 \
    --model /models/nemotron \
    --served-model-name nemotron3-nano-omni \
    --host 0.0.0.0 --port 8000 \
    --quantization modelopt_fp4 \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

Para probar la multimodalidad:

```bash
# Imagen
python3 benchmarks/test_multimodal.py nemotron3-nano-omni image /ruta/a/imagen.png

# Audio (aún con problemas de formato en este contenedor)
python3 benchmarks/test_multimodal.py nemotron3-nano-omni audio /ruta/a/audio.wav
```

### 8. Gemma 4 31B IT (si se necesita el modelo denso)

Requiere descargar `nvidia/Gemma-4-31B-IT-NVFP4`.

```bash
#!/usr/bin/env bash
# run-gemma4-31b.sh

docker run -d --name gemma4-31b \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size 64gb \
  -v ~/vllm/gemma4-31b-it-nvfp4:/models/gemma31 \
  -e HF_HOME=/models \
  vllm/vllm-openai:gemma4-0505-cu130 \
    --model /models/gemma31 \
    --served-model-name gemma-4-31b-it \
    --host 0.0.0.0 --port 8000 \
    --quantization modelopt \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser pythonic
```

### 9. Qwen 3.6 35B-A3B con contexto extremo (262K × 2 sesiones)

Requiere descargar `nvidia/Qwen3.6-35B-A3B-NVFP4`. Esta es la configuración recomendada para agentes que necesitan el máximo contexto posible en el Spark. El script principal `run-qwen36-35b-a3b.sh` ya incluye esta configuración; `run-qwen36-35b-a3b-extreme-context-2seq.sh` es un alias para el mismo script.

```bash
./scripts/run-qwen36-35b-a3b.sh
```

> **Importante**: Qwen 3.6 requiere `--tool-call-parser qwen3_coder` (más robusto para multi-turn que el anterior `qwen3_xml`); sin parser, vLLM devuelve XML en `content` y el array nativo `tool_calls` queda vacío, por lo que agentes como Hermes/OpenClaw no ejecutan las herramientas.
>
> **Configuración estable**: `--max-num-seqs 2` con `gpu-memory-utilization 0.92` deja margen para LiteLLM, ASR y otros servicios auxiliares. Configuraciones con 3 o más secuencias a 262K no están validadas con el checkpoint nvidia + vLLM nightly.

---

## Nemotron-3 Super 120B-A12B con vLLM: por qué no funcionó

Se hicieron dos intentos de servir `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` con `vllm/vllm-openai:gemma4-0505-cu130` usando el comando:

```bash
docker run -d --name vllm-nemotron3-super --gpus all --ipc host --network host --shm-size 64gb \
  -v ~/vllm/nemotron3-super-120b-a12b-nvfp4:/models/nemotron \
  -e HF_HOME=/models \
  vllm/vllm-openai:gemma4-0505-cu130 \
    --model /models/nemotron --served-model-name nemotron3-super-120b-a12b \
    --host 0.0.0.0 --port 8000 --quantization modelopt_fp4 \
    --moe-backend marlin --kv-cache-dtype fp8 --max-model-len 8192 \
    --max-num-seqs 1 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 --trust-remote-code
```

### Primer intento

El contenedor cargó los 17 shards del checkpoint (~75 GB en disco) en ~8 min. Los últimos logs mostraron:

```
Loading weights took 489.67 seconds
WARNING ... Your GPU does not have native support for FP4 computation ...
Using MoEPrepareAndFinalizeNoDPEPModular
Using uncalibrated q_scale 1.0 ...
```

Antes de que el servidor HTTP levantara, el Spark se colgó y tuvo que reiniciarse. La causa fue agotamiento de memoria: en paralelo corrían dos servidores `llama-server` con modelos Qwen GGUF que ocupaban ~76 GB de la memoria unificada.

### Segundo intento (después de liberar memoria)

Tras detener los servicios `qwen27-local.service` y `qwen35-local.service` (llama-server), el sistema quedó con ~117 GB libres. Sin embargo, vLLM falló inmediatamente al inicializar el `EngineCore`:

```
torch.AcceleratorError: CUDA error: out of memory
  File "/usr/local/lib/python3.12/dist-packages/vllm/utils/mem_utils.py", line 108, in measure
    self.free_memory, self.total_memory = current_platform.mem_get_info(device)
```

Es decir, vLLM no pudo ni siquiera medir la memoria disponible sin encontrar un OOM.

### Diagnóstico

El modelo tiene una arquitectura híbrida Mamba-MoE (`NemotronHForCausalLM`) con:

- 88 capas
- 512 expertos enrutados
- 22 expertos activos por token
- 1 experto compartido
- Hidden size 4096, head dim 128, GQA con 2 KV heads

Aunque los pesos en disco ocupan ~75 GB, al descomprimir FP4 → BF16 vía Marlin y reservar bloques de KV cache según `--gpu-memory-utilization 0.90`, el footprint supera el margen seguro de 128 GB unificados. El V1 engine de vLLM también compila kernels con `torch.compile`/`inductor`, lo que añade overhead de memoria adicional.

### Comparación con TensorRT-LLM

El mismo checkpoint con `trtllm-serve --backend pytorch --max_seq_len 8192 --max_batch_size 1 --kv_cache_dtype fp8` carga y sirve establemente a **~14.7 tok/s** usando ~110 GB. TRT-LLM reserva un presupuesto de memoria fijo (modelo + KV cache para seq_len/batch dados + activaciones) en lugar del esquema porcentual y dinámico de vLLM.

**Conclusión**: en DGX Spark, **Nemotron-3 Super 120B-A12B solo debe usarse con TensorRT-LLM**. vLLM es viable para Nemotron-3 Nano (30B) y Nano Omni (30B multimodal), pero no para el Super.

---

## Escalado extremo de contexto: Qwen 3.6 35B-A3B

Para flujos de agentes que necesitan ingerir contextos muy largos (bases de código, historial de conversación, RAG, trazas multi-turno), probamos hasta dónde escala `nvidia/Qwen3.6-35B-A3B-NVFP4` en el DGX Spark. Usamos vLLM nightly con:

```bash
--model /models/qwen3.6 \
--trust-remote-code \
--tensor-parallel-size 1 \
--attention-backend flashinfer \
--moe-backend marlin \
--kv-cache-dtype fp8 \
--gpu-memory-utilization 0.92 \
--max-model-len 262144 \
--max-num-seqs 2 \
--max-num-batched-tokens 32768 \
--enable-chunked-prefill \
--async-scheduling \
--enable-prefix-caching \
--load-format fastsafetensors \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--reasoning-parser qwen3
```

**2 sesiones concurrentes es la configuración estable recomendada**, ya que deja margen para LiteLLM, ASR y otros servicios auxiliares. La variante de 3 sesiones se probó con el checkpoint RedHatAI anterior, pero dejaba poca memoria libre; no ha sido revalidada con el checkpoint nvidia + vLLM nightly.

### Escalado de contexto en una sola sesión

| Tokens de entrada | Tokens de salida | TTFT | Decode tok/s | Notas |
|-------------------|------------------|------|--------------|-------|
| 1,000 | 32 | 0.28 s | 45.57 | Línea base caliente. |
| 50,000 | 64 | 27.1 s | 45.66 | Primera llamada grande; incluye algo de JIT warmup. |
| 100,000 | 64 | 22.87 s | 39.80 | TTFT más rápido que 50K porque los kernels ya están calientes. |
| 200,000 | 64 | 65.45 s | 33.17 | Estable, memoria ~120 GB. |
| 262,000 | 64 | 56.49 s | 30.22 | Cerca del límite duro del modelo (262,144 tokens). |

### Escalado de contexto concurrente (2 sesiones)

Los datos históricos de 3 sesiones se mantienen como referencia; con 2 sesiones el prefill total baja de 786K a 524K tokens, por lo que el TTFT en 262K por sesión debería ser menor.

| Tokens/sesión | Sesiones | Notas |
|---------------|----------|-------|
| 50,000 | 2 | Excelente interactividad. |
| 100,000 | 2 | Aún muy responsivo. |
| 200,000 | 2 | Prefill por chunks mantiene el tiempo total bajo. |
| 262,000 | 2 | Funciona; TTFT menor que con 3 sesiones. |

### Comportamiento de memoria

- **En reposo después de cargar (2 sesiones)**: ~117–119 GB usados / ~121 GB totales, dejando margen para servicios auxiliares.
- **Estable**: sin OOM, sin colgarse, sin reinicios obligatorios durante estas pruebas.

### Guía práctica para agentes

- **Turno típico de agente**: agentes tipo OpenClaw / Hermes suelen usar **8K–32K tokens** de contexto activo por sesión.
- **Configuración de producción conservadora**: **2 sesiones paralelas × 64K de contexto** corren con TTFT sub-segundo y dejan margen para LiteLLM/ASR.
- **Máximo contexto por sesión**: se alcanza **~262K tokens** con 2 sesiones concurrentes por defecto; la variante de 3 sesiones es posible pero deja poco margen.
- **No uses 4 sesiones a 262K** a menos que la máquina esté dedicada a un solo modelo y puedas tolerar colgues por picos de memoria.

---

## Integración con Open WebUI / n8n

Todos los servidores exponen la API OpenAI-compatible en `http://localhost:8000/v1`. Usa la URL base y el `served-model-name` como modelo.

| Herramienta | URL base | Modelo |
|-------------|----------|--------|
| Open WebUI | `http://localhost:8000/v1` | `gemma-4-26b-a4b` / `qwen3.6-35b-a3b` |
| n8n (OpenAI node) | `http://localhost:8000/v1` | igual que arriba |
| API key | cualquier valor (vLLM no valida por defecto) | – |

> Nota: solo un contenedor puede escuchar en el puerto 8000 a la vez. Para correr varios modelos simultáneamente, usa puertos distintos (ej. 8001, 8002) o un proxy como LiteLLM.

---

## Notas técnicas importantes

1. **ARM64**: verifica siempre que la imagen Docker tenga manifest `arm64`.
2. **Marlin obligatorio para MoE NVFP4 en GB10**: `--moe-backend marlin`.
3. **KV cache FP8**: ahorra memoria, pero usa factores de escala del checkpoint; si no existen, vLLM advierte que puede haber pérdida de precisión.
4. **Prefix caching**: acelera requests con contexto compartido o prompts largos repetidos.
5. **Tool calling**: Qwen 3.6 requiere `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`; el parser `qwen3_coder` es más robusto para multi-turn que el anterior `qwen3_xml`. Sin parser, vLLM devuelve XML en `content` y el array nativo `tool_calls` queda vacío, por lo que Hermes/OpenClaw no ejecutan las herramientas. Gemma 4 también soporta `gemma4` nativo, no probado a fondo aquí.
6. **max-num-batched-tokens**: para modelos con input multimodal (Gemma 4), debe ser >= `max_tokens_per_mm_item` (ej. 2496 para Gemma 4, 4096 por defecto).
7. **TensorRT Model Optimizer en GB10**: la cuantización de Qwen 3.6 desde BF16 requiere convertir el checkpoint VLM a text-only y usar la memoria total de GPU (no la libre) porque `accelerate` no entiende el pool unificado de 128 GB.
8. **TRT-LLM PyTorch backend** lee `hf_quant_config.json`; usa `--backend pytorch` y `--kv_cache_dtype fp8` para checkpoints NVFP4 HF.

---

## Intento con TensorRT-LLM oficial

Se probaron los contenedores y flujos oficiales de NVIDIA para Spark:

| Contenedor | Versión TRT-LLM | Gemma 4 26B | Qwen 3.6 35B | Nemotron 3 Nano | Nemotron 3 Super |
|------------|-----------------|-------------|--------------|-------------------|------------------|
| `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` | 1.1.0rc3 | No soporta `gemma4` | Error de argumentos / quantização | – | – |
| `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10` | 1.3.0rc10 | No soporta `gemma4` | `AssertionError` quant_algo | – | – |
| `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | 1.3.0rc13 | No soporta `gemma4` | `AssertionError` quant_algo (pre-cuantizado), OK cuantizado MLP-only | **OK** BF16 | **OK** NVFP4 |

### Errores encontrados

- **Gemma 4**: `ValueError: model type 'gemma4' but Transformers does not recognize this architecture`. El `transformers` empaquetado en los contenedores de NVIDIA (4.55–4.57) no incluye soporte para Gemma 4.
- **Qwen 3.6 35B-A3B NVFP4 (NVIDIA)**: `AssertionError` en `QuantMode.from_quant_algo`, indicando que el backend PyTorch de TRT-LLM no reconoce el algoritmo de cuantización del checkpoint `modelopt` NVFP4 de NVIDIA.

### Resultado de la cuantización propia de Qwen 3.6

Se siguió el flujo oficial: partimos del modelo base BF16 (`Qwen/Qwen3.6-35B-A3B`), convertimos el checkpoint VLM a text-only (`qwen3_5_moe_text`), y cuantizamos con **TensorRT Model Optimizer 0.44.0** a formato HF NVFP4.

| Intento | Quant config | Tamaño | Resultado al servir con TRT-LLM 1.3.0rc13 |
|---------|--------------|--------|-------------------------------------------|
| Full NVFP4 | `--qformat nvfp4` | ~20 GB | `NotImplementedError`: Qwen3.5 split linear-attention packing no soporta tensores `input_scale`/`weight_scale` de la atención lineal cuantizada. |
| **MLP-only NVFP4** | `--qformat nvfp4_mlp_only` | ~22 GB | **Carga y sirve correctamente**: ~34.4 decode tok/s, TTFT caliente ~0.09 s, ~41 GB de memoria unificada. |

El checkpoint MLP-only cuantiza los expertos MoE y las capas MLP a NVFP4, dejando la atención (incluyendo la atención lineal híbrida de Qwen3.5/3.6) en BF16. Es compatible con el backend PyTorch de TRT-LLM y ofrece una velocidad intermedia entre Gemma 4 31B denso y Qwen 3.6 RedHatAI/vLLM.

### Resultado de Nemotron 3 con TensorRT-LLM 1.3.0rc13

| Modelo | Checkpoint | Formato | Decode tok/s | TTFT caliente | Memoria pico | Notas |
|--------|------------|---------|--------------|---------------|--------------|-------|
| **Nemotron-3-Nano-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | BF16 | **~28.8** | ~0.22 s | **~118 GB** | Carga directa. Casi llena el pool unificado. |
| **Nemotron-3-Super-120B-A12B** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | NVFP4 (`MIXED_PRECISION`) | **~14.7** | ~0.29 s | **~110 GB** | Mayor modelo, más expertos activos (A12B), por eso es más lento que Nano. |
| **Nemotron-3-Nano-Omni-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | NVFP4 (`MIXED_PRECISION`) | **~40.0** | ~0.10 s | **~40 GB** | Multimodal vía vLLM. **Imagen probada y funcional**. Audio no funcionó por decodificación del contenedor. |

Los checkpoints oficiales de NVIDIA funcionan out-of-the-box con `trtllm-serve --backend pytorch`. El Nano en BF16 está al límite de la memoria unificada; el Super en NVFP4 deja un poco más de margen. El Omni, por su parte, requiere vLLM para chat multimodal; TRT-LLM 1.3.0rc13 falló al parsear el formato OpenAI multimodal.

**Comparativa vLLM vs TensorRT-LLM para Nemotron 3**:

| Modelo | vLLM tok/s | TRT-LLM tok/s | vLLM memoria | TRT-LLM memoria | Recomendación |
|--------|------------|---------------|--------------|-----------------|---------------|
| Nano-30B-A3B BF16 | ~28.3 | ~28.8 | ~72 GB | ~118 GB | Ambos funcionan. TRT-LLM usa más memoria pero es el stack oficial. |
| Super-120B-A12B NVFP4 | **No viable** (OOM/colgado) | ~14.7 | — | ~110 GB | **Solo TRT-LLM** en GB10. |
| Nano-Omni-30B-A3B NVFP4 | ~40.0 | No probado / parse multimodal falla | ~40 GB | — | **Solo vLLM** para multimodal. |

La razón principal por la que el Super no funciona con vLLM es la gestión de memoria: vLLM V1 reserva memoria agresivamente (`--gpu-memory-utilization 0.90`) para PagedAttention y compila kernels con `torch.compile`, mientras que TRT-LLM reserva un presupuesto fijo calculado a partir de `max_seq_len` y `max_batch_size`. Para un modelo de 120B con 512 expertos y arquitectura híbrida Mamba-MoE, el overhead de vLLM excede los 128 GB unificados.

### Interpretación

Los blueprints oficiales de NVIDIA para Spark (`build.nvidia.com/spark/trt-llm/instructions`) funcionan con modelos específicos como `nvidia/Llama-3.1-8B-Instruct-FP4`, `openai/gpt-oss-*` y `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, pero **no directamente** con los checkpoints NVFP4 pre-cuantizados de Gemma 4 26B-A4B ni Qwen 3.6 35B-A3B tal como están publicados en HuggingFace.

Para usar TensorRT-LLM con Qwen 3.6 hoy hay dos caminos:
1. **Opción funcional probada**: cuantizar el modelo base BF16 con Model Optimizer usando `--qformat nvfp4_mlp_only` y servir con `trtllm-serve --backend pytorch`.
2. **Full NVFP4**: requiere que TRT-LLM agregue soporte para los tensores de escala de la atención lineal de Qwen3.5/3.6; actualmente falla.

Para Gemma 4 26B-A4B sería necesario esperar una versión de TRT-LLM con soporte nativo de `gemma4` en su stack de transformers, o cuantizar el modelo base desde cero.

Dado lo anterior, **vLLM es la opción más rápida y sencilla hoy** para Gemma 4 y Qwen 3.6 pre-cuantizados; **TRT-LLM es viable para Qwen 3.6 si se cuantiza manualmente en formato MLP-only**.

---

## Recomendación final

Para uso con agentes locales en DGX Spark, la configuración ganadora es:

- **Qwen 3.6 35B-A3B nvidia NVFP4 + vLLM nightly** → **~75–77 tok/s**, tool calling con `qwen3_coder`, soporte imagen/video y el **contexto máximo del modelo (262K tokens)** con 2 sesiones paralelas. **Esta es la recomendación actual.**
- **Gemma 4 26B-A4B community + parche** → ~49.5 tok/s, tool calling, bajo uso de VRAM.
- Alternativa estable anterior: **Qwen 3.6 35B-A3B RedHatAI** → ~42.2 tok/s, tool calling, soporte imagen/video y hasta 2 sesiones paralelas de 262K tokens de contexto.
- Alternativa oficial NVIDIA (TensorRT-LLM): **Qwen 3.6 35B-A3B MLP-only NVFP4** cuantizado desde BF16 → ~34.4 tok/s.

Gemma 4 31B dense debe reservarse solo para tareas donde la calidad del modelo denso justifique los ~7 tok/s.

**Si tu framework de agentes (OpenClaw, Hermes, etc.) necesita la ventana de contexto más grande posible en un solo GPU local**, Qwen 3.6 35B-A3B en vLLM es la opción clara: entrega el límite de 262K tokens por sesión con 2 sesiones concurrentes por defecto (3 sesiones son posibles pero dejan poco margen de memoria).

**Nemotron 3**:
- **Nano 30B-A3B BF16**: ~28.8 tok/s con TRT-LLM (~118 GB) o ~28.3 tok/s con vLLM (~72 GB). Elige TRT-LLM si prefieres el stack oficial, vLLM si quieres dejar más memoria libre.
- **Super 120B-A12B NVFP4**: ~14.7 tok/s con TRT-LLM (~110 GB). **No es viable con vLLM**; el engine falla por OOM o colgaba el Spark.
- **Nano Omni 30B-A3B NVFP4**: ~40.0 tok/s con vLLM (~40 GB), **soporta imagen** y es la opción más equilibrada si quieres multimodalidad oficial de NVIDIA.

**Multimodalidad (imagen/audio)**:
- **Imagen/video**: Gemma 4, Qwen 3.6 y **Nemotron-3 Nano Omni** lo soportan vía vLLM.
- **Audio**: Nemotron-3 Nano Omni debería soportarlo nativamente, pero en este contenedor de vLLM la decodificación de audio falló. Requiere más investigación o un contenedor con codecs de audio correctos.
