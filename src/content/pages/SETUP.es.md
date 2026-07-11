# Registro de optimización de LLMs en DGX Spark

> Para una guía rápida de uso, véase [Inicio](/local-llm-agentic-workflows/). Este documento es el log detallado de trabajo.

## Objetivo
Encontrar la mejor forma de ejecutar modelos locales optimizados para el **NVIDIA DGX Spark** (GB10 Grace Blackwell, ARM64/aarch64, 128 GB memoria unificada, CUDA 13.0, sm_121), enfocado en uso con agentes (Hermes/OpenClaw/n8n/Open WebUI).

Modelos objetivo:
- Google Gemma 4 31B IT (dense)
- Google Gemma 4 26B-A4B IT (MoE)
- Qwen 3.6 35B-A3B (MoE)
- NVIDIA Nemotron-3 Nano 30B-A3B (MoE, BF16)
- NVIDIA Nemotron-3 Super 120B-A12B (MoE, NVFP4)

---

## 1. Estado del sistema analizado

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
| Disco | 3.7 TB, 2.7 TB libres |

Modelos/contenedores previos corriendo:
- Open WebUI, n8n, qdrant, searxng, browserless
- NIMs: bigcode/starcoder2-7b, deepseek-coder-v2-lite-instruct, llama-3.1-8b-instruct

---

## 2. Lecciones clave sobre el DGX Spark

1. **ARM64/aarch64 es crítico**: muchos contenedores x86_64 fallan con `exec format error`. Hay que verificar que las imágenes tengan manifest `arm64`.
2. **GB10 (sm_121) no tiene compute FP4 nativo**: a diferencia de B200. Los pesos NVFP4 se ejecutan vía **Marlin** (`--moe-backend marlin`), que descomprime a BF16 en runtime. Esto limita la velocidad respecto a lo que se vería en B200.
3. **Ollama/llama.cpp no exprimen el Spark**: usan backends genéricos sin kernels específicos de Blackwell ni NVFP4 nativo.
4. **Para agentes, el modelo importa más que el framework**: Gemma 4 31B dense está limitado por ancho de banda (~6-7 tok/s en GB10). Gemma 4 26B-A4B MoE (~3.8B activos) es el candidato a 50 tok/s.

---

## 3. Resultados de pruebas

### 3.1 Gemma 4 26B-A4B NVFP4 con vLLM

**Modelos probados**:
- `nvidia/Gemma-4-26B-A4B-NVFP4` (~16.5 GB)
- `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` community + `gemma4_patched.py`

**Contenedores que funcionan**:
- Modelo oficial: `vllm/vllm-openai:gemma4-0505-cu130` (vLLM 0.20.2rc1)
- Community parcheado: `vllm/vllm-openai:gemma4-cu130`

| Configuración | Decode tok/s | TTFT caliente | Notas |
|---------------|--------------|---------------|-------|
| Oficial base (`gemma4-0505-cu130`) | ~30.1 | ~0.20s | Estable, tool calling activado |
| Community + parche (`gemma4-cu130`) | **~49.5** | ~0.08s | Mejor opción para agentes |
| Community, gpu_util 0.92, max-seqs 4, batched 8192 | ~49.3 | ~1.9s | Sin mejora real |
| Community, gpu_util 0.90, batched 2048 | ~49.3 | ~2.4s | Sin mejora real |
| n-gram speculative decoding | ~24.6 | ~2.5s | Empeoró (prompt no repetitivo) |
| MTP (`google/gemma-4-26B-A4B-it-assistant`) | Error | - | `AssertionError` en shape del drafter con esta build |

**Comando recomendado** (community + parche):
```bash
docker run -d --name gemma4-26b-a4b \
  --gpus all --ipc host --network host --shm-size 64gb \
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

**Problemas encontrados**:
- Sin `--max-num-batched-tokens 4096` falla por chunked multimodal input.
- `vllm/vllm-openai:gemma4-cu130` no carga el modelo oficial: `KeyError: 'layers.0.experts.0.down_proj.input_scale'`.
- `vllm/vllm-openai:gemma4-0505-cu130` carga el oficial pero es ~20 tok/s más lento que el community parcheado.
- `nvcr.io/nvidia/vllm:26.04-py3` (vLLM 0.19.0) no reconoce arquitectura `gemma4`.

### 3.2 Gemma 4 31B IT (dense)

**Modelo probado**: `nvidia/Gemma-4-31B-IT-NVFP4` (~31 GB)
**Contenedor**: `vllm/vllm-openai:gemma4-0505-cu130`

| Configuración | Decode tok/s | TTFT caliente | Notas |
|---------------|--------------|---------------|-------|
| Base NVFP4 | **~6.7** | ~1.8s | Limitado por ancho de banda de memoria |

**Nota**: no es candidato a 50 tok/s por ser denso y limitado por ancho de banda de memoria.

### 3.3 Qwen 3.6 35B-A3B

**Modelos probados**:
- `nvidia/Qwen3.6-35B-A3B-NVFP4` → **funciona con vLLM nightly** (recomendación actual).
- `RedHatAI/Qwen3.6-35B-A3B-NVFP4` → funciona con `vllm/vllm-openai:gemma4-0505-cu130` (fallback estable).

**Contenedores**:
- `vllm/vllm-openai:nightly@sha256:a671d5fcda70fe9ac6f245f9780821de459fb4ee22c018fd07a0f10a55279bf9` para el checkpoint nvidia.
- `vllm/vllm-openai:gemma4-0505-cu130` para el checkpoint RedHatAI.

| Configuración | Checkpoint | Contenedor | Decode tok/s | TTFT caliente | Notas |
|---------------|------------|------------|--------------|---------------|-------|
| **NVIDIA NVFP4 W4A16 + marlin + flashinfer** | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:nightly` | **~75–77** | ~0.10s | **Recomendación actual.** `modelopt` W4A16, parser `qwen3_coder`, `fastsafetensors`, `async-scheduling`, 262K contexto. |
| Base (`compressed-tensors` + `marlin`) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~42.2 | ~0.10s | Fallback estable, tool calling activado |
| max-seqs 4, batched 8192, gpu_util 0.92 | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~42.2 | ~1.4s | Sin mejora real |
| n-gram speculative (`num_spec_tokens=5`) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | ~34–37 | ~0.10s | Empeora para texto no repetitivo |
| MTP (`qwen3_5_mtp`) | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | Error | - | Drafter no cuantizado no soporta `moe_backend='marlin'` |
| TRT-LLM 1.3.0rc13 (MLP-only NVFP4 propio) | Cuantizado desde BF16 | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~34.4** | ~0.09s | Cuantizado con Model Optimizer 0.44.0 desde BF16. Ver sección 4. |

**Comando recomendado**:
```bash
./scripts/run-qwen36-35b-a3b.sh
```

El script usa el checkpoint `nvidia/Qwen3.6-35B-A3B-NVFP4`, vLLM nightly, 262K de contexto, `flashinfer`, `async-scheduling`, `fastsafetensors` y el parser `qwen3_coder` para tool calling.

### 3.4 NVIDIA Nemotron 3

**Modelos probados**:
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (~89 GB)
- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (~75 GB)

#### TensorRT-LLM 1.3.0rc13

**Contenedor**: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13`

| Configuración | Decode tok/s | TTFT caliente | Memoria pico | Notas |
|---------------|--------------|---------------|--------------|-------|
| Nano-30B-A3B BF16 | **~28.8** | ~0.22 s | **~118 GB** | Carga directa; casi llena el pool unificado. |
| Super-120B-A12B NVFP4 | **~14.7** | ~0.29 s | **~110 GB** | Checkpoint NVFP4 oficial; más lento por más expertos activos. |

#### vLLM `gemma4-0505-cu130`

**Contenedor**: `vllm/vllm-openai:gemma4-0505-cu130`

| Configuración | Decode tok/s | TTFT caliente | Memoria pico | Notas |
|---------------|--------------|---------------|--------------|-------|
| Nano-30B-A3B BF16 | ~28.3 | ~0.20 s | ~72 GB | Funciona; requiere detener otros servicios grandes. |
| Super-120B-A12B NVFP4 | — | — | — | **No viable**: `CUDA OOM` al inicializar engine; colgó el Spark en el primer intento. |

**Comando recomendado (Nano)**:
```bash
docker run -d --name nemotron3-nano-30b-a3b-trtllm \
  --gpus all --ipc host --network host --shm-size 64gb \
  -v ~/vllm/nemotron3-nano-30b-a3b-bf16:/models/nemotron \
  -e HF_HOME=/models \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 \
    trtllm-serve /models/nemotron --host 0.0.0.0 --port 8000 \
    --backend pytorch --max_seq_len 8192 --max_batch_size 1 --kv_cache_dtype fp8
```

**Comando recomendado (Super)**:
```bash
docker run -d --name nemotron3-super-120b-a12b-trtllm \
  --gpus all --ipc host --network host --shm-size 64gb \
  -v ~/vllm/nemotron3-super-120b-a12b-nvfp4:/models/nemotron \
  -e HF_HOME=/models \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 \
    trtllm-serve /models/nemotron --host 0.0.0.0 --port 8000 \
    --backend pytorch --max_seq_len 8192 --max_batch_size 1 --kv_cache_dtype fp8
```

**Comando alternativo (Nano con vLLM)**:
```bash
docker run -d --name nemotron3-nano-30b-a3b-vllm \
  --gpus all --ipc host --network host --shm-size 64gb \
  -v ~/vllm/nemotron3-nano-30b-a3b-bf16:/models/nemotron \
  -e HF_HOME=/models \
  vllm/vllm-openai:gemma4-0505-cu130 \
    --model /models/nemotron --served-model-name nemotron3-nano-30b-a3b \
    --host 0.0.0.0 --port 8000 --kv-cache-dtype fp8 \
    --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 --trust-remote-code
```

#### Intento fallido: Nemotron-3 Super 120B-A12B con vLLM

Se intentó servir `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` con `vllm/vllm-openai:gemma4-0505-cu130`.

**Primer intento**: el contenedor cargó los 17 shards (~75 GB en disco) en ~489 s. Los últimos logs antes del colgado:
```
Loading weights took 489.67 seconds
WARNING ... Your GPU does not have native support for FP4 computation ...
Using MoEPrepareAndFinalizeNoDPEPModular
Using uncalibrated q_scale 1.0 ...
```
El Spark se colgó. Causa raíz: dos servidores `llama-server` (Qwen3.6-27B-Q8_0 y Qwen3.6-35B-A3B-Q8_0) ocupaban ~76 GB de VRAM, dejando insuficiente memoria para vLLM.

**Segundo intento** (tras reinicio y liberar memoria): vLLM falló inmediatamente con:
```
torch.AcceleratorError: CUDA error: out of memory
  File ".../vllm/utils/mem_utils.py", line 108, in measure
    self.free_memory, self.total_memory = current_platform.mem_get_info(device)
```
Incluso con ~117 GB libres en el host, vLLM no pudo inicializar el `EngineCore`.

**Diagnóstico**: el modelo `NemotronHForCausalLM` tiene 88 capas, 512 expertos enrutados, 22 expertos por token, 1 experto compartido, hidden size 4096. El overhead de descompresión FP4 vía Marlin, la reserva de KV cache por `--gpu-memory-utilization 0.90` y la compilación de kernels del V1 engine exceden el margen de 128 GB unificados. TensorRT-LLM, en cambio, usa un presupuesto de memoria fijo (`--max_seq_len 8192 --max_batch_size 1`) que resulta estable.

**Conclusión**: Nemotron-3 Super 120B-A12B **solo es viable con TensorRT-LLM** en GB10.

### 3.5 NVIDIA Nemotron 3 Nano Omni (multimodal)

**Modelo probado**:
- `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (~21 GB)

**Contenedor**: `vllm/vllm-openai:gemma4-0505-cu130`

| Configuración | Decode tok/s | TTFT caliente | Memoria pico | Multimodal | Notas |
|---------------|--------------|---------------|--------------|------------|-------|
| vLLM, `--quantization modelopt_fp4`, `--moe-backend marlin` | **~40.0** | ~0.10 s | **~40 GB** | Imagen ✅ | Texto rápido; imagen responde correctamente. |
| Audio (vía OpenAI `input_audio`) | – | – | – | Audio ❌ | Falló con `Invalid or unsupported audio file` en la decodificación del contenedor. |

**TRT-LLM 1.3.0rc13** también fue probado, pero falló al parsear mensajes multimodales:
> `AttributeError: 'NoneType' object has no attribute 'model_type'` en `parse_chat_messages_coroutines`.

**Comando recomendado**:
```bash
docker run -d --name nemotron3-nano-omni-vllm \
  --gpus all --ipc host --network host --shm-size 64gb \
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

**Prueba de imagen**:
```bash
python3 benchmarks/test_multimodal.py nemotron3-nano-omni image /ruta/a/imagen.png
```

---

## 4. Intento con TensorRT-LLM oficial de NVIDIA

Objetivo: validar si los modelos oficiales de NVIDIA corren mejor con los contenedores TensorRT-LLM diseñados para DGX Spark.

### Contenedores probados

| Contenedor | TRT-LLM | Gemma 4 26B-A4B oficial | Qwen 3.6 35B-A3B oficial | Nemotron 3 Nano | Nemotron 3 Super |
|------------|---------|-------------------------|--------------------------|-----------------|------------------|
| `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` | 1.1.0rc3 | No soporta `gemma4` | Error de opción/quantização | – | – |
| `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10` | 1.3.0rc10 | No soporta `gemma4` | `AssertionError` quant_algo | – | – |
| `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | 1.3.0rc13 | No soporta `gemma4` | `AssertionError` quant_algo (pre-cuantizado), OK MLP-only propio | **OK** BF16 | **OK** NVFP4 | Omni: error parse multimodal |

### Errores clave

- **Gemma 4**: `ValueError: model type 'gemma4' but Transformers does not recognize this architecture`. El `transformers` empaquetado en los contenedores no incluye Gemma 4.
- **Qwen 3.6 35B-A3B NVFP4**: `AssertionError` en `QuantMode.from_quant_algo` al cargar el checkpoint `modelopt` NVFP4 de NVIDIA. El backend PyTorch de TRT-LLM no reconoce ese esquema de cuantización directamente.

### Cuantización propia con TensorRT Model Optimizer

Se intentó el flujo oficial para Qwen 3.6 desde el modelo base BF16 (`Qwen/Qwen3.6-35B-A3B`) usando **nvidia-modelopt 0.44.0** (los ejemplos de 0.35.0 no son compatibles con PyTorch 2.12 disponible en el sistema).

Pasos ejecutados:
1. Descargar `Qwen/Qwen3.6-35B-A3B` BF16 (~70 GB).
2. Convertir el checkpoint VLM a text-only (`qwen3_5_moe_text`) porque Model Optimizer no carga `Qwen3_5MoeForConditionalGeneration` directamente.
3. Parchear `modelopt.torch.quantization.plugins.huggingface._QuantFusedExperts.iter_weights_for_calibration` para soportar quantizers por experto de Qwen3.5/3.6.
4. Parchear `example_utils.py` para usar la memoria **total** de la GPU en lugar de la libre reportada por `accelerate` (necesario en memoria unificada del GB10).
5. Cuantizar con `--qformat nvfp4` y `--qformat nvfp4_mlp_only`.

Resultados:

| Quant config | Tamaño | Error/success al servir con TRT-LLM 1.3.0rc13 |
|--------------|--------|-----------------------------------------------|
| Full NVFP4 | ~20 GB | `NotImplementedError`: split linear-attention packing no soporta `input_scale`/`weight_scale` de atención lineal cuantizada. |
| **MLP-only NVFP4** | ~22 GB | **Sirve correctamente** con `trtllm-serve --backend pytorch`. |

Benchmark TRT-LLM MLP-only:
- **Decode tok/s**: ~34.4 (max_tokens=1024)
- **TTFT caliente**: ~0.09 s
- **Memoria usada**: ~41 GB del pool unificado

### Conclusión del intento

Los blueprints oficiales de Spark funcionan con modelos como `nvidia/Llama-3.1-8B-Instruct-FP4`, `openai/gpt-oss-*` y `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, pero **no out-of-the-box** con `nvidia/Gemma-4-26B-A4B-NVFP4` ni `nvidia/Qwen3.6-35B-A3B-NVFP4`.

Para TensorRT-LLM con Qwen 3.6 hay un camino funcional hoy:
- Cuantizar el modelo base BF16 con Model Optimizer usando `--qformat nvfp4_mlp_only`.
- Servir con `trtllm-serve --backend pytorch --kv_cache_dtype fp8`.

El formato **full NVFP4** aún no es compatible con TRT-LLM 1.3.0rc13 por la atención lineal de Qwen3.5/3.6. Para Gemma 4, seguir esperando soporte nativo de `gemma4` en TRT-LLM.

**vLLM sigue siendo la opción más rápida y sencilla** para los checkpoints pre-cuantizados; **TRT-LLM es viable para Qwen 3.6 con cuantización MLP-only manual**.

---

## 5. Configuraciones finales recomendadas

| Modelo | Checkpoint | Contenedor | Decode tok/s | Uso recomendado |
|--------|------------|------------|--------------|-----------------|
| **Qwen 3.6 35B-A3B** | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:nightly` | **~75–77** | **Mejor calidad-velocidad; contexto máximo 262K; tool calling robusto.** |
| **Gemma 4 26B-A4B** | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` + parche | `vllm/vllm-openai:gemma4-cu130` | **~49.5** | Máxima velocidad para agentes |
| **Qwen 3.6 35B-A3B** | `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~42.2** | Fallback estable |
| **Gemma 4 31B** | `nvidia/Gemma-4-31B-IT-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~6.7** | Solo si se necesita el denso |
| **Qwen 3.6 35B-A3B** | Cuantiizado propio MLP-only NVFP4 | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~34.4** | Alternativa TRT-LLM; usa stack oficial |
| **Nemotron-3-Nano-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~28.8** | Modelo denso BF16 de NVIDIA; usa casi toda la memoria |
| Nemotron-3-Nano-30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `vllm/vllm-openai:gemma4-0505-cu130` | ~28.3 | Alternativa vLLM; requiere liberar VRAM de otros servicios |
| **Nemotron-3-Super-120B-A12B** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` | **~14.7** | Modelo grande NVFP4; prioriza calidad sobre velocidad. **No usar con vLLM** |
| **Nemotron-3-Nano-Omni-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | `vllm/vllm-openai:gemma4-0505-cu130` | **~40.0** | **Mejor opción multimodal**: texto + imagen funcionan; audio pendiente |

---

## 6. Notas técnicas

- **HF_TOKEN requerido** para descargar modelos Gemma/Qwen de HuggingFace.
- **Memoria**: el modelo Gemma 4 26B-A4B NVFP4 ocupa ~18 GB en memoria al cargar; KV cache FP8 deja ~82 GB disponibles.
- **Marlin backend**: obligatorio en GB10 para MoE NVFP4. Backends nativos FP4 (CUTLASS/FlashInfer) pueden fallar o dar NaN en sm_121.
- **Tool calling**: Qwen 3.6 requiere `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`; el parser `qwen3_coder` es más robusto para multi-turn que el anterior `qwen3_xml`. Sin parser, vLLM devuelve XML en `content` y el array nativo `tool_calls` queda vacío, por lo que agentes como Hermes/OpenClaw no ejecutan herramientas. Gemma 4 también tiene parser nativo `gemma4` por probar.
- **TRT-LLM con Nemotron 3**: los checkpoints oficiales cargan directamente con `trtllm-serve --backend pytorch --kv_cache_dtype fp8`. El Nano BF16 usa ~118 GB del pool unificado; el Super NVFP4 ~110 GB.
- **vLLM con Nemotron 3**: el Nano BF16 y el Omni NVFP4 funcionan con `vllm/vllm-openai:gemma4-0505-cu130`. El Super 120B-A12B no es viable: el V1 engine reserva memoria agresivamente y provoca `CUDA OOM` o colgado del sistema cuando hay otros consumidores de VRAM.
- **Servicios en segundo plano**: antes de lanzar modelos grandes, verifica que no haya `llama-server`, contenedores u otros procesos ocupando memoria GPU. En este trabajo, dos servidores Qwen GGUF en llama.cpp usaban ~76 GB y causaban OOM al iniciar vLLM.

---

## 7. Archivos entregables

- `README.md`: resumen ejecutivo y guía rápida.
- `resultados-gemma4-spark.md`: tabla resumen y conclusiones.
- `registro-gemma4-spark.md`: este log detallado.
- `scripts/run-gemma4-26b-a4b.sh`: script para lanzar Gemma 4 26B-A4B community.
- `scripts/run-qwen36-35b-a3b.sh`: script principal para lanzar Qwen 3.6 35B-A3B nvidia NVFP4 con vLLM nightly (262K contexto).
- `scripts/run-qwen36-35b-a3b-extreme-context-2seq.sh`: alias a `run-qwen36-35b-a3b.sh` para contexto extremo.
- `scripts/run-qwen36-35b-a3b-trtllm.sh`: script para lanzar Qwen 3.6 35B-A3B con TensorRT-LLM (checkpoint MLP-only).
- `scripts/run-gemma4-31b.sh`: script para lanzar Gemma 4 31B.
- `scripts/run-nemotron3-nano-30b-a3b-trtllm.sh`: script para lanzar Nemotron-3-Nano con TRT-LLM.
- `scripts/run-nemotron3-nano-30b-a3b-vllm.sh`: script para lanzar Nemotron-3-Nano con vLLM.
- `scripts/run-nemotron3-super-120b-a12b-trtllm.sh`: script para lanzar Nemotron-3-Super con TRT-LLM.
- `scripts/run-nemotron3-nano-omni-vllm.sh`: script para lanzar Nemotron-3-Nano-Omni multimodal con vLLM.
- `chat-templates/qwen3.6-miaai.jinja`: template Jinja para Qwen 3.6 con thinking/tool support.
- `benchmarks/bench_model.py`: script de benchmark reproducible.
- `benchmarks/test_multimodal.py`: script para probar imagen/audio con modelos multimodales.
- `scripts/quantize-qwen36-nvfp4.sh`: script para cuantizar Qwen 3.6 BF16 a NVFP4 MLP-only con Model Optimizer.
- `scripts/convert-qwen36-vlm-to-text.py`: helper que extrae la parte text-only del checkpoint VLM de Qwen 3.6.

## 8. Próximos pasos opcionales

1. Validar MTP speculative decoding con el checkpoint nvidia + vLLM nightly (en pruebas anteriores con RedHatAI empeoró el decode single-user).
2. Probar `--tool-call-parser gemma4` en Gemma 4 para tool calling nativo.
3. Evaluar calidad de los modelos en tareas de agentes (Hermes/OpenClaw).
4. Probar full NVFP4 de Qwen 3.6 en TRT-LLM cuando se soporten los scales de atención lineal.
5. Comparar calidad de Qwen 3.6 nvidia W4A16 vs RedHatAI compressed-tensors vs MLP-only NVFP4 propio.
6. Probar GPT-OSS con TRT-LLM.
