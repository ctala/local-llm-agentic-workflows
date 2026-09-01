#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 (176B MoE, 6B activos) — DEFAULT desde 2026-09-01.
# Adaptado de https://github.com/blazux/qwen3.8-Flash-DGX (Apache-2.0).
#
# MEDIDO en este Spark (vLLM release/qwen38next + 7 parches GB10, 2026-09-01):
#   chat_short fresh       36.66 tok/s warm (vs Qwen 3.8 27B fresh 13.93 → +163%)
#   code_quicksort fresh   28.52 tok/s
#   agent_with_tools       23.59 tok/s (tool calling + planning)
#   c=1 aggregate          30.29 tok/s
#   c=4 aggregate          45.09 tok/s
#   c=8 aggregate          116.75 tok/s
#   c=16 aggregate         129.65 tok/s
#   Prefill cold (8K)      1018 tok/s
#   Prefill warm (cache hit) 1.46 s (vs 10.47 s cold → 7.2× speedup)
#   Determinism ✓ (first-token logprobs idénticos con EXACT_TOPK=1)
#   GSM8K 97.27%, AIME26 98.75% (referencia, checkpoint RadixArk)
#
# Cold start: ~14 min (76 GiB de pesos + autotune). Disco: ~139 GB (NVFP4 + hybrid).
# Servicio systemd `qwen38-flash-next-vllm.service` (auto-arranque en boot, port 8001).
#
# Para setup completo y tuning ver:
#   /home/ctala/Playground/gemma4-optimizado/qwen38-flash-next/resultados-qwen38-flash-next-2026-09-01.md
#   /home/ctala/Playground/gemma4-optimizado/qwen38-flash-next/tuning-2026-09-01.md

set -euo pipefail

NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8001}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next}"

MODE="${MODE:-hybrid}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
EXACT_TOPK="${EXACT_TOPK:-1}"
CTX="${CTX:-262144}"
YARN="${YARN:-0}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.80}"   # 0.80 recomendado para servicio long-running
MTP="${MTP:-2}"
KV_DTYPE="${KV_DTYPE:-auto}"
PREWARM="${PREWARM:-0}"
WORKERS="${WORKERS:-32}"
SERVICE_MODE="${SERVICE_MODE:-}"
CC="${CC:-}"

# Resolver el snapshot local
REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
SNAP_HOST="$(ls -d "$REPO_DIR"/snapshots/*/ 2>/dev/null | grep -v -- '-fp8hybrid' | head -1 || true)"
if [ -z "$SNAP_HOST" ]; then
  echo "ERROR: checkpoint no encontrado en $REPO_DIR"
  echo "Descarga con:"
  echo "  docker run --rm -v $HF_CACHE:/hf -e HF_HUB_DISABLE_XET=1 qwen38-flash-dgx \\"
  echo "    hf download '$MODEL' --max-workers 8"
  echo "O vía scripts/download-weights.sh del repo upstream."
  exit 1
fi
SNAP_NAME="$(basename "$SNAP_HOST")"
HYBRID_ENV=()
case "$MODE" in
  nvfp4) ;;
  hybrid)
    if [ ! -f "$REPO_DIR/snapshots/${SNAP_NAME}-fp8hybrid/.prepared" ]; then
      echo "ERROR: hybrid checkpoint no preparado."
      echo "  1. Ejecutar una vez: scripts/prepare-hybrid.sh (convierte 4 shards a fp8, ~10 min, +13 GB)"
      echo "  2. Volver a arrancar con MODE=hybrid"
      exit 1
    fi
    SNAP_NAME="${SNAP_NAME}-fp8hybrid"
    HYBRID_ENV=(-e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0)
    ;;
  *) echo "ERROR: MODE debe ser nvfp4 o hybrid"; exit 1 ;;
esac
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$SNAP_NAME"

# PLE gather fuera de CUDA graphs (splitting op + PIECEWISE).
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
if [ -z "$CC" ]; then
  CC="-cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT"
fi

# YaRN para >262k
OVR_ARGS=()
YARN_OVR='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
ALLOW_LONG=0
if [ "$YARN" != 0 ]; then OVR_ARGS=(--hf-overrides "$YARN_OVR"); ALLOW_LONG=1; fi

# MTP + YaRN requiere max_model_len explícito para el draft model.
SPEC=()
if [ "$MTP" != 0 ]; then
  if [ "$YARN" != 0 ]; then
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"max_model_len\":${CTX}}")
  else
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
  fi
fi

PC_ARG=--no-enable-prefix-caching
[ "$PREFIX_CACHE" = 1 ] && PC_ARG=--enable-prefix-caching

docker rm -f "$NAME" >/dev/null 2>&1 || true

DOCKER_ARGS=(
  --name "$NAME"
  --restart unless-stopped
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000"
  -v "$HF_CACHE:/hf"
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS}" -e VLLM_PLE_MMAP_PREWARM="$PREWARM"
  -e VLLM_QSA_EXACT_TOPK="$EXACT_TOPK"
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="$ALLOW_LONG"
  "${HYBRID_ENV[@]}"
)

VLLM_ARGS=(
  "$SNAP_IN"
  --served-model-name "${SERVED_NAME}"
  --host 0.0.0.0 --port 8000
  --load-format safetensors
  --max-model-len "$CTX"
  --max-num-seqs "$SEQS"
  --gpu-memory-utilization "$GPU_MEM"
  $PC_ARG
  --enable-chunked-prefill
  --max-num-batched-tokens 8192
  $CC
  --no-enable-flashinfer-autotune
  --kv-cache-dtype "$KV_DTYPE"
  "${OVR_ARGS[@]}"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  "${SPEC[@]}"
)

# Modo servicio: foreground para que systemd supervise.
if [[ "${SERVICE_MODE}" == "1" ]]; then
  echo ">> $NAME (mode=$MODE ctx=$CTX yarn=$YARN mtp=$MTP seqs=$SEQS gpu_mem=$GPU_MEM) en :$PORT..."
  exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" "${VLLM_ARGS[@]}"
fi

# Modo interactivo: daemon + healthcheck.
echo ">> $NAME (mode=$MODE ctx=$CTX yarn=$YARN mtp=$MTP seqs=$SEQS gpu_mem=$GPU_MEM) arrancando en :$PORT..."
docker run -d "${DOCKER_ARGS[@]}" "$IMAGE" "${VLLM_ARGS[@]}"
echo ">> Esperando healthcheck (8-13 min primera vez por carga de ~76 GiB + autotune)..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  sleep 15
done
echo ">> Listo en http://localhost:${PORT}/v1 (model: ${SERVED_NAME})"
echo ">> Logs: docker logs -f $NAME"