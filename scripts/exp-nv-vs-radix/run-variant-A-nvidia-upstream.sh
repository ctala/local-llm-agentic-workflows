#!/usr/bin/env bash
# Variante A — Qwen3.8-Flash-Next NVFP4 OFICIAL NVIDIA
# Imagen: nvcr.io/nvidia/vllm:26.04-py3 (vllm upstream, SIN parches GB10)
# Checkpoint: nvidia/Qwen3.8-Flash-Next-NVFP4 (cualquiera que esté en cache HF)
# Solo cambia: MODEL + imagen Docker vs el Radix Ark que está corriendo.

set -euo pipefail

NAME="${NAME:-qwen38-flash-expA}"
IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.04-py3}"
MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8001}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next-nv}"

PREFIX_CACHE="${PREFIX_CACHE:-1}"
EXACT_TOPK="${EXACT_TOPK:-1}"
CTX="${CTX:-262144}"
YARN="${YARN:-0}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.80}"
MTP="${MTP:-2}"
KV_DTYPE="${KV_DTYPE:-auto}"
SERVICE_MODE="${SERVICE_MODE:-}"
WORKERS="${WORKERS:-32}"
CC="${CC:-}"

# Resolver snapshot local
REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
SNAP_HOST="$(ls -d "$REPO_DIR"/snapshots/*/ 2>/dev/null | head -1 || true)"
if [ -z "$SNAP_HOST" ]; then
  echo "!! checkpoint NVIDIA oficial no encontrado en $REPO_DIR"
  echo "   corré: hf download nvidia/Qwen3.8-Flash-Next-NVFP4"
  exit 1
fi
SNAP_NAME="$(basename "$SNAP_HOST")"
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$SNAP_NAME"
echo ">> usando snapshot: $SNAP_NAME"

# Empezamos conservador: cudagraphs los mismos splitting_ops que usa el recipe oficial,
# PIECEWISE como pide NVIDIA en su README, y MTP=1 (lo que recomienda para TP=1).
# OJO: SIN ple_mmap, SIN fla fixes, SIN mamba guard, SIN qsa exact top-k — eso es la A.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
if [ -z "$CC" ]; then
  CC="-cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT"
fi

# YaRN > 262k (no usado en este exp)
OVR_ARGS=()
YARN_OVR='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
ALLOW_LONG=0
if [ "$YARN" != 0 ]; then OVR_ARGS=(--hf-overrides "$YARN_OVR"); ALLOW_LONG=1; fi

# MTP
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
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000"
  -v "$HF_CACHE:/hf"
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1
  -e NVIDIA_VLLM_USE_FLASHINFER_SAMPLER=1
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
  --quantization modelopt
  --trust-remote-code
  "${OVR_ARGS[@]}"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  "${SPEC[@]}"
)

if [[ "${SERVICE_MODE}" == "1" ]]; then
  echo ">> $NAME (variant=A: SIN parches GB10, MODEL=nvidia) en :$PORT..."
  exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" vllm serve "${VLLM_ARGS[@]}"
fi

echo ">> $NAME (variant=A: SIN parches GB10, MODEL=nvidia) arrancando en :$PORT..."
docker run -d "${DOCKER_ARGS[@]}" "$IMAGE" vllm serve "${VLLM_ARGS[@]}"
T0=$(date +%s)
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  sleep 15
done
T1=$(date +%s)
echo ">> Listo en http://localhost:${PORT}/v1 tras $((T1-T0))s (model: ${SERVED_NAME})"
echo ">> Logs: docker logs -f $NAME"