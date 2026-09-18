# Pendiente: Re-test NVIDIA Qwen3.8-Flash-Next-NVFP4 cuando el upstream lo soporte

## Trigger para re-test (cualquiera de los 3):
1. `blazux/qwen3.8-Flash-DGX` suba un recipe con vLLM `>=0.20` (con vllm#55513 mergeado) → Docker image nueva
2. NVIDIA suba `vllm-openai:qwen38-flash-next` con vLLM upstream que soporte `qwen4_exp` y ModelOpt `0.46.0.dev281`
3. Alguien haga el patch manual (~10 líneas en `vllm/models/qwen3_8_flash_next/nvidia/mtp.py:343`) para registrar `w2_weight_scale_inv`

## Quick check semanal (5 min)
```bash
# ¿salió vllm upstream que soporte qwen4_exp?
curl -sf https://api.github.com/repos/vllm-project/vllm/commits?path=vllm/model_executor/models/qwen3_8_flash_next | jq -r '.[].sha' | head -1
# ¿se actualizó blazux/qwen3.8-Flash-DGX?
curl -sf https://api.github.com/repos/blazux/qwen3.8-Flash-DGX/commits | jq -r '.[0] | .commit.message'
```

## Si hay update, cómo retomar
```bash
cd /home/ctala/Playground/gemma4-optimizado/qwen38-flash-next/exp-nv
# rebuild imagen con recipe nuevo
docker build -t qwen38-flash-dgx-nv -f ../upstream/Dockerfile ..   # o el nuevo Dockerfile
SERVICE_MODE=1 PORT=8001 GPU_MEM=0.80 MTP=2 bash run-variant-B-nvidia-patched.sh
# medir (mismo battery de bench.py)
python3 bench.py --base-url http://localhost:8001 --model qwen3.8-flash-next-nv-b --variant NV-v2-$(date +%Y%m%d)
# comparar con baseline
diff /tmp/nv-exp/bench-RadixArk-baseline.json /tmp/nv-exp/bench-NV-v2-*.json | head -50
```

## Estado actual (2026-09-18)
- Radix Ark = default operativo. Medido: 23.5/24.6/21.0/19.6 tok/s single-stream, 74.6 c=8.
- NV oficial NVFP4 = 124 GiB en disco, listo para re-test.
- Variantes A y B fallan hoy (FAIL A: qwen4_exp; FAIL B: w2_weight_scale_inv).
