# Agent prompt

Replace `<RECIPE_ID>` with a file name from this folder (without `.yaml`).

```text
You are deploying a local LLM on an NVIDIA DGX Spark (GB10, 128 GB unified memory, aarch64, Ubuntu).
Use the tested recipe `<RECIPE_ID>` from https://github.com/ctala/local-llm-agentic-workflows and follow these steps. Stop and ask me before anything destructive.

1. Check the host: `nvidia-smi`, `free -g`, `docker info`. If another model container is using more than 40 GB, tell me and ask before stopping it.
2. Clone the repo (or `git pull` it) and read `recipes/<RECIPE_ID>.yaml`: `measured` has the tested context, users, KV cache, gpu_util and expected speed; `entry` has the image, env and vLLM args.
3. Download the weights with every command in `download` (needs `pip install "huggingface_hub[cli]"` and, for gated models, `HF_TOKEN`). NIM recipes need `NGC_API_KEY` in the environment; never print it.
4. Generate the command with `python3 scripts/recipe_to_docker.py recipes/<RECIPE_ID>.yaml --port 8000` and run it. Keep `--gpu-memory-utilization` as the recipe says: on unified memory, vLLM's 0.9 default takes ~110 GB.
5. Wait until `curl -s localhost:8000/v1/models` lists the model (follow `docker logs -f <RECIPE_ID>`; large models take 3 to 12 minutes). If it fails, read the last 80 log lines and report the cause before changing flags.
6. Smoke test: one chat completion with a short prompt and `max_tokens: 256`; report tokens/s. If the recipe lists `tools`, also send one tool call.
7. Report: the endpoint URL, the served model name, memory used (`nvidia-smi`), measured speed vs the recipe's `measured.single_user_tok_s`, and the exact command you ran.

Do not change ports of other services, do not run `docker system prune`, and do not expose the port beyond localhost unless I ask.
```
