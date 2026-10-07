#!/usr/bin/env python3
"""Turn a recipe from recipes/*.yaml into a `docker run` command for a DGX Spark (GB10).

  python3 scripts/recipe_to_docker.py recipes/qwen36-35b-a3b-nvfp4-mtp.yaml --ctx 131072 --users 4

Paths: weights are expected under ~/vllm/<folder> and are mounted at /vllm (read-only);
the Hugging Face cache is mounted at /root/.cache/huggingface. Requires PyYAML.
"""
import argparse
import os
import shlex

import yaml

FLAGS = {"ctx": "--max-model-len", "users": "--max-num-seqs", "kv": "--kv-cache-dtype", "gpu_util": "--gpu-memory-utilization"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--users", type=int)
    ap.add_argument("--kv", choices=["auto", "fp8"])
    ap.add_argument("--gpu-util", dest="gpu_util", type=float)
    a = ap.parse_args()
    r = yaml.safe_load(open(a.recipe))
    e = r["entry"]
    measured = r.get("measured") or {}
    over = {"ctx": a.ctx or measured.get("ctx"), "users": a.users or measured.get("users"),
            "kv": a.kv or measured.get("kv_cache"), "gpu_util": a.gpu_util or measured.get("gpu_util")}
    args = [str(x) for x in e.get("args", [])]
    for k, flag in FLAGS.items():
        if over.get(k):
            args = [x for x in args if not x.startswith(flag + "=") and x != flag] + [f"{flag}={over[k]}"]
    home = "$HOME"
    cmd = ["docker", "run", "-d", "--name", r.get("id", os.path.basename(a.recipe).removesuffix(".yaml")),
           "--gpus", "all", "--ipc", "host", "--shm-size", str(e.get("shm_size", "16g")),
           "-p", f"{a.port}:8000", "-v", f"{home}/.cache/huggingface:/root/.cache/huggingface",
           "-v", f"{home}/vllm:/vllm:ro"]
    for k, v in (e.get("env") or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(e["image"])
    if e.get("engine") == "nim":
        cmd[cmd.index(e["image"]):cmd.index(e["image"])] = ["-e", "NGC_API_KEY", "-v", f"{home}/.cache/nim:/opt/nim/.cache"]
    if e.get("engine") == "vllm":
        cmd += [e["model"], "--host", "0.0.0.0", "--port", "8000", "--served-model-name", e["served_name"]] + args
    else:
        cmd += args
    # una línea por flag (con su valor) para que se lea y se copie bien
    lines, i = ["docker run -d"], 3
    q = lambda c: c if c.startswith("$HOME") or c.startswith("-") and "=" not in c else shlex.quote(c)
    while i < len(cmd):
        c = cmd[i]
        if c.startswith("-") and "=" not in c and i + 1 < len(cmd) and not cmd[i + 1].startswith("-") and c not in ("-d",):
            v = cmd[i + 1]
            lines.append(f"{c} " + (f'"{v}"' if v.startswith("$HOME") else shlex.quote(v)))
            i += 2
        else:
            lines.append(q(c))
            i += 1
    print(" \\\n  ".join(lines))


if __name__ == "__main__":
    main()
