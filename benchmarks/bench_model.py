#!/usr/bin/env python3
"""Benchmark simple para un servidor vLLM OpenAI-compatible.

Uso:
    python3 bench_model.py <model_name> [max_tokens]

Mide TTFT y decode tok/s en un prompt en español.
"""
import json
import sys
import time
import urllib.request

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-26b-a4b"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 512

PROMPT = (
    "Explica en 5 párrafos detallados cómo funciona la memoria unificada "
    "en arquitecturas CPU-GPU como NVIDIA Grace Blackwell, incluyendo ventajas "
    "para inferencia de modelos de lenguaje grandes y comparaciones con sistemas "
    "tradicionales de memoria separada."
)

payload = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": MAX_TOKENS,
    "temperature": 0.7,
    "stream": True,
}).encode("utf-8")

req = urllib.request.Request(
    API_URL,
    data=payload,
    headers={"Content-Type": "application/json"},
)

tokens = 0
first_token_time = None
start = time.perf_counter()

try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                tokens += 1
                if first_token_time is None:
                    first_token_time = time.perf_counter()
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

end = time.perf_counter()
ttft = first_token_time - start if first_token_time else 0.0
decode_time = end - first_token_time if first_token_time else end - start
decode_tok_s = tokens / decode_time if decode_time > 0 else 0.0

print(
    f"Modelo: {MODEL} | max_tokens: {MAX_TOKENS}\n"
    f"Tokens: {tokens} | TTFT: {ttft:.2f}s | Total: {end-start:.2f}s | "
    f"Decode tok/s: {decode_tok_s:.2f}"
)
