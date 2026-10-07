#!/usr/bin/env python3
"""Standard round benchmark for an OpenAI-compatible server (vLLM, SGLang, NIM).

Reproduces the "128K x 4 users" round published in RESULTS.md:
  1. single user: 3 streamed requests, median TTFT and decode tok/s
  2. N concurrent users: aggregate tok/s and per-user tok/s
  3. long prompt (~46K tokens): prefill time
  4. stability: 3 rounds of N concurrent users, counts failed requests
  5. optional --stress-minutes: varied prompts (code, writing, JSON/tools, summary) at
     temperature 0 and 0.7, then sustained load; reads speculative-decoding acceptance
     from /metrics when available (MTP, DSpark, DFlash, Eagle3)

Only the Python standard library is required.

Example:
  python3 standard_round.py --base-url http://127.0.0.1:8000 --model qwen3.6-35b-a3b --users 4 --stress-minutes 15
"""
import argparse
import json
import re
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("Explica en unas 200 palabras, en español neutro y en un solo párrafo, qué es la memoria unificada "
          "de un computador y por qué importa para correr modelos de lenguaje localmente.")
LONG = "Resume este registro: " + " ".join(f"evento {n}: el servicio respondió en {n % 97} ms." for n in range(3000))
VARIED = {
    "code": "Escribe una función en Python que reciba una lista de transacciones (fecha, monto, categoría) y devuelva "
            "el total por categoría y por mes, con pruebas unitarias con pytest. Solo código.",
    "writing": "Escribe un post de LinkedIn de 200 palabras, en español neutro, para founders de LATAM sobre por qué "
               "medir el CAC por canal antes de escalar la inversión en anuncios. Tono directo, sin emojis.",
    "json_tools": "Devuelve SOLO un JSON válido con esta forma: {\"tareas\":[{\"titulo\":str,\"prioridad\":1-3,"
                  "\"herramienta\":\"calendario\"|\"correo\"|\"crm\"}]} con 8 tareas para preparar el lanzamiento de un curso online.",
    "summary": "Resume en 8 viñetas los puntos clave de este texto: " + " ".join(
        f"En la sesión {n} el equipo revisó el embudo, detectó una caída del {n % 17 + 3}% en la conversión del paso "
        f"{n % 5 + 1} y acordó probar un cambio en el mensaje del correo número {n % 4 + 1}." for n in range(400)),
}


def stream(base, model, prompt, temp=0.0, max_tokens=300, key=None):
    """Returns (ttft_ms, decode_tok_s, completion_tokens, ok)."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "temperature": temp, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    t0 = time.time(); first = None; tokens = chunks = 0
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{base}/v1/chat/completions", data=body, headers=headers),
                                    timeout=900) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    tokens = d["usage"].get("completion_tokens") or tokens
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        chunks += 1
                        first = first or time.time()
    except Exception:
        return 0.0, 0.0, 0, False
    t1 = time.time(); first = first or t1
    tokens = tokens or chunks
    return (first - t0) * 1000, tokens / max(t1 - first, 1e-6), tokens, True


def concurrent(base, model, n, key=None, prompt=PROMPT):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda _: stream(base, model, prompt, key=key), range(n)))
    wall = time.time() - t0
    return {"total_tok_s": round(sum(r[2] for r in res) / wall, 1),
            "per_user_tok_s": round(statistics.median(r[1] for r in res), 1),
            "ttft_ms": round(statistics.median(r[0] for r in res)), "failed": sum(1 for r in res if not r[3])}


def spec_counts(base):
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return 0.0, 0.0
    tot = lambda name: sum(float(m) for m in re.findall(rf"^{name}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$", text, re.M))
    return tot("vllm:spec_decode_num_accepted_tokens_total"), tot("vllm:spec_decode_num_draft_tokens_total")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--api-key")
    ap.add_argument("--stress-minutes", type=float, default=0)
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args()
    base, m, k = a.base_url.rstrip("/"), a.model, a.api_key
    out = {"model": m, "users": a.users, "date": time.strftime("%Y-%m-%d %H:%M")}
    singles = [stream(base, m, PROMPT, key=k) for _ in range(3)]
    out["single"] = {"tok_s": round(statistics.median(s[1] for s in singles), 1),
                     "ttft_ms": round(statistics.median(s[0] for s in singles))}
    out["concurrent"] = concurrent(base, m, a.users, k)
    t0 = time.time(); ok = stream(base, m, LONG, max_tokens=100, key=k)[3]
    out["long_prompt_s"] = round(time.time() - t0, 1) if ok else None
    rounds = [concurrent(base, m, a.users, k) for _ in range(3)]
    out["stability"] = {"rounds_total_tok_s": [r["total_tok_s"] for r in rounds],
                        "failed": sum(r["failed"] for r in rounds) + (0 if ok else 1)}
    if a.stress_minutes:
        by = {}
        for temp in (0.0, 0.7):
            for name, p in VARIED.items():
                a0, p0 = spec_counts(base)
                _, tps, n, good = stream(base, m, p, temp, 400, k)
                a1, p1 = spec_counts(base)
                by[f"{name}@{temp}"] = {"tok_s": round(tps, 1), "ok": good,
                                        "acceptance_pct": round(100 * (a1 - a0) / (p1 - p0), 1) if p1 > p0 else None}
        end = time.time() + a.stress_minutes * 60
        stats = {"done": 0, "failed": 0, "tokens": 0}
        items = list(VARIED.values())
        a0, p0 = spec_counts(base); t0 = time.time()
        def worker(i):
            j = i
            while time.time() < end:
                _, _, n, good = stream(base, m, items[j % len(items)], 0.7 if j % 2 else 0.0, 400, k)
                stats["done"] += 1; stats["tokens"] += n; stats["failed"] += 0 if good else 1; j += 1
        with ThreadPoolExecutor(max_workers=a.users) as ex:
            list(ex.map(worker, range(a.users)))
        a1, p1 = spec_counts(base)
        out["stress"] = {"minutes": a.stress_minutes, "by_prompt": by, "requests": stats["done"], "failed": stats["failed"],
                         "sustained_tok_s": round(stats["tokens"] / (time.time() - t0), 1),
                         "acceptance_pct": round(100 * (a1 - a0) / (p1 - p0), 1) if p1 > p0 else None}
    if a.json:
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return
    print(f"{m}: 1 user {out['single']['tok_s']} tok/s (TTFT {out['single']['ttft_ms']} ms) · "
          f"{a.users} users {out['concurrent']['total_tok_s']} tok/s total · long prompt {out['long_prompt_s']} s · "
          f"failed {out['stability']['failed']}")
    if "stress" in out:
        s = out["stress"]
        print(f"stress {s['minutes']} min: {s['sustained_tok_s']} tok/s sustained · {s['requests']} requests · "
              f"{s['failed']} failed · speculative acceptance {s['acceptance_pct']} %")


if __name__ == "__main__":
    main()
