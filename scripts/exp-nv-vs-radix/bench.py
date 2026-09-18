#!/usr/bin/env python3
"""Batería unificada para comparar variantes Qwen3.8-Flash-Next."""
import argparse, json, time, base64, urllib.request, statistics, os, sys
from concurrent.futures import ThreadPoolExecutor

PROMPTS = {
    "chat_short": [
        {"role":"user","content":"Hola, cómo estás? Contame brevemente qué modelos conocés y por qué te conviene correr en una DGX Spark. Respondé en español rioplatense, conciso, ~80 tokens."}
    ],
    "code_quicksort": [
        {"role":"user","content":"Implement quicksort in Python 3 with random pivot, docstring, type hints, and 3 doctests including an empty list. Then explain the average-case complexity."}
    ],
    "agent_with_tools": [
        {"role":"user","content":"You are a senior data engineer. Given a CSV with columns `ts,user_id,event_type`, write a Python script that:\n1) loads it with pandas,\n2) computes daily active users,\n3) emits a matplotlib chart to /tmp/dau.png,\n4) returns path on stdout.\nUse only stdlib + pandas + matplotlib. Include a tool call."}
    ],
}
VISION_PROMPT_TEXT = "Describe briefly the architecture in the image (1 sentence each: what's the input, what's the output, what's unusual)."
VISION_IMG = "/home/ctala/VoxCPM/assets/voxcpm_model.png"
CONCURRENCY_LEVELS = [1, 4, 8]
AUTH = "Bearer sk-spark-local"

def chat(base_url, model, messages, max_tokens=512, timeout=180):
    body = json.dumps({"model":model,"messages":messages,"max_tokens":max_tokens,"stream":False}).encode()
    req = urllib.request.Request(base_url+"/v1/chat/completions", data=body,
                                 headers={"Content-Type":"application/json","Authorization":AUTH})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req,timeout=timeout).read().decode())
    dt = time.time()-t0
    comp = r["usage"]["completion_tokens"]
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {"dt":dt,"tok_s":comp/dt if dt>0 else 0,
            "content":content,"reasoning":reasoning,
            "tokens":comp,
            "reasoning_tokens":r["usage"].get("completion_tokens_details",{}).get("reasoning_tokens",0)}

def vision_chat(base_url, model, b64, max_tokens=256, timeout=180):
    messages = [{"role":"user","content":[
        {"type":"text","text":VISION_PROMPT_TEXT},
        {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}
    ]}]
    return chat(base_url, model, messages, max_tokens=max_tokens, timeout=timeout)

def concurrency(base_url, model, prompt_messages, n, max_tokens=200):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _: chat(base_url, model, prompt_messages, max_tokens=max_tokens, timeout=180), range(n)))
    dt = time.time()-t0
    toks = [r["tokens"] for r in results]
    return {"n":n,"wall_dt":dt,"total_tokens":sum(toks),
            "aggregate_tok_s":sum(toks)/dt,
            "per_request_tok_s":[r["tok_s"] for r in results],
            "median_tok_s":statistics.median([r["tok_s"] for r in results])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="ej: http://localhost:4000 o http://localhost:8001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--skip-conc", action="store_true")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"# Variante: {args.variant} | URL: {base_url} | model: {args.model}")
    try:
        r = urllib.request.urlopen(urllib.request.Request(base_url+"/v1/models", headers={"Authorization":AUTH}), timeout=5)
        print(f"# /v1/models: HTTP {r.status} OK")
    except Exception as e:
        print(f"# /v1/models FAIL: {e}"); sys.exit(1)

    out = {"variant":args.variant,"base_url":base_url,"model":args.model,
           "single_stream":{},"concurrency":{},"vision":{}}

    print("# Warmup…", end=" ", flush=True)
    chat(base_url, args.model, PROMPTS["chat_short"], max_tokens=64)
    print("OK")

    for name, msgs in PROMPTS.items():
        print(f"# {name}…", end=" ", flush=True)
        r = chat(base_url, args.model, msgs, max_tokens=512)
        r["content"] = (r["content"] or "")[:200]
        out["single_stream"][name] = r
        print(f"{r['tokens']} tok en {r['dt']:.2f}s → {r['tok_s']:.2f} tok/s")

    print("# vision…", end=" ", flush=True)
    b64 = base64.b64encode(open(VISION_IMG,"rb").read()).decode()
    r = vision_chat(base_url, args.model, b64, max_tokens=256)
    r["content"] = (r["content"] or "")[:200]
    out["vision"]["voxcpm"] = r
    print(f"{r['tokens']} tok en {r['dt']:.2f}s → {r['tok_s']:.2f} tok/s | {(r['content'] or '')[:80]!r}")

    if not args.skip_conc:
        for n in CONCURRENCY_LEVELS:
            print(f"# concurrency c={n}…", end=" ", flush=True)
            r = concurrency(base_url, args.model, PROMPTS["chat_short"], n=n, max_tokens=200)
            out["concurrency"][f"c{n}"] = r
            print(f"agg={r['aggregate_tok_s']:.1f} median={r['median_tok_s']:.2f} wall={r['wall_dt']:.2f}s")

    os.makedirs("/tmp/nv-exp", exist_ok=True)
    outpath = f"/tmp/nv-exp/bench-{args.variant}.json"
    open(outpath,"w").write(json.dumps(out, indent=2))
    print(f"\n## Dump: {outpath}")

if __name__=="__main__":
    main()