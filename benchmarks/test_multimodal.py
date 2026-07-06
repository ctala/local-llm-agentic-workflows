#!/usr/bin/env python3
"""Test multimodal requests (image and audio) against local vLLM OpenAI API."""
import base64, json, sys, time, urllib.request

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "nemotron3-nano-omni"
MODE = sys.argv[2] if len(sys.argv) > 2 else "image"
FILE_PATH = sys.argv[3] if len(sys.argv) > 3 else None

def encode_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

if MODE == "image":
    if not FILE_PATH:
        FILE_PATH = "~/VoxCPM/assets/voxcpm_model.png"
    b64 = encode_file(FILE_PATH)
    ext = FILE_PATH.split(".")[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    content = [
        {"type": "text", "text": "Describe esta imagen en español en una oración."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
elif MODE == "audio":
    if not FILE_PATH:
        FILE_PATH = "~/voice-cloning-repo/outputs/voxcpm2_test_01.wav"
    b64 = encode_file(FILE_PATH)
    ext = FILE_PATH.split(".")[-1].lower()
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(ext, "audio/wav")
    content = [
        {"type": "text", "text": "Transcribe y resume en español el audio."},
        {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
else:
    print("Mode must be 'image' or 'audio'")
    sys.exit(1)

payload = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 256,
    "temperature": 0.6,
    "top_p": 0.95,
}).encode("utf-8")

req = urllib.request.Request(API_URL, data=payload, headers={
    "Content-Type": "application/json",
}, method="POST")

t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        dur = time.time() - t0
        print(f"Mode: {MODE} | File: {FILE_PATH}")
        print(f"Total time: {dur:.2f}s")
        print(f"Response: {text}")
except Exception as e:
    print(f"ERROR: {e}")
    if hasattr(e, "read"):
        print(e.read().decode("utf-8"))
