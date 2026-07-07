"""OpenAI-compatible local ASR server for Hermes.

Runs faster-whisper on the DGX Spark. Hermes can be configured to use it
as the STT provider by pointing its OpenAI-compatible audio endpoint here.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import PlainTextResponse
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_SIZE = os.environ.get("ASR_MODEL", "large-v3")
DEVICE = os.environ.get("ASR_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "int8")
HOST = os.environ.get("ASR_HOST", "0.0.0.0")
PORT = int(os.environ.get("ASR_PORT", "8001"))

whisper_model: Optional[WhisperModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model
    print(f"Loading faster-whisper model: {MODEL_SIZE} ({DEVICE}, {COMPUTE_TYPE})")
    whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model loaded.")
    yield
    whisper_model = None


app = FastAPI(title="Local ASR Server", lifespan=lifespan)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    prompt: Optional[str] = Form(None),
):
    if whisper_model is None:
        return PlainTextResponse("Model not loaded", status_code=503)

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language or None,
            initial_prompt=prompt or None,
            condition_on_previous_text=True,
        )
        texts = [segment.text.strip() for segment in segments]
        text = " ".join(texts)
    finally:
        os.unlink(tmp_path)

    # Hermes' openai STT provider expects plain text by default.
    if response_format in ("text", "verbose_text"):
        return PlainTextResponse(text)

    return {"text": text}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_SIZE, "loaded": whisper_model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
