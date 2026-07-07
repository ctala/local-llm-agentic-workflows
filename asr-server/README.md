# Local ASR Server for Hermes / OpenClaw

OpenAI-compatible speech-to-text server that runs **faster-whisper** locally on the DGX Spark (or any compatible machine). Hermes can use it as its STT provider, so voice messages sent through Telegram, WhatsApp or other connected channels are transcribed locally without leaving the machine.

## Features

- OpenAI-compatible `/v1/audio/transcriptions` endpoint.
- Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for efficient CPU inference.
- Default model `large-v3` tuned for Spanish and English.
- `/health` endpoint for monitoring and systemd integration.

## Requirements

- Python 3.10+
- ~6 GB of disk space for the `large-v3` model (downloaded automatically on first start).
- CPU inference is the default; GPU is possible but not recommended while vLLM is running on the Spark.

## Quick start

```bash
cd asr-server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Wait for the model to load, then test:

```bash
curl -s -X POST http://localhost:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-local-asr" \
  -F file=@/path/to/audio.wav \
  -F model=whisper-1 \
  -F response_format=text
```

## Run as a systemd user service

```bash
mkdir -p ~/.config/systemd/user
cp local-asr-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now local-asr-server.service
systemctl --user status local-asr-server.service
```

The service listens on `0.0.0.0:8001` and starts automatically on login.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL` | `large-v3` | faster-whisper model size |
| `ASR_DEVICE` | `cpu` | `cpu` or `cuda` |
| `ASR_COMPUTE_TYPE` | `int8` | `int8`, `float16`, `int8_float16` |
| `ASR_HOST` | `0.0.0.0` | Bind address |
| `ASR_PORT` | `8001` | Bind port |

## Hermes configuration

Add this to `~/.hermes/config.yaml`:

```yaml
stt:
  enabled: true
  provider: openai
  openai:
    model: whisper-1
    base_url: http://localhost:8001/v1
    api_key: sk-local-asr
```

When a voice message arrives through a connected platform (e.g. Telegram), Hermes will send it to this local endpoint and receive the transcript.

## Notes

- The first start downloads the Whisper model; keep the machine online.
- If vLLM is using most of the GPU memory, keep `ASR_DEVICE=cpu`.
- For lower latency on shorter messages you can try `ASR_MODEL=medium` or `ASR_MODEL=small`, at the cost of accuracy.
