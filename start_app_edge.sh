#!/bin/bash

# Activate the virtual environment
conda activate rt

# Optional HF configuration
export HF_ENDPOINT="https://hf-mirror.com"

# Start the app
HF_ENDPOINT=https://hf-mirror.com python3 -m src.main --certfile cf.pem --keyfile cf.key --port 8765 --vad-type pyannote --vad-args '{"auth_token": "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"}'

#gunicorn --bind 0.0.0.0:8765 --workers 1 --reload --timeout 0 --keyfile cf.key --certfile cf.pem  src.main -k uvicorn.workers.UvicornWorker --vad-type pyannote --vad-args '{"auth_token": "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"}'
