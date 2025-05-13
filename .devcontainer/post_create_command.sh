#!/bin/bash

cd frontend && npm install
pipx install poetry

echo 'alias start-backend="cd /workspaces/VoiceAgent && python3 -m src.main --certfile cert.pem --keyfile private.key --port 8765 --llm-type openai --tts-type xtts-v2 --vad-type silero "' >> ~/.bashrc

echo 'alias start-frontend="cd /workspaces/VoiceAgent/frontend && npm run dev"' >> ~/.bashrc

source /home/vscode/.bashrc
