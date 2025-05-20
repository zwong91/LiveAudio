#!/bin/bash

cd frontend && npm install
pipx install poetry

echo 'alias start-backend="cd /workspaces/VoiceAgent && python3 -m src.main "' >> ~/.bashrc

source /home/vscode/.bashrc
