# FYPServer

1. Unity version

This project uses Unity 2022.3.62f3

2. Prerequisites

- Creating a venv
- install httpx, uvicorn, python-multipart, whisperx and fastapi within the venv
  - If whisperx install doesn't work, try older Python versions like 3.10

command: uvicorn server:app --host 127.0.0.1 --port 8000

Ollama: ollama pull qwen2.5vl:latest, ollama run qwen2.5vl
