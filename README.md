# FYPServer

1. Unity version

This project uses Unity 2022.3.62f3. Please open the project with this specified version.

2. Prerequisites

- Creating a venv
- install httpx, uvicorn, python-multipart, whisperx and fastapi within the venv (pip install _: fill _ with each of them)
  - If whisperx install doesn't work, try older Python versions like 3.10
- Install Ollama (https://ollama.com/download)

3. Running the program
Before you run the Unity scene, run the server and pull & run Qwen via Ollama.
- Server start command: uvicorn server:app --host 127.0.0.1 --port 8000
- Ollama commands: ollama pull qwen2.5vl:latest (about 6GB) -> ollama run qwen2.5vl
