# Repository improvement audit

## Summary

After reviewing the repository, the biggest improvement areas are dependency management, operational setup, backend reliability, security, and project hygiene.

## Suggested improvements

### 1. Fix Python dependency management
- `requirements.txt` appears to be saved in UTF-16 instead of standard UTF-8, which can break `pip install -r requirements.txt`.
- The current dependency list does not match the backend code: `emotion-server/server.py` imports `fastapi`, `whisperx`, `torch`, `httpx`, and `pydantic`, but these are not declared in `requirements.txt`.
- `requirements.txt` also contains packages such as `Flask`, `Jinja2`, and `Werkzeug`, although the backend is implemented with FastAPI.

**Why this matters:** the project is hard to install and reproduce reliably.

**References:** `requirements.txt`, `emotion-server/server.py`

### 2. Improve setup documentation and path accuracy
- `README.md` tells users to run `pip install -r emotion-server/requirements.txt`, but the repository currently contains `requirements.txt` at the root.
- `README.md` refers to `unity-scripts/EmotionSystem/` and `unity-scripts/Integration/`, but the repository folders are `emotion-server/Scripts/` and `driving-simulator/Scripts/`.
- `emotion-server/mediapipe_service.py` expects a local `face_landmarker.task` file, but the README does not explain where to get it or where to place it.

**Why this matters:** a new contributor cannot follow the documented setup end to end.

**References:** `README.md`, `emotion-server/mediapipe_service.py`

### 3. Reduce backend statefulness and concurrency risk
- `emotion-server/server.py` stores iteration data in module-level mutable globals such as `iteration_state`, `iteration_logs`, and `latest_emotion`.
- These values are updated across multiple async endpoints (`/iteration_start`, `/iteration_end`, `/emotion_from_logs`) without isolation or locking.

**Why this matters:** concurrent requests or overlapping sessions could corrupt state or return the wrong emotion result.

**References:** `emotion-server/server.py`

### 4. Tighten security defaults
- `emotion-server/server.py` enables CORS for every origin with `allow_origins=["*"]`.
- The backend accepts and returns sensitive behavioural/emotion data, but there is no visible authentication or origin restriction.

**Why this matters:** permissive defaults are risky even for a research prototype.

**References:** `emotion-server/server.py`

### 5. Add backend validation and automated tests
- The repository has Unity editor tests for `FaceFeatureExtractor`, but there are no Python tests covering the FastAPI backend.
- Pydantic request models in `emotion-server/server.py` use permissive defaults and do not validate important fields such as empty IDs or oversized payloads.

**Why this matters:** the most failure-prone logic currently has the least protection against regressions.

**References:** `Tests/Editor/FaceFeatureExtractorTests.cs`, `Tests/Editor/FaceFeatureExtractorBehaviourTests.cs`, `Tests/Editor/FaceFeatureExtractorEdgeCaseTests.cs`, `emotion-server/server.py`

### 6. Make configuration more portable
- Several service URLs and ports are hardcoded across the codebase, for example:
  - `http://127.0.0.1:8000` in `EmotionBOBridge.cs` and `EmotionWindowAggregator.cs`
  - `http://127.0.0.1:8000/transcribe` in `WhisperXClient.cs`
  - `http://127.0.0.1:8010/mediapipe_face` in `OfficialMediaPipeClient.cs`
- Only the Ollama settings in `server.py` currently use environment variables.

**Why this matters:** local-only defaults make deployment, testing, and teammate setup harder than necessary.

**References:** `driving-simulator/Scripts/EmotionBOBridge.cs`, `emotion-server/Scripts/EmotionWindowAggregator.cs`, `emotion-server/Scripts/WhisperXClient.cs`, `emotion-server/Scripts/OfficialMediaPipeClient.cs`, `emotion-server/server.py`

### 7. Add project hygiene basics
- There is no `.gitignore` in the repository root.
- The repository also has no visible GitHub Actions workflow for automated checks.

**Why this matters:** generated files, local environments, logs, and model assets are easier to commit accidentally, and changes are not automatically validated.

**References:** repository root, `.github/` (not present)

## Recommended next steps

1. Fix `requirements.txt` format and replace it with an accurate dependency set.
2. Update `README.md` so setup instructions match the actual repository structure.
3. Refactor backend iteration state into a dedicated state manager.
4. Restrict CORS and define environment-based configuration.
5. Add Python tests and a minimal CI workflow.
