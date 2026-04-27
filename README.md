# LLM-Based Implicit Feedback for Driving Style Optimisation

A real-time multimodal emotion recognition system that produces driving-style feedback for a Human-in-the-Loop Bayesian optimisation loop. Built with:
- Unity (C#)
- MediaPipe Face Landmarker (468 landmarks + 52 blendshapes)
- WhisperX (speech-to-text)
- Gemma 4 via Ollama
- FastAPI backend

The system fuses:
- Facial geometry (9 derived features)
- Cropped face images
- Speech transcription (ASR)
- MediaPipe blendshapes (filtered subset)

and produces structured outputs containing:
- Per-iteration objective ratings: safety, naturalness, progress (each $-3$ to $+3$)
- Emotion label (one of 7 categories)
- Valence and arousal (each $1$ to $9$, Russell's circumplex)
- Confidence
- Signals & reasoning notes

The objective ratings are written into the BO objectives so the optimiser can adapt the AV's driving style across iterations without an explicit questionnaire.

## What's included

This repository contains the parts of the system implemented for the dissertation:
- The Python backend (`server.py`, `mediapipe_service.py`)
- The Unity-side C# scripts that capture face/audio data and integrate the pipeline into the BO loop

The Unity driving simulator project itself is not included. It was developed by Pascal Jansen and Tim Eckstein at Ulm University; access to that codebase should be requested from them. The C# scripts in this repository are designed to be added into that simulator's `Assets/Scripts/` folder.

## Repository structure

```
.
├── emotion-server/                 # Python backend
│   ├── server.py                   # Main FastAPI server (port 8000)
│   ├── mediapipe_service.py        # MediaPipe FaceLandmarker service (port 8010)
│   ├── Scripts   
│   │   ├── OfficialMediaPipeRunner.cs
│   │   ├── OfficialMediaPipeClient.cs
│   │   ├── EmotionWindowAggregator.cs
│   │   ├── EmotionClient.cs
│   │   ├── WhisperXClient.cs
│   │   ├── FaceCropper.cs
│   │   ├── FaceFeatureExtractor.cs
│   │   ├── FaceLandmarksJsonlLogger.cs
│   │   ├── LogTailerEmotionBridge.cs
│   │   ├── EmotionHUD.cs
│   │   └── CsvLog.cs
│
├── driving simulator/                  # C# scripts to add to the simulator
│   └── Scripts/                # Hooks into the existing simulator
│       ├── StudyConditionManager.cs
│       ├── EmotionBOBridge.cs
│       ├── LoopForQT.cs
│       ├── ConditionHooks_UPDATE.cs
│       └── LogPaths.cs
│
└── README.md
```

## System Overview

### Unity Client (core components)
- OfficialMediaPipeRunner
- OfficialMediaPipeClient
- EmotionWindowAggregator
- EmotionClient
- WhisperXClient
- FaceCropper, FaceFeatureExtractor
- FaceLandmarksJsonlLogger
- LogTailerEmotionBridge
- EmotionBOBridge (integration with the BO loop)
- LoopForQT, ConditionHooks_UPDATE (modified BOforUnity loop)
- StudyConditionManager (condition routing)

### Backend (server.py)
- FastAPI
- WhisperX
- Ollama (Gemma 4) → multimodal text + image inference

## Conditions

The system supports three feedback conditions, set via the **Condition ID** field on the simulator's start screen:

| ID | Name | Behaviour |
|----|------|-----------|
| 1  | LLM-only | Per-iteration questionnaire is replaced by a 10-second processing pop-up. The LLM's ratings drive the BO. |
| 2  | Questionnaire-only | Standard BOforUnity behaviour. The LLM still runs and is logged for offline analysis, but does not drive the BO. |
| 3  | Combination | Both signals are collected. The BO objective values are a 50/50 blend of the LLM and questionnaire ratings. |

The **Group ID** must be set to `RuralUrbanHighway` to reproduce the study setup. Only the rural and urban environments are used; the third (Highway) entry is vestigial code inherited from an earlier study and is never instantiated.

## Per-iteration data flow

1. The Unity simulator signals `iteration_start` to the backend, which switches to a `collecting` state.
2. During the 30-second iteration:
   - The MediaPipe runner captures one webcam frame every ~3 seconds, sends it to `mediapipe_service.py`, receives the 468 landmarks and 52 blendshapes, and crops the face image.
   - The WhisperX client records 3-second audio chunks and posts them to `/transcribe`, receiving back per-chunk transcripts.
   - Every 5 seconds, the aggregator pushes its accumulated face features, blendshapes, transcript, and most recent face image to `/emotion_from_logs`. The server appends to a per-iteration buffer.
3. At iteration end, the simulator signals `iteration_end`. The backend takes the most recent buffered payload, builds the multimodal prompt, and queries Gemma 4. The structured JSON response is parsed, validated, and exposed via `/latest_emotion`.
4. The Unity client fetches the result via `/latest_emotion`, blends it with the questionnaire response according to the active condition, and writes the final objective values into the BO.

## Unity scripts

The C# scripts are split into two folders. `EmotionSystem/` is the self-contained capture pipeline written for this project. `Integration/` contains scripts that hook into the existing BOforUnity simulator's flow — some are new, some replace existing simulator scripts, and one is a patch file with edits to be pasted into an existing simulator file.

### EmotionSystem/

These are the standalone capture pipeline. None of them depend on BOforUnity; they could be lifted out and used in any Unity project.

#### OfficialMediaPipeRunner
Captures webcam frames at 0.333 Hz, sends them to the local MediaPipe service, and forwards results to the rest of the pipeline.

Responsibilities:
- Run webcam capture
- POST cropped frames to `mediapipe_service.py`
- Receive 468 landmarks + 52 blendshapes
- Crop the face region from the original frame
- Forward face features and crop to the aggregator and `EmotionClient`

Key settings:
- Capture Hz: 0.333 (one frame every 3 seconds)
- Crop Padding: 0.15
- JPEG Quality: 80

#### OfficialMediaPipeClient
Thin HTTP client for the MediaPipe service. Used by `OfficialMediaPipeRunner`.

#### FaceFeatureExtractor
Converts 468 landmarks into nine normalised features:
- mouth_open
- smile
- brow_raise
- brow_furrow
- eye_open
- head_yaw
- head_pitch
- head_roll
- blink_rate_10s

Pure geometry-based extraction.

#### EmotionWindowAggregator
Polls the backend's iteration status. While in `collecting` state, accumulates face features and posts to `/emotion_from_logs` every 5 seconds. Skips emotion capture entirely in the questionnaire-only condition (saves GPU on the LLM call).

Features:
- Iteration-status polling at 1 Hz
- Time-window averaging of face features
- Blink detection
- Time-based sending
- ASR-triggered sending
- Prevents overlapping requests
- Condition-aware skip logic

#### EmotionClient
Responsible for:
- Building the JSON request
- Attaching:
  - ASR text
  - Face features
  - Filtered blendshapes
  - Base64-encoded cropped face image
- Sending POST to `/emotion_from_logs`
- Parsing the response
- Writing CSV logs

CSV columns:
- Timestamp, Session ID
- Iteration metadata (iteration ID, condition ID, group ID)
- ASR text
- All face features
- HTTP status
- Emotion, Valence, Arousal, Confidence
- Signals, Notes

#### WhisperXClient
Loop:
1. Records a 3-second microphone chunk at 16 kHz
2. Converts to WAV
3. Sends to `/transcribe`
4. Updates:
   - On-screen transcript
   - `EmotionClient`
   - `EmotionWindowAggregator`
5. Appends to `transcripts.jsonl`

#### FaceCropper
Static helper. Computes a padded face bounding box from the 468-point mesh and crops the original webcam frame to a fixed-size JPEG suitable for the LLM.

#### FaceLandmarksJsonlLogger
Logs raw 468 MediaPipe landmarks to `face_landmarks_468.jsonl`. Each line includes:
- UTC timestamp
- Unity time
- Face index
- Full landmark array

Raw landmarks are not sent to the server; only the derived features are. The JSONL log is used for:
- Dataset generation
- Offline replay
- Debugging
- Feature validation

#### LogTailerEmotionBridge
Optional fallback. Reads `transcripts.jsonl` and `face_landmarks_468.jsonl` and pushes parsed data back into the aggregator. Allows offline replay and decoupled pipelines. Poll interval: 0.2 seconds.

#### EmotionHUD
Optional on-screen overlay showing the latest emotion label and confidence.

#### CsvLog
Tiny utility for appending CSV rows.

### Integration/

These connect the emotion pipeline to the existing BOforUnity loop. They fall into three groups: new files to add, existing files to replace, and a patch file with edits to be pasted into an existing simulator file.

**New scripts.** Drop these into `Assets/Scripts/`:

#### StudyConditionManager
Single source of truth for condition logic. Maps Condition ID (1, 2, 3) to one of `llm_only`, `value_only`, `combination`, and exposes predicate methods (`ShouldUseEmotion`, `ShouldUseQuestionnaireForBO`, `ShouldBlend`, `ShouldEmotionSystemCapture`) that the rest of the integration consults. Every other integration script asks this class what to do, so the routing rules live in one place.

#### EmotionBOBridge
At iteration end, fetches the latest LLM result via `/latest_emotion`, blends it with the questionnaire values according to the active condition, and writes the final objective values into the BO. This is where the 50/50 blend in the combination condition actually happens.

#### LogPaths
Centralised resolver for the two `Application.persistentDataPath/Logs/` roots (one for the emotion system, one for the simulator). Used by both subsystems to discover each other's logs without hardcoding paths.

**Modified scripts.** These replace BOforUnity's existing versions of the same file:

#### LoopForQT
The simulator's per-iteration coroutine. Modified to:
- Read the active condition from `StudyConditionManager` at the start of each iteration
- Skip the questionnaire entirely in the LLM-only condition and instead show a 10-second processing pop-up while the LLM runs
- Trigger `EmotionBOBridge.ApplyEmotionToObjectives()` at iteration end so the LLM result gets written into the BO before the next iteration starts

The original control flow for the questionnaire-only condition is unchanged.

**Patch file.** Paste the snippets in this file into the indicated existing simulator file:

#### ConditionHooks_UPDATE
Not a standalone class. Contains drop-in replacements for methods in BOforUnity's `QTQuestionnaireManager.cs` (part of the QuestionnaireToolkit subsystem). The replacements add condition-aware behaviour: when an iteration ends, the patched `EndOfQt()` calls `EmotionBOBridge.ApplyEmotionToObjectives()` so the emotion result is folded into the BO objectives in the LLM-only and combination conditions. Inline comments in the file mark where each snippet goes.

### Logging System

All logs are stored in:
```
Application.persistentDataPath/Logs
```

Check Unity logs if unsure about the path; the logger prints the resolved path on startup.

Structure:
```
Logs/
├── transcripts.jsonl
├── face_landmarks_468.jsonl
├── emotion_log.csv
└── images/
    └── face_<session>_<timestamp>.jpg
```

The simulator side writes per-iteration BO state and final selected parameters into its own `Logs/` folder; QuestionnaireToolkit additionally writes per-iteration questionnaire responses into its own `Results/` folder.

## Backend (server.py)

Built with FastAPI, WhisperX, and Ollama (Gemma 4). Key endpoints:

### /transcribe
Uses WhisperX to produce:
```json
{
  "text": "...",
  "segments": [...]
}
```

### /iteration_start, /iteration_end, /iteration_status
Manage the iteration state machine (`idle` → `collecting` → `processing` → `ready`). The Unity client polls `/iteration_status` at 1 Hz to coordinate with the simulator's iteration timing.

### /emotion_from_logs
Accepts:
```json
{
  "session_id": "...",
  "t_utc": "...",
  "asr_text": "...",
  "face_features": { ... },
  "blendshapes": { ... },
  "face_jpeg_b64": "..."
}
```
Behaviour depends on iteration state:
- During `collecting`: appends payload to the iteration buffer and returns immediately. No LLM call.
- During `idle` (backwards-compatible): processes immediately and returns the LLM result.

### /latest_emotion
Returns the most recent LLM result. Used by `EmotionBOBridge` to fetch ratings at iteration end.

### /health
Returns service status, including which model and Ollama URL are configured.

### Iteration-end LLM call
At `iteration_end`, the server:
1. Pulls the most recent buffered payload.
2. Constructs a multimodal prompt (text + face image).
3. Sends it to Ollama.
4. Forces JSON-only output via the system message.
5. Validates and clamps the response (numeric ranges, allowed emotion list).
6. Returns the structured emotion object.

### Ollama + Gemma 4

Environment variables:
```
OLLAMA_CHAT_URL=http://127.0.0.1:11434/api/chat
OLLAMA_MODEL=gemma4:latest
```

Generation parameters: `temperature=0.2`, `stream=false`, `think=false`.

The model is used in true multimodal mode:
- Text (prompt with task definition, modality declarations, scale anchors)
- Numeric features (face features, filtered blendshapes as JSON)
- Face image (attached as a separate `images` field on the user message)

Of MediaPipe's 52 blendshapes, only a curated subset of 21 (those most relevant to the seven emotion categories) is forwarded; an activation threshold of 0.01 further filters out inactive blendshapes from each prompt to keep token count low.

## Output format

```json
{
  "safety": -1.5,
  "naturalness": 0.5,
  "progress": 1.0,
  "emotion": "neutral",
  "valence": 5.5,
  "arousal": 3.0,
  "confidence": 0.85,
  "signals": ["calm facial expression", "steady speech"],
  "notes": "Driver appears unconcerned"
}
```

A more simplified version is printed out on the Unity console (emotion and confidence).

Field constraints:
- `safety` / `naturalness` / `progress`: floats in $[-3, +3]$
- `valence` / `arousal`: floats in $[1, 9]$
- `confidence`: float in $[0, 1]$
- `emotion`: one of `neutral, happy, angry, fear, disgust, surprised, sad`

## How to run

This project requires a CUDA-capable GPU. Without one, model inference will be too slow and may error out.

### 1. Backend

Create a Python 3.12 venv (3.10 also works if WhisperX has compatibility issues with newer versions).

Terminal 1 — MediaPipe service on port 8010:
```
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r emotion-server/requirements.txt
python emotion-server/mediapipe_service.py
```

Terminal 2 — main FastAPI server (Gemma 4, WhisperX) on port 8000:
```
venv\Scripts\activate
python emotion-server/server.py
```

Terminal 3 — Ollama:
```
ollama pull gemma4:latest
ollama run gemma4:latest
```

Notes:
- If WhisperX install fails, try Python 3.10 or 3.11.
- Install Ollama from https://ollama.com/download.
- Health-check both services: `http://127.0.0.1:8000/health` and `http://127.0.0.1:8010/health`.

### 2. Unity

Requirements:
- Unity 2022.3 LTS (project was developed using 2022.3.62f3)
- The BOforUnity-based driving simulator project (request access from Ulm University)
- TextMeshPro
- Newtonsoft JSON

Steps:
1. Copy the contents of `unity-scripts/EmotionSystem/` and the **new** scripts in `unity-scripts/Integration/` (`StudyConditionManager.cs`, `EmotionBOBridge.cs`, `LogPaths.cs`) into the simulator project's `Assets/Scripts/` folder.
2. Replace the simulator's existing `LoopForQT.cs` with the version in `unity-scripts/Integration/`.
3. Open `unity-scripts/Integration/ConditionHooks_UPDATE.cs` and follow the inline instructions: paste the labelled snippets into the indicated locations in `QTQuestionnaireManager.cs` (part of the QuestionnaireToolkit subsystem).
4. In the main scene:
   - Create a GameObject called `EmotionSystem` and attach `EmotionWindowAggregator`, `EmotionClient`, and `LogTailerEmotionBridge` to it.
   - Create a GameObject called `WhisperXClient` and attach the `WhisperXClient` component (sample rate 16000, record seconds 3).
   - Attach `OfficialMediaPipeRunner` to the GameObject hosting the webcam input.
   - Attach `EmotionBOBridge` to a manager GameObject so the BO loop can find it.
5. Configure the server URLs on `EmotionClient` and `WhisperXClient` if running services on a non-default host or port.
6. Press Play. On the start screen, set Condition ID (1, 2, or 3), enter a User ID, set Group ID to `RuralUrbanHighway`, and press Initialize.
