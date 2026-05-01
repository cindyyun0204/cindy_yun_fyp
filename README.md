# LLMs as Implicit Feedback for Bayesian Optimization in Personalized Automated Driving

This repository contains the emotion recognition system developed for the dissertation *From Faces to Feedback: LLM-Based Implicit Evaluation for Driving Style Optimisation in Automated Vehicles* (Yun, 2026, UCL).

The system investigates whether a multimodal Large Language Model can replace or supplement explicit questionnaire feedback in human-in-the-loop multi-objective Bayesian optimisation (HITL MOBO) for automated vehicle driving style personalisation. It captures facial landmarks and blendshapes via MediaPipe, transcribes speech via WhisperX, and processes both modalities through Gemma 4 (via Ollama) to output the same safety, naturalness, and progress ratings as the questionnaire.

The pipeline is integrated into a Unity-based HITL MOBO driving simulator that uses the BOforUnity framework. A within-subjects user study (N = 15) compared three feedback conditions: questionnaire-only, LLM-only, and a 50/50 combination, across rural and urban environments.

## What's included

This repository contains two parts:

- **Python backend** (`emotion-server/`): the FastAPI server (`server.py`) and MediaPipe service (`mediapipe_service.py`) that run locally to receive sensor data, query Gemma 4 via Ollama, and return structured emotion ratings.
- **Unity C# scripts**: the emotion capture client (`emotion-server/Scripts/`) and the BO-loop integration scripts (`driving-simulator/Scripts/`) that hook into the existing simulator.

The Unity driving simulator project itself is **not** included in this repository. It was developed by Pascal Jansen and Tim Eckstein at Ulm University; access should be requested from them. The scripts under `driving-simulator/Scripts/` are designed to be added into that simulator's `Assets/Scripts/` folder.

## Repository structure

```
.
├── emotion-server/                 # Python backend + Unity emotion capture scripts
│   ├── server.py                   # Main FastAPI server (port 8000)
│   ├── mediapipe_service.py        # MediaPipe FaceLandmarker service (port 8010)
│   ├── tests/                      # pytest suite
│   └── Scripts/                    # Unity emotion capture pipeline
│       ├── OfficialMediaPipeRunner.cs
│       ├── OfficialMediaPipeClient.cs
│       ├── EmotionWindowAggregator.cs
│       ├── EmotionClient.cs
│       ├── WhisperXClient.cs
│       ├── FaceCropper.cs
│       ├── FaceFeatureExtractor.cs
│       ├── FaceLandmarksJsonlLogger.cs
│       ├── LogTailerEmotionBridge.cs
│       ├── EmotionHUD.cs
│       ├── LogPaths.cs
│       └── CsvLog.cs
│
├── driving-simulator/              # BO-loop integration scripts for the simulator
│   └── Scripts/
│       ├── StudyConditionManager.cs
│       ├── EmotionBOBridge.cs
│       ├── LoopForQT.cs
│       └── ConditionHooks_UPDATE.cs
│
├── Tests/                          # Unity editor tests for FaceFeatureExtractor
├── .github/workflows/ci.yml
├── .gitignore
├── requirements.txt
└── README.md
```

## Conditions

The system supports three feedback conditions (Section 3.2.1 of the dissertation), set via the **Condition ID** field on the simulator's start screen:

| ID | Name | Behaviour |
| --- | --- | --- |
| 1 | LLM-only | Per-iteration questionnaire is replaced by a 10-second processing pop-up. The LLM's ratings drive the BO. |
| 2 | Questionnaire-only | Standard BOforUnity behaviour. The LLM still runs but does not capture anything (idle state). |
| 3 | Combination | Both signals are collected. The BO objective values are a 50/50 blend of the LLM and questionnaire ratings. |

The **Group ID** must be set to `RuralUrbanHighway` to reproduce the study setup. Only the rural and urban environments are used; the third (Highway) entry is vestigial code inherited from an earlier study and is never instantiated.

## Per-iteration data flow

1. The Unity simulator signals `iteration_start` to the backend, which switches to a `collecting` state.
2. During the 30-second iteration:
   - The MediaPipe runner captures one webcam frame every ~3 seconds, sends it to `mediapipe_service.py`, and receives the 468 landmarks and 52 blendshapes.
   - The WhisperX client records 3-second audio chunks at 16 kHz mono and posts them to `/transcribe`, receiving back per-chunk transcripts.
   - Every 5 seconds, the aggregator pushes accumulated face features, blendshapes, transcript, and the most recent cropped face image to `/emotion_from_logs`. The server appends to a per-iteration buffer.
3. At iteration end, the simulator signals `iteration_end`. The backend takes the most recent buffered payload, builds the multimodal prompt, and queries Gemma 4. The structured JSON response is parsed, validated, and exposed via `/latest_emotion`.
4. The Unity client fetches the result, blends it with the questionnaire response according to the active condition, and writes the final objective values into the BO.

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

Field constraints (enforced server-side):

- `safety` / `naturalness` / `progress`: floats in $[-3, +3]$
- `valence` / `arousal`: floats in $[1, 9]$
- `confidence`: float in $[0, 1]$
- `emotion`: one of `neutral, happy, angry, fear, disgust, surprised, sad`

The full prompt template (system message, user template, allowed emotion categories, blendshape whitelist, generation parameters) is reproduced in Appendix F of the dissertation.

## Prerequisites

- **Unity 2022.3 LTS.** Both the simulator project and this emotion capture project must be opened separately in Unity Hub.
- **Python 3.12.** Required for the FastAPI backend and MediaPipe service. Older versions (3.10 or 3.11) may be needed if WhisperX encounters compatibility issues.
- **Ollama**, available at <https://ollama.com/download>.
- **CUDA-capable GPU.** Required for acceptable LLM inference speeds and WhisperX transcription. Development was conducted on an NVIDIA RTX 4070 and an NVIDIA RTX 5080; testing was conducted on an NVIDIA RTX 5080.
- **Operating system.** Developed and tested on Windows. The terminal commands below assume a Windows environment with PowerShell.

## Installation

### 1. Get the MediaPipe FaceLandmarker model

`mediapipe_service.py` loads a `face_landmarker.task` file at startup. Download it from Google's MediaPipe model garden:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
  -OutFile "face_landmarker.task"
```

The path can be overridden via the `FACE_LANDMARKER_TASK` environment variable.

### 2. Set up the Python environment

Navigate to the project directory:

```powershell
cd <path-to-this-repository>
```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
```

Install dependencies for the MediaPipe service:

```powershell
pip install mediapipe fastapi uvicorn opencv-python numpy --upgrade
```

Install dependencies for the main server (in the same virtual environment):

```powershell
pip install fastapi uvicorn whisperx torch httpx numpy ffmpeg --upgrade
```

Alternatively, install everything at once via `requirements.txt`.

### 3. Set up Ollama and Gemma 4

Install Ollama from <https://ollama.com/download>, then pull the model:

```powershell
ollama pull gemma4:latest
```

The `gemma4:latest` tag at the time of writing corresponds to the `e4b` variant.

## Running the system

Three terminal sessions and two Unity instances are required. Start the services in the following order.

### Terminal 1: MediaPipe Service

```powershell
cd <path-to-this-repository>
venv\Scripts\activate
uvicorn mediapipe_service:app --host 127.0.0.1 --port 8010
```

### Terminal 2: Main Server (LLM and WhisperX)

```powershell
cd <path-to-this-repository>
venv\Scripts\activate
uvicorn server:app --host 127.0.0.1 --port 8000
```

### Terminal 3: Ollama

```powershell
ollama serve
```

If the error `bind: Only one usage of each socket address` appears, Ollama is already running in the background and no action is needed.

### Unity: Emotion Capture System

Open the emotion capture Unity project in Unity Hub and start the emotion capture scene. **This must be started before the driving simulator.**

### Unity: Driving Simulator

Open the driving simulator project (the BOforUnity-based simulator) in Unity Hub, then follow this procedure:

1. Run the `StartScene`.
2. Input the **User ID**.
3. Input the **Condition ID** (1 = LLM-only, 2 = Questionnaire-only, 3 = Combination).
4. Input the **Group ID**: set this to `RuralUrbanHighway` for the study setup.
5. Click **Update IDs** and wait for the debug log `Python process started successfully`.
6. Click **Load Map** after the debug log `Initialization DONE` appears. A further log displays the BO configuration.
7. Wait approximately 20 seconds for the scene to load. The scene must be included in Unity's build settings and its name must match the corresponding `SceneName`-tagged GameObject in the `StartScene`.
8. Click **Initialize** to begin the driving session with the MOBO active.

To monitor emotion recognition activity within the driving simulator, type `Emotion` in the Unity console search bar to filter for `[EmotionBridge]` log entries.

## Configuration (environment variables)

All service URLs and ports are read from environment variables with localhost defaults.

| Variable | Default | Used by |
| --- | --- | --- |
| `OLLAMA_CHAT_URL` | `http://127.0.0.1:11434/api/chat` | server.py |
| `OLLAMA_MODEL` | `gemma4:latest` | server.py |
| `EMOTION_CORS_ORIGINS` | `http://localhost,http://127.0.0.1` | server.py CORS allow-list (comma-separated) |
| `MEDIAPIPE_CORS_ORIGINS` | `http://localhost,http://127.0.0.1` | mediapipe_service.py CORS allow-list |
| `FACE_LANDMARKER_TASK` | `face_landmarker.task` | mediapipe_service.py |
| `MEDIAPIPE_SERVICE_HOST` | `127.0.0.1` | mediapipe_service.py bind host |
| `MEDIAPIPE_SERVICE_PORT` | `8010` | mediapipe_service.py bind port |
| `EMOTION_SERVER_BASE_URL` | `http://127.0.0.1:8000` | Unity (`EmotionWindowAggregator`, `WhisperXClient`, `EmotionBOBridge`) |
| `MEDIAPIPE_BASE_URL` | `http://127.0.0.1:8010` | Unity (`OfficialMediaPipeClient`) |

## Running the tests

```powershell
pip install pytest pytest-asyncio
cd emotion-server
python -m pytest tests\ -v
```

The same suite runs on every push and PR via `.github/workflows/ci.yml`.

## Log file locations

Unity applications on Windows store persistent data under:

```
%USERPROFILE%\AppData\LocalLow\DefaultCompany\<ApplicationName>\
```

The following log directories are relative to this base path (Section D.5 of the dissertation):

- **Questionnaire responses** — per-iteration CSV files generated by the driving simulator's questionnaire system:
  `ContextBOStudyResults\Assets\QuestionnaireToolkit\Results`
- **Blended and raw emotion responses** — combined log entries recording both LLM and questionnaire values as used by the MOBO:
  `ContextBOStudyResults\Logs`
- **Emotion system data** — transcripts, captured face images, facial landmarks, and raw LLM responses logged by the emotion recognition system:
  `<EmotionSystemProjectName>\Logs`
- **Driving parameters** — per-iteration driving simulator parameters (speed, offset, braking, distance), stored relative to the driving simulator project root:
  `Assets\Logs`

These directories contain webcam crops, transcripts, and other personally-identifying data and are excluded by `.gitignore`.

## Notes

- Audio and facial data are captured every 3 seconds during each 30-second segment, then sent to the server in bulk at the end of the iteration. Only one LLM assessment is produced per iteration.
- Speech transcription operates on rolling 3-second audio chunks at 16 kHz mono.
- The condition router (`StudyConditionManager.cs`) maps Condition IDs 1/2/3 to `llm_only`, `value_only`, and `combination` respectively, and is the single source of truth for which feedback channel is active.

## Citation

If you use this code or build on this work, please cite the dissertation:

> Yun, C. (2026). *From Faces to Feedback: LLM-Based Implicit Evaluation for Driving Style Optimisation in Automated Vehicles.* BSc dissertation, University College London.
