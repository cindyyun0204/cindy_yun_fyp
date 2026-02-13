# Driver emotion Recognition (Unity + WhisperX + Qwen)

## Summary
A real-time multimodal emotion recognition system for driver monitoring built with:
- Unity (C#)
- MediaPipe Face Landmarker (468 landmarks)
- WhisperX (speech-to-text)
- Qwen2.5-VL via Ollama
- FastAPI backend

The system fuses:
- Facial geometry
- Cropped face images
- Speech transcription (ASR)
- Numerical face features
and produces structured emotion outputs including:
- Emotion label
- Valence (-1 → 1)
- Arousal (0 → 1)
- Confidence
- Signals & reasoning

## System Overview
### Unity Client (core components)
- FaceLandmarkerRunner
- EmotionWindowAggregator
- EmotionClient
- WhisperXClient
- LogTailerEmotionBridge
  
### Backend (server.py)
- FastAPI
- WhisperX
- Ollama (Qwen2.5-VL)

## Full Data Flow
Webcam → MediaPipe → 468 Landmarks
                ↓
         Feature Extraction
                ↓
         Window Aggregation
                ↓
Mic → WhisperX → ASR Text
                ↓
      EmotionClient → FastAPI
                ↓
         Ollama (Qwen-VL)
                ↓
         Emotion JSON Response
                ↓
            CSV Logging
            
## Unity Components Breakdown
All C# scripts are in Assets -> Scripts except for FaceLandmarkerRunner.cs, which is in Assets -> MediaPipeUnity -> Samples -> Scenes -> FaceLandmarkDetection

### FaceLandmarkerRunner
Uses MediaPipe FaceLandmarker (468-point mesh). Added features to the existing script.

Responsibilities:
- Detect face landmarks
- Log raw 468 landmarks to JSONL
- Extract geometric features
- Crop face region from frame
- Send cropped image to EmotionClient

Uses:
- FaceCropper
- FaceFeatureExtractor
- FaceLandmarksJsonlLogger

### FaceFeatureExtractor
Converts 468 landmarks into normalised features:
- mouth_open
- smile
- brow_raise
- brow_furrow
- eye_open
- head_yaw
- head_pitch
- head_roll
- blink_rate_10s

Pure geometery-based extraction.

### EmotionWindowAggregator
Maintains sliding window buffer
On code, default is 9 seconds but can be modified using the Unity object 'EmotionSystem''s 'Emotion Window Aggregator' component. Currently set as 6 seconds

Features:
- Time-window averaging
- Blink detection
- Time-based sending
- ASR-triggered sending
- Prevents overlapping requests

This stabilises predictions

### EmotionClient
Responsible for:
- Creating JSON request
- Attaching:
  - ASR text
  - Face features
  - Base64 cropped face image
- Sending post to /emotion_from_logs
- Parsingh response
- Writing CSV logs

CSV Columns:
- Timestamp
- Session ID
- Driving flag
- ASR text
- All face features
- HTTP status
- Emotion
- Valence
- Arousal
- Confidence
- Signals
- Notes

### WhisperXClient
Loop:
1. Records microphone chunk
2. Converts to WAV
3. Sends to /transcribe
4. Updates:
   - UI transcript
   - EmotionClient
   - EmotionWindowAggregator
5. Appends transcripts.jsonl

### LogTailerEmotionBridge
Reads:
- transcripts.jsonl
- face_landmarks_468.jsonl
Pushes parsed data back into the aggregator

This allows:
- Offline replay
- Dataset-driven evaluation
- Decoupled pipelines

### Logging System
All logs stored in:
Application.persistentDataPath/Logs
Check Unity logs if unsure about the path. Logs print out paths.

Structure:
Logs/
  transcripts.jsonl
  face_landmarks_468.jsonl
  emotion_log.csv
  images/
      face_session_timestamp.jpg

## Backend (server.py)
Built with:
- FastAPI
- WhisperX
- Ollama (Qwen2.5-VL)

### /transcribe
Uses WhisperX to produce:
{
  "text": "...",
  "segments": [...]
}

### /emotions_from_logs
Accepts:
{
  "session_id": "...",
  "t_utc": "...",
  "asr_text": "...",
  "face_features": { ... },
  "driving_session": true,
  "face_jpeg_b64": "..."
}

Then:
1. Constructs multimodal prompt
2. Sends to Ollama
3. Forces JSON-only output
4. Validates response
5. Returns structured emotion object

### Ollama + Qwen2.5-VL
Environment variables:
OLLAMA_CHAT_URL=http://127.0.0.1:11434/api/chat
OLLAMA_MODEL=qwen2.5vl:latest

Model is used in true multimodal mode:
- Text
- Numeric features
- Face Image

## Emotion Output Format
{
  "emotion": "happy",
  "valence": 0.62,
  "arousal": 0.71,
  "confidence": 0.88,
  "signals": [
    "smile detected",
    "positive speech tone"
  ],
  "notes": "Driver appears engaged"
}

More simplified version is printed out on Unity logs (emotion and confidence)

## Scene Architecture
The main demo scene (main) is structured into modular systems:
- Main Canvas
- Solution
- WhisperXClient
- FaceLog
- EmotionSystem

Default Unity objects (Main Camera, Direction Light, EventSystem) are not part of the emotion pipeline.

### Solution (FaceLandmarker Runner)
Component: FaceLandmarkerRunner
This object runs MediaPipe Face Landmarker and acts as the primary visual + capture pipeline.

#### Responsibilities:
- Runs MediaPipe Face Landmarker (Async mode)
- Draws face landmark annotations
- Logs 468 landmarks to JSONL
- Extracts face features
- Feeds EmotionWindowAggregator
- Crops face image and sends it to EmotionClient

#### Key Settings:
- Capture Hz: 0.333 (i.e. every 3 seconds)
- Crop Padding: 0.15
- JPEG Quality: 80
- Flip Output Vertically: enabled
- Face Crop: enabled

This object is the vision entry point of the system

### WhisperXClient
Component: WhisperXClient
Handes real-time microphone recording and speech transcription

#### Responsibilities:
- Records 3-second audio chunks
- Sends audio to:
  http://127.0.0.1:8000/transcribe
- Updates on-screen transcript
- Pushes ASR text into EmotionClient
- Triggers EmotionWindowAggregator on new speech
- Logs transcripts to transcripts.jsonl

#### Key Settings:
- Sample Rate: 16000
- Record Seconds: 3
- Transcript Logging: enabled

This object is the audio entry point of the system.

### FaceLog
Component: FaceLandmarksJsonlLogger
Logs raw 468 MediaPipe landmarks.

#### Output
Logs/face_landmarks_468.jsonl

Each frame includes:
- UTC timestamp
- Unity time
- Face index
- Full landmark array

This is used for:
- Dataset generation
- Offline replay
- Debugging
- Feature validation

Raw landmarks are not sent to the server. Instead, face features are (as described previously).

### EmotionSystem
This object contains:
- LogTailerEmotionBridge
- EmotionClient
- EmotionWindowAggregator

It is the core multimodal fusion system.

#### LogTailerEmotionBridge
Polls:
- transcripts.jsonl
- face_landmarks_468.jsonl

Feeds parsed data into the aggregator.

Poll interval: 0.2 seconds

Allows:
- Decoupled logging
- Replay capability
- File-driven integration

#### EmotionClient
Sends fused data to backend:
http://127.0.0.1:8000/emotion_from_logs

Sends:
- ASR text
- Aggregated face features
- Cropped face image (base64 JPEG)
- Driving session flag

Logs final result to:
Logs/emotion_log.csv

Key Settings:
- Server Vision Side: 256
- JPEG Quality: 70
- Save Face Images: enabled
- CSV Logging: enabled

#### EmotionWindowAggregator
Maintains sliding window of face features.

Configuration:
- Window: 6 seconds
- Keep: 8 seconds
- Blink threshold: 0.18
- Send on Timer: enabled
- Timer interval: 3 seconds
- Send on ASR Chunk: enabled

Prevents overlapping server requests and smooths predictions

## How the Scene Works Together
Webcam → Solution (FaceLandmarkerRunner)
           ↓
     468 landmarks
           ↓
   FaceLog (JSONL logging)
           ↓
   EmotionWindowAggregator
           ↓
Mic → WhisperXClient
           ↓
   EmotionClient
           ↓
FastAPI → Ollama (Qwen-VL)
           ↓
emotion_log.csv

## How to Run:
Please run on a device that has a GPU. If not it would be too slow, and might return an error instead.

### 1. Backend
Install dependencies (create a venv if needed):
pip install fastapi uvicorn whisperx torch httpx numpy ffmpeg
  - If whisperx install doesn't work, try older Python versions like 3.10
- Install Ollama (https://ollama.com/download)

Start server:
uvicorn server:app --host 127.0.0.1 --port 8000
(or any address you want)

Pull the model and ensure Ollama is running:
- ollama pull qwen2.5vl:latest
  (about 6GB)
- ollama run qwen2.5vl

### 2. Unity
Requirements:
- Unity 2022+
  This project was developed using Unity 2022.3.62f3
- MediaPipe Unity plugin
- TextMeshPro
- Newtonsoft JSON

Steps:
1. Add the Unity package (emotion_response.unitypackage) to the project
   - Open Unity project
   - Go to Assets -> Import Package -> Custom Package
   - Select the .unitypackage file
   - Click Import All
2. Configure EmotionClient.serverUrl (can keep as given)
3. Configure WhisperXClient.serverUrl (can keep as given)
4. Press Play
