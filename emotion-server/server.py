from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import whisperx
import torch
import tempfile
import shutil
import os
import json
import re
import numpy as np
import time
import logging
import asyncio

import httpx
from pydantic import BaseModel
from typing import Dict, Any, Optional, List


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WhisperX setup
device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisperx.load_model(
    "base",
    device=device,
    compute_type="float16" if device == "cuda" else "int8",
    vad_method="silero",
)


LABELS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]

# Ollama config
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

ALLOWED_EMOTIONS = [
    "neutral", "happy", "angry",
    "fear", "disgust", "surprised", "sad",
]

# Blendshape keys we care about - keeps the prompt concise
BLENDSHAPE_KEYS = [
    "browDownLeft", "browDownRight",
    "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawOpen",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthOpen",
    "mouthPucker",
    "noseSneerLeft", "noseSneerRight",
    "cheekPuff",
]

_http_client: Optional[httpx.AsyncClient] = None

# Store latest emotion result for driving simulator polling
latest_emotion: Optional[Dict[str, Any]] = None

# ==========================================================================
# ITERATION STATE — signaled by driving simulator
# ==========================================================================
# States: "idle" -> "collecting" -> "processing" -> "ready" -> "idle"
#   idle:       no iteration running
#   collecting: iteration in progress, emotion system should collect data
#   processing: iteration ended, emotion system should send data to Ollama
#   ready:      emotion result available for driving sim to fetch
iteration_state: str = "idle"
iteration_number: int = 0
iteration_start_time: float = 0.0
iteration_end_time: float = 0.0
iteration_user_id: str = ""
iteration_condition_id: str = ""
iteration_group_id: str = ""
iteration_environment_index: int = 0
iteration_condition: str = ""

# Accumulated emotion logs received during the iteration window
iteration_logs: List[Dict[str, Any]] = []


@app.on_event("startup")
async def _startup():
    global _http_client
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
    _http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    logging.info("Server startup complete. Async HTTP client ready.")


@app.on_event("shutdown")
async def _shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None
    logging.info("Server shutdown complete.")


# Helpers

def _safe_temp_wav(upload: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(upload.file, tmp)
        return tmp.name


def _transcribe_wav_path(tmp_path: str) -> dict:
    audio = whisperx.load_audio(tmp_path)
    result = model.transcribe(audio)

    segments = result.get("segments", [])
    text = result.get("text")
    if text is None:
        text = " ".join([s.get("text", "").strip() for s in segments]).strip()

    return {
        "text": text,
        "segments": segments,
        "raw_keys": list(result.keys()),
    }


def dummy_probs() -> dict:
    probs = {l: 0.0 for l in LABELS}
    probs["neutral"] = 1.0
    return probs


def predict_audio_emotion(wav_path: str) -> dict:
    probs = dummy_probs()
    label = max(probs, key=probs.get)
    conf = float(probs[label])
    return {"label": label, "confidence": conf, "probs": probs}


def predict_face_emotion(face_features: np.ndarray) -> dict:
    probs = dummy_probs()
    label = max(probs, key=probs.get)
    conf = float(probs[label])
    return {"label": label, "confidence": conf, "probs": probs}


def fuse_probs(audio_probs: dict, face_probs: Optional[dict], w_audio: float, w_face: float) -> dict:
    a = np.array([audio_probs[l] for l in LABELS], dtype=np.float32)
    f = np.array([face_probs[l] for l in LABELS], dtype=np.float32) if face_probs else np.zeros_like(a)

    fused = w_audio * a + w_face * f
    s = float(fused.sum())
    if s > 0:
        fused = fused / s

    out = {LABELS[i]: float(fused[i]) for i in range(len(LABELS))}
    label = max(out, key=out.get)
    return {"label": label, "confidence": float(out[label]), "probs": out}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _strip_think_tags(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()


def _json_recover(s: str) -> Dict[str, Any]:
    s = _strip_think_tags(s or "")
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            return json.loads(s[start:end + 1])
        raise


def _filter_blendshapes(raw: Dict[str, float]) -> Dict[str, float]:
    """Keep only the emotionally relevant blendshapes, drop near-zero values."""
    out = {}
    for k in BLENDSHAPE_KEYS:
        v = raw.get(k)
        if v is not None and v > 0.01:
            out[k] = round(v, 3)
    return out


# Request model from Unity

class EmotionLogRequest(BaseModel):
    session_id: str
    t_utc: str
    asr_text: str = ""
    face_features: Dict[str, float] = {}
    blendshapes: Dict[str, float] = {}
    driving_session: bool = True
    face_jpeg_b64: str = ""


class IterationSignal(BaseModel):
    iteration: int = 0
    user_id: str = ""
    condition_id: str = ""
    group_id: str = ""
    environment_index: int = 0
    condition: str = ""


# Ollama call (async) - modality-aware prompt


async def call_ollama_emotion_json(payload: EmotionLogRequest) -> Dict[str, Any]:
    if _http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")

    has_face = bool(
        payload.face_features
        and any(v != 0.0 for v in payload.face_features.values())
    )
    has_blendshapes = bool(payload.blendshapes)
    has_image = bool(payload.face_jpeg_b64 and len(payload.face_jpeg_b64) > 100)
    has_asr = bool(payload.asr_text and payload.asr_text.strip())

    modalities = []
    if has_asr:           modalities.append("speech transcription")
    if has_face:          modalities.append("geometric face features")
    if has_blendshapes:   modalities.append("face blendshapes")
    if has_image:         modalities.append("face image")
    if not modalities:    modalities.append("limited context")

    modality_str = " + ".join(modalities)

    prompt = f"""
Classify driver emotion AND estimate their perceived safety, naturalness, and progress using the provided inputs ({modality_str}).
Only base your analysis on the modalities listed above. Do not infer signals from modalities marked as not available.

Definitions:
- safety: how safe the driver feels right now (-3 = feels very unsafe, +3 = feels completely safe)
- naturalness: how natural the autonomous vehicle's behavior appears to the driver (-3 = very unnatural/jerky/unpredictable, +3 = completely natural/smooth/human-like)
- progress: how well the driver perceives the route or task is progressing (-3 = no progress/stuck/going wrong way, +3 = excellent progress/on track/completing efficiently)

Choose EXACTLY ONE emotion from:
{ALLOWED_EMOTIONS}

Return ONLY valid JSON:
{{
  "safety": <float -3..3>,
  "naturalness": <float -3..3>,
  "progress": <float -3..3>,
  "emotion": "<allowed emotion>",
  "valence": <float 1..9>,
  "arousal": <float 1..9>,
  "confidence": <float 0..1>,
  "signals": ["<short reason>", "..."],
  "notes": "<optional short note>"
}}

ASR: {payload.asr_text if has_asr else "(no speech)"}
"""

    if has_face:
        prompt += f"\nGeometric face features: {json.dumps(payload.face_features, ensure_ascii=False)}"
    else:
        prompt += "\nGeometric face features: not available"

    if has_blendshapes:
        filtered = _filter_blendshapes(payload.blendshapes)
        prompt += f"\nFace blendshapes (MediaPipe, 0=inactive 1=full activation): {json.dumps(filtered, ensure_ascii=False)}"
    else:
        prompt += "\nFace blendshapes: not available"

    prompt = prompt.strip()

    user_msg: Dict[str, Any] = {"role": "user", "content": prompt}
    if has_image:
        user_msg["images"] = [payload.face_jpeg_b64]

    body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": "You output JSON only. No markdown. /nothink"},
            user_msg,
        ],
        "options": {"temperature": 0.2},
    }

    t0 = time.time()
    logging.info(
        f"Ollama call start model={OLLAMA_MODEL} "
        f"modalities=[{modality_str}] "
        f"asr_len={len(payload.asr_text or '')} "
        f"blendshapes={len(payload.blendshapes)} "
        f"img={'yes' if has_image else 'no'}"
    )

    try:
        r = await _http_client.post(OLLAMA_CHAT_URL, json=body)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {e}")

    dt = time.time() - t0
    logging.info(f"Ollama call done status={r.status_code} in {dt:.2f}s")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text[:800]}")

    data = r.json()
    content = (data.get("message") or {}).get("content", "")

    try:
        return _json_recover(content)
    except Exception:
        raise HTTPException(status_code=502, detail=f"Model did not return valid JSON. Raw: {content[:800]}")


# ==========================================================================
# ITERATION SIGNAL ENDPOINTS — called by driving simulator
# ==========================================================================

@app.post("/iteration_start")
async def iteration_start(signal: IterationSignal):
    """Driving sim calls this when a new iteration begins (driver starts driving)."""
    global iteration_state, iteration_number, iteration_start_time, iteration_logs
    global iteration_user_id, iteration_condition_id, iteration_group_id
    global iteration_environment_index, iteration_condition

    iteration_state = "collecting"
    iteration_number = signal.iteration
    iteration_start_time = time.time()
    iteration_logs = []  # clear accumulated logs for new iteration
    iteration_user_id = signal.user_id
    iteration_condition_id = signal.condition_id
    iteration_group_id = signal.group_id
    iteration_environment_index = signal.environment_index
    iteration_condition = signal.condition

    logging.info(
        f"/iteration_start iter={signal.iteration} "
        f"user={signal.user_id} cond={signal.condition_id} group={signal.group_id} "
        f"env={signal.environment_index} condition={signal.condition}"
    )
    return {
        "status": "collecting",
        "iteration": iteration_number,
        "message": "Iteration started. Emotion system should begin collecting data."
    }


@app.post("/iteration_end")
async def iteration_end(signal: IterationSignal):
    """Driving sim calls this when iteration ends (questionnaire appears).
    This triggers emotion processing with the latest accumulated data."""
    global iteration_state, iteration_end_time, latest_emotion

    iteration_end_time = time.time()
    duration = iteration_end_time - iteration_start_time
    iteration_state = "processing"

    logging.info(
        f"/iteration_end iter={signal.iteration} "
        f"duration={duration:.1f}s accumulated_logs={len(iteration_logs)}"
    )

    # If we have accumulated emotion logs, use the latest one to produce the result
    # (it should have the most complete data — full window of face features + ASR)
    if iteration_logs:
        latest_log = iteration_logs[-1]  # use the last/most complete submission
        logging.info(f"Processing emotion from {len(iteration_logs)} accumulated logs")

        try:
            # Build an EmotionLogRequest from the latest accumulated log
            req = EmotionLogRequest(**latest_log)
            out = await call_ollama_emotion_json(req)

            emotion = out.get("emotion", "neutral")
            if emotion not in ALLOWED_EMOTIONS:
                emotion = "neutral"

            safety      = _clamp(float(out.get("safety",      0.0)), -3.0, 3.0)
            naturalness = _clamp(float(out.get("naturalness",  0.0)), -3.0, 3.0)
            progress    = _clamp(float(out.get("progress",     0.0)), -3.0, 3.0)
            valence     = _clamp(float(out.get("valence",      0.0)), 1.0, 9.0)
            arousal     = _clamp(float(out.get("arousal",      0.0)), 1.0, 9.0)
            confidence  = _clamp(float(out.get("confidence",   0.5)), 0.0, 1.0)

            signals = out.get("signals", [])
            if not isinstance(signals, list):
                signals = []
            notes = str(out.get("notes", ""))[:240]

            resp = {
                "safety": safety, "naturalness": naturalness, "progress": progress,
                "emotion": emotion,
                "valence": valence, "arousal": arousal,
                "confidence": confidence,
                "signals": [str(s)[:80] for s in signals][:6],
                "notes": notes,
                "iteration": iteration_number,
                "window_duration": round(duration, 1),
                "logs_used": len(iteration_logs),
            }

            latest_emotion = resp
            iteration_state = "ready"

            logging.info(
                f"Iteration {iteration_number} emotion ready: {emotion} "
                f"conf={confidence:.2f} S={safety:.2f} N={naturalness:.2f} P={progress:.2f}"
            )
            return resp

        except Exception as e:
            iteration_state = "ready"  # still mark ready so sim doesn't hang
            logging.error(f"Emotion processing failed for iteration {iteration_number}: {e}")
            return {"status": "error", "detail": str(e), "iteration": iteration_number}
    else:
        iteration_state = "ready"
        logging.warning(f"Iteration {iteration_number} ended but no emotion logs were accumulated.")
        return {
            "status": "no_data",
            "iteration": iteration_number,
            "message": "No emotion data was collected during this iteration."
        }


@app.get("/iteration_status")
async def get_iteration_status():
    """Emotion system polls this to know when to collect/send data."""
    return {
        "state": iteration_state,
        "iteration": iteration_number,
        "start_time": iteration_start_time,
        "logs_count": len(iteration_logs),
        "user_id": iteration_user_id,
        "condition_id": iteration_condition_id,
        "group_id": iteration_group_id,
        "environment_index": iteration_environment_index,
        "condition": iteration_condition,
    }


# ==========================================================================
# EMOTION LOG ENDPOINT — called by emotion Unity system during collecting
# ==========================================================================

@app.post("/emotion_from_logs")
async def emotion_from_logs(req: EmotionLogRequest):
    """Called by the emotion Unity system with face/voice/blendshape data.
    During 'collecting' state, data is accumulated for iteration_end processing.
    Otherwise processes immediately (backwards compatible)."""
    global latest_emotion, iteration_logs

    t0 = time.time()
    logging.info(
        f"/emotion_from_logs start session={req.session_id} t_utc={req.t_utc} "
        f"blendshapes={len(req.blendshapes)} state={iteration_state}"
    )

    # If we're in collecting state, accumulate the log for later processing
    if iteration_state == "collecting":
        iteration_logs.append(req.dict())
        logging.info(
            f"Accumulated log #{len(iteration_logs)} for iteration {iteration_number} "
            f"(will process at iteration_end)"
        )
        return {
            "status": "accumulated",
            "iteration": iteration_number,
            "logs_count": len(iteration_logs),
            "message": "Data stored. Will process when iteration ends."
        }

    # Otherwise process immediately (backwards compatible / initial capture)
    out = await call_ollama_emotion_json(req)

    emotion = out.get("emotion", "neutral")
    if emotion not in ALLOWED_EMOTIONS:
        emotion = "neutral"

    safety      = _clamp(float(out.get("safety",      0.0)), -3.0, 3.0)
    naturalness = _clamp(float(out.get("naturalness",  0.0)), -3.0, 3.0)
    progress    = _clamp(float(out.get("progress",     0.0)), -3.0, 3.0)
    valence     = _clamp(float(out.get("valence",      0.0)), 1.0, 9.0)
    arousal     = _clamp(float(out.get("arousal",      0.0)), 1.0, 9.0)
    confidence  = _clamp(float(out.get("confidence",   0.5)), 0.0, 1.0)

    signals = out.get("signals", [])
    if not isinstance(signals, list):
        signals = []
    notes = str(out.get("notes", ""))[:240]

    resp = {
        "safety": safety, "naturalness": naturalness, "progress": progress,
        "emotion": emotion,
        "valence": valence, "arousal": arousal,
        "confidence": confidence,
        "signals": [str(s)[:80] for s in signals][:6],
        "notes": notes,
    }

    latest_emotion = resp

    logging.info(
        f"/emotion_from_logs done in {time.time()-t0:.2f}s -> {emotion} "
        f"conf={confidence:.2f} safety={safety:.2f} nat={naturalness:.2f} prog={progress:.2f}"
    )
    return resp


# ==========================================================================
# OTHER ENDPOINTS
# ==========================================================================

@app.get("/health")
async def health():
    return {
        "ok": True,
        "device": device,
        "ollama_chat_url": OLLAMA_CHAT_URL,
        "ollama_model": OLLAMA_MODEL,
        "image_mode": "enabled_if_face_jpeg_b64_sent",
        "iteration_state": iteration_state,
        "iteration_number": iteration_number,
    }


@app.get("/latest_emotion")
async def get_latest_emotion():
    if latest_emotion is None:
        return {
            "safety": 0.0, "naturalness": 0.0, "progress": 0.0,
            "emotion": "neutral", "valence": 0.0, "arousal": 0.0,
            "confidence": 0.0, "signals": [], "notes": "no data yet",
        }
    return latest_emotion


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tmp_path = _safe_temp_wav(file)
    try:
        return _transcribe_wav_path(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/predict_emotion")
async def predict_emotion(
    file: UploadFile = File(...),
    face_features_json: str = Form(...),
    face_detected: int = Form(...),
):
    tmp_path = _safe_temp_wav(file)
    try:
        face_ok = bool(face_detected)
        face_features = np.array(json.loads(face_features_json), dtype=np.float32) if face_ok else None

        audio_pred = predict_audio_emotion(tmp_path)
        face_pred = predict_face_emotion(face_features) if face_ok and face_features is not None else None

        w_audio = 1.0
        w_face = 1.0 if face_pred is not None else 0.0
        fused = fuse_probs(audio_pred["probs"], face_pred["probs"] if face_pred else None, w_audio, w_face)

        return {
            "final": {"label": fused["label"], "confidence": fused["confidence"]},
            "audio": {"label": audio_pred["label"], "confidence": audio_pred["confidence"]},
            "face":  {"label": (face_pred["label"] if face_pred else "none"),
                      "confidence": (face_pred["confidence"] if face_pred else 0.0),
                      "used": bool(face_pred is not None)},
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/transcribe_and_emotion")
async def transcribe_and_emotion(
    file: UploadFile = File(...),
    face_features_json: str = Form(...),
    face_detected: int = Form(...),
):
    tmp_path = _safe_temp_wav(file)
    try:
        transcript = _transcribe_wav_path(tmp_path)

        face_ok = bool(face_detected)
        face_features = np.array(json.loads(face_features_json), dtype=np.float32) if face_ok else None

        audio_pred = predict_audio_emotion(tmp_path)
        face_pred = predict_face_emotion(face_features) if face_ok and face_features is not None else None

        w_audio = 1.0
        w_face = 1.0 if face_pred is not None else 0.0
        fused = fuse_probs(audio_pred["probs"], face_pred["probs"] if face_pred else None, w_audio, w_face)

        return {
            "transcript": transcript,
            "emotion": {
                "final": {"label": fused["label"], "confidence": fused["confidence"]},
                "audio": {"label": audio_pred["label"], "confidence": audio_pred["confidence"]},
                "face":  {"label": (face_pred["label"] if face_pred else "none"),
                          "confidence": (face_pred["confidence"] if face_pred else 0.0),
                          "used": bool(face_pred is not None)},
            }
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass