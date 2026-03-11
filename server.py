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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:latest")

ALLOWED_EMOTIONS = [
    "neutral", "happy", "angry",
    "fear", "disgust", "surprised", "sad",
]

_http_client: Optional[httpx.AsyncClient] = None

# Store latest emotion result for driving simulator polling
latest_emotion: Optional[Dict[str, Any]] = None


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Remove <think>...</think> blocks that some models emit even with /nothink."""
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


# ---------------------------------------------------------------------------
# Request model from Unity
# ---------------------------------------------------------------------------

class EmotionLogRequest(BaseModel):
    session_id: str
    t_utc: str
    asr_text: str = ""
    face_features: Dict[str, float] = {}
    driving_session: bool = True
    face_jpeg_b64: str = ""


# ---------------------------------------------------------------------------
# Ollama call (async) - modality-aware prompt
# ---------------------------------------------------------------------------

async def call_ollama_emotion_json(payload: EmotionLogRequest) -> Dict[str, Any]:
    if _http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")

    # ---- Detect which modalities are actually present ----
    has_face = bool(
        payload.face_features
        and any(v != 0.0 for v in payload.face_features.values())
    )
    has_image = bool(payload.face_jpeg_b64 and len(payload.face_jpeg_b64) > 100)
    has_asr = bool(payload.asr_text and payload.asr_text.strip())

    modalities = []
    if has_asr:
        modalities.append("speech transcription")
    if has_face:
        modalities.append("face features")
    if has_image:
        modalities.append("face image")

    if not modalities:
        modalities.append("limited context")

    modality_str = " + ".join(modalities)

    # ---- Build prompt ----
    prompt = f"""
Classify driver emotion AND estimate their perceived trust, safety, and risk using the provided inputs ({modality_str}).
Only base your analysis on the modalities listed above. Do not infer signals from modalities that are marked as not available.

Definitions (all values use the range -1 to +1):
- trust: how much the driver trusts the vehicle / autonomous system (-1 = no trust at all, +1 = complete trust)
- safety: how safe the driver feels right now (-1 = feels very unsafe, +1 = feels completely safe)
- risk: how much risk the driver perceives (-1 = extreme perceived risk, +1 = no perceived risk at all)
  Note: risk is POSITIVELY stated, so +1 means the driver feels minimal risk.

Choose EXACTLY ONE emotion from:
{ALLOWED_EMOTIONS}

Return ONLY valid JSON:
{{
  "trust": <float -1..1>,
  "safety": <float -1..1>,
  "risk": <float -1..1>,
  "emotion": "<allowed emotion>",
  "valence": <float -1..1>,
  "arousal": <float 0..1>,
  "confidence": <float 0..1>,
  "signals": ["<short reason>", "..."],
  "notes": "<optional short note>"
}}

ASR: {payload.asr_text if has_asr else "(no speech)"}
"""

    if has_face:
        prompt += f"\nFace features (numeric): {json.dumps(payload.face_features, ensure_ascii=False)}"
    else:
        prompt += "\nFace features: not available"

    prompt = prompt.strip()

    # ---- Build message (conditionally attach image) ----
    user_msg: Dict[str, Any] = {"role": "user", "content": prompt}

    if has_image:
        user_msg["images"] = [payload.face_jpeg_b64]

    body = {
        "model": OLLAMA_MODEL,
        "stream": False,
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "ok": True,
        "device": device,
        "ollama_chat_url": OLLAMA_CHAT_URL,
        "ollama_model": OLLAMA_MODEL,
        "image_mode": "enabled_if_face_jpeg_b64_sent",
    }


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


@app.post("/emotion_from_logs")
async def emotion_from_logs(req: EmotionLogRequest):
    global latest_emotion

    t0 = time.time()
    logging.info(f"/emotion_from_logs start session={req.session_id} t_utc={req.t_utc}")

    out = await call_ollama_emotion_json(req)

    emotion = out.get("emotion", "neutral")
    if emotion not in ALLOWED_EMOTIONS:
        emotion = "neutral"

    trust = _clamp(float(out.get("trust", 0.0)), -1.0, 1.0)
    safety = _clamp(float(out.get("safety", 0.0)), -1.0, 1.0)
    risk = _clamp(float(out.get("risk", 0.0)), -1.0, 1.0)

    valence = _clamp(float(out.get("valence", 0.0)), -1.0, 1.0)
    arousal = _clamp(float(out.get("arousal", 0.0)), 0.0, 1.0)
    confidence = _clamp(float(out.get("confidence", 0.5)), 0.0, 1.0)

    signals = out.get("signals", [])
    if not isinstance(signals, list):
        signals = []
    notes = str(out.get("notes", ""))[:240]

    resp = {
        "trust": trust,
        "safety": safety,
        "risk": risk,
        "emotion": emotion,
        "valence": valence,
        "arousal": arousal,
        "confidence": confidence,
        "signals": [str(s)[:80] for s in signals][:6],
        "notes": notes,
    }

    # Store latest emotion for the driving simulator to poll
    latest_emotion = resp

    logging.info(
        f"/emotion_from_logs done in {time.time()-t0:.2f}s -> {emotion} conf={confidence:.2f} "
        f"trust={trust:.2f} safety={safety:.2f} risk={risk:.2f}"
    )
    return resp


# Endpoint for driving simulator to poll
@app.get("/latest_emotion")
async def get_latest_emotion():
    if latest_emotion is None:
        return {
            "trust": 0.0, "safety": 0.0, "risk": 0.0,
            "emotion": "neutral", "valence": 0.0, "arousal": 0.0,
            "confidence": 0.0, "signals": [], "notes": "no data yet",
        }
    return latest_emotion