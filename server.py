from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import whisperx
import torch
import tempfile
import shutil
import os
import json
import numpy as np
import time
import logging

import httpx
from pydantic import BaseModel
from typing import Dict, Any, Optional, List


# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# WhisperX setup
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisperx.load_model(
    "base",
    device=device,
    compute_type="int8",
    vad_method="silero",
)

LABELS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]

# -----------------------
# Ollama config
# -----------------------
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")

ALLOWED_EMOTIONS = [
    "neutral", "happy", "angry",
    "fear", "disgust", "surprised", "sad",
]

_http_client: Optional[httpx.AsyncClient] = None


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


# -----------------------
# Helpers
# -----------------------
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


def _json_recover(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            return json.loads(s[start:end + 1])
        raise


# -----------------------
# Request model from Unity
# -----------------------
class EmotionLogRequest(BaseModel):
    session_id: str
    t_utc: str
    asr_text: str = ""
    face_features: Dict[str, float] = {}
    driving_session: bool = True
    face_jpeg_b64: str = ""  # base64 JPEG (no data: prefix)


# -----------------------
# Ollama call (async) - MULTIMODAL
# -----------------------
async def call_ollama_emotion_json(payload: EmotionLogRequest) -> Dict[str, Any]:
    """
    Calls Ollama /api/chat with:
    - ASR text
    - face_features
    - OPTIONAL face image (base64)
    """
    if _http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")

    prompt = f"""
Classify driver emotion using ALL provided inputs (speech + face features + image if present).

Choose EXACTLY ONE emotion from:
{ALLOWED_EMOTIONS}

Return ONLY valid JSON:
{{
  "emotion": "<allowed emotion>",
  "valence": <float -1..1>,
  "arousal": <float 0..1>,
  "confidence": <float 0..1>,
  "signals": ["<short reason>", "..."],
  "notes": "<optional short note>"
}}

ASR: {payload.asr_text}

Face features (numeric): {json.dumps(payload.face_features, ensure_ascii=False)}
""".strip()

    user_msg: Dict[str, Any] = {"role": "user", "content": prompt}

    # Attach image if present (THIS is what makes Qwen-VL truly multimodal)
    if payload.face_jpeg_b64 and len(payload.face_jpeg_b64) > 100:
        user_msg["images"] = [payload.face_jpeg_b64]

    body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You output JSON only. No markdown."},
            user_msg,
        ],
        "options": {"temperature": 0.2},
    }

    t0 = time.time()
    logging.info(
        f"Ollama call start model={OLLAMA_MODEL} asr_len={len(payload.asr_text or '')} "
        f"img={'yes' if 'images' in user_msg else 'no'}"
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


# -----------------------
# Endpoints
# -----------------------
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
    t0 = time.time()
    logging.info(f"/emotion_from_logs start session={req.session_id} t_utc={req.t_utc}")

    out = await call_ollama_emotion_json(req)

    emotion = out.get("emotion", "neutral")
    if emotion not in ALLOWED_EMOTIONS:
        emotion = "neutral"

    valence = _clamp(float(out.get("valence", 0.0)), -1.0, 1.0)
    arousal = _clamp(float(out.get("arousal", 0.0)), 0.0, 1.0)
    confidence = _clamp(float(out.get("confidence", 0.5)), 0.0, 1.0)

    signals = out.get("signals", [])
    if not isinstance(signals, list):
        signals = []
    notes = str(out.get("notes", ""))[:240]

    resp = {
        "emotion": emotion,
        "valence": valence,
        "arousal": arousal,
        "confidence": confidence,
        "signals": [str(s)[:80] for s in signals][:6],
        "notes": notes
    }

    logging.info(f"/emotion_from_logs done in {time.time()-t0:.2f}s -> {emotion} conf={confidence:.2f}")
    return resp
