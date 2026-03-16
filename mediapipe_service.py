import base64
import time
from typing import List, Optional, Dict

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI()

MODEL_PATH = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,          # ← enabled
    output_facial_transformation_matrixes=False,
)

landmarker = FaceLandmarker.create_from_options(options)


class FaceRequest(BaseModel):
    image_b64: str
    timestamp_ms: Optional[int] = None


class LandmarkPoint(BaseModel):
    i: int
    x: float
    y: float
    z: float


class FaceResponse(BaseModel):
    ok: bool
    face_detected: bool
    timestamp_ms: int
    landmarks: List[LandmarkPoint]
    blendshapes: Dict[str, float] = {}     # ← new: name → score (0..1)
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/mediapipe_face", response_model=FaceResponse)
def mediapipe_face(req: FaceRequest):
    ts = req.timestamp_ms or int(time.time() * 1000)

    try:
        if not req.image_b64:
            return FaceResponse(
                ok=False, face_detected=False,
                timestamp_ms=ts, landmarks=[], blendshapes={},
                error="image_b64 missing",
            )

        img_bytes = base64.b64decode(req.image_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if bgr is None:
            return FaceResponse(
                ok=False, face_detected=False,
                timestamp_ms=ts, landmarks=[], blendshapes={},
                error="failed to decode image",
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return FaceResponse(
                ok=True, face_detected=False,
                timestamp_ms=ts, landmarks=[], blendshapes={},
            )

        # ── Landmarks ────────────────────────────────────────────────────
        face = result.face_landmarks[0]
        landmarks = [
            LandmarkPoint(i=i, x=float(p.x), y=float(p.y), z=float(p.z))
            for i, p in enumerate(face)
        ]

        # ── Blendshapes ──────────────────────────────────────────────────
        blendshapes: Dict[str, float] = {}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for cat in result.face_blendshapes[0]:
                blendshapes[cat.category_name] = round(float(cat.score), 4)

        return FaceResponse(
            ok=True, face_detected=True,
            timestamp_ms=ts,
            landmarks=landmarks,
            blendshapes=blendshapes,
        )

    except Exception as e:
        return FaceResponse(
            ok=False, face_detected=False,
            timestamp_ms=ts, landmarks=[], blendshapes={},
            error=str(e),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)