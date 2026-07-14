"""MediaPipe model asset management with auto-download."""

from __future__ import annotations

import urllib.request
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = ASSETS_DIR / "face_landmarker.task"


def ensure_face_landmarker() -> Path:
    """Return path to face_landmarker.task, downloading if missing."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if FACE_LANDMARKER_PATH.exists() and FACE_LANDMARKER_PATH.stat().st_size > 1_000_000:
        return FACE_LANDMARKER_PATH

    print("Downloading face_landmarker.task (~4 MB)...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
    print("Download complete.")
    return FACE_LANDMARKER_PATH
