"""MediaPipe Face Landmarker — 25-dim feature extraction with mouth-open ratio."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from gazer.head_pose import estimate_head_pose_numpy
from gazer.models import ensure_face_landmarker

# Iris landmarks (478-landmark model)
_LEFT_IRIS = [468, 469, 470, 471, 472]
_RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye corners and lids
_LEFT_INNER = 133
_LEFT_OUTER = 33
_LEFT_UPPER = 159
_LEFT_LOWER = 145
_RIGHT_INNER = 362
_RIGHT_OUTER = 263
_RIGHT_UPPER = 386
_RIGHT_LOWER = 374

# EAR landmarks (6 points per eye)
_LEFT_EAR_IDX = [33, 160, 158, 133, 153, 144]
_RIGHT_EAR_IDX = [362, 385, 387, 263, 373, 380]

# Nose
_NOSE_TIP = 1
_NOSE_BRIDGE = 6
_FACE_CENTER_X = 234  # between the eyes, approximate face center x
_FACE_CENTER_Y = 234  # approximate face center y

# Mouth landmarks for MAR
_MOUTH_UPPER_LIP = 13
_MOUTH_LOWER_LIP = 14
_MOUTH_LEFT_CORNER = 61
_MOUTH_RIGHT_CORNER = 291

# ---- 25-dimensional feature vector ----
# Indices 0-7:   Original features (eye ratios + head pose + scale)
# Indices 8-11:  Absolute iris position in face frame
# Indices 12-13: Iris radii (left, right)
# Indices 14-15: Eye asymmetry (vergence cue)
# Indices 16-19: Individual eyelid positions
# Indices 20-21: Nose direction relative to face center
# Indices 22-24: Face 3D position (tvec from solvePnP)
FEATURE_DIM = 25

# Feature names for debugging / documentation
FEATURE_NAMES = [
    "l_rx", "l_ry", "r_rx", "r_ry",        # 0-3
    "yaw", "pitch", "roll", "norm_scale",   # 4-7
    "l_iris_x", "l_iris_y",                  # 8-9
    "r_iris_x", "r_iris_y",                  # 10-11
    "l_iris_r", "r_iris_r",                  # 12-13
    "eye_asym_x", "eye_asym_y",              # 14-15
    "l_upper_lid", "l_lower_lid",            # 16-17
    "r_upper_lid", "r_lower_lid",            # 18-19
    "nose_dx", "nose_dy",                    # 20-21
    "face_tx", "face_ty", "face_tz",         # 22-24
]


@dataclass
class GazeFeatures:
    vector: np.ndarray       # shape (FEATURE_DIM,)
    left_ear: float
    right_ear: float
    ear: float
    mar: float               # mouth aspect ratio (for mouth-open click)
    yaw: float               # for outlier rejection
    pitch: float
    valid: bool


def _iris_center(landmarks: np.ndarray, indices: list[int], w: int, h: int) -> tuple[float, float]:
    xs = landmarks[indices, 0] * w
    ys = landmarks[indices, 1] * h
    return float(xs.mean()), float(ys.mean())


def _iris_radius(landmarks: np.ndarray, indices: list[int], w: int, h: int) -> float:
    """Compute iris radius as average distance from center to boundary landmarks."""
    cx, cy = _iris_center(landmarks, indices, w, h)
    pts = np.column_stack([landmarks[indices, 0] * w, landmarks[indices, 1] * h])
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return float(dists.mean())


def _ratio(val: float, a: float, b: float) -> float:
    denom = b - a
    if abs(denom) < 1e-6:
        return 0.5
    return (val - a) / denom


def _eye_ratios(
    landmarks: np.ndarray, w: int, h: int,
    inner: int, outer: int, upper: int, lower: int, iris_idx: list[int],
) -> tuple[float, float]:
    ix, iy = _iris_center(landmarks, iris_idx, w, h)
    inner_x = landmarks[inner, 0] * w
    outer_x = landmarks[outer, 0] * w
    upper_y = landmarks[upper, 1] * h
    lower_y = landmarks[lower, 1] * h
    rx = _ratio(ix, inner_x, outer_x)
    ry = _ratio(iy, upper_y, lower_y)
    return rx, ry


def _ear(landmarks: np.ndarray, w: int, h: int, indices: list[int]) -> float:
    pts = np.column_stack([landmarks[indices, 0] * w, landmarks[indices, 1] * h])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h_dist = np.linalg.norm(pts[0] - pts[3])
    if h_dist < 1e-6:
        return 0.3
    return (v1 + v2) / (2.0 * h_dist)


def _mouth_open_ratio(landmarks: np.ndarray, w: int, h: int) -> float:
    upper_y = landmarks[_MOUTH_UPPER_LIP, 1] * h
    lower_y = landmarks[_MOUTH_LOWER_LIP, 1] * h
    left_x = landmarks[_MOUTH_LEFT_CORNER, 0] * w
    right_x = landmarks[_MOUTH_RIGHT_CORNER, 0] * w
    vertical = abs(lower_y - upper_y)
    horizontal = abs(right_x - left_x)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def _raw_to_numpy(raw_landmarks) -> np.ndarray:
    n = len(raw_landmarks)
    arr = np.empty((n, 3), dtype=np.float64)
    for i, lm in enumerate(raw_landmarks):
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = getattr(lm, "z", 0.0)
    return arr


class FaceTracker:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        model_path = str(ensure_face_landmarker())
        base = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._image_format = mp.ImageFormat.SRGB
        self._start_time: float | None = None

        # Previous iris centers for temporal features (future use)
        self._prev_l_iris: tuple[float, float] | None = None
        self._prev_r_iris: tuple[float, float] | None = None

        # Phase 3: Store raw landmarks for camera preview visualization
        self._last_raw_landmarks: np.ndarray | None = None

    def close(self) -> None:
        self._landmarker.close()

    def _get_timestamp_ms(self) -> int:
        if self._start_time is None:
            self._start_time = time.monotonic()
        return int((time.monotonic() - self._start_time) * 1000)

    def extract(self, frame_rgb: np.ndarray, frame_width: int, frame_height: int) -> GazeFeatures | None:
        if frame_rgb is None or frame_width <= 0 or frame_height <= 0:
            return None

        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)

        mp_image = self._mp_image_cls(image_format=self._image_format, data=frame_rgb)
        ts_ms = self._get_timestamp_ms()
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks:
            self._prev_l_iris = None
            self._prev_r_iris = None
            self._last_raw_landmarks = None
            return None

        raw = result.face_landmarks[0]
        if len(raw) < 478:
            return None

        lm = _raw_to_numpy(raw)

        # Phase 3: Store raw landmarks for camera preview
        self._last_raw_landmarks = lm.copy()

        w, h = frame_width, frame_height

        # ---- Original 4 features: eye ratios ----
        l_rx, l_ry = _eye_ratios(lm, w, h, _LEFT_INNER, _LEFT_OUTER, _LEFT_UPPER, _LEFT_LOWER, _LEFT_IRIS)
        r_rx, r_ry = _eye_ratios(lm, w, h, _RIGHT_INNER, _RIGHT_OUTER, _RIGHT_UPPER, _RIGHT_LOWER, _RIGHT_IRIS)

        # ---- Head pose (7 returns now: yaw, pitch, roll, scale, tx, ty, tz) ----
        yaw, pitch, roll, face_scale, face_tx, face_ty, face_tz = estimate_head_pose_numpy(lm, w, h)
        norm_scale = face_scale / max(w * 0.25, 1.0)

        # ---- EAR (per eye + average) ----
        left_ear = _ear(lm, w, h, _LEFT_EAR_IDX)
        right_ear = _ear(lm, w, h, _RIGHT_EAR_IDX)
        ear = (left_ear + right_ear) / 2.0

        # ---- MAR (mouth open) ----
        mar = _mouth_open_ratio(lm, w, h)

        # ---- NEW: Absolute iris positions in normalized face frame ----
        # Centered on nose tip for invariance
        nose_x = lm[_NOSE_TIP, 0]
        nose_y = lm[_NOSE_TIP, 1]
        l_ix, l_iy = _iris_center(lm, _LEFT_IRIS, 1, 1)  # normalized
        r_ix, r_iy = _iris_center(lm, _RIGHT_IRIS, 1, 1)
        l_iris_x = l_ix - nose_x  # relative to nose
        l_iris_y = l_iy - nose_y
        r_iris_x = r_ix - nose_x
        r_iris_y = r_iy - nose_y

        # ---- NEW: Iris radii ----
        l_iris_r = _iris_radius(lm, _LEFT_IRIS, w, h) / max(face_scale, 1.0)
        r_iris_r = _iris_radius(lm, _RIGHT_IRIS, w, h) / max(face_scale, 1.0)

        # ---- NEW: Eye asymmetry (vergence cue) ----
        eye_asym_x = l_rx - r_rx
        eye_asym_y = l_ry - r_ry

        # ---- NEW: Individual eyelid positions (normalized) ----
        l_upper_lid = lm[_LEFT_UPPER, 1] - nose_y
        l_lower_lid = lm[_LEFT_LOWER, 1] - nose_y
        r_upper_lid = lm[_RIGHT_UPPER, 1] - nose_y
        r_lower_lid = lm[_RIGHT_LOWER, 1] - nose_y

        # ---- NEW: Nose bridge direction ----
        nose_bridge_x = lm[_NOSE_BRIDGE, 0] - nose_x
        nose_bridge_y = lm[_NOSE_BRIDGE, 1] - nose_y

        # ---- NEW: Face 3D position (normalized tvec) ----
        # Normalize by face_scale to make distance-invariant
        norm_tz = face_tz / max(abs(face_tz), 1.0) if face_tz != 0 else 0.0

        # ---- Assemble 25-dim vector ----
        vector = np.array([
            # Original 8
            l_rx, l_ry, r_rx, r_ry,
            yaw, pitch, roll, norm_scale,
            # Absolute iris positions (face-frame, nose-centered)
            l_iris_x, l_iris_y, r_iris_x, r_iris_y,
            # Iris radii (scale-normalized)
            l_iris_r, r_iris_r,
            # Eye asymmetry
            eye_asym_x, eye_asym_y,
            # Individual eyelid positions
            l_upper_lid, l_lower_lid, r_upper_lid, r_lower_lid,
            # Nose direction
            nose_bridge_x, nose_bridge_y,
            # Face 3D position
            face_tx / max(w, 1.0), face_ty / max(h, 1.0), norm_tz,
        ], dtype=np.float32)

        # Store previous for potential temporal features
        self._prev_l_iris = (l_ix, l_iy)
        self._prev_r_iris = (r_ix, r_iy)

        return GazeFeatures(
            vector=vector,
            left_ear=left_ear,
            right_ear=right_ear,
            ear=ear,
            mar=mar,
            yaw=yaw,
            pitch=pitch,
            valid=True,
        )