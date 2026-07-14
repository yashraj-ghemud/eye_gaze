"""Head pose estimation via solvePnP with refined camera intrinsics."""

from __future__ import annotations

import math

import cv2
import numpy as np

# MediaPipe FaceMesh landmark indices for head pose (expanded set)
_NOSE_TIP = 1
_NOSE_BRIDGE = 6
_CHIN = 152
_FOREHEAD = 10
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291
_LEFT_BROW = 70
_RIGHT_BROW = 300

# 3D face model (mm-scale, average adult proportions)
# More points = better solvePnP conditioning
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # 0  nose tip
        (0.0, -7.5, -6.0),         # 1  nose bridge
        (0.0, -63.6, -12.5),       # 2  chin
        (0.0, 25.0, -15.0),        # 3  forehead
        (-43.3, 32.7, -26.0),      # 4  left eye outer
        (43.3, 32.7, -26.0),       # 5  right eye outer
        (-16.0, 30.0, -25.0),      # 6  left eye inner
        (16.0, 30.0, -25.0),       # 7  right eye inner
        (-28.9, -28.9, -24.1),     # 8  left mouth
        (28.9, -28.9, -24.1),      # 9  right mouth
        (-35.0, 48.0, -25.0),      # 10 left brow
        (35.0, 48.0, -25.0),       # 11 right brow
    ],
    dtype=np.float64,
)

# Mapping: landmark index -> _MODEL_POINTS row index
_LANDMARK_INDICES = [
    _NOSE_TIP, _NOSE_BRIDGE, _CHIN, _FOREHEAD,
    _LEFT_EYE_OUTER, _RIGHT_EYE_OUTER, _LEFT_EYE_INNER, _RIGHT_EYE_INNER,
    _LEFT_MOUTH, _RIGHT_MOUTH, _LEFT_BROW, _RIGHT_BROW,
]

# Inter-pupillary landmark indices (for focal length estimation)
_IPD_LEFT = _LEFT_EYE_INNER   # 133
_IPD_RIGHT = _RIGHT_EYE_INNER  # 362

# Real-world average IPD in mm (adults)
_REAL_IPD_MM = 63.0


def _rotation_matrix_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    rot, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-rot[2, 0], sy)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        roll = math.atan2(rot[2, 1], rot[2, 2])
    else:
        pitch = math.atan2(-rot[2, 0], sy)
        yaw = 0.0
        roll = math.atan2(-rot[1, 2], rot[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _estimate_focal(
    landmarks: list | np.ndarray,
    frame_width: int,
    frame_height: int,
) -> float:
    """Estimate focal length from observed inter-pupillary distance.

    Uses the known average IPD (63 mm) and the observed pixel distance
    between inner eye corners to refine the focal length estimate.
    """
    if isinstance(landmarks, np.ndarray):
        lx = landmarks[_IPD_LEFT, 0] * frame_width
        rx = landmarks[_IPD_RIGHT, 0] * frame_width
        ly = landmarks[_IPD_LEFT, 1] * frame_height
        ry = landmarks[_IPD_RIGHT, 1] * frame_height
    else:
        lx, ly = landmarks[_IPD_LEFT].x * frame_width, landmarks[_IPD_LEFT].y * frame_height
        rx, ry = landmarks[_IPD_RIGHT].x * frame_width, landmarks[_IPD_RIGHT].y * frame_height

    ipd_px = math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
    if ipd_px < 10:
        return float(frame_width)

    # focal = ipd_px * (face_model_ipd) / real_world_ipd
    # Face model IPD between inner corners is ~32mm
    model_ipd = abs(_MODEL_POINTS[6, 0] - _MODEL_POINTS[7, 0])  # ~32mm
    focal = ipd_px * _REAL_IPD_MM / model_ipd
    # Clamp to reasonable range
    return max(frame_width * 0.5, min(focal, frame_width * 3.0))


def estimate_head_pose(
    landmarks: list,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    """Returns yaw, pitch, roll (degrees) and face_scale."""
    result = _solve(landmarks, frame_width, frame_height)
    if result is None:
        return 0.0, 0.0, 0.0, 1.0
    yaw, pitch, roll, face_scale, _, _, _ = result
    return yaw, pitch, roll, face_scale


def estimate_head_pose_numpy(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float, float, float, float]:
    """Numpy version. Returns (yaw, pitch, roll, face_scale, face_tx, face_ty, face_tz)."""
    result = _solve(landmarks, frame_width, frame_height)
    if result is None:
        return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    return result


def _solve(
    landmarks: list | np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float, float, float, float] | None:
    """Core solvePnP. Returns (yaw, pitch, roll, face_scale, tx, ty, tz) or None."""
    image_points = []
    for idx in _LANDMARK_INDICES:
        if isinstance(landmarks, np.ndarray):
            image_points.append((landmarks[idx, 0] * frame_width, landmarks[idx, 1] * frame_height))
        else:
            image_points.append((landmarks[idx].x * frame_width, landmarks[idx].y * frame_height))
    image_points = np.array(image_points, dtype=np.float64)

    # Refine focal length from observed face proportions
    focal = _estimate_focal(landmarks, frame_width, frame_height)
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    # Use a simple radial distortion model
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    yaw, pitch, roll = _rotation_matrix_to_euler(rvec)

    # Face scale: inter-ocular distance in pixels
    face_scale = float(np.linalg.norm(image_points[4] - image_points[5]))

    # Face 3D position from translation vector (normalized)
    tx = float(tvec[0, 0])
    ty = float(tvec[1, 0])
    tz = float(tvec[2, 0])

    return yaw, pitch, roll, face_scale, tx, ty, tz