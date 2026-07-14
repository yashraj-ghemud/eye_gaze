"""Camera preview widget with MediaPipe face mesh landmark visualization.

Shows a live camera feed during calibration with ALL 478 landmarks drawn in
distinct RGB colors, grouped by facial region, with labels explaining what
each landmark group represents and its significance for gaze tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


# ---- Landmark group definitions with colors and descriptions ----

@dataclass
class LandmarkGroup:
    name: str
    indices: list[int]
    color_bgr: tuple[int, int, int]
    color_rgb: tuple[int, int, int]
    radius: int = 2
    description: str = ""
    significance: str = ""


def _build_landmark_groups() -> list[LandmarkGroup]:
    """Define all landmark groups with colors, descriptions, and significance."""
    groups = [
        LandmarkGroup(
            name="Face Oval",
            indices=list(range(10, 33)) + [234] + list(range(454, 464)),
            color_bgr=(255, 200, 0),
            color_rgb=(0, 200, 255),
            radius=2,
            description="Face contour outline",
            significance="Defines face bounding region for head pose estimation",
        ),
        LandmarkGroup(
            name="Left Eye",
            indices=[33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246],
            color_bgr=(255, 0, 0),
            color_rgb=(0, 0, 255),
            radius=3,
            description="Left eye contour (16 landmarks)",
            significance="Iris position within eye = primary gaze direction signal",
        ),
        LandmarkGroup(
            name="Right Eye",
            indices=[362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398],
            color_bgr=(0, 0, 255),
            color_rgb=(255, 0, 0),
            radius=3,
            description="Right eye contour (16 landmarks)",
            significance="Combined with left eye for vergence and depth perception",
        ),
        LandmarkGroup(
            name="Left Iris",
            indices=[468, 469, 470, 471, 472],
            color_bgr=(0, 255, 255),
            color_rgb=(255, 255, 0),
            radius=4,
            description="Left iris boundary (5 landmarks)",
            significance="Iris center position = where the eye is looking",
        ),
        LandmarkGroup(
            name="Right Iris",
            indices=[473, 474, 475, 476, 477],
            color_bgr=(0, 255, 0),
            color_rgb=(0, 255, 0),
            radius=4,
            description="Right iris boundary (5 landmarks)",
            significance="Iris center = gaze vector; radius = pupil dilation",
        ),
        LandmarkGroup(
            name="Left Eyebrow",
            indices=[46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            color_bgr=(200, 0, 200),
            color_rgb=(200, 0, 200),
            radius=2,
            description="Left eyebrow shape (10 landmarks)",
            significance="Brow position helps estimate head tilt and vertical gaze",
        ),
        LandmarkGroup(
            name="Right Eyebrow",
            indices=[276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
            color_bgr=(200, 100, 200),
            color_rgb=(200, 100, 200),
            radius=2,
            description="Right eyebrow shape (10 landmarks)",
            significance="Brow asymmetry indicates head roll angle",
        ),
        LandmarkGroup(
            name="Nose Bridge",
            indices=[6, 197, 195, 5, 4, 1],
            color_bgr=(0, 165, 255),
            color_rgb=(255, 165, 0),
            radius=3,
            description="Nose bridge and tip (6 landmarks)",
            significance="Nose tip = 3D anchor for solvePnP head pose estimation",
        ),
        LandmarkGroup(
            name="Nose Bottom",
            indices=[94, 2, 164, 0, 13, 14, 17, 84, 312, 378],
            color_bgr=(255, 165, 0),
            color_rgb=(0, 165, 255),
            radius=2,
            description="Nose bottom contour (10 landmarks)",
            significance="Nose width helps estimate distance to camera",
        ),
        LandmarkGroup(
            name="Lips Outer",
            indices=[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                     409, 270, 269, 267, 0, 37, 39, 40, 185],
            color_bgr=(0, 255, 128),
            color_rgb=(128, 255, 0),
            radius=2,
            description="Outer lip contour (20 landmarks)",
            significance="Mouth aspect ratio (MAR) = mouth-open click detection",
        ),
        LandmarkGroup(
            name="Lips Inner",
            indices=[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                     324, 318, 402, 317, 14, 87, 178, 88, 95],
            color_bgr=(255, 128, 0),
            color_rgb=(0, 128, 255),
            radius=2,
            description="Inner lip contour (20 landmarks)",
            significance="Inner lip distance = precise mouth-open measurement",
        ),
        LandmarkGroup(
            name="Left Pupil",
            indices=[468],
            color_bgr=(255, 255, 255),
            color_rgb=(255, 255, 255),
            radius=6,
            description="Left pupil center",
            significance="PRIMARY: pupil position = eye direction",
        ),
        LandmarkGroup(
            name="Right Pupil",
            indices=[473],
            color_bgr=(255, 255, 255),
            color_rgb=(255, 255, 255),
            radius=6,
            description="Right pupil center",
            significance="PRIMARY: pupil = where you are looking",
        ),
        LandmarkGroup(
            name="Chin/Jaw",
            indices=[152, 377, 400, 152, 148, 176, 149, 150, 136, 172,
                     58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            color_bgr=(180, 180, 0),
            color_rgb=(0, 180, 180),
            radius=1,
            description="Chin and jawline (21 landmarks)",
            significance="Jaw shape = head pitch estimation",
        ),
    ]
    return groups


LANDMARK_GROUPS = _build_landmark_groups()


def draw_landmarks_on_frame(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    show_labels: bool = True,
    show_connections: bool = True,
    frame_width: int = 640,
    frame_height: int = 480,
) -> np.ndarray:
    """Draw all MediaPipe landmarks on a BGR frame with RGB colors and labels."""
    annotated = frame_bgr.copy()
    h, w = frame_height, frame_width

    for group in LANDMARK_GROUPS:
        pts = []
        for idx in group.indices:
            if idx < len(landmarks):
                lx = int(landmarks[idx, 0] * w)
                ly = int(landmarks[idx, 1] * h)
                lx = max(0, min(w - 1, lx))
                ly = max(0, min(h - 1, ly))
                pts.append((lx, ly))
                cv2.circle(annotated, (lx, ly), group.radius, group.color_bgr, -1)

        # Draw connections within group
        if show_connections and len(pts) > 1:
            cv2.polylines(annotated, [np.array(pts, dtype=np.int32)],
                          isClosed=False, color=group.color_bgr, thickness=1)

        # Draw label near the group's center
        if show_labels and pts:
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
            label_y = cy - group.radius - 8

            font_scale = 0.35
            if "Iris" in group.name or "Pupil" in group.name:
                font_scale = 0.4

            cv2.putText(annotated, group.name, (cx, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        group.color_bgr, 1, cv2.LINE_AA)

    return annotated


def draw_legend(annotated: np.ndarray) -> np.ndarray:
    """Draw a compact legend in the bottom-right corner."""
    h, w = annotated.shape[:2]

    legend_w = 195
    legend_h = 17 * len(LANDMARK_GROUPS) + 10
    legend_x = w - legend_w - 4
    legend_y = h - legend_h - 4

    # Semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (legend_x - 2, legend_y - 2),
                  (legend_x + legend_w + 2, legend_y + legend_h + 2),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
    cv2.rectangle(annotated, (legend_x - 2, legend_y - 2),
                  (legend_x + legend_w + 2, legend_y + legend_h + 2),
                  (80, 80, 80), 1)

    y_off = legend_y + 11
    for group in LANDMARK_GROUPS:
        cv2.circle(annotated, (legend_x + 8, y_off - 3), 3, group.color_bgr, -1)
        cv2.putText(annotated, f"{group.name}: {group.significance[:28]}",
                    (legend_x + 16, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (200, 200, 200), 1, cv2.LINE_AA)
        y_off += 17

    return annotated


def draw_info_panel(annotated: np.ndarray, info_text: list[str]) -> np.ndarray:
    """Draw an info panel at the top-left with status text."""
    y = 14
    for line in info_text:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(annotated, (4, y - 12), (tw + 14, y + 3), (0, 0, 0), -1)
        cv2.putText(annotated, line, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 255, 0), 1, cv2.LINE_AA)
        y += 17
    return annotated


class CameraPreviewWidget(QWidget):
    """Live camera preview with MediaPipe landmark visualization.

    Shows a small camera window during calibration with:
    - Live video feed from the webcam
    - All 478 MediaPipe face landmarks drawn in distinct RGB colors
    - Group labels showing what each landmark set represents
    - A legend explaining all landmark groups and their significance
    - Status info panel showing face detection state
    """

    def __init__(self, width: int = 420, height: int = 316, parent=None):
        super().__init__(parent)
        self.preview_width = width
        self.preview_height = height
        self.show_legend = True
        self.show_labels = True
        self.show_connections = True

        self._landmarks: np.ndarray | None = None
        self._raw_frame_bgr: np.ndarray | None = None
        self._info_lines: list[str] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Gazer - Camera Preview (MediaPipe Landmarks)")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setFixedSize(self.preview_width, self.preview_height)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        self._label = QLabel()
        self._label.setFixedSize(self.preview_width - 4, self.preview_height - 4)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

    def set_frame_and_landmarks(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray | None = None,
        info_lines: list[str] | None = None,
    ) -> None:
        """Update the preview with a new frame and optional landmarks."""
        self._raw_frame_bgr = frame_bgr
        self._landmarks = landmarks
        if info_lines is not None:
            self._info_lines = info_lines

    def refresh(self) -> None:
        """Render the current frame + landmarks to the QLabel.
        Call this from the main thread (e.g., from a QTimer).
        """
        if self._raw_frame_bgr is None:
            return

        frame = self._raw_frame_bgr.copy()
        fh, fw = frame.shape[:2]

        # Draw landmarks if available
        if self._landmarks is not None and len(self._landmarks) >= 478:
            frame = draw_landmarks_on_frame(
                frame, self._landmarks,
                show_labels=self.show_labels,
                show_connections=self.show_connections,
                frame_width=fw, frame_height=fh,
            )
            if self.show_legend:
                frame = draw_legend(frame)

        # Draw info panel
        if self._info_lines:
            frame = draw_info_panel(frame, self._info_lines)

        # Resize to fit preview widget
        frame = cv2.resize(frame, (self.preview_width - 4, self.preview_height - 4))

        # Convert BGR -> RGB for Qt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._label.setPixmap(pixmap)

    def update_info(self, info_lines: list[str]) -> None:
        """Update just the info text lines without changing the frame."""
        self._info_lines = info_lines