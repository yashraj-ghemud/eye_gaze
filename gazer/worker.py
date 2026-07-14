"""Gaze tracking worker thread — Phase 3 with Magic Cursor + frame export."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from gazer.camera import Camera
from gazer.face_tracker import FaceTracker, GazeFeatures
from gazer.gaze_model import GazeModel
from gazer.mouth_open_detector import MouthOpenDetector
from gazer.one_euro_filter import OneEuroFilter2D
from gazer.screen import ScreenBounds

logger = logging.getLogger(__name__)


@dataclass
class GazeResult:
    screen_x: float
    screen_y: float
    pred_nx: float
    pred_ny: float
    features: GazeFeatures | None
    valid: bool
    # Phase 3: raw frame and landmarks for camera preview
    raw_frame_bgr: np.ndarray | None = None
    raw_landmarks: np.ndarray | None = None


class GazeWorker:
    def __init__(
        self,
        screen: ScreenBounds,
        model: GazeModel,
        blink_detector=None,  # kept for API compat; unused
        dwell_clicker=None,
        enable_cursor: bool = False,
        enable_blink: bool = False,
        enable_dwell: bool = False,
    ):
        self.screen = screen
        self.model = model
        self.enable_cursor = enable_cursor
        self.enable_blink = False  # blink-click is DISABLED — mouth-open replaces it
        self.enable_dwell = enable_dwell

        self._camera = Camera()
        self._tracker = FaceTracker()
        self._filter = OneEuroFilter2D(min_cutoff=1.2, beta=0.008)

        # Mouth-open click detector
        self.mouth_detector = MouthOpenDetector()

        # Clicks are flagged here (worker thread) and executed on main thread
        self.pending_click: bool = False

        self._queue: queue.Queue[GazeResult | None] = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._latest: GazeResult | None = None

        # Error recovery state
        self._consecutive_errors: int = 0
        self._max_consecutive_errors: int = 50
        self._error_pause_s: float = 0.5

    def start(self) -> bool:
        if not self._camera.open():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._camera.release()
        self._tracker.close()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def get_latest(self) -> GazeResult | None:
        while True:
            try:
                item = self._queue.get_nowait()
                if item is None:
                    continue
                self._latest = item
            except queue.Empty:
                break
        return self._latest

    def _run(self) -> None:
        """Main worker loop with error recovery."""
        while not self._stop.is_set():
            try:
                ok, frame = self._camera.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                features = self._tracker.extract(rgb, w, h)

                # Get raw landmarks from the tracker for camera preview
                raw_lm = None
                if hasattr(self._tracker, '_last_raw_landmarks'):
                    raw_lm = self._tracker._last_raw_landmarks

                if features is None or not self.model.is_trained:
                    result = GazeResult(
                        0, 0, 0.5, 0.5, features, False,
                        raw_frame_bgr=frame.copy(),
                        raw_landmarks=raw_lm.copy() if raw_lm is not None else None,
                    )
                else:
                    pred = self.model.predict(features.vector)
                    if pred is None:
                        result = GazeResult(
                            0, 0, 0.5, 0.5, features, False,
                            raw_frame_bgr=frame.copy(),
                            raw_landmarks=raw_lm.copy() if raw_lm is not None else None,
                        )
                    else:
                        # Phase 3: Magic Cursor handles smoothing, but we still
                        # apply OneEuro filter for the prediction pipeline
                        # (MagicCursor runs on top in cursor_controller.py)
                        nx, ny = pred
                        sx, sy = self.screen.norm_to_screen(nx, ny)
                        sx, sy = self._filter.filter(sx, sy)
                        sx, sy = self.screen.clamp(sx, sy)

                        # Mouth-open click detection (replaces blink)
                        if features is not None:
                            if self.mouth_detector.update(features.mar):
                                self.pending_click = True  # flag for main thread

                        result = GazeResult(
                            sx, sy, nx, ny, features, True,
                            raw_frame_bgr=frame.copy(),
                            raw_landmarks=raw_lm.copy() if raw_lm is not None else None,
                        )

                try:
                    self._queue.put_nowait(result)
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(result)
                    except queue.Full:
                        pass

                # Reset error counter on successful frame
                self._consecutive_errors = 0

            except cv2.error as e:
                self._consecutive_errors += 1
                logger.warning("OpenCV error in worker (count=%d): %s", self._consecutive_errors, e)
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error("Too many consecutive CV errors — pausing then retrying")
                    time.sleep(self._error_pause_s)
                    self._consecutive_errors = 0
                else:
                    time.sleep(0.01)

            except Exception as e:
                self._consecutive_errors += 1
                logger.error("Unexpected error in worker (count=%d): %s", self._consecutive_errors, e, exc_info=True)
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.error("Worker in error state — sleeping before retry")
                    time.sleep(self._error_pause_s)
                    self._consecutive_errors = 0
                else:
                    time.sleep(0.01)

        logger.info("Worker loop ended")

    def reset_filter(self) -> None:
        self._filter.reset()
        self.mouth_detector.reset()