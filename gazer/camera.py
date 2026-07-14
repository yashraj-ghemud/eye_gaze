"""Camera capture with platform-appropriate backend."""

from __future__ import annotations

import platform
import sys

import cv2


class Camera:
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None

    def _backend(self) -> int:
        if sys.platform == "win32":
            return cv2.CAP_DSHOW
        if platform.system() == "Linux":
            return cv2.CAP_V4L2
        return cv2.CAP_ANY

    def open(self) -> bool:
        indices = [self.device_index] + [i for i in range(4) if i != self.device_index]
        for idx in indices:
            cap = cv2.VideoCapture(idx, self._backend())
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # FIX: Set resolution BEFORE warmup so warmup grabs use correct size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Warm up — some drivers need a few grabs before first frame
            ok = False
            for _ in range(8):
                if cap.grab():
                    ok = True
                    break
            if ok:
                self._cap = cap
                self.device_index = idx
                break
            cap.release()

        if self._cap is None:
            return False

        return True

    def read(self):
        if self._cap is None:
            return False, None
        # Grab latest frame; discard 1 buffered frame to reduce latency
        self._cap.grab()
        return self._cap.retrieve()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()