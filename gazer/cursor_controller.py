"""Cursor control with Magic Cursor — physics-based warp toward gaze point.

Phase 3: Instead of teleporting the cursor directly to the gaze point (which
causes the "Midas touch" problem — clicking everything in the path), the cursor
now smoothly warps toward the gaze point with momentum, dead-zone, and damping.
This feels natural — like the cursor is magnetically attracted to where you look.
"""

from __future__ import annotations

import logging
import time

from pynput.mouse import Button, Controller

from gazer.magic_cursor import MagicCursor, MagicCursorConfig
from gazer.worker import GazeWorker

logger = logging.getLogger(__name__)


class CursorController:
    def __init__(self, worker: GazeWorker, screen=None, max_hz: float = 60.0):
        self.worker = worker
        self.min_interval = 1.0 / max_hz
        self._mouse = Controller()
        self._last_move = 0.0
        self.enabled = False

        # Phase 3: Magic Cursor replaces direct positioning
        from gazer.screen import ScreenBounds
        scr: ScreenBounds = screen if screen is not None else self.worker.screen
        config = MagicCursorConfig(
            attraction=8.0,
            damping=4.5,
            dead_zone_px=25.0,
            max_speed=3500.0,
            warmup_frames=5,
            settled_threshold=15.0,
        )
        self._magic = MagicCursor(scr, config)

        # Initialize cursor at screen center
        self._magic.init_position(scr.center_x, scr.center_y)

        # Track if magic cursor has been initialized with real position
        self._initialized = False

    def tick(self) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        if now - self._last_move < self.min_interval:
            return

        dt = now - self._last_move
        self._last_move = now

        result = self.worker.get_latest()
        if result is not None and result.valid:
            gaze_x = result.screen_x
            gaze_y = result.screen_y

            # Initialize magic cursor to current mouse position on first valid frame
            if not self._initialized:
                try:
                    mx, my = self._mouse.position
                    self._magic.init_position(float(mx), float(my))
                except Exception:
                    self._magic.init_position(gaze_x, gaze_y)
                self._initialized = True

            # Magic Cursor: warp toward gaze point with physics
            cursor_x, cursor_y = self._magic.update(gaze_x, gaze_y, dt)

            try:
                self._mouse.position = (int(cursor_x), int(cursor_y))
            except Exception as e:
                logger.warning("Failed to move cursor: %s", e)

        # Execute mouth-open click on main thread (set by worker)
        if self.worker.pending_click:
            self.worker.pending_click = False
            try:
                self._mouse.click(Button.left, 1)
            except Exception as e:
                logger.warning("Failed to execute mouth-open click: %s", e)

    def reset(self) -> None:
        """Reset magic cursor state. Call when starting a new session."""
        self._magic.reset()
        self._initialized = False