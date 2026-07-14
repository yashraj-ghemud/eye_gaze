"""Mouth-open detection for click triggering via MediaPipe face landmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class MouthOpenConfig:
    # MAR threshold: mouth is "open" when MAR exceeds this
    mar_threshold: float = 0.4
    # Minimum duration the mouth must stay open (seconds)
    min_open_s: float = 0.15
    # Cooldown between consecutive clicks (seconds)
    cooldown_s: float = 0.5


@dataclass
class MouthOpenDetector:
    """Detects deliberate mouth-open gestures and signals a click.

    Uses Mouth Aspect Ratio (MAR) computed from MediaPipe landmarks:
      MAR = |upper_lip - lower_lip| / |left_corner - right_corner|

    A click fires when the mouth opens past *mar_threshold* for at least
    *min_open_s* seconds, then closes again.
    """

    config: MouthOpenConfig = field(default_factory=MouthOpenConfig)
    _open_since: float | None = None
    _last_click: float = 0.0
    _mar_smoothed: float = 0.0

    # Exponential moving average factor for smoothing raw MAR
    _ema_alpha: float = 0.3

    def update(self, mar: float, now: float | None = None) -> bool:
        """Feed a new MAR value. Returns True if a click should be dispatched."""
        if now is None:
            now = time.perf_counter()

        # Smooth the raw MAR to avoid jitter
        self._mar_smoothed = (
            self._ema_alpha * mar + (1 - self._ema_alpha) * self._mar_smoothed
            if self._mar_smoothed > 0
            else mar
        )

        mar_val = self._mar_smoothed

        if mar_val >= self.config.mar_threshold:
            if self._open_since is None:
                self._open_since = now
        else:
            if self._open_since is not None:
                open_duration = now - self._open_since
                self._open_since = None
                if (
                    open_duration >= self.config.min_open_s
                    and (now - self._last_click) >= self.config.cooldown_s
                ):
                    self._last_click = now
                    return True
        return False

    @property
    def current_mar(self) -> float:
        return self._mar_smoothed

    def reset(self) -> None:
        self._open_since = None
        self._last_click = 0.0
        self._mar_smoothed = 0.0