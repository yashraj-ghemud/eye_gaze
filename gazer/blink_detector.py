"""Personalized blink detection via EAR threshold."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pynput.mouse import Button, Controller


@dataclass
class BlinkConfig:
    closed_ratio: float = 0.65  # fraction of open baseline
    min_closed_ms: float = 80.0
    cooldown_ms: float = 300.0


@dataclass
class BlinkDetector:
    config: BlinkConfig = field(default_factory=BlinkConfig)
    ear_open_baseline: float | None = None
    ear_closed_baseline: float | None = None
    _closed_since: float | None = None
    _last_click: float = 0.0
    _mouse: Controller = field(default_factory=Controller)

    def set_baseline(self, open_ear: float, closed_ear: float | None = None) -> None:
        self.ear_open_baseline = open_ear
        self.ear_closed_baseline = closed_ear

    @property
    def threshold(self) -> float | None:
        if self.ear_open_baseline is None:
            return None
        if self.ear_closed_baseline is not None:
            return (self.ear_open_baseline + self.ear_closed_baseline) / 2.0
        return self.ear_open_baseline * self.config.closed_ratio

    def is_blink(self, ear: float) -> bool:
        th = self.threshold
        if th is None:
            return False
        return ear < th

    def update(self, ear: float, now: float | None = None) -> bool:
        """Returns True if a click was dispatched."""
        if now is None:
            now = time.perf_counter()

        th = self.threshold
        if th is None:
            return False

        if ear < th:
            if self._closed_since is None:
                self._closed_since = now
        else:
            if self._closed_since is not None:
                closed_ms = (now - self._closed_since) * 1000
                self._closed_since = None
                if closed_ms >= self.config.min_closed_ms:
                    if (now - self._last_click) * 1000 >= self.config.cooldown_ms:
                        self._mouse.click(Button.left, 1)
                        self._last_click = now
                        return True
        return False


@dataclass
class DwellClicker:
    radius_px: float = 15.0
    dwell_ms: float = 800.0
    _stable_since: float | None = None
    _last_pos: tuple[float, float] | None = None
    _last_click: float = 0.0
    _mouse: Controller = field(default_factory=Controller)

    def update(self, x: float, y: float, now: float | None = None) -> bool:
        if now is None:
            now = time.perf_counter()

        if self._last_pos is not None:
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= self.radius_px:
                if self._stable_since is None:
                    self._stable_since = now
                elif (now - self._stable_since) * 1000 >= self.dwell_ms:
                    if (now - self._last_click) * 1000 >= 1000:
                        self._mouse.click(Button.left, 1)
                        self._last_click = now
                        self._stable_since = None
                        return True
            else:
                self._stable_since = None

        self._last_pos = (x, y)
        return False
