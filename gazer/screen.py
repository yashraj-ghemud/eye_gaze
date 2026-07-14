"""Screen geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

from screeninfo import get_monitors


@dataclass(frozen=True)
class ScreenBounds:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def norm_to_screen(self, nx: float, ny: float) -> tuple[float, float]:
        """Map normalized [0,1] coords to screen pixels."""
        sx = self.x + max(0.0, min(1.0, nx)) * self.width
        sy = self.y + max(0.0, min(1.0, ny)) * self.height
        return sx, sy

    def screen_to_norm(self, sx: float, sy: float) -> tuple[float, float]:
        nx = (sx - self.x) / self.width
        ny = (sy - self.y) / self.height
        return nx, ny

    def clamp(self, sx: float, sy: float) -> tuple[float, float]:
        return (
            max(self.x, min(self.right - 1, sx)),
            max(self.y, min(self.bottom - 1, sy)),
        )


def get_primary_screen() -> ScreenBounds:
    monitors = get_monitors()
    primary = next((m for m in monitors if m.is_primary), monitors[0])
    return ScreenBounds(primary.x, primary.y, primary.width, primary.height)


def get_virtual_desktop() -> ScreenBounds:
    monitors = get_monitors()
    if not monitors:
        return ScreenBounds(0, 0, 1920, 1080)
    min_x = min(m.x for m in monitors)
    min_y = min(m.y for m in monitors)
    max_x = max(m.x + m.width for m in monitors)
    max_y = max(m.y + m.height for m in monitors)
    return ScreenBounds(min_x, min_y, max_x - min_x, max_y - min_y)
