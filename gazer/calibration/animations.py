"""Easing and path animation utilities for calibration targets."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from gazer.screen import ScreenBounds


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def ease_in_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - pow(1 - t, 3)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp2(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return lerp(a[0], b[0], t), lerp(a[1], b[1], t)


def norm_point(screen: ScreenBounds, nx: float, ny: float) -> tuple[float, float]:
    return screen.norm_to_screen(nx, ny)


def perimeter_points(margin: float = 0.05) -> list[tuple[float, float]]:
    m = margin
    c = 0.5
    return [
        (m, m), (c, m), (1 - m, m),
        (1 - m, c), (1 - m, 1 - m),
        (c, 1 - m), (m, 1 - m),
        (m, c), (c, c), (m, m),
    ]


def corner_points(margin: float = 0.05) -> list[tuple[float, float]]:
    m = margin
    c = 0.5
    return [(m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m), (c, c)]


def grid_points(rows: int = 5, cols: int = 5, margin: float = 0.15) -> list[tuple[float, float]]:
    pts = []
    for r in range(rows):
        for c in range(cols):
            nx = margin + (1 - 2 * margin) * c / max(cols - 1, 1)
            ny = margin + (1 - 2 * margin) * r / max(rows - 1, 1)
            pts.append((nx, ny))
    return pts


def lissajous_points(n: int = 40, a: float = 3.0, b: float = 2.0, margin: float = 0.08) -> list[tuple[float, float]]:
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 0.5 + (0.5 - margin) * math.sin(a * t + math.pi / 2)
        y = 0.5 + (0.5 - margin) * math.sin(b * t)
        pts.append((x, y))
    return pts


def circle_points(n: int = 40, margin: float = 0.08) -> list[tuple[float, float]]:
    pts = []
    r = 0.5 - margin
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 0.5 + r * math.cos(t)
        y = 0.5 + r * math.sin(t)
        pts.append((x, y))
    return pts


def edge_trace_points(margin: float = 0.05) -> list[tuple[float, float]]:
    m = margin
    return [
        (m, m), (1 - m, m), (1 - m, 1 - m), (m, 1 - m), (m, m),
    ]


def micro_cluster_centers() -> list[tuple[float, float]]:
    return [
        (0.5, 0.5),
        (0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8),
        (0.5, 0.15),
    ]


def micro_cluster_grid(center: tuple[float, float], spread: float = 0.02) -> list[tuple[float, float]]:
    cx, cy = center
    pts = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            pts.append((cx + dc * spread, cy + dr * spread))
    return pts


def zigzag_points(n: int = 12, margin: float = 0.08) -> list[tuple[float, float]]:
    pts = []
    for i in range(n):
        t = i / max(n - 1, 1)
        x = margin + (1 - 2 * margin) * t
        y = margin if i % 2 == 0 else 1 - margin
        pts.append((x, y))
    return pts


@dataclass
class Segment:
    start: tuple[float, float]
    end: tuple[float, float]
    duration_ms: float
    pause_ms: float = 0.0
    easing: Callable[[float], float] = ease_in_out_cubic


@dataclass
class PathAnimation:
    segments: list[Segment]
    loop: bool = False

    def total_duration_ms(self) -> float:
        return sum(s.duration_ms + s.pause_ms for s in self.segments)

    def position_at(self, elapsed_ms: float) -> tuple[float, float, bool]:
        """Returns (nx, ny, active). active=False during pauses."""
        if not self.segments:
            return 0.5, 0.5, False

        total = self.total_duration_ms()
        if self.loop and total > 0:
            elapsed_ms = elapsed_ms % total
        elif elapsed_ms >= total:
            last = self.segments[-1]
            return last.end[0], last.end[1], True

        t_acc = 0.0
        for seg in self.segments:
            if elapsed_ms < t_acc + seg.duration_ms:
                local = (elapsed_ms - t_acc) / max(seg.duration_ms, 1)
                eased = seg.easing(local)
                x, y = lerp2(seg.start, seg.end, eased)
                return x, y, True
            t_acc += seg.duration_ms
            if elapsed_ms < t_acc + seg.pause_ms:
                return seg.end[0], seg.end[1], False
            t_acc += seg.pause_ms

        last = self.segments[-1]
        return last.end[0], last.end[1], True


def build_segments_from_points(
    points: list[tuple[float, float]],
    move_ms: float,
    pause_ms: float = 0.0,
    easing: Callable[[float], float] = ease_in_out_cubic,
) -> list[Segment]:
    segs = []
    for i in range(1, len(points)):
        segs.append(Segment(points[i - 1], points[i], move_ms, pause_ms, easing))
    return segs


def figure8_verification_points(n: int = 200, margin: float = 0.1) -> list[tuple[float, float]]:
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 0.5 + (0.5 - margin) * (0.6 * math.sin(3.5 * t) + 0.3 * math.sin(2.5 * t))
        y = 0.5 + (0.5 - margin) * (0.6 * math.sin(2.5 * t + 1.2) + 0.3 * math.cos(3.5 * t))
        pts.append((x, y))
    return pts


def random_jump_points(n: int = 20, margin: float = 0.08, seed: int = 42) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    pts = [(0.5, 0.5)]
    for _ in range(n - 1):
        pts.append((rng.uniform(margin, 1 - margin), rng.uniform(margin, 1 - margin)))
    return pts
