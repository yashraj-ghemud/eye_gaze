"""Calibration phase definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from gazer.calibration.animations import (
    PathAnimation,
    Segment,
    build_segments_from_points,
    circle_points,
    corner_points,
    ease_in_cubic,
    ease_in_out_cubic,
    ease_out_cubic,
    edge_trace_points,
    grid_points,
    lissajous_points,
    micro_cluster_centers,
    micro_cluster_grid,
    perimeter_points,
    random_jump_points,
    zigzag_points,
)


class PhaseId(IntEnum):
    SMOOTH_PERIMETER = 1
    EXTREME_CORNERS = 2
    SLOW_CENTER_PURSUIT = 3
    DENSE_INNER_MATRIX = 4
    RANDOM_ERRATIC = 5
    EXTENDED_LISSAJOUS = 6
    EDGE_TRACING = 7
    MICRO_SACCADE = 8
    CONTINUOUS_LINEAR_MEDIUM = 9
    CONTINUOUS_FAST = 10
    SLOW_TO_FAST = 11
    FAST_TO_SLOW = 12
    CONTINUOUS_CIRCULAR = 13
    HEAD_ROBUSTNESS = 14
    BLINK_EAR_BASELINE = 15


@dataclass
class PhaseSpec:
    phase_id: PhaseId
    name: str
    weight: float  # relative time share
    animation: PathAnimation | None
    prompt: str = ""
    collect_samples: bool = True
    move_cursor: bool = True
    is_fixed_target: bool = False
    is_ear_baseline: bool = False


def _perimeter_phase() -> PhaseSpec:
    pts = perimeter_points(0.05)
    segs = build_segments_from_points(pts, 1750, 1000, ease_in_out_cubic)
    return PhaseSpec(PhaseId.SMOOTH_PERIMETER, "Smooth Perimeter", 1.0, PathAnimation(segs))


def _corners_phase() -> PhaseSpec:
    pts = corner_points(0.05)
    segs = build_segments_from_points(pts, 200, 1500, ease_in_out_cubic)
    return PhaseSpec(PhaseId.EXTREME_CORNERS, "Extreme Corners", 0.8, PathAnimation(segs))


def _center_pursuit_phase() -> PhaseSpec:
    pts = [(0.2, 0.2), (0.8, 0.8), (0.8, 0.2), (0.2, 0.8), (0.5, 0.5)]
    segs = build_segments_from_points(pts, 4000, 500, ease_in_out_cubic)
    return PhaseSpec(PhaseId.SLOW_CENTER_PURSUIT, "Slow Center Pursuit", 0.9, PathAnimation(segs))


def _dense_matrix_phase() -> PhaseSpec:
    pts = grid_points(6, 6, 0.12)
    segs = build_segments_from_points(pts, 1200, 200, ease_in_out_cubic)
    return PhaseSpec(PhaseId.DENSE_INNER_MATRIX, "Dense Inner Matrix", 1.2, PathAnimation(segs))


def _random_phase() -> PhaseSpec:
    pts = random_jump_points(25)
    segs = build_segments_from_points(pts, 300, 1000, ease_in_out_cubic)
    return PhaseSpec(PhaseId.RANDOM_ERRATIC, "Random Erratic Jumps", 0.7, PathAnimation(segs))


def _lissajous_phase() -> PhaseSpec:
    pts = lissajous_points(40)
    segs = build_segments_from_points(pts, 800, 0, ease_in_out_cubic)
    return PhaseSpec(PhaseId.EXTENDED_LISSAJOUS, "Extended Lissajous", 1.0, PathAnimation(segs, loop=True))


def _edge_phase() -> PhaseSpec:
    pts = edge_trace_points(0.05)
    segs = build_segments_from_points(pts, 1500, 1000, ease_in_out_cubic)
    return PhaseSpec(PhaseId.EDGE_TRACING, "Edge Tracing", 0.8, PathAnimation(segs))


def _micro_saccade_phase() -> PhaseSpec:
    pts = []
    for center in micro_cluster_centers():
        pts.extend(micro_cluster_grid(center, 0.018))
    segs = build_segments_from_points(pts, 400, 150, ease_in_out_cubic)
    return PhaseSpec(PhaseId.MICRO_SACCADE, "Micro-Saccade Precision", 1.1, PathAnimation(segs))


def _linear_medium_phase() -> PhaseSpec:
    pts = perimeter_points(0.05)
    segs = build_segments_from_points(pts, 1500, 0, ease_in_out_cubic)
    return PhaseSpec(PhaseId.CONTINUOUS_LINEAR_MEDIUM, "Continuous Linear Medium", 0.7, PathAnimation(segs, loop=True))


def _fast_zigzag_phase() -> PhaseSpec:
    pts = zigzag_points(14)
    segs = build_segments_from_points(pts, 600, 0, ease_in_out_cubic)
    return PhaseSpec(PhaseId.CONTINUOUS_FAST, "Continuous Fast Zig-Zag", 0.6, PathAnimation(segs, loop=True))


def _slow_to_fast_phase() -> PhaseSpec:
    segs = [Segment((0.08, 0.5), (0.92, 0.5), 2000, 0, ease_in_cubic)]
    return PhaseSpec(PhaseId.SLOW_TO_FAST, "Slow-to-Fast Acceleration", 0.5, PathAnimation(segs, loop=True))


def _fast_to_slow_phase() -> PhaseSpec:
    segs = [Segment((0.5, 0.08), (0.5, 0.92), 2000, 0, ease_out_cubic)]
    return PhaseSpec(PhaseId.FAST_TO_SLOW, "Fast-to-Slow Deceleration", 0.5, PathAnimation(segs, loop=True))


def _circular_phase() -> PhaseSpec:
    pts = circle_points(40)
    segs = build_segments_from_points(pts, 200, 0, ease_in_out_cubic)
    return PhaseSpec(PhaseId.CONTINUOUS_CIRCULAR, "Continuous Circular Sweep", 0.6, PathAnimation(segs, loop=True))


def _head_robustness_phase() -> PhaseSpec:
    pts = [(0.5, 0.5), (0.08, 0.08), (0.92, 0.08), (0.92, 0.92), (0.08, 0.92)]
    segs = []
    for p in pts:
        segs.append(Segment(p, p, 4000, 0))  # hold still 4s each
    return PhaseSpec(
        PhaseId.HEAD_ROBUSTNESS,
        "Head-Robustness Fixed-Target",
        1.0,
        PathAnimation(segs),
        prompt="Keep eyes on the dot. Slowly turn head left/right, then up/down.",
        is_fixed_target=True,
    )


def _ear_baseline_phase() -> PhaseSpec:
    return PhaseSpec(
        PhaseId.BLINK_EAR_BASELINE,
        "Blink/EAR Baseline",
        0.4,
        None,
        prompt="Blink naturally for a few seconds, then open eyes wide.",
        collect_samples=False,
        move_cursor=False,
        is_ear_baseline=True,
    )


ALL_PHASES: list[PhaseSpec] = [
    _perimeter_phase(),
    _corners_phase(),
    _center_pursuit_phase(),
    _dense_matrix_phase(),
    _random_phase(),
    _lissajous_phase(),
    _edge_phase(),
    _micro_saccade_phase(),
    _linear_medium_phase(),
    _fast_zigzag_phase(),
    _slow_to_fast_phase(),
    _fast_to_slow_phase(),
    _circular_phase(),
    _head_robustness_phase(),
    _ear_baseline_phase(),
]


def phase_durations_ms(total_ms: float) -> dict[PhaseId, float]:
    """Distribute total calibration time across phases by weight."""
    total_weight = sum(p.weight for p in ALL_PHASES)
    return {p.phase_id: (p.weight / total_weight) * total_ms for p in ALL_PHASES}
