"""Animated Magic Cursor — warps toward gaze point with physics-based momentum.

Solves the "Midas touch" problem: the cursor doesn't teleport directly to where
the eyes look (which would click everything in the path). Instead, it smoothly
warps toward the gaze point with configurable attraction force, velocity damping,
and a dead-zone where the cursor holds still.

The feel is similar to a magnetic attraction — natural and intuitive.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from gazer.screen import ScreenBounds


@dataclass
class MagicCursorConfig:
    # Attraction strength: how fast the cursor accelerates toward the gaze point
    # Higher = more responsive but more "Midas touch" prone
    attraction: float = 8.0

    # Damping factor: velocity multiplied by (1 - damping * dt) each frame
    # Higher = more sluggish, Lower = more momentum/overshoot
    damping: float = 4.5

    # Dead-zone radius in pixels: cursor won't move if gaze is within this radius
    # Prevents jitter when user is looking at the cursor's current position
    dead_zone_px: float = 25.0

    # Max velocity in pixels/second — prevents cursor from flying across screen
    max_speed: float = 3500.0

    # Smooth-start: minimum frames before full attraction kicks in
    # Prevents initial jump when cursor control starts
    warmup_frames: int = 5

    # Velocity threshold to consider cursor "settled" (for visual effects)
    settled_threshold: float = 15.0

    # Gaze confidence threshold: if gaze is unreliable, reduce attraction
    # This is used externally; the cursor itself doesn't compute confidence
    # low_confidence_factor: float = 0.3  # used via set_confidence()


@dataclass
class MagicCursorState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    frame_count: int = 0
    last_time: float = 0.0
    speed: float = 0.0  # current speed in px/s (for visual effects)
    is_settled: bool = True  # True when cursor is nearly stationary


class MagicCursor:
    """Physics-based cursor that warps toward the gaze point.

    Instead of setting cursor position directly to gaze prediction (which causes
    the "Midas touch" problem where the cursor clicks everything in its path),
    this cursor uses a spring-damper system:

    1. The gaze point acts as an **attractor** (like a magnet)
    2. The cursor has **velocity and momentum** — it accelerates toward the gaze
    3. A **dead zone** prevents micro-jitter when gaze is near the cursor
    4. **Velocity damping** prevents oscillation and provides natural deceleration

    Usage:
        mc = MagicCursor(screen)
        mc.init_position(screen_center_x, screen_center_y)
        # Each frame:
        target_x, target_y = mc.update(gaze_screen_x, gaze_screen_y)
        mouse.position = (int(target_x), int(target_y))
    """

    def __init__(self, screen: ScreenBounds, config: MagicCursorConfig | None = None):
        self.screen = screen
        self.config = config or MagicCursorConfig()
        self.state = MagicCursorState()
        self._confidence: float = 1.0  # 0.0 to 1.0, externally settable

    def init_position(self, x: float, y: float) -> None:
        """Set initial cursor position. Call once when starting cursor control."""
        self.state.x = float(x)
        self.state.y = float(y)
        self.state.vx = 0.0
        self.state.vy = 0.0
        self.state.frame_count = 0
        self.state.last_time = time.perf_counter()
        self.state.speed = 0.0
        self.state.is_settled = True

    def set_confidence(self, confidence: float) -> None:
        """Set gaze confidence [0,1]. Low confidence reduces attraction force."""
        self._confidence = max(0.0, min(1.0, confidence))

    def update(self, gaze_x: float, gaze_y: float, dt: float | None = None) -> tuple[float, float]:
        """Update cursor position toward gaze point.

        Args:
            gaze_x: Target screen X coordinate (where eyes are looking)
            gaze_y: Target screen Y coordinate
            dt: Time delta in seconds. If None, auto-measured.

        Returns:
            (cursor_x, cursor_y) — where the cursor should be drawn
        """
        if dt is None:
            now = time.perf_counter()
            dt = now - self.state.last_time
            self.state.last_time = now

        dt = max(dt, 0.001)  # clamp to avoid division issues
        dt = min(dt, 0.1)    # clamp to avoid huge jumps after lag

        self.state.frame_count += 1

        # Warmup: gradually increase attraction over first few frames
        warmup = min(1.0, self.state.frame_count / max(self.config.warmup_frames, 1))

        # Compute direction and distance to gaze target
        dx = gaze_x - self.state.x
        dy = gaze_y - self.state.y
        dist = math.sqrt(dx * dx + dy * dy)

        # Dead zone: don't move if gaze is very close to cursor
        if dist < self.config.dead_zone_px:
            self.state.vx *= 0.5  # bleed off velocity quickly
            self.state.vy *= 0.5
            self._update_speed()
            return self._clamped_pos()

        # Normalize direction
        nx = dx / dist
        ny = dy / dist

        # Apply confidence scaling (low confidence = weaker attraction)
        effective_attraction = self.config.attraction * warmup * self._confidence

        # Spring-damper physics:
        # Force = attraction * direction
        # Velocity += force * dt
        # Velocity *= (1 - damping * dt)  [exponential damping]
        # Position += velocity * dt

        # Acceleration toward target
        ax = effective_attraction * nx * (dist / 200.0 + 0.5)  # stronger pull when farther
        ay = effective_attraction * ny * (dist / 200.0 + 0.5)

        # Apply acceleration
        self.state.vx += ax * dt
        self.state.vy += ay * dt

        # Exponential damping
        damp = max(0.0, 1.0 - self.config.damping * dt)
        self.state.vx *= damp
        self.state.vy *= damp

        # Clamp velocity to max speed
        speed = math.sqrt(self.state.vx ** 2 + self.state.vy ** 2)
        if speed > self.config.max_speed:
            scale = self.config.max_speed / speed
            self.state.vx *= scale
            self.state.vy *= scale

        # Update position
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt

        self._update_speed()
        return self._clamped_pos()

    def _update_speed(self) -> None:
        self.state.speed = math.sqrt(self.state.vx ** 2 + self.state.vy ** 2)
        self.state.is_settled = self.state.speed < self.config.settled_threshold

    def _clamped_pos(self) -> tuple[float, float]:
        cx = max(self.screen.x, min(self.screen.right - 1, self.state.x))
        cy = max(self.screen.y, min(self.screen.bottom - 1, self.state.y))
        return cx, cy

    def reset(self) -> None:
        """Reset cursor state. Call before starting a new session."""
        self.state = MagicCursorState()
        self._confidence = 1.0

    @property
    def speed(self) -> float:
        """Current cursor speed in pixels/second."""
        return self.state.speed

    @property
    def is_settled(self) -> bool:
        """Whether the cursor has come to rest."""
        return self.state.is_settled

    @property
    def position(self) -> tuple[float, float]:
        """Current cursor position."""
        return self.state.x, self.state.y

    def get_trail_alpha(self) -> float:
        """Get opacity for cursor trail effect (0.0 to 1.0).
        Higher when moving fast, fades when settled.
        """
        if self.state.is_settled:
            return 0.0
        return min(1.0, self.state.speed / 1000.0)