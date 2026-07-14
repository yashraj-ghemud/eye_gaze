"""Frameless transparent calibration overlay."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from gazer.calibration.session import CalibrationState, VerificationState
from gazer.screen import ScreenBounds


class CalibrationOverlay(QWidget):
    def __init__(self, screen: ScreenBounds):
        super().__init__()
        self.screen = screen
        self.cal_state: CalibrationState | None = None
        self.verify_state: VerificationState | None = None
        self.show_hud = True
        self._setup_window()

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setGeometry(self.screen.x, self.screen.y, self.screen.width, self.screen.height)

    def set_calibration_state(self, state: CalibrationState) -> None:
        self.cal_state = state
        self.verify_state = None
        self.update()

    def set_verification_state(self, state: VerificationState) -> None:
        self.verify_state = state
        self.cal_state = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.cal_state is not None:
            self._paint_calibration(painter, self.cal_state)
        elif self.verify_state is not None:
            self._paint_verification(painter, self.verify_state)

    def _to_local(self, nx: float, ny: float) -> tuple[float, float]:
        x = nx * self.screen.width
        y = ny * self.screen.height
        return x, y

    def _paint_calibration(self, painter: QPainter, state: CalibrationState) -> None:
        if state.phase_name != "Blink/EAR Baseline":
            tx, ty = self._to_local(state.target_nx, state.target_ny)
            painter.setBrush(QColor(255, 255, 255, 240))
            painter.setPen(QPen(QColor(0, 0, 0, 80), 1))
            painter.drawEllipse(int(tx - 14), int(ty - 14), 28, 28)

            if state.model_trained:
                px, py = self._to_local(state.pred_nx, state.pred_ny)
                painter.setBrush(QColor(0, 220, 80, 210))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(px - 8), int(py - 8), 16, 16)

        if self.show_hud:
            self._paint_hud(
                painter,
                phase=state.phase_name,
                samples=len(state.samples),
                avg_err=state.running_avg_error_px,
                remaining_s=state.time_remaining_ms / 1000,
                prompt=state.prompt,
                extra=state.status_line,
            )

    def _paint_verification(self, painter: QPainter, state: VerificationState) -> None:
        tx, ty = self._to_local(state.target_nx, state.target_ny)
        painter.setBrush(QColor(255, 255, 255, 240))
        painter.setPen(QPen(QColor(0, 0, 0, 80), 1))
        painter.drawEllipse(int(tx - 12), int(ty - 12), 24, 24)

        px, py = self._to_local(state.pred_nx, state.pred_ny)
        painter.setBrush(QColor(0, 220, 80, 210))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(px - 8), int(py - 8), 16, 16)

        if self.show_hud:
            remaining = max(0, (state.duration_ms - state.elapsed_ms) / 1000)
            live_avg = sum(state.errors_px) / len(state.errors_px) if state.errors_px else 0
            self._paint_hud(
                painter,
                phase="Verification",
                samples=len(state.errors_px),
                avg_err=live_avg,
                remaining_s=remaining,
                prompt="Follow the white dot with your eyes",
                extra="",
            )

    def _paint_hud(
        self,
        painter: QPainter,
        phase: str,
        samples: int,
        avg_err: float,
        remaining_s: float,
        prompt: str,
        extra: str = "",
    ) -> None:
        lines = [
            f"Phase: {phase}",
            f"Samples: {samples}",
            f"Avg Error: {avg_err:.1f} px",
            f"Time left: {remaining_s:.0f}s",
        ]
        if extra:
            lines.append(extra)
        if prompt:
            lines.append(prompt)

        box_h = 16 + len(lines) * 22
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.drawRoundedRect(10, 10, 340, box_h, 8, 8)

        painter.setPen(QPen(QColor(255, 255, 255, 230)))
        font = QFont("Segoe UI", 11)
        painter.setFont(font)

        y = 30
        for line in lines:
            painter.drawText(20, y, line)
            y += 22
