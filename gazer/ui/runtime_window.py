"""Runtime control window while gaze cursor is active — Phase 3 with Magic Cursor info."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gazer.profile_manager import ProfileMeta
from gazer.worker import GazeWorker


class RuntimeControlWindow(QWidget):
    quit_requested = pyqtSignal()

    def __init__(self, meta: ProfileMeta, worker: GazeWorker | None = None, parent=None):
        super().__init__(parent)
        self.worker = worker
        self._cursor_ref = None
        self._cursor_paused = False

        self.setWindowTitle("Gazer - Active (Magic Cursor)")
        self.setMinimumWidth(380)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        err = f"{meta.last_avg_error_px:.1f} px" if meta.last_avg_error_px else "N/A"
        layout.addWidget(
            QLabel(
                f"<b>Eye gaze cursor control active</b><br><br>"
                f"Profile: <b>{meta.name}</b><br>"
                f"Sessions: {meta.sessions} | Samples: {meta.total_samples}<br>"
                f"Last avg error: {err}<br><br>"
                f"<b style='color:#00cc66;'>Magic Cursor ON</b> — cursor warps toward gaze<br>"
                f"<b style='color:#ff9900;'>Open mouth = left click</b><br>"
                f"Keep your face visible to the webcam."
            )
        )

        # Magic Cursor status indicator
        self._status_label = QLabel("Cursor: Active | Speed: 0 px/s")
        self._status_label.setStyleSheet("color: #00cc66; font-weight: bold;")
        layout.addWidget(self._status_label)

        # Phase 3: Camera preview toggle
        preview_row = QHBoxLayout()
        self._preview_checkbox = QCheckBox("Show Camera Preview (runtime)")
        self._preview_checkbox.stateChanged.connect(self._toggle_runtime_preview)
        preview_row.addWidget(self._preview_checkbox)
        layout.addLayout(preview_row)

        # Attraction strength slider
        attr_row = QHBoxLayout()
        attr_row.addWidget(QLabel("Cursor Speed:"))
        self._attraction_slider = QSlider(Qt.Orientation.Horizontal)
        self._attraction_slider.setRange(2, 20)
        self._attraction_slider.setValue(8)
        self._attraction_slider.valueChanged.connect(self._update_attraction)
        attr_row.addWidget(self._attraction_slider)
        self._attraction_label = QLabel("8")
        attr_row.addWidget(self._attraction_label)
        layout.addLayout(attr_row)

        # Dead zone slider
        dz_row = QHBoxLayout()
        dz_row.addWidget(QLabel("Dead Zone:"))
        self._dz_slider = QSlider(Qt.Orientation.Horizontal)
        self._dz_slider.setRange(5, 80)
        self._dz_slider.setValue(25)
        self._dz_slider.valueChanged.connect(self._update_dead_zone)
        dz_row.addWidget(self._dz_slider)
        self._dz_label = QLabel("25 px")
        dz_row.addWidget(self._dz_label)
        layout.addLayout(dz_row)

        self._pause_btn = QPushButton("Pause cursor")
        self._pause_btn.clicked.connect(self._toggle_cursor)
        layout.addWidget(self._pause_btn)

        quit_btn = QPushButton("Quit Gazer")
        quit_btn.clicked.connect(self.quit_requested.emit)
        layout.addWidget(quit_btn)

        # Status update timer (10 Hz)
        self._status_timer = QTimer()
        self._status_timer.setInterval(100)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start()

        # Runtime preview widget (hidden by default)
        self._runtime_preview = None

    def _toggle_runtime_preview(self, state) -> None:
        """Show/hide camera preview during runtime."""
        if state == Qt.CheckState.Checked.value:
            if self._runtime_preview is None:
                from gazer.ui.camera_preview import CameraPreviewWidget
                self._runtime_preview = CameraPreviewWidget(width=320, height=240)
                self._runtime_preview.show_legend = False
            self._runtime_preview.show()
        else:
            if self._runtime_preview is not None:
                self._runtime_preview.hide()

    def _update_attraction(self, value: int) -> None:
        """Update magic cursor attraction strength."""
        self._attraction_label.setText(str(value))
        if self._cursor_ref is not None and hasattr(self._cursor_ref, '_magic'):
            self._cursor_ref._magic.config.attraction = float(value)

    def _update_dead_zone(self, value: int) -> None:
        """Update magic cursor dead zone."""
        self._dz_label.setText(f"{value} px")
        if self._cursor_ref is not None and hasattr(self._cursor_ref, '_magic'):
            self._cursor_ref._magic.config.dead_zone_px = float(value)

    def _update_status(self) -> None:
        """Update cursor status display."""
        if self._cursor_ref is None:
            return

        # Get magic cursor state
        if hasattr(self._cursor_ref, '_magic'):
            mc = self._cursor_ref._magic
            speed = mc.speed
            settled = mc.is_settled

            if self._cursor_paused:
                status = "PAUSED"
                color = "#ff4444"
            elif settled:
                status = "Settled"
                color = "#00cc66"
            else:
                status = "Moving"
                color = "#ffaa00"

            self._status_label.setText(
                f"Cursor: <span style='color:{color}'>{status}</span> | "
                f"Speed: {speed:.0f} px/s"
            )

            # Update runtime preview if visible
            if self._runtime_preview is not None and self._runtime_preview.isVisible():
                if self.worker is not None:
                    result = self.worker.get_latest()
                    if result is not None:
                        info = [
                            f"Speed: {speed:.0f} px/s",
                            f"State: {status}",
                        ]
                        if result.features is not None:
                            info.append(f"MAR: {result.features.mar:.2f}")
                        self._runtime_preview.set_frame_and_landmarks(
                            result.raw_frame_bgr,
                            result.raw_landmarks,
                            info,
                        )
                        self._runtime_preview.refresh()

    def _toggle_cursor(self) -> None:
        self._cursor_paused = not self._cursor_paused
        self._pause_btn.setText("Resume cursor" if self._cursor_paused else "Pause cursor")
        if self._cursor_ref is not None:
            self._cursor_ref.enabled = not self._cursor_paused

    def bind_cursor(self, cursor_controller) -> None:
        self._cursor_ref = cursor_controller

    def closeEvent(self, event) -> None:
        """Clean up on close."""
        self._status_timer.stop()
        if self._runtime_preview is not None:
            self._runtime_preview.close()
        super().closeEvent(event)