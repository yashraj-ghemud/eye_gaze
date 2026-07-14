"""Main application entry point — Phase 3 with Camera Preview + Magic Cursor."""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from gazer.calibration.session import CalibrationSession, VerificationSession
from gazer.cursor_controller import CursorController
from gazer.gaze_model import GazeModel
from gazer.models import ensure_face_landmarker
from gazer.mouth_open_detector import MouthOpenDetector
from gazer.profile_manager import ProfileManager
from gazer.screen import get_primary_screen
from gazer.training_thread import TrainingThread
from gazer.ui.camera_preview import CameraPreviewWidget
from gazer.ui.dialogs import ProfileDialog, ScorecardDialog, TrainingDurationDialog
from gazer.ui.overlay import CalibrationOverlay
from gazer.worker import GazeWorker

logger = logging.getLogger(__name__)


class GazerApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Gazer")
        self.screen = get_primary_screen()
        self.profiles = ProfileManager()
        self.profile_name = ""
        self.duration_sec = 180
        self.skip_calibration = False
        self.retrain_mode = False

        self.overlay: CalibrationOverlay | None = None
        self.worker: GazeWorker | None = None
        self.cursor: CursorController | None = None
        self.cal_session: CalibrationSession | None = None
        self.verify_session: VerificationSession | None = None
        self.runtime_window = None
        self._model = GazeModel()

        # Mouth-open detector (used during cursor control)
        self.mouth_detector = MouthOpenDetector()

        self._tick_timer = QTimer()
        self._tick_timer.setInterval(16)
        self._tick_timer.timeout.connect(self._on_tick)

        self._cursor_timer = QTimer()
        self._cursor_timer.setInterval(16)
        self._cursor_timer.timeout.connect(self._on_cursor_tick)

        # Background training thread management
        self._training_thread: TrainingThread | None = None
        self._training_busy: bool = False

        # Real-time dt measurement
        self._last_tick_time: float = 0.0

        # Phase 3: Camera preview widget for calibration
        self._camera_preview: CameraPreviewWidget | None = None

        # Phase 3: Preview refresh timer (runs at ~15fps to save CPU)
        self._preview_timer = QTimer()
        self._preview_timer.setInterval(66)  # ~15 fps
        self._preview_timer.timeout.connect(self._on_preview_tick)

    def run(self) -> int:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        try:
            ensure_face_landmarker()
        except Exception as exc:
            QMessageBox.critical(
                None,
                "Model Download Failed",
                f"Could not download MediaPipe face model.\n\n{exc}\n\nCheck your internet connection.",
            )
            return 1

        if not self._setup_profile():
            return 0

        self.worker = GazeWorker(self.screen, self._model)
        if not self.worker.start():
            QMessageBox.critical(
                None,
                "Camera Error",
                "Could not open webcam.\n\n"
                "- Check camera permissions (Settings > Privacy > Camera)\n"
                "- Close other apps using the camera\n"
                "- Try a different USB port",
            )
            return 1

        if self.skip_calibration:
            return self._start_from_saved_profile()

        dur_dialog = TrainingDurationDialog()
        if dur_dialog.exec() != dur_dialog.DialogCode.Accepted:
            self._shutdown()
            return 0
        self.duration_sec = dur_dialog.duration_sec()

        self.overlay = CalibrationOverlay(self.screen)
        self.overlay.show()

        # Phase 3: Create camera preview widget
        self._camera_preview = CameraPreviewWidget(width=420, height=316)
        self._camera_preview.show()

        self.cal_session = CalibrationSession(self.screen, self.duration_sec, self.profile_name)
        self._model = self.cal_session.model
        self.worker.model = self._model

        if self.retrain_mode:
            loaded = self.profiles.load_model(self.profile_name)
            if loaded.is_trained:
                self._model = loaded
                self.cal_session.model = self._model
                self.worker.model = self._model

        self._last_tick_time = time.perf_counter()
        self._tick_timer.start()
        self._preview_timer.start()  # Start preview refresh
        return self.app.exec()

    def _setup_profile(self) -> bool:
        names = self.profiles.list_profiles()
        dialog = ProfileDialog(names)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return False
        self.profile_name = dialog.selected_profile()
        self.skip_calibration = dialog.skip_calibration()
        self.retrain_mode = dialog.is_retrain()
        return True

    def _start_from_saved_profile(self) -> int:
        meta = self.profiles.load_meta(self.profile_name)
        model = self.profiles.load_model(self.profile_name)

        # Check feature dimension compatibility
        if model.is_trained and model.feature_dim != self._model.feature_dim:
            QMessageBox.warning(
                None,
                "Profile Incompatible",
                f"Profile '{self.profile_name}' was created with {model.feature_dim}-dim features.\n"
                f"Current version uses {self._model.feature_dim}-dim features.\n\n"
                "Please run a new calibration to create an updated profile.",
            )
            # Fall through to calibration
            model = GazeModel()

        if not model.is_trained:
            QMessageBox.warning(
                None,
                "No trained model",
                f"Profile '{self.profile_name}' has no trained model yet.\n"
                "Running calibration instead.",
            )
            self.skip_calibration = False
            dur_dialog = TrainingDurationDialog()
            if dur_dialog.exec() != dur_dialog.DialogCode.Accepted:
                self._shutdown()
                return 0
            self.duration_sec = dur_dialog.duration_sec()
            self.overlay = CalibrationOverlay(self.screen)
            self.overlay.show()

            # Phase 3: Camera preview for calibration
            self._camera_preview = CameraPreviewWidget(width=420, height=316)
            self._camera_preview.show()

            self.cal_session = CalibrationSession(self.screen, self.duration_sec, self.profile_name)
            self._model = self.cal_session.model
            self.worker.model = self._model
            self._last_tick_time = time.perf_counter()
            self._tick_timer.start()
            self._preview_timer.start()
            return self.app.exec()

        self._model = model
        self.worker.model = self._model
        self._start_cursor_control(meta)
        return self.app.exec()

    def _on_tick(self) -> None:
        if self.worker is None or self.overlay is None:
            return

        result = self.worker.get_latest()
        features = result.features if result else None

        # FIX: Measure real elapsed time instead of hardcoded 16.0
        now = time.perf_counter()
        dt_ms = (now - self._last_tick_time) * 1000.0
        dt_ms = max(0.0, min(dt_ms, 100.0))  # clamp to avoid huge jumps
        self._last_tick_time = now

        if self.cal_session is not None and not self.cal_session.state.finished:
            state = self.cal_session.tick(dt_ms, features)
            self.overlay.set_calibration_state(state)

            # Phase 3: Feed frame + landmarks to camera preview
            if self._camera_preview is not None and result is not None:
                info = self._build_preview_info(state.phase_name, state, features)
                self._camera_preview.set_frame_and_landmarks(
                    result.raw_frame_bgr,
                    result.raw_landmarks,
                    info,
                )

            # Check if calibration session wants to retrain — launch in background
            if not self._training_busy and self.cal_session.needs_retrain:
                self._launch_background_retrain(self.cal_session, self.worker)

            if state.finished:
                self._start_verification()
            return

        if self.verify_session is not None and not self.verify_session.state.finished:
            state = self.verify_session.tick(dt_ms, features)
            self.overlay.set_verification_state(state)

            # Phase 3: Feed frame + landmarks to camera preview during verification
            if self._camera_preview is not None and result is not None:
                info = self._build_verify_info(state, features)
                self._camera_preview.set_frame_and_landmarks(
                    result.raw_frame_bgr,
                    result.raw_landmarks,
                    info,
                )

            if state.finished:
                self._show_scorecard()
            return

    def _build_preview_info(self, phase_name: str, state, features) -> list[str]:
        """Build info text lines for the camera preview during calibration."""
        lines = [
            f"Phase: {phase_name}",
            f"Samples: {len(state.samples)}",
            f"Face: {'YES' if state.face_detected else 'NO'}",
        ]
        if state.model_trained:
            lines.append(f"Error: {state.running_avg_error_px:.1f} px")
        if features is not None:
            lines.append(f"MAR: {features.mar:.2f} {'[OPEN]' if features.mar > 0.4 else ''}")
            lines.append(f"EAR: {features.ear:.2f}")
        remaining = state.time_remaining_ms / 1000
        lines.append(f"Time: {remaining:.0f}s left")
        return lines

    def _build_verify_info(self, vstate, features) -> list[str]:
        """Build info text lines during verification."""
        lines = ["Mode: Verification"]
        if vstate.errors_px:
            avg = sum(vstate.errors_px) / len(vstate.errors_px)
            lines.append(f"Avg Error: {avg:.1f} px")
        if features is not None:
            lines.append(f"MAR: {features.mar:.2f}")
        remaining = max(0, (vstate.duration_ms - vstate.elapsed_ms) / 1000)
        lines.append(f"Time: {remaining:.0f}s left")
        return lines

    def _on_preview_tick(self) -> None:
        """Refresh the camera preview widget at ~15fps."""
        if self._camera_preview is not None:
            self._camera_preview.refresh()

    # ------------------------------------------------------------------
    # Background training
    # ------------------------------------------------------------------

    def _launch_background_retrain(
        self,
        cal_session: CalibrationSession,
        worker: GazeWorker,
        epochs: int | None = None,
    ) -> None:
        """Launch model training in a QThread so the UI stays responsive."""
        req = cal_session.consume_retrain_request()
        if req is None:
            return
        X, y, req_epochs = req
        if epochs is not None:
            req_epochs = epochs

        self._training_busy = True
        self._training_thread = TrainingThread(cal_session.model, X, y, epochs=req_epochs, parent=self.app)
        self._training_thread.done.connect(
            lambda model, loss: self._on_training_done(cal_session, worker, model, loss)
        )
        self._training_thread.start()

    def _on_training_done(
        self,
        cal_session: CalibrationSession,
        worker: GazeWorker,
        model: GazeModel,
        loss: float,
    ) -> None:
        """TrainingThread finished — update worker model (on main thread)."""
        self._training_busy = False
        self._training_thread = None

        if loss == float("inf"):
            logger.error("Background training failed")
            return

        # Update prediction immediately after retrain
        if self.worker is not None:
            self.worker.model = model

        # Update live prediction overlay
        if self.cal_session is not None and self.worker is not None:
            result = self.worker.get_latest()
            if result and result.features is not None and model.is_trained:
                pred = model.predict(result.features.vector)
                if pred is not None:
                    self.cal_session.state.pred_nx, self.cal_session.state.pred_ny = pred

    def _start_verification(self) -> None:
        assert self.cal_session is not None

        # Final training also runs in background
        if not self.cal_session.model.is_trained:
            self._retrain_final()
        else:
            self._verify_session = VerificationSession(
                self.screen,
                self.cal_session.model,
                duration_sec=30.0,
            )

    def _retrain_final(self) -> None:
        """Prepare final training data and launch in background thread."""
        assert self.cal_session is not None
        samples = self.cal_session.state.samples
        if len(samples) < 10:
            # Not enough data — skip to verification anyway
            self._verify_session = VerificationSession(
                self.screen, self.cal_session.model, duration_sec=30.0
            )
            return

        new_feats = np.array([s.features for s in samples], dtype=np.float32)
        new_tx = np.array([s.target_x for s in samples], dtype=np.float32)
        new_ty = np.array([s.target_y for s in samples], dtype=np.float32)
        old_feats, old_tx, old_ty, _ = self.profiles.load_dataset(self.profile_name)
        if len(old_feats) and self.retrain_mode:
            feats = np.vstack([old_feats, new_feats])
            tx = np.concatenate([old_tx, new_tx])
            ty = np.concatenate([old_ty, new_ty])
        else:
            feats, tx, ty = new_feats, new_tx, new_ty
        y = np.column_stack([tx, ty])

        # Launch background training; start verification when done
        self._training_busy = True
        self._training_thread = TrainingThread(
            self.cal_session.model, feats, y, epochs=150, parent=self.app
        )
        self._training_thread.done.connect(self._on_final_training_done)
        self._training_thread.start()

    def _on_final_training_done(self, model: GazeModel, loss: float) -> None:
        self._training_busy = False
        self._training_thread = None
        if self.worker is not None:
            self.worker.model = model
        # Now start verification
        self._verify_session = VerificationSession(
            self.screen, model, duration_sec=30.0
        )

    def _show_scorecard(self) -> None:
        assert self.cal_session is not None and self.verify_session is not None
        vstate = self.verify_session.state

        if not vstate.errors_px and self.cal_session.model.is_trained:
            vstate.grade = "Fair"
            vstate.avg_error_px = 999.0
            vstate.max_error_px = 999.0

        ear_open, ear_closed = self.cal_session.get_ear_baselines()

        meta = self.profiles.record_session(
            self.profile_name,
            self.cal_session.state.samples,
            self.cal_session.model,
            avg_error_px=vstate.avg_error_px if vstate.errors_px else None,
            ear_open=ear_open,
            ear_closed=ear_closed,
        )

        dialog = ScorecardDialog(
            meta,
            vstate.avg_error_px,
            vstate.max_error_px,
            vstate.valid_samples,
            vstate.grade or ("Fair" if not vstate.errors_px else "N/A"),
        )
        dialog.exec()

        if dialog.result_action == "retrain":
            self._restart_calibration(extra_sec=60)
        elif dialog.result_action in ("done", "save"):
            self._start_cursor_control(meta)
        else:
            self._shutdown()

    def _restart_calibration(self, extra_sec: int = 60) -> None:
        self.verify_session = None
        self.retrain_mode = True
        self.duration_sec = extra_sec
        self.cal_session = CalibrationSession(self.screen, self.duration_sec, self.profile_name)
        loaded = self.profiles.load_model(self.profile_name)
        if loaded.is_trained:
            self.cal_session.model = loaded
        self._model = self.cal_session.model
        self.worker.model = self._model
        if self.overlay:
            self.overlay.show()
            self.overlay.show_hud = True

        # Phase 3: Show camera preview during re-calibration too
        if self._camera_preview:
            self._camera_preview.show()

        self._last_tick_time = time.perf_counter()
        self._tick_timer.start()
        self._preview_timer.start()

    def _start_cursor_control(self, meta) -> None:
        self._tick_timer.stop()
        self._preview_timer.stop()  # Stop preview during runtime

        if self.overlay:
            self.overlay.hide()

        # Phase 3: Hide camera preview during cursor control
        if self._camera_preview:
            self._camera_preview.hide()

        if self.worker:
            self.worker.model = self._model
            self.worker.reset_filter()

        # Phase 3: CursorController now uses Magic Cursor internally
        self.cursor = CursorController(self.worker, screen=self.screen)
        self.cursor.enabled = True
        self._cursor_timer.start()

        from gazer.ui.runtime_window import RuntimeControlWindow

        self.runtime_window = RuntimeControlWindow(meta, self.worker)
        self.runtime_window.show()
        self.runtime_window.bind_cursor(self.cursor)
        self.runtime_window.quit_requested.connect(self._shutdown)

    def _on_cursor_tick(self) -> None:
        if self.cursor:
            self.cursor.tick()

    def _shutdown(self) -> None:
        # Wait for any in-flight training
        if self._training_thread is not None and self._training_thread.isRunning():
            self._training_thread.quit()
            self._training_thread.wait(3000)

        self._cursor_timer.stop()
        self._tick_timer.stop()
        self._preview_timer.stop()

        # Phase 3: Clean up camera preview
        if self._camera_preview is not None:
            self._camera_preview.close()
            self._camera_preview = None

        if self.worker:
            self.worker.stop()
        self.app.quit()


def main():
    app = GazerApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()