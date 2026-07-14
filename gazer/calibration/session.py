"""Calibration session orchestrator with outlier rejection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from gazer.calibration.animations import figure8_verification_points, build_segments_from_points
from gazer.calibration.phases import ALL_PHASES, PhaseId, PhaseSpec, phase_durations_ms
from gazer.face_tracker import GazeFeatures, FEATURE_DIM
from gazer.gaze_model import GazeModel
from gazer.profile_manager import CalibrationSample, ProfileManager
from gazer.screen import ScreenBounds

logger = logging.getLogger(__name__)

# Outlier rejection thresholds
_MIN_EAR_FOR_SAMPLE = 0.15       # reject if eyes nearly closed
_MAX_YAW_FOR_SAMPLE = 40.0       # reject if head turned too far
_MAX_PITCH_FOR_SAMPLE = 35.0     # reject if head tilted too far
_TRANSITION_IGNORE_MS = 300.0    # ignore first N ms of each phase (transition)


@dataclass
class CalibrationState:
    phase_index: int = 0
    phase_elapsed_ms: float = 0.0
    total_elapsed_ms: float = 0.0
    samples: list[CalibrationSample] = field(default_factory=list)
    rejected_count: int = 0       # how many samples were rejected
    running_avg_error_px: float = 0.0
    _error_count: int = 0
    target_nx: float = 0.5
    target_ny: float = 0.5
    pred_nx: float = 0.5
    pred_ny: float = 0.5
    prompt: str = ""
    phase_name: str = ""
    time_remaining_ms: float = 0.0
    ear_samples_open: list[float] = field(default_factory=list)
    ear_samples_blink: list[float] = field(default_factory=list)
    finished: bool = False
    model_trained: bool = False
    status_line: str = ""
    face_detected: bool = False


@dataclass
class VerificationState:
    elapsed_ms: float = 0.0
    duration_ms: float = 30000.0
    errors_px: list[float] = field(default_factory=list)
    target_nx: float = 0.5
    target_ny: float = 0.5
    pred_nx: float = 0.5
    pred_ny: float = 0.5
    finished: bool = False
    avg_error_px: float = 0.0
    max_error_px: float = 0.0
    valid_samples: int = 0
    grade: str = ""


def grade_from_error(avg_px: float) -> str:
    if avg_px < 30:
        return "Excellent"
    if avg_px < 60:
        return "Good"
    if avg_px < 100:
        return "Fair"
    return "Redo calibration"


def _is_valid_sample(features: GazeFeatures, phase_elapsed_ms: float) -> bool:
    """Reject low-quality calibration samples.

    Returns True if the sample should be kept.
    """
    # Reject during phase transitions
    if phase_elapsed_ms < _TRANSITION_IGNORE_MS:
        return False

    # Reject if eyes nearly closed
    if features.ear < _MIN_EAR_FOR_SAMPLE:
        return False

    # Reject extreme head angles (user not looking at target)
    if abs(features.yaw) > _MAX_YAW_FOR_SAMPLE:
        return False
    if abs(features.pitch) > _MAX_PITCH_FOR_SAMPLE:
        return False

    return True


def _reject_feature_outliers(
    samples: list[CalibrationSample],
    iqr_multiplier: float = 2.0,
) -> list[CalibrationSample]:
    """IQR-based outlier rejection on feature vectors.

    Removes samples whose features are statistical outliers in any dimension.
    """
    if len(samples) < 20:
        return samples  # not enough data for IQR

    feats = np.array([s.features for s in samples], dtype=np.float32)

    # Compute per-feature IQR bounds
    q1 = np.percentile(feats, 25, axis=0)
    q3 = np.percentile(feats, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    # Keep samples where all features are within bounds
    mask = np.all((feats >= lower) & (feats <= upper), axis=1)
    kept = [s for s, m in zip(samples, mask) if m]
    rejected = len(samples) - len(kept)

    if rejected > 0:
        logger.info(
            "Feature outlier rejection: kept %d / %d samples (%d rejected)",
            len(kept), len(samples), rejected,
        )

    return kept


class CalibrationSession:
    def __init__(self, screen: ScreenBounds, total_duration_sec: float, profile_name: str):
        self.screen = screen
        self.total_duration_ms = total_duration_sec * 1000
        self.profile_name = profile_name
        self.phase_durations = phase_durations_ms(self.total_duration_ms)
        self.phases = ALL_PHASES
        self.model = GazeModel()
        self.state = CalibrationState(time_remaining_ms=self.total_duration_ms)
        self._last_retrain = 0.0
        self._retrain_interval_ms = 2000.0
        self._min_samples_retrain = 15
        self._profiles = ProfileManager()

        # Cache old dataset in memory
        self._cached_old_feats: np.ndarray
        self._cached_old_tx: np.ndarray
        self._cached_old_ty: np.ndarray
        self._cached_old_feats, self._cached_old_tx, self._cached_old_ty, _ = (
            self._profiles.load_dataset(profile_name)
        )

        # If old data has different feature dim, discard it (requires recalibration)
        if len(self._cached_old_feats) > 0 and self._cached_old_feats.shape[1] != FEATURE_DIM:
            logger.warning(
                "Old profile has %d-dim features, current is %d-dim. Discarding old data — full recalibration needed.",
                self._cached_old_feats.shape[1], FEATURE_DIM,
            )
            self._cached_old_feats = np.zeros((0, FEATURE_DIM), dtype=np.float32)
            self._cached_old_tx = np.zeros(0, dtype=np.float32)
            self._cached_old_ty = np.zeros(0, dtype=np.float32)

        # Train on cached old data at start (if compatible)
        if len(self._cached_old_feats) >= 10:
            y = np.column_stack([self._cached_old_tx, self._cached_old_ty])
            self.model.train(self._cached_old_feats, y, epochs=100)

        # Background retrain signaling
        self.needs_retrain: bool = False
        self.retrain_data: tuple[np.ndarray, np.ndarray] | None = None
        self.retrain_epochs: int = 80

    @property
    def current_phase(self) -> PhaseSpec:
        return self.phases[self.state.phase_index]

    def _advance_phase_if_needed(self) -> None:
        while self.state.phase_index < len(self.phases):
            phase = self.current_phase
            budget = self.phase_durations[phase.phase_id]
            if self.state.phase_elapsed_ms < budget:
                break
            self.state.phase_index += 1
            self.state.phase_elapsed_ms = 0.0
            if self.state.phase_index >= len(self.phases):
                self.state.finished = True
                break

    def tick(self, dt_ms: float, features: GazeFeatures | None) -> CalibrationState:
        if self.state.finished:
            return self.state

        self.state.total_elapsed_ms += dt_ms
        self.state.time_remaining_ms = max(0, self.total_duration_ms - self.state.total_elapsed_ms)
        self._advance_phase_if_needed()
        if self.state.finished:
            return self.state

        phase = self.current_phase
        self.state.phase_name = phase.name
        self.state.prompt = phase.prompt
        self.state.phase_elapsed_ms += dt_ms
        self.state.model_trained = self.model.is_trained
        self.state.face_detected = features is not None

        if features is None:
            self.state.status_line = "No face detected - look at the camera"
        elif not self.model.is_trained:
            need = max(0, self._min_samples_retrain - len(self.state.samples))
            self.state.status_line = f"Collecting samples... ({need} more for first prediction)"
        else:
            self.state.status_line = f"Model active - {len(self.state.samples)} samples collected"

        if phase.is_ear_baseline:
            self.state.target_nx = 0.5
            self.state.target_ny = 0.5
            if features is not None:
                half = self.phase_durations[phase.phase_id] / 2
                if self.state.phase_elapsed_ms < half:
                    self.state.ear_samples_blink.append(features.ear)
                else:
                    self.state.ear_samples_open.append(features.ear)
            return self.state

        if phase.animation is not None:
            nx, ny, _ = phase.animation.position_at(self.state.phase_elapsed_ms)
            self.state.target_nx = nx
            self.state.target_ny = ny

        # Collect sample with outlier rejection
        if features is not None and phase.collect_samples:
            if _is_valid_sample(features, self.state.phase_elapsed_ms):
                self.state.samples.append(
                    CalibrationSample(
                        features=features.vector.copy(),
                        target_x=self.state.target_nx,
                        target_y=self.state.target_ny,
                        timestamp=time.time(),
                        phase_id=int(phase.phase_id),
                    )
                )
            else:
                self.state.rejected_count += 1

            if self.model.is_trained:
                pred = self.model.predict(features.vector)
                if pred is not None:
                    self.state.pred_nx, self.state.pred_ny = pred
                    tx, ty = self.screen.norm_to_screen(self.state.target_nx, self.state.target_ny)
                    px, py = self.screen.norm_to_screen(pred[0], pred[1])
                    err = ((tx - px) ** 2 + (ty - py) ** 2) ** 0.5
                    self.state._error_count += 1
                    self.state.running_avg_error_px += (err - self.state.running_avg_error_px) / self.state._error_count

        # Incremental retrain check
        if len(self.state.samples) >= self._min_samples_retrain:
            since = self.state.total_elapsed_ms - self._last_retrain
            if since >= self._retrain_interval_ms:
                self._prepare_retrain_data()
                self._last_retrain = self.state.total_elapsed_ms

        return self.state

    def _prepare_retrain_data(self) -> None:
        """Prepare training data with IQR outlier rejection."""
        # First pass: reject feature outliers
        clean_samples = _reject_feature_outliers(self.state.samples)

        new_feats = np.array([s.features for s in clean_samples], dtype=np.float32)
        new_tx = np.array([s.target_x for s in clean_samples], dtype=np.float32)
        new_ty = np.array([s.target_y for s in clean_samples], dtype=np.float32)

        if len(self._cached_old_feats) and self._cached_old_feats.shape[1] == FEATURE_DIM:
            feats = np.vstack([self._cached_old_feats, new_feats])
            tx = np.concatenate([self._cached_old_tx, new_tx])
            ty = np.concatenate([self._cached_old_ty, new_ty])
        else:
            feats, tx, ty = new_feats, new_tx, new_ty

        if len(feats) >= 10:
            y = np.column_stack([tx, ty])
            self.retrain_data = (feats, y)
            self.retrain_epochs = 80
            self.needs_retrain = True

    def consume_retrain_request(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        if not self.needs_retrain or self.retrain_data is None:
            return None
        self.needs_retrain = False
        X, y = self.retrain_data
        epochs = self.retrain_epochs
        self.retrain_data = None
        return X, y, epochs

    def get_ear_baselines(self) -> tuple[float | None, float | None]:
        open_vals = self.state.ear_samples_open
        blink_vals = self.state.ear_samples_blink
        open_b = float(np.mean(open_vals)) if open_vals else None
        closed_b = float(np.percentile(blink_vals, 20)) if blink_vals else None
        return open_b, closed_b


class VerificationSession:
    def __init__(self, screen: ScreenBounds, model: GazeModel, duration_sec: float = 30.0):
        self.screen = screen
        self.model = model
        pts = figure8_verification_points(120)
        segs = build_segments_from_points(pts, 250, 0)
        from gazer.calibration.animations import PathAnimation
        self._anim = PathAnimation(segs, loop=True)
        self.state = VerificationState(duration_ms=duration_sec * 1000)

    def tick(self, dt_ms: float, features: GazeFeatures | None) -> VerificationState:
        if self.state.finished:
            return self.state

        self.state.elapsed_ms += dt_ms
        nx, ny, _ = self._anim.position_at(self.state.elapsed_ms)
        self.state.target_nx = nx
        self.state.target_ny = ny

        if features is not None and self.model.is_trained:
            pred = self.model.predict(features.vector)
            if pred is not None:
                self.state.pred_nx, self.state.pred_ny = pred
                tx, ty = self.screen.norm_to_screen(nx, ny)
                px, py = self.screen.norm_to_screen(pred[0], pred[1])
                err = ((tx - px) ** 2 + (ty - py) ** 2) ** 0.5
                self.state.errors_px.append(err)

        if self.state.elapsed_ms >= self.state.duration_ms:
            self.state.finished = True
            if self.state.errors_px:
                self.state.avg_error_px = float(np.mean(self.state.errors_px))
                self.state.max_error_px = float(np.max(self.state.errors_px))
                self.state.valid_samples = len(self.state.errors_px)
                self.state.grade = grade_from_error(self.state.avg_error_px)

        return self.state