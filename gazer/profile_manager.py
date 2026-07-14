"""Profile persistence — datasets, models, metadata."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gazer.face_tracker import FEATURE_DIM
from gazer.gaze_model import GazeModel

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


@dataclass
class CalibrationSample:
    features: np.ndarray
    target_x: float  # normalized
    target_y: float
    timestamp: float
    phase_id: int


@dataclass
class ProfileMeta:
    name: str
    sessions: int = 0
    total_samples: int = 0
    last_avg_error_px: float | None = None
    ear_open_baseline: float | None = None
    ear_closed_baseline: float | None = None
    feature_dim: int = FEATURE_DIM  # track which dim this profile was created with
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ProfileManager:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or PROFILES_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> list[str]:
        return sorted(
            p.name for p in self.base_dir.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        )

    def profile_dir(self, name: str) -> Path:
        d = self.base_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_meta(self, name: str) -> ProfileMeta:
        path = self.profile_dir(name) / "meta.json"
        if not path.exists():
            return ProfileMeta(name=name, feature_dim=FEATURE_DIM)
        data = json.loads(path.read_text(encoding="utf-8"))
        # Ensure feature_dim exists (backward compat)
        if "feature_dim" not in data:
            data["feature_dim"] = 8  # old profiles were 8-dim
        return ProfileMeta(**data)

    def save_meta(self, meta: ProfileMeta) -> None:
        meta.updated_at = time.time()
        meta.feature_dim = FEATURE_DIM  # always update to current dim
        path = self.profile_dir(meta.name) / "meta.json"
        path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    def load_dataset(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        path = self.profile_dir(name) / "dataset.npz"
        if not path.exists():
            return (
                np.zeros((0, FEATURE_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
            )
        data = np.load(path)
        feats = data["features"]
        # Check dimension compatibility
        if feats.shape[1] != FEATURE_DIM:
            logger.warning(
                "Profile '%s' has %d-dim features, current is %d-dim. Data will be padded/discarded.",
                name, feats.shape[1], FEATURE_DIM,
            )
            if feats.shape[1] < FEATURE_DIM:
                # Pad with zeros (old 8-dim -> new 25-dim)
                padded = np.zeros((len(feats), FEATURE_DIM), dtype=np.float32)
                padded[:, :feats.shape[1]] = feats
                feats = padded
            else:
                # Truncate (shouldn't happen but handle it)
                feats = feats[:, :FEATURE_DIM]
        return feats, data["target_x"], data["target_y"], data["phase_id"]

    def append_samples(self, name: str, samples: list[CalibrationSample]) -> int:
        if not samples:
            return 0

        feats, tx, ty, pid = self.load_dataset(name)
        new_feats = np.array([s.features for s in samples], dtype=np.float32)
        new_tx = np.array([s.target_x for s in samples], dtype=np.float32)
        new_ty = np.array([s.target_y for s in samples], dtype=np.float32)
        new_pid = np.array([s.phase_id for s in samples], dtype=np.int32)

        all_feats = np.vstack([feats, new_feats]) if len(feats) else new_feats
        all_tx = np.concatenate([tx, new_tx])
        all_ty = np.concatenate([ty, new_ty])
        all_pid = np.concatenate([pid, new_pid])

        path = self.profile_dir(name) / "dataset.npz"
        np.savez(path, features=all_feats, target_x=all_tx, target_y=all_ty, phase_id=all_pid)
        return len(all_feats)

    def save_model(self, name: str, model: GazeModel) -> None:
        model.save(self.profile_dir(name) / "model.pt")

    def load_model(self, name: str) -> GazeModel:
        model = GazeModel()
        path = self.profile_dir(name) / "model.pt"
        if path.exists() or path.with_suffix(".pkl").exists():
            model.load(path)
        return model

    def train_from_profile(self, name: str, model: GazeModel | None = None) -> GazeModel:
        feats, tx, ty, _ = self.load_dataset(name)
        if model is None:
            model = GazeModel()
        if len(feats) >= 10:
            y = np.column_stack([tx, ty])
            model.train(feats, y)
        return model

    def record_session(
        self,
        name: str,
        samples: list[CalibrationSample],
        model: GazeModel,
        avg_error_px: float | None = None,
        ear_open: float | None = None,
        ear_closed: float | None = None,
    ) -> ProfileMeta:
        total = self.append_samples(name, samples)
        self.save_model(name, model)
        meta = self.load_meta(name)
        meta.sessions += 1
        meta.total_samples = total
        if avg_error_px is not None:
            meta.last_avg_error_px = avg_error_px
        if ear_open is not None:
            meta.ear_open_baseline = ear_open
        if ear_closed is not None:
            meta.ear_closed_baseline = ear_closed
        self.save_meta(meta)
        return meta


# Need json import at module level
import json  # noqa: E402