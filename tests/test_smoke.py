"""Smoke tests for core Gazer components."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_one_euro_filter():
    from gazer.one_euro_filter import OneEuroFilter2D

    f = OneEuroFilter2D()
    x, y = f.filter(100.0, 200.0)
    assert 90 < x < 110 and 190 < y < 210
    print("  [OK] One Euro Filter")


def test_gaze_model_train_predict():
    from gazer.gaze_model import GazeModel

    rng = np.random.default_rng(42)
    X = rng.random((200, 8)).astype(np.float32)
    y = rng.random((200, 2)).astype(np.float32)
    model = GazeModel()
    loss = model.train(X, y, epochs=30)
    assert model.is_trained
    pred = model.predict(X[0])
    assert pred is not None
    assert 0 <= pred[0] <= 1 and 0 <= pred[1] <= 1
    print(f"  [OK] Gaze model train/predict (loss={loss:.4f})")


def test_profile_persistence():
    from gazer.gaze_model import GazeModel
    from gazer.profile_manager import CalibrationSample, ProfileManager

    with tempfile.TemporaryDirectory() as tmp:
        pm = ProfileManager(Path(tmp) / "profiles")
        samples = [
            CalibrationSample(
                features=np.random.rand(8).astype(np.float32),
                target_x=0.5,
                target_y=0.5,
                timestamp=1.0,
                phase_id=1,
            )
            for _ in range(20)
        ]
        model = GazeModel()
        feats = np.array([s.features for s in samples])
        y = np.array([[s.target_x, s.target_y] for s in samples])
        model.train(feats, y, epochs=20)
        meta = pm.record_session("testuser", samples, model, avg_error_px=45.0, ear_open=0.3)
        assert meta.sessions == 1
        loaded = pm.load_model("testuser")
        assert loaded.is_trained
    print("  [OK] Profile persistence")


def test_calibration_phases():
    from gazer.calibration.phases import ALL_PHASES, phase_durations_ms

    assert len(ALL_PHASES) == 15
    durations = phase_durations_ms(180_000)
    assert abs(sum(durations.values()) - 180_000) < 1.0
    print("  [OK] 15 calibration phases")


def test_face_landmarker_model():
    from gazer.models import ensure_face_landmarker

    path = ensure_face_landmarker()
    assert path.exists()
    assert path.stat().st_size > 1_000_000
    print(f"  [OK] Face landmarker model ({path.stat().st_size // 1024} KB)")


def test_face_tracker_init():
    from gazer.face_tracker import FaceTracker

    tracker = FaceTracker()
    tracker.close()
    print("  [OK] FaceTracker init (MediaPipe Tasks API)")


def test_camera_open():
    import threading

    from gazer.camera import Camera

    result = {"ok": False, "msg": ""}

    def _try():
        cam = Camera()
        if cam.open():
            ret, frame = cam.read()
            cam.release()
            if ret and frame is not None:
                result["ok"] = True
                result["msg"] = f"{frame.shape[1]}x{frame.shape[0]}"
        else:
            result["msg"] = "not found"

    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout=8.0)
    if result["ok"]:
        print(f"  [OK] Camera opened ({result['msg']})")
    elif t.is_alive():
        print("  [SKIP] Camera timed out (driver issue)")
    else:
        print(f"  [SKIP] Camera not available ({result['msg']})")


def main():
    print("Gazer smoke tests\n")
    tests = [
        test_one_euro_filter,
        test_gaze_model_train_predict,
        test_profile_persistence,
        test_calibration_phases,
        test_face_landmarker_model,
        test_face_tracker_init,
        test_camera_open,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"  [FAIL] {t.__name__}: {exc}")
            failed += 1
    print()
    if failed:
        print(f"{failed} test(s) failed.")
        sys.exit(1)
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    main()
