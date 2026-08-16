# eye_gaze
> Gazer: a Python desktop app that uses a webcam and MediaPipe face landmarker to estimate gaze and control the system mouse via profile-based calibration and a regression model.

## Overview
A modular desktop application (Windows/Linux) that captures webcam frames, extracts a 25-dimensional facial feature vector (iris + face/pose cues), trains or loads a regression model (PyTorch with a scikit-learn fallback) to predict normalized screen coordinates, smooths predictions (One Euro filter), and drives the OS cursor with a physics-based "Magic Cursor". Supports calibration profiles, blink/dwell/mouth-open clickers, and a Qt-based UI.

## What it does
- Tracks face and iris with MediaPipe and extracts features for gaze estimation.
- Trains or loads a gaze regression model and predicts screen (x,y) coordinates.
- Smooths outputs and moves the system cursor via pynput, using a Magic Cursor to reduce accidental activation.
- Provides calibration profiles saved under profiles/<name>/ with persistent artifacts (dataset, model, metadata).
- Offers blink-to-click (EAR), dwell-click, and mouth-open clickers, plus a camera preview and calibration overlay UI.

## Key capabilities
- MediaPipe face landmarker + custom 25-dim feature extraction.
- Gaze regression with PyTorch neural net and sklearn fallback (scaler & augmentation).
- One Euro 2D filter for smoothing predictions.
- Physics-based Magic Cursor to handle attraction/damping and reduce midas-touch.
- Blink (EAR), dwell, and mouth-open click detection modules.
- Camera capture abstraction supporting platform backends.
- Profile management and persistence (profiles/<name>/ contains dataset.npz, model.pt, meta.json).
- Auto-download of the MediaPipe face landmarker asset to gazer/assets/ on first run.

## Technology
Primary technologies documented in the repository:
- Python 3.10+
- OpenCV (opencv-python)
- MediaPipe Tasks API (mediapipe)
- NumPy
- PyTorch (optional; sklearn fallback)
- scikit-learn
- PyQt6 (UI)
- pynput (mouse control)
- screeninfo
- joblib

(These requirements are enumerated in requirements.txt in the repository.)

## Repository structure
Top-level items (as present in the supplied snapshot):
- README.md
- gazer/  (package with core modules: camera, face_tracker, gaze_model, one_euro_filter, magic_cursor, blink_detector, mouth_open_detector, models.py, etc.)
- main.py
- requirements.txt
- setup.bat
- tests/ (contains tests/test_smoke.py)

Notable modules (referenced in docs and code):
- gazer/camera.py
- gazer/face_tracker.py
- gazer/gaze_model.py
- gazer/one_euro_filter.py
- gazer/magic_cursor.py
- gazer/blink_detector.py
- gazer/mouth_open_detector.py
- gazer/models.py (auto-download helper for MediaPipe model)
- gazer/cursor_controller.py
- gazer/app.py (references ProfileManager; ProfileManager implementation not included in the supplied snapshot)

## Getting started
Evidence in the repository includes a short quick-start and a setup script. Minimal steps (as found in the existing README excerpt):
- On Windows:
  - Run setup.bat (present in the repo snapshot).
  - Run python main.py to start the application.
- Alternatively:
  - Install dependencies from requirements.txt: pip install -r requirements.txt
  - Verify with python tests/test_smoke.py (smoke test script present)
  - Run python main.py

On first run, the MediaPipe face landmarker model is automatically downloaded to gazer/assets/.

Note: the repository snapshot contains a requirements.txt listing the Python dependencies. See the Configuration section for its contents.

## Configuration
- requirements.txt (as present in the repository snapshot):
  - opencv-python>=4.8.0
  - mediapipe>=0.10.9,<0.11
  - numpy>=1.24.0
  - torch>=2.0.0
  - scikit-learn>=1.3.0
  - PyQt6>=6.5.0
  - pynput>=1.7.6
  - screeninfo>=0.8.1
  - joblib>=1.3.0

- Profiles:
  - Saved under profiles/<name>/ with files:
    - dataset.npz — calibration samples (features + targets)
    - model.pt — PyTorch model weights
    - meta.json — session metadata (sample counts, EAR baseline, last error, etc.)

- MediaPipe model:
  - Auto-downloaded to gazer/assets/ by gazer/models.py on first use.

If you need to inspect manifests and configuration before running anything, examine:
- requirements.txt (dependency list)
- gazer/models.py (download behavior)
- tests/test_smoke.py (what components the smoke tests exercise)
- gazer/ (module files mentioned above) and profiles/ (if any profiles exist in your clone)

## Development and quality notes
- Smoke tests: tests/test_smoke.py exercises OneEuroFilter, gaze model training/prediction, profile persistence (ProfileManager referenced), calibration phases, face landmarker download, face tracker init, and camera open. The test file in the snapshot is truncated at the end; it may need review to run cleanly.
- Truncated files: several source files provided in the snapshot are truncated (notably gazer/face_tracker.py, gazer/gaze_model.py, gazer/head_pose.py, gazer/magic_cursor.py). Full implementations are required to verify runtime behavior and should be completed or restored.
- No CI or packaging metadata is present in the supplied dossier.
- No LICENSE or CONTRIBUTING.md present in the snapshot.

Recommended immediate hygiene (based on repository evidence):
- Review and complete truncated modules so unit tests and static review can validate behavior.
- Run smoke tests locally; camera-dependent tests may be flaky or block on headless CI — consider marking or mocking hardware tests.

## Safety and responsible use
Relevant safety/security observations from the supplied snapshot:
- Model download uses urllib.request.urlretrieve without checksum verification. The code in gazer/models.py does not include a SHA256/ signature check for the downloaded MediaPipe model.
- The application controls the system mouse via pynput. This is a powerful system-level operation; ensure users are informed and consent to mouse control on their machines.
- Profile names are used to create directories under profiles/<name>/. The ProfileManager implementation was not included in the snapshot, so validate/sanitize profile names to avoid path traversal or unsafe filesystem operations.
- No telemetry or privacy controls are visible in the provided snapshot; camera data is inherently sensitive. Handle it carefully and avoid transmitting raw frames without explicit opt-in.

## Contributing
- The repository snapshot does not include a CONTRIBUTING.md or a license file. To contribute:
  - Inspect the code in gazer/ and tests/test_smoke.py to understand module boundaries and current test coverage.
  - Run the smoke test locally (python tests/test_smoke.py) to observe current behavior; be aware camera tests may require a local webcam and permissions.
  - Open issues or pull requests in the repository to propose fixes (e.g., complete truncated files, add checksum verification for model downloads, sanitize profile names, add CI and packaging).
  - When proposing changes that affect model download or cursor control, include security and user-consent considerations in the PR description.

If you want to audit configuration and manifests before executing anything, review:
- requirements.txt
- gazer/models.py
- tests/test_smoke.py
- any files under gazer/ referenced above
These files contain the primary observable configuration and behavior from the supplied snapshot.

(There is no LICENSE file included in the supplied snapshot; no license statement is made here.)
