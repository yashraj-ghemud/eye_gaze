"""Main window dialogs and scorecard."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from gazer.profile_manager import ProfileManager, ProfileMeta

MODE_USE_AS_IS = 0
MODE_CALIBRATE = 1
MODE_NEW = 2
MODE_RETRAIN = 3


class TrainingDurationDialog(QDialog):
    PRESETS = {
        "Quick (1 min)": 60,
        "Standard (3 min)": 180,
        "Deep (5 min)": 300,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gazer — Training Duration")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("How long do you want to train?"))

        self.preset = QComboBox()
        for name in self.PRESETS:
            self.preset.addItem(name)
        self.preset.addItem("Custom")
        self.preset.currentIndexChanged.connect(self._on_preset)
        layout.addWidget(self.preset)

        custom_row = QHBoxLayout()
        custom_row.addWidget(QLabel("Custom (seconds):"))
        self.custom_sec = QSpinBox()
        self.custom_sec.setRange(30, 1800)
        self.custom_sec.setValue(180)
        self.custom_sec.setEnabled(False)
        custom_row.addWidget(self.custom_sec)
        layout.addLayout(custom_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_preset(self, idx: int) -> None:
        self.custom_sec.setEnabled(self.preset.currentText() == "Custom")

    def duration_sec(self) -> int:
        text = self.preset.currentText()
        if text == "Custom":
            return self.custom_sec.value()
        return self.PRESETS[text]


class ProfileDialog(QDialog):
    def __init__(self, profiles: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gazer — Profile")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Welcome to Gazer</b> — eye gaze cursor control"))

        self.mode = QComboBox()
        if profiles:
            self.mode.addItems([
                "Use profile — skip calibration",
                "Calibrate profile (fresh training)",
                "Retrain profile (append more data)",
                "New profile",
            ])
        else:
            self.mode.addItems(["New profile"])
        layout.addWidget(self.mode)

        form = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(profiles if profiles else [])
        self.profile_combo.setEnabled(bool(profiles))
        form.addRow("Profile:", self.profile_combo)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. yash, work, home")
        form.addRow("New name:", self.name_edit)
        layout.addLayout(form)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.mode.currentIndexChanged.connect(self._update_ui)
        if profiles:
            self.profile_combo.currentTextChanged.connect(self._show_meta)
            self._show_meta(profiles[0])
        else:
            self.mode.setCurrentIndex(0)
            self._update_ui()

        self._pm = ProfileManager()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_ui(self) -> None:
        if not self.profile_combo.count():
            self.name_edit.setEnabled(True)
            self.profile_combo.setEnabled(False)
            self.info_label.setText("Create your first gaze profile. Calibration takes 1–5 minutes.")
            return

        mode = self.mode.currentIndex()
        is_new = mode == MODE_NEW
        self.profile_combo.setEnabled(not is_new)
        self.name_edit.setEnabled(is_new)

        hints = {
            MODE_USE_AS_IS: "Use saved model immediately. Recalibrate if accuracy drifts.",
            MODE_CALIBRATE: "Full calibration session with a freshly trained model.",
            MODE_RETRAIN: "Append new samples to existing data and retrain on combined set.",
            MODE_NEW: "Create a new gaze profile.",
        }
        self.info_label.setText(hints.get(mode, ""))
        if not is_new and self.profile_combo.currentText():
            self._show_meta(self.profile_combo.currentText())

    def _show_meta(self, name: str) -> None:
        if not name:
            return
        meta = self._pm.load_meta(name)
        err = f"{meta.last_avg_error_px:.1f} px" if meta.last_avg_error_px else "N/A"
        self.info_label.setText(
            f"Trained on {meta.sessions} sessions, {meta.total_samples} total samples, "
            f"last validated avg error {err}"
        )

    def _validate(self) -> None:
        if self.is_new():
            name = self.name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Name required", "Enter a profile name.")
                return
            if name in self._pm.list_profiles():
                QMessageBox.warning(self, "Name taken", f"Profile '{name}' already exists. Pick another name.")
                return
        elif not self.profile_combo.currentText():
            QMessageBox.warning(self, "Profile required", "Select a profile.")
            return
        self.accept()

    def selected_profile(self) -> str:
        if self.is_new():
            return self.name_edit.text().strip()
        return self.profile_combo.currentText()

    def is_new(self) -> bool:
        if not self.profile_combo.count():
            return True
        if self.mode.count() == 1:
            return True
        return self.mode.currentIndex() == MODE_NEW

    def skip_calibration(self) -> bool:
        return self.mode.count() > 1 and self.mode.currentIndex() == MODE_USE_AS_IS

    def is_retrain(self) -> bool:
        return self.mode.count() > 1 and self.mode.currentIndex() == MODE_RETRAIN

    def needs_calibration(self) -> bool:
        return self.is_new() or self.is_retrain() or (
            self.mode.count() > 1 and self.mode.currentIndex() == MODE_CALIBRATE
        )


class ScorecardDialog(QDialog):
    def __init__(self, meta: ProfileMeta, avg_px: float, max_px: float, samples: int, grade: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gazer — Results")
        self.setMinimumWidth(420)
        self.result_action = "done"

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<h2>Grade: {grade or 'N/A'}</h2>"))

        stats = QLabel(
            f"Average Error: <b>{avg_px:.1f} px</b><br>"
            f"Max Error: <b>{max_px:.1f} px</b><br>"
            f"Valid Samples: <b>{samples}</b><br><br>"
            f"Profile <b>{meta.name}</b>: {meta.sessions} sessions, "
            f"{meta.total_samples} total samples"
        )
        stats.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(stats)

        btn_row = QHBoxLayout()
        done_btn = QPushButton("Done — Start Cursor Control")
        retrain_btn = QPushButton("Retrain (add more time)")
        save_btn = QPushButton("Save Profile")

        done_btn.clicked.connect(lambda: self._finish("done"))
        retrain_btn.clicked.connect(lambda: self._finish("retrain"))
        save_btn.clicked.connect(lambda: self._finish("save"))

        btn_row.addWidget(done_btn)
        btn_row.addWidget(retrain_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _finish(self, action: str) -> None:
        self.result_action = action
        self.accept()
