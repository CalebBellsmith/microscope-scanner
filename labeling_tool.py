"""
Training data labeler for the quality classifier (labeling_tool.py)
====================================================================
Run via train.bat or:  py -3.12 labeling_tool.py

PURPOSE
───────
Helps you build the training dataset for the ML quality classifier
(model.pt / model.onnx used by capture_pipeline to decide whether to nudge).

The camera live-feeds to the screen.  Point the microscope at a slide,
press Space to freeze a frame, then label it as GOOD or BAD.
Saved images land in:
  training_data/good/  — clean slide area (horizontal scratches are fine)
  training_data/bad/   — dust, debris, watermarks, non-horizontal artefacts

Once you have ≥ 10 of each class, click "Train model.pt" to start training.
Training runs as a subprocess (train.py) to avoid Windows DLL issues with torch.

KEYBOARD SHORTCUTS
──────────────────
  Space   — freeze / capture the current frame (switches to FROZEN mode)
  D       — save frozen frame as GOOD  → training_data/good/<n>.jpg
  A       — save frozen frame as BAD   → training_data/bad/<n>.jpg
  W       — run model prediction on frozen frame + show defect centroid overlay
  S       — undo: delete the last saved file
  Enter   — dismiss prediction overlay (return to plain frozen frame)

CAMERA MODE
───────────
Analysis mode ON  — fixed exposure/gain/negative; greyscale output
                    (matches what the full scanner captures)
Analysis mode OFF — auto-exposure; full colour raw view for reference

The camera mode is mirrored in main.py so both GUIs show consistent images.
"""
import sys
import os
import threading

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QProgressBar, QStatusBar,
    QSpinBox, QCheckBox, QComboBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont

from camera import open_camera
from ml_inference import QualityClassifier

_HERE        = os.path.dirname(os.path.abspath(__file__))   # folder this script lives in
TRAINING_DIR = os.path.join(_HERE, "training_data")          # root for all label images
GOOD_DIR     = os.path.join(TRAINING_DIR, "good")            # clean/acceptable frames
BAD_DIR      = os.path.join(TRAINING_DIR, "bad")             # dusty/defective frames


# ── Cross-thread signals ──────────────────────────────────────────────────────
# Background threads (camera, inference, training) post results through these
# signals so Qt delivers them safely on the GUI thread.

class Signals(QObject):
    frame_ready    = pyqtSignal(np.ndarray)                        # new camera frame to display
    # Prediction result: (frame, cx_frac, cy_frac, label, confidence)
    # cx_frac / cy_frac are 0–1 image coordinates; -1 means no defect found
    overlay_ready  = pyqtSignal(np.ndarray, float, float, str, float)
    train_progress = pyqtSignal(int, int, float)   # (current_epoch, total_epochs, avg_loss)
    train_done     = pyqtSignal(float)             # validation accuracy 0.0–1.0
    train_error    = pyqtSignal(str)               # error message if training fails


# ── Main window ───────────────────────────────────────────────────────────────

class LabelingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Data Labeler")
        self.resize(1200, 760)

        self._sig = Signals()
        self._sig.frame_ready.connect(self._on_frame)
        self._sig.overlay_ready.connect(self._on_overlay)
        self._sig.train_progress.connect(self._on_train_progress)
        self._sig.train_done.connect(self._on_train_done)
        self._sig.train_error.connect(
            lambda e: self._status.showMessage(f"Training error: {e}")
        )

        self._camera        = None               # ToupTekCamera / OpenCVCamera / MSSCamera
        self._clf           = QualityClassifier(mode="hybrid")  # mode changed by dropdown
        self._frozen_frame  = None   # numpy frame; None = live preview mode, set = frozen
        self._last_saved    = None   # path of most recently saved file (used by undo)
        self._analysis_on   = True   # whether analysis exposure/gain settings are active

        self._preview_timer = QTimer()
        self._preview_timer.timeout.connect(self._update_preview)

        os.makedirs(GOOD_DIR, exist_ok=True)
        os.makedirs(BAD_DIR,  exist_ok=True)

        self._build_ui()
        self._connect_camera()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Connecting to camera…")

        # ── Left: camera feed ─────────────────────────────────────────────────
        left = QVBoxLayout()

        self._feed_label = QLabel("Camera feed will appear here")
        self._feed_label.setAlignment(Qt.AlignCenter)
        self._feed_label.setMinimumSize(700, 560)
        self._feed_label.setStyleSheet(
            "background:#111; color:#888; border:1px solid #444;"
        )
        left.addWidget(self._feed_label, stretch=1)

        # Analysis mode ON / OFF — two explicit buttons so state is always clear
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Camera mode:")
        mode_lbl.setStyleSheet("color:#aaa;")
        mode_row.addWidget(mode_lbl)

        self._analysis_on_btn  = QPushButton("🔬  Analysis ON")
        self._analysis_off_btn = QPushButton("👁  Raw OFF")

        for btn in (self._analysis_on_btn, self._analysis_off_btn):
            btn.setFocusPolicy(Qt.NoFocus)   # prevent Space from clicking buttons
            btn.setFixedHeight(32)
            mode_row.addWidget(btn)

        self._analysis_on_btn.clicked.connect(lambda: self._set_analysis_mode(True))
        self._analysis_off_btn.clicked.connect(lambda: self._set_analysis_mode(False))
        left.addLayout(mode_row)

        # ── Analysis adjusters (only active when analysis mode is ON) ─────────
        adj_row = QHBoxLayout()

        adj_row.addWidget(QLabel("Exposure (ms):"))
        self._expo_spin = QSpinBox()
        self._expo_spin.setFocusPolicy(Qt.NoFocus)
        self._expo_spin.setRange(10, 5000)
        self._expo_spin.setValue(1300)
        self._expo_spin.setSuffix(" ms")
        self._expo_spin.setFixedWidth(90)
        self._expo_spin.valueChanged.connect(self._apply_analysis_settings)
        adj_row.addWidget(self._expo_spin)

        adj_row.addSpacing(12)
        adj_row.addWidget(QLabel("Gain:"))
        self._gain_spin = QSpinBox()
        self._gain_spin.setFocusPolicy(Qt.NoFocus)
        self._gain_spin.setRange(100, 3200)
        self._gain_spin.setValue(300)
        self._gain_spin.setSingleStep(50)
        self._gain_spin.setFixedWidth(75)
        self._gain_spin.valueChanged.connect(self._apply_analysis_settings)
        adj_row.addWidget(self._gain_spin)

        adj_row.addSpacing(12)
        self._negative_chk = QCheckBox("Negative")
        self._negative_chk.setFocusPolicy(Qt.NoFocus)
        self._negative_chk.setChecked(True)
        self._negative_chk.stateChanged.connect(self._apply_analysis_settings)
        adj_row.addWidget(self._negative_chk)

        adj_row.addStretch()
        left.addLayout(adj_row)

        # Store adjuster widgets so we can enable/disable them as a group
        self._adj_widgets = [
            self._expo_spin, self._gain_spin, self._negative_chk,
        ]
        self._set_analysis_mode(True)   # start in analysis mode (must be after _adj_widgets)

        root.addLayout(left, stretch=3)

        # ── Right: controls ───────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)
        right.setContentsMargins(10, 10, 10, 10)

        # Label counts
        counts_box = QGroupBox("Label counts")
        counts_lay = QVBoxLayout(counts_box)
        big = QFont(); big.setPointSize(14); big.setBold(True)
        self._good_lbl = QLabel("Good:  0")
        self._bad_lbl  = QLabel("Bad:   0")
        for w, color in [(self._good_lbl, "#2E7D32"), (self._bad_lbl, "#C62828")]:
            w.setFont(big)
            w.setStyleSheet(f"color:{color};")
            counts_lay.addWidget(w)
        right.addWidget(counts_box)

        # State indicator
        self._state_lbl = QLabel("● LIVE")
        self._state_lbl.setFont(big)
        self._state_lbl.setAlignment(Qt.AlignCenter)
        self._state_lbl.setStyleSheet("color:#888; padding:8px;")
        right.addWidget(self._state_lbl)

        # Prediction / direction result
        self._pred_lbl = QLabel("")
        self._pred_lbl.setAlignment(Qt.AlignCenter)
        self._pred_lbl.setWordWrap(True)
        self._pred_lbl.setStyleSheet("font-size:13px;")
        right.addWidget(self._pred_lbl)

        # Key guide
        keys_box = QGroupBox("Controls")
        keys_lay = QVBoxLayout(keys_box)
        for key, desc, color in [
            ("SPACE", "Capture frame",    "#555555"),
            ("D",     "Label: GOOD ✓",   "#2E7D32"),
            ("A",     "Label: BAD ✗",    "#C62828"),
            ("W",     "Model prediction","#1565C0"),
            ("S",     "Undo last label", "#E65100"),
        ]:
            row = QHBoxLayout()
            k = QLabel(key)
            k.setFixedWidth(55)
            k.setAlignment(Qt.AlignCenter)
            k.setStyleSheet(
                f"background:{color}; color:white; border-radius:4px;"
                " padding:4px 2px; font-weight:bold;"
            )
            row.addWidget(k)
            row.addWidget(QLabel(desc))
            row.addStretch()
            keys_lay.addLayout(row)
        right.addWidget(keys_box)

        right.addStretch()

        # ── Classifier mode selector ──────────────────────────────────────────
        # Lets you compare rule-based vs ML vs hybrid live by pressing W.
        # Changing the dropdown updates self._clf.mode immediately.
        clf_box = QGroupBox("Classifier (W key)")
        clf_lay = QVBoxLayout(clf_box)
        clf_lay.setSpacing(4)

        self._clf_combo = QComboBox()
        self._clf_combo.setFocusPolicy(Qt.NoFocus)
        self._clf_combo.addItem("Hybrid  (rules + ML)",  "hybrid")
        self._clf_combo.addItem("Rules only  (shape)",   "rules")
        self._clf_combo.addItem("ML only  (trained model)", "ml")
        self._clf_combo.currentIndexChanged.connect(self._on_clf_mode_changed)

        clf_lay.addWidget(self._clf_combo)

        # Brief explanation of each mode shown as a hint label
        self._clf_hint = QLabel("")
        self._clf_hint.setWordWrap(True)
        self._clf_hint.setStyleSheet("color:#888; font-size:10px;")
        clf_lay.addWidget(self._clf_hint)

        right.addWidget(clf_box)
        self._on_clf_mode_changed(0)   # populate hint for default selection

        # Train section
        train_box = QGroupBox("Train model")
        train_lay = QVBoxLayout(train_box)

        self._train_btn = QPushButton("▶  Train model.pt")
        self._train_btn.setFocusPolicy(Qt.NoFocus)
        self._train_btn.setStyleSheet(
            "QPushButton         { background:#1565C0; color:white;"
            "  font-weight:bold; padding:8px; border-radius:4px; }"
            "QPushButton:disabled{ background:#333; color:#777; }"
        )
        self._train_btn.clicked.connect(self._start_training)
        train_lay.addWidget(self._train_btn)

        self._train_bar = QProgressBar()
        self._train_bar.setVisible(False)
        train_lay.addWidget(self._train_bar)

        self._train_result = QLabel("")
        self._train_result.setAlignment(Qt.AlignCenter)
        self._train_result.setWordWrap(True)
        train_lay.addWidget(self._train_result)

        right.addWidget(train_box)
        root.addLayout(right, stretch=1)

    # ── Classifier mode ───────────────────────────────────────────────────────

    _CLF_HINTS = {
        "hybrid": "Rules handle clear cases; ML is consulted when the shape test is uncertain. Best default.",
        "rules":  "Shape analysis only — horizontal lines pass, round blobs/dust fail. No model needed.",
        "ml":     "Trained MobileNetV3 model only. Requires model.onnx. Use to compare against rules.",
    }

    def _on_clf_mode_changed(self, _idx: int = 0):
        """Update classifier mode and hint text when the dropdown changes."""
        mode = self._clf_combo.currentData()
        self._clf.mode = mode
        self._clf_hint.setText(self._CLF_HINTS.get(mode, ""))

    # ── Camera ────────────────────────────────────────────────────────────────

    def _connect_camera(self):
        try:
            self._camera = open_camera()
            # Re-apply analysis settings now that camera is open
            # (_set_analysis_mode was called during _build_ui when camera was None)
            self._set_analysis_mode(self._analysis_on)
            # Wait for camera to stabilise at new exposure before showing frames
            QTimer.singleShot(2000, lambda: self._preview_timer.start(80))
            self._status.showMessage(
                "Camera connected.  Press Space to capture a frame."
            )
        except Exception as e:
            self._status.showMessage(f"Camera error: {e}  —  running without camera.")
        self._refresh_counts()

    def _update_preview(self):
        if self._frozen_frame is not None:
            return
        if self._camera is None:
            return
        frame = self._camera.grab()
        if frame is not None:
            self._sig.frame_ready.emit(frame)

    def _set_analysis_mode(self, on: bool):
        """Switch camera mode, update button styles, enable/disable adjusters."""
        self._analysis_on = on
        active   = "background:#1565C0; color:white; font-weight:bold; border-radius:4px;"
        inactive = "background:#333;    color:#888; font-weight:normal; border-radius:4px;"
        self._analysis_on_btn.setStyleSheet( active   if on  else inactive)
        self._analysis_off_btn.setStyleSheet(inactive if on  else active)
        for w in self._adj_widgets:
            w.setEnabled(on)
        if on:
            # Set the flag on the camera first so greyscale conversion fires
            if self._camera and hasattr(self._camera, "set_analysis_mode"):
                self._camera.set_analysis_mode(True)
            self._apply_analysis_settings()
        else:
            # Hand full control back to the camera's auto defaults
            if self._camera and hasattr(self._camera, "set_analysis_mode"):
                self._camera.set_analysis_mode(False)
        self.setFocus()

    def _apply_analysis_settings(self, *_):
        """Push current spinbox/checkbox values to the camera."""
        if not getattr(self, "_analysis_on", True):
            return
        cam = self._camera
        if cam is None or not hasattr(cam, "_cam") or cam._cam is None:
            return
        expo_us = self._expo_spin.value() * 1000   # ms → µs
        gain    = self._gain_spin.value()
        neg     = self._negative_chk.isChecked()
        try:
            cam._cam.put_AutoExpoEnable(False)
            cam._cam.put_ExpoTime(expo_us)
            cam._cam.put_ExpoAGain(gain)
            cam._preview_expo_us = expo_us   # keep grab_fresh in sync
        except Exception:
            pass
        try:
            cam._cam.put_Negative(neg)
            cam._negative_fallback = False
        except Exception:
            cam._negative_fallback = neg   # software fallback

    # ── Frame display ─────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        self._show_frame(frame)

    def _on_overlay(self, frame: np.ndarray,
                    cx_frac: float, cy_frac: float,
                    label: str, conf: float):
        """Draw defect centroid crosshair + prediction text on the frozen frame."""
        import cv2
        display = frame.copy()
        color = (50, 200, 50) if label == "good" else (220, 60, 60)

        if cx_frac >= 0:
            h, w = display.shape[:2]
            cx = int(cx_frac * w)
            cy = int(cy_frac * h)
            cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 44, 3)
            cv2.circle(display, (cx, cy), 22, color, 2)

        cv2.putText(
            display,
            f"{label.upper()}  {conf*100:.0f}%",
            (15, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA,
        )
        self._show_frame(display)

        color_hex = "#2E7D32" if label == "good" else "#C62828"
        direction = _direction_hint(cx_frac, cy_frac) if cx_frac >= 0 else ""
        html = f"Model: <b>{label.upper()}</b> ({conf*100:.0f}%)"
        if direction:
            html += f"<br><small>Defect detected → nudge <b>{direction}</b></small>"
        self._pred_lbl.setText(html)
        self._pred_lbl.setStyleSheet(f"color:{color_hex}; font-size:13px;")

    def _show_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self._feed_label.width(), self._feed_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self._feed_label.setPixmap(pix)

    # ── Key events ────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        k = event.key()
        if   k == Qt.Key_Space:                          self._capture_frame()
        elif k == Qt.Key_D:                              self._label("good")
        elif k == Qt.Key_A:                              self._label("bad")
        elif k == Qt.Key_W:                              self._predict()
        elif k == Qt.Key_S:                              self._undo()
        elif k in (Qt.Key_Return, Qt.Key_Enter):         self._dismiss_prediction()
        else: super().keyPressEvent(event)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _capture_frame(self):
        if self._camera is None:
            self._status.showMessage("No camera connected.")
            return
        frame = (self._camera.grab_fresh()
                 if hasattr(self._camera, "grab_fresh")
                 else self._camera.grab())
        if frame is None:
            self._status.showMessage("No frame available — try again.")
            return
        self._frozen_frame = frame
        self._sig.frame_ready.emit(frame)
        self._pred_lbl.setText("")
        self._state_lbl.setText("● FROZEN — label it")
        self._state_lbl.setStyleSheet(
            "color:#E65100; font-weight:bold; padding:8px;"
        )
        self._status.showMessage(
            "Frame captured.  D = good  ·  A = bad  ·  W = predict  ·  S = undo"
        )

    def _label(self, label: str):
        if self._frozen_frame is None:
            self._status.showMessage("Capture a frame first (Space).")
            return
        from PIL import Image
        target = GOOD_DIR if label == "good" else BAD_DIR
        n      = len([f for f in os.listdir(target)
                      if f.lower().endswith((".jpg", ".png"))])
        path   = os.path.join(target, f"{n+1:05d}.jpg")
        Image.fromarray(self._frozen_frame).save(path, quality=95)
        self._last_saved   = path
        self._frozen_frame = None
        self._refresh_counts()

        color = "#2E7D32" if label == "good" else "#C62828"
        self._state_lbl.setText(f"● Saved as {label.upper()}")
        self._state_lbl.setStyleSheet(
            f"color:{color}; font-weight:bold; padding:8px;"
        )
        self._status.showMessage(f"Saved {label.upper()}: {os.path.basename(path)}")
        self.setFocus()
        QTimer.singleShot(1200, self._back_to_live)

    def _predict(self):
        if self._frozen_frame is None:
            self._status.showMessage("Capture a frame first (Space).")
            return
        frame = self._frozen_frame
        threading.Thread(
            target=self._run_predict, args=(frame,), daemon=True
        ).start()

    def _run_predict(self, frame: np.ndarray):
        label, conf      = self._clf.predict(frame)
        cx_frac, cy_frac = _defect_centroid(frame)
        self._sig.overlay_ready.emit(frame, cx_frac, cy_frac, label, conf)

    def _dismiss_prediction(self):
        """Clear prediction overlay — show plain frozen frame again (or go live)."""
        self._pred_lbl.setText("")
        if self._frozen_frame is not None:
            self._sig.frame_ready.emit(self._frozen_frame)   # redraw without crosshair
            self._status.showMessage(
                "Prediction dismissed.  D = good  ·  A = bad  ·  S = undo"
            )

    def _undo(self):
        if self._last_saved is None:
            self._status.showMessage("Nothing to undo.")
            return
        try:
            os.remove(self._last_saved)
            self._status.showMessage(f"Deleted: {os.path.basename(self._last_saved)}")
            self._last_saved = None
            self._refresh_counts()
        except Exception as e:
            self._status.showMessage(f"Undo failed: {e}")

    def _back_to_live(self):
        self._state_lbl.setText("● LIVE")
        self._state_lbl.setStyleSheet("color:#888; padding:8px;")

    def _refresh_counts(self):
        def _n(d):
            return len([f for f in os.listdir(d)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]) \
                   if os.path.isdir(d) else 0
        self._good_lbl.setText(f"Good:  {_n(GOOD_DIR)}")
        self._bad_lbl.setText( f"Bad:   {_n(BAD_DIR)}")

    # ── Training ──────────────────────────────────────────────────────────────

    def _start_training(self):
        def _n(d):
            return len([f for f in os.listdir(d)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]) \
                   if os.path.isdir(d) else 0
        gn, bn = _n(GOOD_DIR), _n(BAD_DIR)
        if gn < 10 or bn < 10:
            self._train_result.setText(
                f"Need ≥ 10 of each class.\nHave: {gn} good, {bn} bad."
            )
            return
        self._train_btn.setEnabled(False)
        self._train_bar.setVisible(True)
        self._train_bar.setValue(0)
        self._train_result.setText("Training…")
        # Run train.py as a subprocess — avoids PyQt5 thread DLL init issues with torch
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        import subprocess, sys, re
        try:
            proc = subprocess.Popen(
                [sys.executable, os.path.join(_HERE, "train.py")],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            # Parse stdout lines for progress and final accuracy
            acc = None
            for line in proc.stdout:
                line = line.strip()
                # "  Epoch  3/15  loss=0.4231"
                m = re.match(r"Epoch\s+(\d+)/(\d+)\s+loss=([\d.]+)", line)
                if m:
                    ep, tot, loss = int(m[1]), int(m[2]), float(m[3])
                    self._sig.train_progress.emit(ep, tot, loss)
                # "Val accuracy: 91.2%"
                m2 = re.match(r"Val accuracy:\s+([\d.]+)%", line)
                if m2:
                    acc = float(m2[1]) / 100.0
            proc.wait()
            if proc.returncode != 0:
                self._sig.train_error.emit(f"train.py exited with code {proc.returncode}")
            elif acc is not None:
                self._sig.train_done.emit(acc)
            else:
                self._sig.train_error.emit("Training finished but no accuracy reported.")
        except Exception as e:
            self._sig.train_error.emit(str(e))

    def _on_train_progress(self, epoch: int, total: int, loss: float):
        self._train_bar.setMaximum(total)
        self._train_bar.setValue(epoch)
        self._train_result.setText(f"Epoch {epoch}/{total}  loss={loss:.4f}")

    def _on_train_done(self, acc: float):
        self._train_btn.setEnabled(True)
        self._train_bar.setValue(self._train_bar.maximum())
        self._train_result.setText(
            f"✓ Done!  Val accuracy: {acc*100:.1f}%\nSaved to model.pt"
        )
        self._clf = QualityClassifier()   # reload fresh weights

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._preview_timer.stop()
        if self._camera:
            self._camera.close()
        super().closeEvent(event)


# ── Helpers (shared with capture_pipeline) ────────────────────────────────────

def _defect_centroid(frame: np.ndarray) -> tuple[float, float]:
    """
    Locate the largest dark blob (dust / debris) in frame.
    Returns (cx_frac, cy_frac) in 0–1 image coordinates,
    or (-1.0, -1.0) if no significant defect found.
    """
    import cv2
    gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mean  = float(gray.mean())
    std   = float(gray.std())
    thr   = max(0, int(mean - 2.0 * std))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return -1.0, -1.0

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return -1.0, -1.0

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return -1.0, -1.0

    h, w = frame.shape[:2]
    return M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h


def _direction_hint(cx_frac: float, cy_frac: float) -> str:
    """Human-readable stage nudge direction (away from defect centroid)."""
    if cx_frac < 0:
        return ""
    dx, dy = cx_frac - 0.5, cy_frac - 0.5
    parts  = []
    if abs(dy) > 0.15:
        parts.append("DOWN" if dy < 0 else "UP")
    if abs(dx) > 0.15:
        parts.append("LEFT" if dx > 0 else "RIGHT")
    return " + ".join(parts) if parts else "centre (any direction)"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LabelingWindow()
    win.show()
    sys.exit(app.exec_())
