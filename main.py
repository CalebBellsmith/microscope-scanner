"""
Automated Microscope Slide Scanner
Double-click run.bat (Windows) or run.sh (Mac) to launch.

Workflow:
  1. App opens → user picks / creates set folder  (e.g. A-08/)
  2. User selects profile + mode, clicks Go
  3. 30 images captured → dialog: "Name this leg:" (free text, typically FR/FL/BR/BL)
  4. Images saved to  set_folder / leg_name / 001.jpg … 030.jpg
  5. Analysis starts in background; user swaps slide and clicks Go again
  6. Repeat for each leg; re-run any leg to overwrite its results
  7. Click Finish → waits for all running analyses → writes Excel
"""
import sys
import os
import threading

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSpinBox,
    QPushButton, QProgressBar, QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QLineEdit, QFileDialog, QMessageBox, QStatusBar,
    QRadioButton, QButtonGroup, QInputDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QFrame,
    QScrollArea, QCheckBox, QSlider,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont

from camera import open_camera
from motor import MotorController
from ml_inference import QualityClassifier
from capture_pipeline import CapturePipeline
from analysis_pipeline import (
    AnalysisPipeline, write_new_format, write_legacy_format,
    LEGS, EXPECTED_IMAGES,
)

DEFAULT_BASE = os.path.join(os.path.expanduser("~"), "Downloads", "alchemy", "abrasion")

PROFILES = [
    ("Standard  3 × 10  (30 images)", 3, 10),
    ("Small     2 × 5   (10 images)", 2, 5),
    ("Medium    4 × 10  (40 images)", 4, 10),
    ("Large     5 × 12  (60 images)", 5, 12),
    ("Custom",                        None, None),
]
_CUSTOM_IDX = len(PROFILES) - 1

MODE_CAPTURE_ONLY    = 0
MODE_ANALYZE_ONLY    = 1
MODE_CAPTURE_ANALYZE = 2

# Leg table column indices
COL_LEG    = 0
COL_IMAGES = 1
COL_STATUS = 2
COL_RERUN  = 3

STATUS_COLORS = {
    "—":         QColor("#888888"),
    "Capturing": QColor("#1565C0"),
    "Analyzing": QColor("#E65100"),
    "Done":      QColor("#2E7D32"),
    "Error":     QColor("#C62828"),
}


# ── Signals ───────────────────────────────────────────────────────────────────

class Signals(QObject):
    frame_ready         = pyqtSignal(np.ndarray)
    capture_progress    = pyqtSignal(int, int)
    analysis_progress   = pyqtSignal(int, int)
    capture_done        = pyqtSignal()
    leg_analysis_done   = pyqtSignal(str, list)   # (leg_name, results)
    error               = pyqtSignal(str)
    status_msg          = pyqtSignal(str)


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Slide Scanner")
        self.resize(1300, 800)

        self._sig = Signals()
        self._sig.frame_ready.connect(self._on_frame)
        self._sig.capture_progress.connect(self._on_capture_progress)
        self._sig.analysis_progress.connect(self._on_analysis_progress)
        self._sig.capture_done.connect(self._on_capture_done)
        self._sig.leg_analysis_done.connect(self._on_leg_analysis_done)
        self._sig.error.connect(self._on_error)
        self._sig.status_msg.connect(lambda m: self._statusbar.showMessage(m))

        self._camera  = None
        self._motor   = None
        self._clf     = QualityClassifier()
        self._capture_pipeline  = None
        self._preview_timer = QTimer()
        self._preview_timer.timeout.connect(self._update_preview)
        self._analysis_on = True

        self._set_dir: str | None = None
        # {leg_name: list_of_result_dicts}  — None means analysis still running
        self._leg_results: dict = {}
        self._analyzing_legs: set = set()
        self._finish_waiting = False
        self._pending_leg_name: str | None = None   # leg being captured right now

        self._build_ui()
        self._prompt_set_folder()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Select a set folder to begin.")

        # ── Left: camera feed ─────────────────────────────────────────────────
        left = QVBoxLayout()
        self._feed_label = QLabel("Camera feed will appear here")
        self._feed_label.setAlignment(Qt.AlignCenter)
        self._feed_label.setMinimumSize(700, 560)
        self._feed_label.setStyleSheet(
            "background:#111; color:#888; border:1px solid #444;"
        )
        left.addWidget(self._feed_label, stretch=1)

        # Camera mode ON / OFF buttons
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Camera mode:"))
        self._analysis_on_btn  = QPushButton("🔬  Analysis ON")
        self._analysis_off_btn = QPushButton("👁  Raw OFF")
        for btn in (self._analysis_on_btn, self._analysis_off_btn):
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setFixedHeight(32)
            mode_row.addWidget(btn)
        self._analysis_on_btn.clicked.connect(lambda: self._set_analysis_mode(True))
        self._analysis_off_btn.clicked.connect(lambda: self._set_analysis_mode(False))
        left.addLayout(mode_row)

        # Exposure / gain / negative adjusters (enabled only in analysis mode)
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

        self._adj_widgets = [self._expo_spin, self._gain_spin, self._negative_chk]
        self._set_analysis_mode(True)

        root.addLayout(left, stretch=3)

        # ── Right: scrollable controls ────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(420)
        scroll.setFrameShape(QFrame.NoFrame)

        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setSpacing(8)
        right.setContentsMargins(6, 6, 6, 6)
        scroll.setWidget(right_widget)
        root.addWidget(scroll, stretch=1)

        def _row(lbl, widget):
            h = QHBoxLayout()
            lbl_w = QLabel(lbl)
            lbl_w.setMinimumWidth(120)
            h.addWidget(lbl_w)
            h.addWidget(widget)
            return h

        # ── Set folder ────────────────────────────────────────────────────────
        set_box = QGroupBox("Set folder")
        set_lay = QHBoxLayout(set_box)
        self._set_label = QLabel("(none selected)")
        self._set_label.setWordWrap(True)
        bold = QFont(); bold.setBold(True)
        self._set_label.setFont(bold)
        set_lay.addWidget(self._set_label, stretch=1)
        change_btn = QPushButton("Change")
        change_btn.setFixedWidth(65)
        change_btn.clicked.connect(self._prompt_set_folder)
        set_lay.addWidget(change_btn)
        right.addWidget(set_box)

        # ── Mode ─────────────────────────────────────────────────────────────
        mode_box = QGroupBox("Mode")
        mode_lay = QVBoxLayout(mode_box)
        mode_lay.setSpacing(3)
        self._mode_group = QButtonGroup(self)
        self._btn_cap_analyze = QRadioButton("Capture + Analyze simultaneously")
        self._btn_cap_only    = QRadioButton("Capture only  (analyze later)")
        self._btn_ana_only    = QRadioButton("Analyze only  (existing folder)")
        self._btn_cap_analyze.setChecked(True)
        for btn, mid in [(self._btn_cap_analyze, MODE_CAPTURE_ANALYZE),
                         (self._btn_cap_only,    MODE_CAPTURE_ONLY),
                         (self._btn_ana_only,    MODE_ANALYZE_ONLY)]:
            self._mode_group.addButton(btn, mid)
            mode_lay.addWidget(btn)
        self._mode_group.buttonToggled.connect(self._on_mode_changed)
        right.addWidget(mode_box)

        # ── Connection ────────────────────────────────────────────────────────
        self._conn_box = QGroupBox("Connection")
        conn_lay = QVBoxLayout(self._conn_box)
        conn_lay.setSpacing(4)

        self._cam_port_edit = QLineEdit("auto")
        self._cam_port_edit.setPlaceholderText("auto  (ToupTek) or 0, 1… (OpenCV)")
        conn_lay.addLayout(_row("Camera:", self._cam_port_edit))

        self._esp_port_edit = QLineEdit("auto")
        self._esp_port_edit.setPlaceholderText("auto  or  COM3 / /dev/cu.usbserial…")
        conn_lay.addLayout(_row("ESP32 (serial):", self._esp_port_edit))

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect)
        conn_lay.addWidget(self._connect_btn)
        right.addWidget(self._conn_box)

        # ── Manual joystick mode ──────────────────────────────────────────────
        manual_box = QGroupBox("Manual control")
        manual_lay = QVBoxLayout(manual_box)
        manual_lay.setSpacing(4)

        self._manual_chk = QCheckBox("Enable arrow-key joystick")
        self._manual_chk.setFocusPolicy(Qt.NoFocus)
        self._manual_chk.setToolTip(
            "When checked: ← → move X stepper, ↑ ↓ move Y stepper"
        )
        manual_lay.addWidget(self._manual_chk)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step size:"))
        self._manual_step_spin = QSpinBox()
        self._manual_step_spin.setFocusPolicy(Qt.NoFocus)
        self._manual_step_spin.setRange(1, 2048)
        self._manual_step_spin.setValue(100)
        self._manual_step_spin.setSuffix(" steps")
        step_row.addWidget(self._manual_step_spin)
        step_row.addStretch()
        manual_lay.addLayout(step_row)

        manual_lay.addWidget(QLabel(
            "← → = X axis    ↑ ↓ = Y axis",
        ))
        right.addWidget(manual_box)

        # ── Scan profile ──────────────────────────────────────────────────────
        self._profile_box = QGroupBox("Scan profile")
        profile_lay = QVBoxLayout(self._profile_box)
        profile_lay.setSpacing(4)
        self._profile_combo = QComboBox()
        for label, _, _ in PROFILES:
            self._profile_combo.addItem(label)
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        profile_lay.addWidget(self._profile_combo)

        self._rows_spin = QSpinBox(); self._rows_spin.setRange(1, 50); self._rows_spin.setValue(3)
        self._cols_spin = QSpinBox(); self._cols_spin.setRange(1, 50); self._cols_spin.setValue(10)
        self._x_spin    = QSpinBox(); self._x_spin.setRange(1, 5000); self._x_spin.setValue(200)
        self._y_spin    = QSpinBox(); self._y_spin.setRange(1, 500);  self._y_spin.setValue(50)
        self._rows_spin.valueChanged.connect(self._on_spinbox_changed)
        self._cols_spin.valueChanged.connect(self._on_spinbox_changed)
        self._total_label = QLabel()
        self._update_total_label()

        profile_lay.addLayout(_row("Rows:", self._rows_spin))
        profile_lay.addLayout(_row("Columns:", self._cols_spin))
        profile_lay.addLayout(_row("X spacing (steps):", self._x_spin))
        profile_lay.addLayout(_row("Y spacing (units):", self._y_spin))
        profile_lay.addWidget(self._total_label)

        # Quality threshold slider
        profile_lay.addWidget(QLabel("Quality threshold (nudge sensitivity):"))
        thresh_row = QHBoxLayout()
        self._thresh_label_lo = QLabel("Lenient")
        self._thresh_label_lo.setStyleSheet("color:#888; font-size:10px;")
        self._thresh_slider = QSlider(Qt.Horizontal)
        self._thresh_slider.setRange(1, 9)   # maps to 0.1 – 0.9
        self._thresh_slider.setValue(5)       # default 0.5
        self._thresh_slider.setTickPosition(QSlider.TicksBelow)
        self._thresh_slider.setTickInterval(1)
        self._thresh_val_lbl = QLabel("0.5")
        self._thresh_val_lbl.setFixedWidth(28)
        self._thresh_slider.valueChanged.connect(
            lambda v: self._thresh_val_lbl.setText(f"{v/10:.1f}")
        )
        self._thresh_label_hi = QLabel("Strict")
        self._thresh_label_hi.setStyleSheet("color:#888; font-size:10px;")
        thresh_row.addWidget(self._thresh_label_lo)
        thresh_row.addWidget(self._thresh_slider, stretch=1)
        thresh_row.addWidget(self._thresh_label_hi)
        thresh_row.addWidget(self._thresh_val_lbl)
        profile_lay.addLayout(thresh_row)

        right.addWidget(self._profile_box)

        # ── Run Slide button (primary action) ─────────────────────────────────
        self._go_btn = QPushButton("▶  Run Slide")
        self._go_btn.setEnabled(False)
        self._go_btn.setFixedHeight(46)
        self._go_btn.setStyleSheet(
            "QPushButton { background:#1565C0; color:white; font-size:14px; font-weight:bold; }"
            "QPushButton:disabled { background:#555; color:#999; }"
        )
        self._go_btn.clicked.connect(self._on_go)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setFixedHeight(46)
        self._stop_btn.clicked.connect(self._on_stop)

        run_row = QHBoxLayout()
        run_row.addWidget(self._go_btn, stretch=3)
        run_row.addWidget(self._stop_btn, stretch=1)
        right.addLayout(run_row)

        # ── Legs table ────────────────────────────────────────────────────────
        leg_box = QGroupBox("Legs  (each named after capture)")
        leg_lay = QVBoxLayout(leg_box)
        leg_lay.setSpacing(4)

        self._leg_table = QTableWidget(0, 4)
        self._leg_table.setHorizontalHeaderLabels(["Leg", "Images", "Status", ""])
        self._leg_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._leg_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._leg_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._leg_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._leg_table.setSelectionMode(QTableWidget.NoSelection)
        self._leg_table.setFixedHeight(150)
        leg_lay.addWidget(self._leg_table)

        self._leg_placeholder = QLabel(
            "After each slide is captured you will be prompted\n"
            "to name the leg (FR / FL / BR / BL).\n"
            "Progress and re-run controls appear here."
        )
        self._leg_placeholder.setAlignment(Qt.AlignCenter)
        self._leg_placeholder.setStyleSheet("color:#888; font-size:11px;")
        leg_lay.addWidget(self._leg_placeholder)
        right.addWidget(leg_box)

        # ── Export format ─────────────────────────────────────────────────────
        export_box = QGroupBox("Export format")
        export_lay = QVBoxLayout(export_box)
        export_lay.setSpacing(3)
        self._export_group = QButtonGroup(self)
        self._btn_new_fmt    = QRadioButton("New — single workbook, 5 tabs")
        self._btn_legacy_fmt = QRadioButton("Legacy — 3 files (MATLAB-compatible)")
        self._btn_new_fmt.setChecked(True)
        self._export_group.addButton(self._btn_new_fmt,    0)
        self._export_group.addButton(self._btn_legacy_fmt, 1)
        export_lay.addWidget(self._btn_new_fmt)
        export_lay.addWidget(self._btn_legacy_fmt)
        right.addWidget(export_box)

        # ── Progress ──────────────────────────────────────────────────────────
        prog_box = QGroupBox("Progress")
        prog_lay = QVBoxLayout(prog_box)
        prog_lay.setSpacing(3)
        self._cap_label = QLabel("Capture:")
        prog_lay.addWidget(self._cap_label)
        self._cap_bar = QProgressBar(); self._cap_bar.setValue(0)
        prog_lay.addWidget(self._cap_bar)
        self._ana_label = QLabel("Analysis (current leg):")
        prog_lay.addWidget(self._ana_label)
        self._ana_bar = QProgressBar(); self._ana_bar.setValue(0)
        prog_lay.addWidget(self._ana_bar)
        right.addWidget(prog_box)

        # ── Finish & Export ───────────────────────────────────────────────────
        self._finish_btn = QPushButton("Finish && Export Excel")
        self._finish_btn.setEnabled(False)
        self._finish_btn.setFixedHeight(44)
        self._finish_btn.setStyleSheet(
            "QPushButton { background:#2E7D32; color:white; font-size:13px; font-weight:bold; }"
            "QPushButton:disabled { background:#444; color:#777; }"
        )
        self._finish_btn.clicked.connect(self._on_finish)
        right.addWidget(self._finish_btn)
        right.addStretch()

        self._on_mode_changed()

    # ── Set folder ────────────────────────────────────────────────────────────

    def _prompt_set_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select or create set folder", DEFAULT_BASE
        )
        if path:
            self._set_dir = path
            self._set_label.setText(path)
            self._leg_results.clear()
            self._analyzing_legs.clear()
            self._leg_table.setRowCount(0)
            self._leg_placeholder.setVisible(True)
            self._finish_btn.setEnabled(False)
            self._statusbar.showMessage(f"Set folder: {path}")
            mode = self._mode_group.checkedId()
            if mode == MODE_ANALYZE_ONLY:
                self._go_btn.setEnabled(True)
            # pre-populate table with any legs already in the folder
            self._scan_existing_legs()

    def _scan_existing_legs(self):
        """If the set folder already has leg sub-folders, show them in the table."""
        if not self._set_dir:
            return
        for entry in sorted(os.listdir(self._set_dir)):
            full = os.path.join(self._set_dir, entry)
            if os.path.isdir(full):
                jpegs = [f for f in os.listdir(full)
                         if f.endswith(".jpg") and "overlay" not in f]
                self._upsert_leg_row(entry, len(jpegs), "—")

    # ── Camera mode ───────────────────────────────────────────────────────────

    def _set_analysis_mode(self, on: bool):
        self._analysis_on = on
        active   = "background:#1565C0; color:white; font-weight:bold; border-radius:4px;"
        inactive = "background:#333;    color:#888; font-weight:normal; border-radius:4px;"
        self._analysis_on_btn.setStyleSheet( active   if on else inactive)
        self._analysis_off_btn.setStyleSheet(inactive if on else active)
        for w in self._adj_widgets:
            w.setEnabled(on)
        if on:
            if self._camera and hasattr(self._camera, "set_analysis_mode"):
                self._camera.set_analysis_mode(True)
            self._apply_analysis_settings()
        else:
            if self._camera and hasattr(self._camera, "set_analysis_mode"):
                self._camera.set_analysis_mode(False)

    def _apply_analysis_settings(self, *_):
        if not self._analysis_on:
            return
        cam = self._camera
        if cam is None or not hasattr(cam, "_cam") or cam._cam is None:
            return
        expo_us = self._expo_spin.value() * 1000
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
            cam._negative_fallback = neg

    # ── Manual joystick ──────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if not self._manual_chk.isChecked():
            super().keyPressEvent(event)
            return
        if self._motor is None:
            self._statusbar.showMessage("Motor not connected — connect first.")
            super().keyPressEvent(event)
            return
        steps = self._manual_step_spin.value()
        key   = event.key()
        if   key == Qt.Key_Left:  self._motor.move("X", -steps)
        elif key == Qt.Key_Right: self._motor.move("X",  steps)
        elif key == Qt.Key_Up:    self._motor.move("Y", -steps)
        elif key == Qt.Key_Down:  self._motor.move("Y",  steps)
        else: super().keyPressEvent(event)

    # ── Profile helpers ───────────────────────────────────────────────────────

    def _on_profile_changed(self, idx):
        _, rows, cols = PROFILES[idx]
        if rows is not None:
            self._rows_spin.blockSignals(True)
            self._cols_spin.blockSignals(True)
            self._rows_spin.setValue(rows)
            self._cols_spin.setValue(cols)
            self._rows_spin.blockSignals(False)
            self._cols_spin.blockSignals(False)
        custom = rows is None
        self._rows_spin.setEnabled(custom)
        self._cols_spin.setEnabled(custom)
        self._update_total_label()

    def _on_spinbox_changed(self):
        self._profile_combo.blockSignals(True)
        self._profile_combo.setCurrentIndex(_CUSTOM_IDX)
        self._profile_combo.blockSignals(False)
        self._rows_spin.setEnabled(True)
        self._cols_spin.setEnabled(True)
        self._update_total_label()

    def _update_total_label(self):
        self._total_label.setText(
            f"Total images: {self._rows_spin.value() * self._cols_spin.value()}"
        )

    # ── Mode visibility ───────────────────────────────────────────────────────

    def _on_mode_changed(self, *_):
        mode    = self._mode_group.checkedId()
        capture = mode in (MODE_CAPTURE_ONLY, MODE_CAPTURE_ANALYZE)
        analyze = mode in (MODE_ANALYZE_ONLY, MODE_CAPTURE_ANALYZE)

        self._conn_box.setVisible(capture)
        self._profile_box.setVisible(capture)
        self._cap_label.setVisible(capture)
        self._cap_bar.setVisible(capture)
        self._ana_label.setVisible(analyze)
        self._ana_bar.setVisible(analyze)

        if mode == MODE_ANALYZE_ONLY and self._set_dir:
            self._go_btn.setEnabled(True)

    # ── Leg table helpers ─────────────────────────────────────────────────────

    def _upsert_leg_row(self, leg_name: str, image_count: int, status: str):
        """Insert or update a row in the leg status table."""
        for row in range(self._leg_table.rowCount()):
            if self._leg_table.item(row, COL_LEG) and \
               self._leg_table.item(row, COL_LEG).text() == leg_name:
                self._set_leg_row(row, leg_name, image_count, status)
                return
        row = self._leg_table.rowCount()
        self._leg_table.insertRow(row)
        self._set_leg_row(row, leg_name, image_count, status)
        self._leg_placeholder.setVisible(False)

    def _set_leg_row(self, row: int, leg_name: str, image_count: int, status: str):
        name_item = QTableWidgetItem(leg_name)
        name_item.setTextAlignment(Qt.AlignCenter)
        self._leg_table.setItem(row, COL_LEG, name_item)

        count_item = QTableWidgetItem(str(image_count))
        count_item.setTextAlignment(Qt.AlignCenter)
        self._leg_table.setItem(row, COL_IMAGES, count_item)

        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignCenter)
        color = STATUS_COLORS.get(status, QColor("#888888"))
        status_item.setForeground(color)
        f = QFont(); f.setBold(True)
        status_item.setFont(f)
        self._leg_table.setItem(row, COL_STATUS, status_item)

        # Re-run button (only shown when Done or Error)
        if status in ("Done", "Error"):
            btn = QPushButton("Re-run")
            btn.setFixedHeight(24)
            btn.clicked.connect(lambda _, n=leg_name: self._on_rerun(n))
            self._leg_table.setCellWidget(row, COL_RERUN, btn)
        else:
            self._leg_table.setCellWidget(row, COL_RERUN, None)

    def _set_leg_status(self, leg_name: str, status: str, image_count: int = None):
        for row in range(self._leg_table.rowCount()):
            if self._leg_table.item(row, COL_LEG) and \
               self._leg_table.item(row, COL_LEG).text() == leg_name:
                if image_count is not None:
                    ci = self._leg_table.item(row, COL_IMAGES)
                    if ci:
                        ci.setText(str(image_count))
                status_item = self._leg_table.item(row, COL_STATUS)
                if status_item:
                    status_item.setText(status)
                    status_item.setForeground(STATUS_COLORS.get(status, QColor("#888")))
                if status in ("Done", "Error"):
                    btn = QPushButton("Re-run")
                    btn.setFixedHeight(24)
                    _n = leg_name
                    btn.clicked.connect(lambda _, n=_n: self._on_rerun(n))
                    self._leg_table.setCellWidget(row, COL_RERUN, btn)
                else:
                    self._leg_table.setCellWidget(row, COL_RERUN, None)
                return

    # ── Camera preview ────────────────────────────────────────────────────────

    def _update_preview(self):
        if self._camera is None:
            return
        frame = self._camera.grab()
        if frame is not None:
            self._sig.frame_ready.emit(frame)

    def _on_frame(self, rgb):
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._feed_label.width(), self._feed_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self._feed_label.setPixmap(pixmap)

    # ── Connection ────────────────────────────────────────────────────────────

    def _on_connect(self):
        try:
            if self._camera is None:
                cam_txt = self._cam_port_edit.text().strip().lower()
                # Pass camera index to OpenCV backend if user typed a digit
                if cam_txt.isdigit():
                    from camera import OpenCVCamera
                    self._camera = OpenCVCamera(int(cam_txt))
                    self._camera.open()
                else:
                    self._camera = open_camera()

            if self._motor is None:
                esp_txt = self._esp_port_edit.text().strip()
                port = None if esp_txt.lower() == "auto" else esp_txt
                self._motor = MotorController(port=port)
                self._motor.open()

            self._clf.load()
            # Re-apply camera settings now that camera is open
            self._set_analysis_mode(self._analysis_on)
            # Delay preview so camera settles at new exposure before first frame shows
            QTimer.singleShot(2000, lambda: self._preview_timer.start(66))
            self._connect_btn.setText("Connected ✓")
            self._connect_btn.setEnabled(False)
            self._go_btn.setEnabled(True)
            self._statusbar.showMessage("Connected — camera and ESP32 ready")
        except Exception as e:
            QMessageBox.critical(self, "Connection error", str(e))

    # ── Go ────────────────────────────────────────────────────────────────────

    def _on_go(self):
        if not self._set_dir:
            self._prompt_set_folder()
            return

        mode = self._mode_group.checkedId()

        if mode == MODE_ANALYZE_ONLY:
            self._run_analyze_only()
            return

        rows  = self._rows_spin.value()
        cols  = self._cols_spin.value()
        total = rows * cols
        expected = EXPECTED_IMAGES

        self._cap_bar.setMaximum(total); self._cap_bar.setValue(0)
        self._ana_bar.setMaximum(total); self._ana_bar.setValue(0)
        self._go_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        # Temporary "Capturing" row — named after capture completes
        self._pending_leg_name = "__capturing__"
        self._upsert_leg_row("(capturing…)", total, "Capturing")

        self._capture_pipeline = CapturePipeline(
            camera=self._camera, motor=self._motor, classifier=self._clf,
            output_dir=self._set_dir, set_name="__tmp__", leg="__tmp__",
            rows=rows, cols=cols,
            x_spacing=self._x_spin.value(),
            y_spacing=self._y_spin.value(),
            quality_threshold=self._thresh_slider.value() / 10.0,
            on_progress=lambda d, t: self._sig.capture_progress.emit(d, t),
            on_frame=lambda f: self._sig.frame_ready.emit(f),
            on_done=lambda: self._sig.capture_done.emit(),
            on_error=lambda e: self._sig.error.emit(str(e)),
        )
        # Override output dir: images go to a temp folder until the user names the leg
        self._tmp_capture_dir = os.path.join(self._set_dir, "__tmp__")
        self._capture_pipeline._out = self._tmp_capture_dir
        os.makedirs(self._tmp_capture_dir, exist_ok=True)

        self._capture_mode = mode
        self._capture_total = total
        self._statusbar.showMessage(f"Capturing {rows}×{cols} images…")
        self._capture_pipeline.start()

    def _on_capture_done(self):
        self._stop_btn.setEnabled(False)

        # Count captured images
        tmp_dir = self._tmp_capture_dir
        jpegs = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".jpg"))
        n_captured = len(jpegs)

        # Check image count
        expected = self._capture_total
        if n_captured != expected:
            result = QMessageBox.warning(
                self, "Image count mismatch",
                f"Expected {expected} images but captured {n_captured}.\n\n"
                "This leg may be incomplete. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if result == QMessageBox.No:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
                self._go_btn.setEnabled(True)
                self._statusbar.showMessage("Capture cancelled.")
                self._leg_table.removeRow(
                    self._find_leg_row("(capturing…)")
                )
                return

        # Ask user to name this leg
        leg_name, ok = QInputDialog.getText(
            self, "Name this leg",
            "Enter leg name  (e.g. FR, FL, BR, BL):",
        )
        if not ok or not leg_name.strip():
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._leg_table.removeRow(self._find_leg_row("(capturing…)"))
            self._go_btn.setEnabled(True)
            self._statusbar.showMessage("Leg discarded — no name given.")
            return
        leg_name = leg_name.strip()

        # Confirm overwrite if leg already exists
        final_dir = os.path.join(self._set_dir, leg_name)
        if os.path.isdir(final_dir) and leg_name in self._leg_results:
            result = QMessageBox.question(
                self, "Overwrite?",
                f"Leg '{leg_name}' already has results. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if result == QMessageBox.No:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
                self._leg_table.removeRow(self._find_leg_row("(capturing…)"))
                self._go_btn.setEnabled(True)
                return
            import shutil
            shutil.rmtree(final_dir, ignore_errors=True)
            if leg_name in self._leg_results:
                del self._leg_results[leg_name]
            self._analyzing_legs.discard(leg_name)

        # Move tmp folder to final name
        import shutil
        shutil.move(tmp_dir, final_dir)

        # Remove temp row and add real one
        tmp_row = self._find_leg_row("(capturing…)")
        if tmp_row >= 0:
            self._leg_table.removeRow(tmp_row)
        self._upsert_leg_row(leg_name, n_captured, "—")

        self._go_btn.setEnabled(True)
        self._statusbar.showMessage(f"Captured {n_captured} images → {leg_name}")

        if self._capture_mode == MODE_CAPTURE_ANALYZE:
            self._start_leg_analysis(leg_name, final_dir, n_captured)
        else:
            self._check_finish_eligibility()

    def _find_leg_row(self, leg_name: str) -> int:
        for row in range(self._leg_table.rowCount()):
            item = self._leg_table.item(row, COL_LEG)
            if item and item.text() == leg_name:
                return row
        return -1

    # ── Analysis launch ───────────────────────────────────────────────────────

    def _start_leg_analysis(self, leg_name: str, leg_dir: str, img_count: int):
        self._leg_results[leg_name] = None  # sentinel: in progress
        self._analyzing_legs.add(leg_name)
        self._set_leg_status(leg_name, "Analyzing", img_count)

        ap = AnalysisPipeline(
            leg_dir=leg_dir,
            total_expected=img_count,
            on_progress=lambda d, t: self._sig.analysis_progress.emit(d, t or img_count),
            on_done=lambda results, _n=leg_name: self._sig.leg_analysis_done.emit(_n, results),
            on_error=lambda e, _n=leg_name: self._sig.error.emit(
                f"Analysis error on {_n}: {e}"
            ),
        )
        ap.start()

    def _on_leg_analysis_done(self, leg_name: str, results: list):
        self._leg_results[leg_name] = results
        self._analyzing_legs.discard(leg_name)
        self._set_leg_status(leg_name, "Done")
        self._statusbar.showMessage(f"Analysis complete: {leg_name}")
        self._check_finish_eligibility()

        if self._finish_waiting and not self._analyzing_legs:
            self._do_export()

    def _check_finish_eligibility(self):
        ready = any(v is not None for v in self._leg_results.values())
        self._finish_btn.setEnabled(ready)

    # ── Re-run ────────────────────────────────────────────────────────────────

    def _on_rerun(self, leg_name: str):
        result = QMessageBox.question(
            self, "Re-run leg",
            f"Re-run capture for '{leg_name}'? This will delete its current images and results.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if result == QMessageBox.No:
            return
        import shutil
        leg_dir = os.path.join(self._set_dir, leg_name)
        shutil.rmtree(leg_dir, ignore_errors=True)
        if leg_name in self._leg_results:
            del self._leg_results[leg_name]
        self._analyzing_legs.discard(leg_name)
        self._set_leg_status(leg_name, "—", 0)
        self._check_finish_eligibility()

    # ── Analyze-only mode ─────────────────────────────────────────────────────

    def _run_analyze_only(self):
        found = [d for d in os.listdir(self._set_dir)
                 if os.path.isdir(os.path.join(self._set_dir, d))]
        if not found:
            QMessageBox.warning(self, "Empty folder",
                                "No sub-folders found in the set folder.")
            return

        # Validate image counts
        bad_legs = []
        for lg in found:
            lg_dir = os.path.join(self._set_dir, lg)
            n = len([f for f in os.listdir(lg_dir)
                     if f.endswith(".jpg") and "overlay" not in f])
            if n != EXPECTED_IMAGES:
                bad_legs.append(f"{lg}: {n} images (expected {EXPECTED_IMAGES})")

        if bad_legs:
            msg = "The following legs do not have exactly " \
                  f"{EXPECTED_IMAGES} images:\n\n" + "\n".join(bad_legs) + \
                  "\n\nAnalyze anyway?"
            if QMessageBox.warning(self, "Image count warning", msg,
                                   QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
                return

        total = sum(
            len([f for f in os.listdir(os.path.join(self._set_dir, lg))
                 if f.endswith(".jpg") and "overlay" not in f])
            for lg in found
        )
        self._ana_bar.setMaximum(total); self._ana_bar.setValue(0)
        self._go_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        done_count = [0]
        for lg in found:
            lg_dir = os.path.join(self._set_dir, lg)
            n = len([f for f in os.listdir(lg_dir)
                     if f.endswith(".jpg") and "overlay" not in f])
            self._upsert_leg_row(lg, n, "Analyzing")
            self._start_leg_analysis(lg, lg_dir, n)

        self._statusbar.showMessage(f"Analyzing {len(found)} leg(s)…")

    # ── Stop ─────────────────────────────────────────────────────────────────

    def _on_stop(self):
        if self._capture_pipeline:
            self._capture_pipeline.stop()
        self._stop_btn.setEnabled(False)
        self._go_btn.setEnabled(True)
        self._statusbar.showMessage("Stopped")

    # ── Finish & Export ───────────────────────────────────────────────────────

    def _on_finish(self):
        if not self._leg_results:
            QMessageBox.information(self, "Nothing to export",
                                    "No leg results available yet.")
            return

        if self._analyzing_legs:
            n = len(self._analyzing_legs)
            legs_str = ", ".join(sorted(self._analyzing_legs))
            QMessageBox.information(
                self, "Analysis still running",
                f"{n} leg(s) still analyzing: {legs_str}\n\n"
                "Excel will be exported automatically when they finish.",
            )
            self._finish_waiting = True
            self._finish_btn.setEnabled(False)
            self._statusbar.showMessage(
                f"Waiting for analysis to finish: {legs_str}…"
            )
            return

        self._do_export()

    def _do_export(self):
        self._finish_waiting = False
        completed = {k: v for k, v in self._leg_results.items() if v is not None}
        if not completed:
            return

        try:
            use_legacy = self._export_group.checkedId() == 1
            if use_legacy:
                paths = write_legacy_format(self._set_dir, completed)
                msg = "Legacy export complete:\n" + "\n".join(
                    os.path.basename(p) for p in paths
                )
            else:
                path = write_new_format(self._set_dir, completed)
                msg = f"Export complete:\n{os.path.basename(path)}"

            QMessageBox.information(self, "Export complete", msg)
            self._statusbar.showMessage("Export complete — " + self._set_dir)
            self._finish_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    # ── Progress callbacks ────────────────────────────────────────────────────

    def _on_capture_progress(self, done, total):
        self._cap_bar.setMaximum(total); self._cap_bar.setValue(done)

    def _on_analysis_progress(self, done, total):
        self._ana_bar.setMaximum(max(total, 1)); self._ana_bar.setValue(done)

    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self._stop_btn.setEnabled(False)
        self._go_btn.setEnabled(True)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._preview_timer.stop()
        if self._capture_pipeline:
            self._capture_pipeline.stop()
        if self._camera:
            self._camera.close()
        if self._motor:
            self._motor.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
