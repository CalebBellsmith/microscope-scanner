"""
Scan Path Simulator (scan_simulator.py)
=======================================
A hardware-free way to verify the capture motion BEFORE running it on the
Windows rig.  It drives the REAL CapturePipeline traversal logic against a mock
motor + camera, then draws and animates the resulting stage path, the 30 numbered
capture points, and the exact serial-command stream the ESP32 firmware would
receive — including the concurrent `MOVE XY` rung transitions.

Run it on the Mac with:
    python3 scan_simulator.py
or double-click  run_simulator.command

Nothing is written to disk and no serial port is opened; this is purely a
visual/timing dry run of capture_pipeline.py + firmware.ino.
"""
import sys
import time

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSpinBox, QPushButton, QVBoxLayout,
    QHBoxLayout, QGroupBox, QPlainTextEdit, QFormLayout,
)
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont

import capture_pipeline as cp

# Firmware step rates (µs per half-step) — keep in sync with firmware.ino.
DEFAULT_X_DELAY_US = 900
DEFAULT_Y_DELAY_US = 1500
DEFAULT_CAM_SEC    = 1.3   # grab_fresh exposure/settle time per image


# ── Mock hardware ─────────────────────────────────────────────────────────────

class MockMotor:
    """Records every command and accumulates motion time from step rates."""
    def __init__(self, x_delay_us, y_delay_us, events):
        self.x = 0
        self.y = 0
        self._xd = x_delay_us
        self._yd = y_delay_us
        self._events = events
        self.motion_sec = 0.0

    def move(self, axis, amount):
        amount = int(amount)
        if axis.upper() == "X":
            self.x += amount
            self.motion_sec += abs(amount) * self._xd / 1e6
        else:
            self.y += amount
            self.motion_sec += abs(amount) * self._yd / 1e6
        self._events.append(("move", self.x, self.y, f"MOVE {axis.upper()} {amount}"))

    def move_xy(self, x_amount, y_amount):
        x_amount = int(x_amount); y_amount = int(y_amount)
        self.x += x_amount
        self.y += y_amount
        # Concurrent: paced by the longer axis at the X (fast) rate.
        self.motion_sec += max(abs(x_amount), abs(y_amount)) * self._xd / 1e6
        self._events.append(("move", self.x, self.y, f"MOVE XY {x_amount} {y_amount}"))

    def home(self):
        self._events.append(("move", self.x, self.y, "HOME"))


class MockCamera:
    """Returns a tiny dummy frame instantly."""
    def grab(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)


class MockClassifier:
    """Always 'good' so no centroid nudge fires (pure grid motion)."""
    def predict(self, frame):
        return ("good", 1.0)


def simulate(rows, cols, x_spacing, y_spacing, x_delay_us, y_delay_us, cam_sec):
    """
    Run the real pipeline traversal with mocks and return:
        events       : ordered list of ('cap', x, y, n) and ('move', x, y, cmd)
        captures     : list of (img_num, x_steps, y_rungs)
        motion_sec   : estimated motor motion time
        capture_sec  : estimated camera time
    Positions in events are in raw half-steps; captures are in step/rung units.
    """
    events = []
    motor = MockMotor(x_delay_us, y_delay_us, events)
    captures = []

    pipe = cp.CapturePipeline(
        camera=MockCamera(), motor=motor, classifier=MockClassifier(),
        output_dir="/tmp/_sim", set_name="sim", leg="sim",
        rows=rows, cols=cols, x_spacing=x_spacing, y_spacing=y_spacing,
    )
    pipe._save = lambda arr, path: None     # never touch disk

    def fake_best_frame():
        n = len(captures) + 1
        captures.append((n, motor.x / x_spacing, motor.y / y_spacing))
        events.append(("cap", motor.x, motor.y, n))
        return np.zeros((4, 4, 3), dtype=np.uint8)

    pipe._best_frame = fake_best_frame

    # Run synchronously with sleeps neutralised so collection is instant.
    real_sleep = cp.time.sleep
    cp.time.sleep = lambda *_a, **_k: None
    try:
        pipe._run()
    finally:
        cp.time.sleep = real_sleep

    capture_sec = len(captures) * cam_sec
    return events, captures, motor.motion_sec, capture_sec


# ── Path canvas ───────────────────────────────────────────────────────────────

class PathCanvas(QWidget):
    """Draws the scan path + numbered captures and animates the stage dot."""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(560, 420)
        self._events = []
        self._captures = []
        self._x_spacing = 1
        self._y_spacing = 1
        self._cursor = 0           # how many events have been "played"
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def load(self, events, captures, x_spacing, y_spacing):
        self._events = events
        self._captures = captures
        self._x_spacing = x_spacing
        self._y_spacing = y_spacing
        self._cursor = 0
        self.update()

    def play(self):
        if not self._events:
            return
        self._cursor = 0
        self._timer.start(110)

    def _advance(self):
        self._cursor += 1
        if self._cursor >= len(self._events):
            self._cursor = len(self._events)
            self._timer.stop()
        self.update()

    # coordinate transform: stage units → widget pixels
    def _bounds(self):
        xs = [c[1] for c in self._captures] or [0, 1]
        ys = [c[2] for c in self._captures] or [0, 1]
        return min(xs), max(xs), min(ys), max(ys)

    def _to_px(self, x_steps, y_rungs):
        xmin, xmax, ymin, ymax = self._bounds()
        m = 46
        w = self.width() - 2 * m
        h = self.height() - 2 * m
        # pad the X range a touch so the 10.5 overshoot is visible
        xr = (xmax - xmin) or 1
        yr = (ymax - ymin) or 1
        px = m + (x_steps - xmin) / (xr + 0.5) * w
        py = m + (y_rungs - ymin) / yr * h    # rung 0 at top
        return QPointF(px, py)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#1c1c20"))
        if not self._captures:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Click “Run simulation”.")
            return

        # Build the full position polyline from move/cap events.
        pts = [(0.0, 0.0)]
        for kind, x, y, _meta in self._events:
            pts.append((x / self._x_spacing, y / self._y_spacing))

        # faint full path
        p.setPen(QPen(QColor("#3a3a44"), 1, Qt.DashLine))
        for a, b in zip(pts, pts[1:]):
            p.drawLine(self._to_px(*a), self._to_px(*b))

        # played path (bright) up to the cursor
        played = pts[: self._cursor + 1]
        p.setPen(QPen(QColor("#5fa8ff"), 2))
        for a, b in zip(played, played[1:]):
            p.drawLine(self._to_px(*a), self._to_px(*b))

        # capture markers
        played_caps = {e[3] for e in self._events[: self._cursor] if e[0] == "cap"}
        f = QFont(); f.setPointSize(8); p.setFont(f)
        for n, xs, yr in self._captures:
            c = self._to_px(xs, yr)
            done = n in played_caps
            p.setBrush(QBrush(QColor("#ffd24a") if done else QColor("#55552e")))
            p.setPen(QPen(QColor("#000"), 1))
            p.drawEllipse(c, 11, 11)
            p.setPen(QColor("#000") if done else QColor("#999"))
            p.drawText(QRectF(c.x() - 11, c.y() - 11, 22, 22),
                       Qt.AlignCenter, str(n))

        # stage dot at the current position
        if self._cursor < len(pts):
            cur = self._to_px(*pts[self._cursor])
        else:
            cur = self._to_px(*pts[-1])
        p.setBrush(QBrush(QColor("#ff5f5f")))
        p.setPen(QPen(QColor("#fff"), 2))
        p.drawEllipse(cur, 7, 7)


# ── Main window ───────────────────────────────────────────────────────────────

class Simulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scan Path Simulator — dry run (no hardware)")
        root = QHBoxLayout(self)

        # left: controls + command log
        left = QVBoxLayout()
        form_box = QGroupBox("Scan parameters")
        form = QFormLayout(form_box)
        self._rows = self._spin(1, 50, 3)
        self._cols = self._spin(1, 50, 10)
        self._xsp  = self._spin(1, 20000, 2714)
        self._ysp  = self._spin(1, 20000, 1463)
        self._xdl  = self._spin(100, 5000, DEFAULT_X_DELAY_US)
        self._ydl  = self._spin(100, 5000, DEFAULT_Y_DELAY_US)
        self._cam  = self._spin(0, 10000, int(DEFAULT_CAM_SEC * 1000))
        form.addRow("Rows (rungs):", self._rows)
        form.addRow("Cols (per rung):", self._cols)
        form.addRow("X spacing (half-steps):", self._xsp)
        form.addRow("Y spacing (half-steps):", self._ysp)
        form.addRow("X step delay (µs):", self._xdl)
        form.addRow("Y step delay (µs):", self._ydl)
        form.addRow("Camera time (ms/img):", self._cam)
        left.addWidget(form_box)

        self._run_btn = QPushButton("▶  Run simulation")
        self._run_btn.clicked.connect(self._run)
        left.addWidget(self._run_btn)

        self._summary = QLabel("—")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color:#ccc;")
        left.addWidget(self._summary)

        left.addWidget(QLabel("Serial command stream (→ ESP32):"))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("font-family:monospace; font-size:11px;")
        left.addWidget(self._log, 1)

        root.addLayout(left, 0)
        self._canvas = PathCanvas()
        root.addWidget(self._canvas, 1)

    @staticmethod
    def _spin(lo, hi, val):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); return s

    def _run(self):
        rows = self._rows.value(); cols = self._cols.value()
        xsp = self._xsp.value(); ysp = self._ysp.value()
        events, captures, motion, capt = simulate(
            rows, cols, xsp, ysp,
            self._xdl.value(), self._ydl.value(), self._cam.value() / 1000.0,
        )

        # command log
        lines = []
        for kind, x, y, meta in events:
            if kind == "move":
                lines.append(f"{meta:<22}  → x={x:>7}  y={y:>6}")
            else:
                lines.append(f"   · capture #{meta:<3} at x={x:>7} y={y:>6}")
        self._log.setPlainText("\n".join(lines))

        n_moves = sum(1 for e in events if e[0] == "move")
        total = motion + capt
        self._summary.setText(
            f"{len(captures)} images · {n_moves} motor commands\n"
            f"Motor motion: {motion:5.1f} s   Camera: {capt:5.1f} s\n"
            f"Estimated total: {total:5.1f} s  ({total/60:.1f} min) per leg\n"
            f"(plus settle pauses, excluded here)"
        )
        self._canvas.load(events, captures, xsp, ysp)
        self._canvas.play()


def main():
    app = QApplication(sys.argv)
    win = Simulator()
    win.resize(1040, 560)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
