"""
Capture pipeline: grid scan with local nudge search.

Output structure:
  output_dir / set_name / leg / 001.jpg … 030.jpg

Image numbers are positional (row * cols + col + 1), always left-to-right,
so filenames are stable regardless of boustrophedon capture order.
This matches what the MATLAB analysis script expects.
"""
import os
import time
import threading

# Spiral nudge offsets in (dx_steps, dy_units)
_NUDGE_SPIRAL = [
    (10, 0), (-20, 0), (10, 5), (0, -10), (10, 5),
    (-10, 5), (0, -10), (10, -5), (-10, 0), (10, 5),
]


class CapturePipeline:
    def __init__(self, camera, motor, classifier,
                 output_dir, set_name, leg,
                 rows, cols, x_spacing, y_spacing,
                 on_progress=None, on_frame=None, on_done=None, on_error=None):
        self._cam = camera
        self._motor = motor
        self._clf = classifier
        # Final save directory: output_dir / set_name / leg
        self._out = os.path.join(output_dir, set_name, leg)
        self._rows = rows
        self._cols = cols
        self._x_spacing = x_spacing  # stepper steps
        self._y_spacing = y_spacing  # servo units
        self._on_progress = on_progress or (lambda done, total: None)
        self._on_frame = on_frame or (lambda img: None)
        self._on_done = on_done or (lambda: None)
        self._on_error = on_error or (lambda e: None)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        os.makedirs(self._out, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        try:
            total = self._rows * self._cols
            done = 0
            for row in range(self._rows):
                if self._stop_event.is_set():
                    break
                if row > 0:
                    self._motor.move("Y", self._y_spacing)
                    time.sleep(0.3)

                # Boustrophedon: odd rows go right-to-left
                col_range = range(self._cols) if row % 2 == 0 else range(self._cols - 1, -1, -1)
                for col in col_range:
                    if self._stop_event.is_set():
                        break
                    if col > 0:
                        direction = 1 if row % 2 == 0 else -1
                        self._motor.move("X", direction * self._x_spacing)
                        time.sleep(0.1)

                    frame = self._best_frame()
                    if frame is not None:
                        # Positional number: always left-to-right, 1-indexed
                        img_num = row * self._cols + col + 1
                        path = os.path.join(self._out, f"{img_num:03d}.jpg")
                        self._save(frame, path)
                        self._on_frame(frame)
                    done += 1
                    self._on_progress(done, total)

            self._on_done()
        except Exception as e:
            self._on_error(e)

    def _best_frame(self):
        """Grab frame; spiral-nudge if bad; return best frame found."""
        frame = self._wait_for_frame()
        if frame is None:
            return None
        if self._clf.is_good(frame):
            return frame

        cumulative_dx, cumulative_dy = 0, 0
        best_frame = frame
        _, best_conf = self._clf.predict(frame)

        for dx, dy in _NUDGE_SPIRAL:
            if self._stop_event.is_set():
                break
            self._motor.move("X", dx)
            cumulative_dx += dx
            if dy != 0:
                self._motor.move("Y", dy)
                cumulative_dy += dy
            time.sleep(0.15)

            candidate = self._wait_for_frame()
            if candidate is None:
                continue
            label, conf = self._clf.predict(candidate)
            if label == "good":
                self._motor.move("X", -cumulative_dx)
                if cumulative_dy != 0:
                    self._motor.move("Y", -cumulative_dy)
                return candidate
            if conf > best_conf:
                best_frame = candidate
                best_conf = conf

        self._motor.move("X", -cumulative_dx)
        if cumulative_dy != 0:
            self._motor.move("Y", -cumulative_dy)
        return best_frame

    def _wait_for_frame(self, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self._cam.grab()
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None

    @staticmethod
    def _save(rgb_array, path):
        from PIL import Image
        # quality=95 subsampling=0 matches ToupView export defaults
        Image.fromarray(rgb_array).save(path, quality=95, subsampling=0)
