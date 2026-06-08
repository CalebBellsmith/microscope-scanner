"""
Capture pipeline: grid scan with centroid-directed nudge search.

Output structure:
  output_dir / set_name / leg / 001.jpg … 030.jpg

Image numbers are positional (row * cols + col + 1), always left-to-right,
so filenames are stable regardless of boustrophedon capture order.
This matches what the MATLAB analysis script expects.

Nudge strategy
──────────────
When the quality classifier flags a frame as bad:
  1. Find the largest dark blob (dust/debris) in the frame via OpenCV.
  2. Compute its centroid relative to the frame centre.
  3. Move the stage *away* from the defect (toward the clean half).
  4. Capture a fresh frame and compare quality scores.
  5. Return the better of the two frames, then restore stage position.
"""
import os
import time
import threading

import numpy as np


class CapturePipeline:
    def __init__(self, camera, motor, classifier,
                 output_dir, set_name, leg,
                 rows, cols, x_spacing, y_spacing,
                 on_progress=None, on_frame=None, on_done=None, on_error=None):
        self._cam       = camera
        self._motor     = motor
        self._clf       = classifier
        self._out       = os.path.join(output_dir, set_name, leg)
        self._rows      = rows
        self._cols      = cols
        self._x_spacing = x_spacing   # stepper steps per grid column
        self._y_spacing = y_spacing   # stepper steps per grid row
        self._on_progress = on_progress or (lambda done, total: None)
        self._on_frame    = on_frame    or (lambda img: None)
        self._on_done     = on_done     or (lambda: None)
        self._on_error    = on_error    or (lambda e: None)
        self._stop_event  = threading.Event()
        self._thread      = None

    def start(self):
        os.makedirs(self._out, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    # ── Grid scan ─────────────────────────────────────────────────────────────

    def _run(self):
        try:
            total = self._rows * self._cols
            done  = 0
            for row in range(self._rows):
                if self._stop_event.is_set():
                    break
                if row > 0:
                    self._motor.move("Y", self._y_spacing)
                    time.sleep(0.3)

                # Boustrophedon: odd rows travel right-to-left
                col_range = (
                    range(self._cols)
                    if row % 2 == 0
                    else range(self._cols - 1, -1, -1)
                )
                for col in col_range:
                    if self._stop_event.is_set():
                        break
                    if col > 0:
                        direction = 1 if row % 2 == 0 else -1
                        self._motor.move("X", direction * self._x_spacing)
                        time.sleep(0.1)

                    frame = self._best_frame()
                    if frame is not None:
                        img_num = row * self._cols + col + 1   # positional, 1-indexed
                        path = os.path.join(self._out, f"{img_num:03d}.jpg")
                        self._save(frame, path)
                        self._on_frame(frame)
                    done += 1
                    self._on_progress(done, total)

            self._on_done()
        except Exception as e:
            self._on_error(e)

    # ── Frame quality & nudge ─────────────────────────────────────────────────

    def _best_frame(self):
        """
        Grab a frame.  If bad, do one centroid-directed nudge toward the clean
        half of the image, capture again, return the better result, then
        restore the stage to its original position.
        """
        frame = self._wait_for_frame()
        if frame is None:
            return None
        if self._clf.is_good(frame):
            return frame

        # Locate defect and compute nudge direction
        nudge_x, nudge_y = _centroid_nudge(
            frame, self._x_spacing, self._y_spacing
        )

        if nudge_x == 0 and nudge_y == 0:
            return frame   # couldn't localise defect; keep original

        # Move toward the clean side
        if nudge_x != 0:
            self._motor.move("X", nudge_x)
        if nudge_y != 0:
            self._motor.move("Y", nudge_y)
        time.sleep(0.2)

        candidate = self._wait_for_frame()

        # Always restore to nominal grid position
        if nudge_x != 0:
            self._motor.move("X", -nudge_x)
        if nudge_y != 0:
            self._motor.move("Y", -nudge_y)

        if candidate is None:
            return frame

        # Return whichever frame scored higher
        _, orig_conf = self._clf.predict(frame)
        _, cand_conf = self._clf.predict(candidate)
        return candidate if cand_conf > orig_conf else frame

    def _wait_for_frame(self, timeout=5.0):
        """
        Get a full-quality frame for saving.
        Uses grab_fresh() on ToupTek (switches to 1300ms exposure, waits for
        a genuinely new frame, then restores fast preview exposure).
        Falls back to grab() for OpenCV/mss cameras.
        """
        if hasattr(self._cam, "grab_fresh"):
            return self._cam.grab_fresh(timeout=timeout)
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
        Image.fromarray(rgb_array).save(path, quality=95, subsampling=0)


# ── Centroid nudge helper ─────────────────────────────────────────────────────

def _centroid_nudge(frame: np.ndarray,
                    x_spacing: int,
                    y_spacing: int) -> tuple[int, int]:
    """
    Find the largest dark blob in *frame* and return (steps_x, steps_y) that
    moves the stage *away* from the defect (toward the cleaner half).

    Stage direction convention
    ──────────────────────────
    Moving the stage in +X shifts the image field in the same +X direction
    (the stage carries the slide; the objective stays fixed).  So to push a
    defect on the *right* side out of frame we move the stage *left* (−X).
    Negate NUDGE_SIGN_X / NUDGE_SIGN_Y below if your microscope is inverted.

    Returns (0, 0) if no significant defect can be localised.
    """
    import cv2

    NUDGE_SIGN_X = -1   # flip to +1 if stage/image X axes are inverted
    NUDGE_SIGN_Y = -1   # flip to +1 if stage/image Y axes are inverted
    NUDGE_SCALE  = 0.4  # fraction of grid spacing to nudge (tunable)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std  = float(gray.std())
    thr  = max(0, int(mean - 2.0 * std))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0, 0

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return 0, 0

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return 0, 0

    h, w    = frame.shape[:2]
    cx_frac = M["m10"] / M["m00"] / w   # 0 = left edge,  1 = right edge
    cy_frac = M["m01"] / M["m00"] / h   # 0 = top  edge,  1 = bottom edge

    # Offset from centre (−0.5 … +0.5)
    dx_frac = cx_frac - 0.5
    dy_frac = cy_frac - 0.5

    # Stage moves opposite to defect direction to push it out of frame
    nudge_x = int(NUDGE_SIGN_X * dx_frac * x_spacing * NUDGE_SCALE)
    nudge_y = int(NUDGE_SIGN_Y * dy_frac * y_spacing * NUDGE_SCALE)

    return nudge_x, nudge_y
