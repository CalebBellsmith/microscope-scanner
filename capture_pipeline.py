"""
Capture pipeline: moves the stage in a grid pattern and saves one image
per position, applying a centroid-directed nudge if the quality classifier
flags a frame as bad (dust/debris in frame).

Output folder structure:
    output_dir / set_name / leg / 001.jpg … 030.jpg

Image numbers are positional (row * cols + col + 1), always left-to-right,
so filenames match what the MATLAB analysis script expects when it does
dir('*.jpg').

Grid traversal pattern (calibrated raster with half-step stagger):
    Every rung scans the SWEEP axis forward (capture, then step one sweep-step),
    then the stage returns along the SWEEP axis and advances one RUNG step.  All
    moves are SEQUENTIAL (one axis at a time — no concurrent diagonal).  The
    returns are deliberately asymmetric so odd rungs are offset half a sweep-step,
    interleaving their samples with the even rungs for finer coverage:

        Rung 0:  capture at sweep-steps 0,1,…,9   (start offset 0.0)
          ↩ sweep return −8.5 steps,  rung +1
        Rung 1:  capture at sweep-steps 0.5,1.5,…,9.5  (start offset 0.5)
          ↩ sweep return −9.5 steps,  rung +1
        Rung 2:  capture at sweep-steps 0,1,…,9   (start offset 0.0)

    Each rung stops ON its last capture — 10 captures use 9 inter-capture steps,
    NOT 10 — so the stage never overshoots one step past the final photo; the
    return distances above already account for ending on capture 9 (not 10).

    Axis roles (see SWEEP_AXIS / RUNG_AXIS below): the fast 10-per-rung SWEEP
    axis is firmware Y (1.4 cm/rot, default 3000 half-steps ≈ 1.0 cm/step); the
    slow 3-rung RUNG axis is firmware X (0.8 cm/rot, default 1350 half-steps
    ≈ 0.26 cm/step).
    On a clean finish the stage walks back to the leg origin; on STOP it halts
    in place (the operator re-zeroes before the next run).

Nudge strategy when a frame is bad:
    1. Find the largest dark blob (dust) in the frame using OpenCV.
    2. Compute where its centre is relative to the frame centre.
    3. Move the stage away from the blob toward the clean half.
    4. Capture a second frame and compare quality scores.
    5. Keep whichever frame scored higher, then restore stage position.
    Only one nudge attempt is made per position — if both frames are bad,
    the better one is saved (we always need exactly 30 images per leg).
"""
import os
import time
import threading

import numpy as np


class CapturePipeline:

    def __init__(self, camera, motor, classifier,
                 output_dir, set_name, leg,
                 rows, cols, x_spacing, y_spacing,
                 quality_threshold=0.5,
                 on_progress=None, on_frame=None, on_done=None, on_error=None):
        """
        camera            : camera object (ToupTekCamera / OpenCVCamera)
        motor             : MotorController instance
        classifier        : QualityClassifier instance
        output_dir        : root folder for saved images
        set_name / leg    : sub-folder names  → output_dir/set_name/leg/
        rows / cols       : grid dimensions (default 3 × 10 = 30 images)
        x_spacing         : stepper steps between adjacent columns
        y_spacing         : stepper steps between adjacent rows
        quality_threshold : classifier confidence required to skip nudge
                            (0.1 = lenient / accept almost anything,
                             0.9 = strict / nudge unless very confident)
        on_progress(done, total) : called after each image is captured
        on_frame(frame)          : called with each saved frame (for GUI display)
        on_done()                : called when all images have been captured
        on_error(exception)      : called if an unhandled exception occurs
        """
        self._cam       = camera
        self._motor     = motor
        self._clf       = classifier
        self._out       = os.path.join(output_dir, set_name, leg)
        self._rows      = rows
        self._cols      = cols
        self._x_spacing = x_spacing
        self._y_spacing = y_spacing
        self._threshold = quality_threshold
        self._on_progress = on_progress or (lambda done, total: None)
        self._on_frame    = on_frame    or (lambda img: None)
        self._on_done     = on_done     or (lambda: None)
        self._on_error    = on_error    or (lambda e: None)
        self._stop_event  = threading.Event()   # set by stop() to abort scan
        self._thread      = None

    def start(self):
        """Create the output folder and start the capture thread."""
        os.makedirs(self._out, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the capture thread to stop after the current image."""
        self._stop_event.set()

    # ── Grid scan ─────────────────────────────────────────────────────────────

    # ── Physical axis roles ────────────────────────────────────────────────
    # The fast SWEEP axis takes 10 captures per rung; the slow RUNG axis indexes
    # 3 times per leg.  These map the scan roles onto the firmware axis letters.
    # If the stage axes are wired the other way, swap these two letters (and the
    # X/Y spacing defaults in main.py).
    SWEEP_AXIS = "Y"   # fast: 10 captures per rung  (firmware Y stepper)
    RUNG_AXIS  = "X"   # slow: 3 rungs per leg       (firmware X stepper)

    # Pause after a move, before capturing, so the stage stops vibrating before
    # the (long-exposure) photo is taken.  Raise if photos still look smeared;
    # lower to shave time once you know the stage settles quickly.
    SETTLE_S = 0.5

    @staticmethod
    def _rung_start(row):
        """Sweep-axis start offset of a rung, in whole sweep-steps.  Odd rungs
        are staggered half a step so their samples interleave with even rungs."""
        return 0.5 if (row % 2 == 1) else 0.0

    def _run(self):
        """
        Main capture loop — runs in a background thread.
        Each rung scans the SWEEP axis forward (capture-then-step); between rungs
        the SWEEP axis returns and the RUNG axis advances one step.  All moves
        are sequential (one axis at a time).  On normal completion the stage
        returns to the leg origin; on STOP it halts in place.
        """
        try:
            total = self._rows * self._cols
            cols  = self._cols
            # Spacing is indexed by firmware axis; pick out each role's step.
            spacing    = {"X": self._x_spacing, "Y": self._y_spacing}
            sweep_step = spacing[self.SWEEP_AXIS]   # half-steps per sweep step
            rung_step  = spacing[self.RUNG_AXIS]    # half-steps per rung step
            done = 0

            # Net travel per firmware axis, so we can return to origin at the
            # end.  (Centroid nudges restore themselves and are not counted.)
            abs_pos = {"X": 0, "Y": 0}

            def move(axis, amount):
                amount = int(amount)
                if amount == 0:
                    return
                self._motor.move(axis, amount)
                abs_pos[axis] += amount

            for row in range(self._rows):
                if self._stop_event.is_set():
                    break

                # Reposition between rungs (sequential): SWEEP axis returns, then
                # RUNG axis advances one step.  The previous rung ended ON its
                # last capture (no trailing step), at sweep position
                # prev_start + (cols − 1).  So the return distance to the next
                # rung's start = next_start − (prev_start + cols − 1), e.g.
                # row 0→1: 0.5 − (0 + 9) = −8.5 steps;  row 1→2: 0 − (0.5 + 9) = −9.5.
                if row > 0:
                    sweep_return = (self._rung_start(row)
                                    - self._rung_start(row - 1) - (cols - 1))
                    move(self.SWEEP_AXIS, round(sweep_return * sweep_step))
                    move(self.RUNG_AXIS, rung_step)
                    time.sleep(self.SETTLE_S)   # let the stage stop ringing

                for col in range(cols):
                    if self._stop_event.is_set():
                        break

                    # Capture FIRST (capture-then-step) at this sweep position.
                    frame = self._best_frame()
                    if frame is not None:
                        img_num = row * cols + col + 1   # positional filename
                        path = os.path.join(self._out, f"{img_num:03d}.jpg")
                        self._save(frame, path)
                        self._on_frame(frame)

                    done += 1
                    self._on_progress(done, total)

                    # Step the SWEEP axis one position between captures only —
                    # 10 captures need 9 steps, so we DON'T step after the last
                    # capture of a rung.  Counting the start position as the first
                    # capture, this keeps the rung from overshooting one step past
                    # its final photo; the inter-rung return (above) covers the
                    # distance back instead.
                    if col < cols - 1:
                        move(self.SWEEP_AXIS, sweep_step)
                        time.sleep(self.SETTLE_S)   # settle before next capture

            # On a clean finish, walk back to the leg origin (sequential).  On a
            # STOP, leave the stage where it is — the operator re-zeroes anyway.
            if not self._stop_event.is_set():
                move(self.RUNG_AXIS,  -abs_pos[self.RUNG_AXIS])
                move(self.SWEEP_AXIS, -abs_pos[self.SWEEP_AXIS])

            self._on_done()

        except Exception as e:
            self._on_error(e)

    # ── Frame quality & nudge ─────────────────────────────────────────────────

    @staticmethod
    def _goodness(label, conf):
        """Signed quality score: +conf if good, −conf if bad.  Higher = better."""
        return conf if label == "good" else -conf

    def _best_frame(self):
        """
        Grab a frame and classify it.
        If the classifier calls it GOOD → return immediately.
        If it calls it BAD  → run _centroid_nudge to move away from the defect,
                              grab a second frame, return whichever scores higher,
                              then restore the stage position.

        Decision is made on the classifier's LABEL (which already reflects the
        sensitivity setting), not on a raw confidence threshold — a confidently
        BAD frame has high confidence too, so the old `conf >= threshold` test
        wrongly accepted it.
        """
        frame = self._wait_for_frame()
        if frame is None:
            return None

        label, conf = self._clf.predict(frame)
        if label == "good":
            return frame   # clean frame — no nudge needed

        # Frame is bad — find the defect and nudge away from it
        nudge_x, nudge_y = _centroid_nudge(
            frame, self._x_spacing, self._y_spacing
        )

        if nudge_x == 0 and nudge_y == 0:
            # Couldn't localise the defect — keep the original frame
            return frame

        # Move toward the clean side of the frame, capture, then ALWAYS move
        # back by the EXACT amount applied.  applied_* records what actually
        # moved, and the finally block reverses precisely that — so a failed
        # capture (or a move that only half-completed) can't leave the stage
        # nudged, which would accumulate as drift across the 30-image scan.
        applied_x = applied_y = 0
        candidate = None
        try:
            if nudge_x != 0:
                self._motor.move("X", nudge_x); applied_x = nudge_x
            if nudge_y != 0:
                self._motor.move("Y", nudge_y); applied_y = nudge_y
            time.sleep(self.SETTLE_S)   # settle before the nudged shot
            candidate = self._wait_for_frame()
        finally:
            # Reverse exactly what was applied (equal magnitude, opposite sign).
            if applied_y != 0:
                self._motor.move("Y", -applied_y)
            if applied_x != 0:
                self._motor.move("X", -applied_x)

        if candidate is None:
            return frame   # nudge gave no result — keep original

        # Return whichever frame the classifier liked more (signed goodness, so a
        # GOOD candidate always beats a BAD original even if both are confident).
        orig_score = self._goodness(label, conf)              # reuse top prediction
        cand_score = self._goodness(*self._clf.predict(candidate))
        return candidate if cand_score > orig_score else frame

    def _wait_for_frame(self, timeout=5.0):
        """
        Get a full-quality frame for saving.
        ToupTek cameras: uses grab_fresh() which switches to 1300ms exposure,
        waits for a genuinely new frame, then restores preview exposure.
        Other cameras: polls grab() until a frame arrives.
        """
        if hasattr(self._cam, "grab_fresh"):
            return self._cam.grab_fresh(timeout=timeout)

        # Fallback for OpenCV / screen-capture cameras
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self._cam.grab()
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None

    @staticmethod
    def _save(rgb_array, path):
        """Save a numpy RGB array as a JPEG at quality 95, no chroma subsampling."""
        from PIL import Image
        Image.fromarray(rgb_array).save(path, quality=95, subsampling=0)


# ── Centroid nudge helper ─────────────────────────────────────────────────────

def _centroid_nudge(frame: np.ndarray,
                    x_spacing: int,
                    y_spacing: int) -> tuple[int, int]:
    """
    Find the largest dark blob (dust/debris) in the frame and return
    (steps_x, steps_y) that moves the stage AWAY from it.

    How it works:
      1. Convert to greyscale.
      2. Threshold at mean - 2*std to isolate unusually dark pixels (dust).
      3. Find contours (connected dark regions).
      4. Take the largest contour — that is the main defect.
      5. Compute its centroid relative to the frame centre.
      6. Scale the centroid offset to a fraction of the grid spacing.

    Stage direction notes:
      Moving the stage +X shifts the image field +X (stage carries the slide).
      To push a defect on the RIGHT out of frame, move the stage LEFT (−X).
      NUDGE_SIGN_X = −1 implements this.  Flip to +1 if your microscope
      maps stage motion in the opposite direction to image motion.

    Returns (0, 0) if no significant defect is found.
    """
    import cv2

    NUDGE_SIGN_X = -1   # −1: stage and image move in same direction
    NUDGE_SIGN_Y = -1   # flip to +1 if image pans opposite to stage
    NUDGE_SCALE  = 0.4  # fraction of one grid spacing to nudge (tunable)

    # Convert to greyscale for blob detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std  = float(gray.std())

    # Pixels darker than mean − 2σ are likely dust/debris
    thr  = max(0, int(mean - 2.0 * std))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    # Find all dark blobs
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0, 0   # no blobs found

    # Take the biggest blob — most likely the defect we want to avoid
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return 0, 0   # blob too small to be a meaningful defect

    # Compute centroid of the blob using image moments
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return 0, 0

    h, w    = frame.shape[:2]
    cx_frac = M["m10"] / M["m00"] / w   # 0.0 = left edge,  1.0 = right edge
    cy_frac = M["m01"] / M["m00"] / h   # 0.0 = top  edge,  1.0 = bottom edge

    # Offset from frame centre (range −0.5 to +0.5)
    dx_frac = cx_frac - 0.5
    dy_frac = cy_frac - 0.5

    # Convert fraction to actual stepper steps, moving AWAY from the defect
    nudge_x = int(NUDGE_SIGN_X * dx_frac * x_spacing * NUDGE_SCALE)
    nudge_y = int(NUDGE_SIGN_Y * dy_frac * y_spacing * NUDGE_SCALE)

    return nudge_x, nudge_y
