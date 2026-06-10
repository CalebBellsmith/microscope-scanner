"""
Capture pipeline: moves the stage in a grid pattern and saves one image
per position, applying a centroid-directed nudge if the quality classifier
flags a frame as bad (dust/debris in frame).

Output folder structure:
    output_dir / set_name / leg / 001.jpg … 030.jpg

Image numbers are positional (row * cols + col + 1), always left-to-right
regardless of capture direction, so filenames match what the MATLAB
analysis script expects when it does dir('*.jpg').

Grid traversal pattern (boustrophedon / snake scan):
    Row 0: → positions  0  1  2  3 … (left to right)
    Row 1: ← positions  9  8  7  6 … (right to left — reverses direction)
    Row 2: → positions 20 21 22 23 … (left to right again)
    …
This minimises total stage travel distance.

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

    def _run(self):
        """
        Main capture loop — runs in a background thread.
        Iterates over every grid position, moves the stage, grabs a frame,
        and saves it.  Calls on_progress after each image.
        """
        try:
            total = self._rows * self._cols
            done  = 0

            for row in range(self._rows):
                if self._stop_event.is_set():
                    break

                # Move to the next row (skip for row 0 — already at start)
                if row > 0:
                    self._motor.move("Y", self._y_spacing)
                    time.sleep(0.3)   # brief pause for stage to settle

                # Boustrophedon: even rows go left→right, odd rows right→left
                col_range = (
                    range(self._cols)
                    if row % 2 == 0
                    else range(self._cols - 1, -1, -1)
                )

                for col in col_range:
                    if self._stop_event.is_set():
                        break

                    # Move one column in the current direction (skip for col 0)
                    if col > 0:
                        direction = 1 if row % 2 == 0 else -1
                        self._motor.move("X", direction * self._x_spacing)
                        time.sleep(0.1)

                    # Capture the best available frame for this position
                    frame = self._best_frame()
                    if frame is not None:
                        # Positional filename: always left-to-right, 1-indexed
                        # e.g. row=1, col=3 → image 14 of a 10-col grid
                        img_num = row * self._cols + col + 1
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
        Grab a frame and check its quality score against the threshold.
        If it passes → return immediately.
        If it fails  → run _centroid_nudge to find and move away from the
                       defect, grab a second frame, compare scores, return
                       the better one, then restore the stage position.
        """
        frame = self._wait_for_frame()
        if frame is None:
            return None

        # Ask the classifier how confident it is that this frame is good
        _, conf = self._clf.predict(frame)
        if conf >= self._threshold:
            return frame   # good enough — no nudge needed

        # Frame is bad — find the defect and nudge away from it
        nudge_x, nudge_y = _centroid_nudge(
            frame, self._x_spacing, self._y_spacing
        )

        if nudge_x == 0 and nudge_y == 0:
            # Couldn't localise the defect — keep the original frame
            return frame

        # Move toward the clean side of the frame
        if nudge_x != 0:
            self._motor.move("X", nudge_x)
        if nudge_y != 0:
            self._motor.move("Y", nudge_y)
        time.sleep(0.2)   # let stage settle

        candidate = self._wait_for_frame()

        # Always return stage to the nominal grid position
        if nudge_x != 0:
            self._motor.move("X", -nudge_x)
        if nudge_y != 0:
            self._motor.move("Y", -nudge_y)

        if candidate is None:
            return frame   # nudge gave no result — keep original

        # Return whichever frame the classifier liked more
        _, orig_conf = self._clf.predict(frame)
        _, cand_conf = self._clf.predict(candidate)
        return candidate if cand_conf > orig_conf else frame

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
