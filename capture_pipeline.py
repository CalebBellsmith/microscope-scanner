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
    axis is firmware Y (1.4 cm/rot, default 2900 half-steps ≈ 1.0 cm/step); the
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
                 review_mode="none", review_fn=None, blur_threshold=3000.0,
                 good_dir=None, bad_dir=None,
                 z_step=300, z_range=10000, nudge_scale=0.4,
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
        review_mode       : how FOCUS is handled — the defect-nudge always runs
                            in every mode (it's the core job).  On top of that:
                            "none"     — no focus handling.
                            "auto"     — auto-keep in-focus frames; pause for the
                                         operator only when a frame is soft.
                            "manual"   — pause for the operator on every frame.
                            "autofocus"— when a frame is soft, drive the Z stepper
                                         to refocus and re-capture (no operator).
        review_fn(frame, reason) -> "good"|"bad"
                            blocking callback into the GUI for auto/manual modes;
                            "good" keeps the frame, "bad" logs it and retakes.
        blur_threshold    : focus-score floor below which a frame is considered
                            out of focus (auto + autofocus modes).
        good_dir / bad_dir: if set, reviewed frames are copied here (kept→good,
                            rejected→bad) to grow the ML training dataset.
        z_step / z_range  : autofocus half-steps per probe, and the max ± travel
                            the search may span.  The Z axis is a continuous
                            roller with no hard travel limit, so z_range is only
                            a runaway guard — the search normally stops itself
                            at the focus peak.  Which direction is "into focus"
                            is discovered by probing (never configured): each
                            search remembers its winning direction and probes
                            that way first on the next field.
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
        self._review_mode = review_mode
        self._review_fn   = review_fn or (lambda frame, reason: "good")
        self._blur_thresh = blur_threshold
        self._good_dir    = good_dir
        self._bad_dir     = bad_dir
        self._z_step      = int(z_step)
        self._z_range     = int(z_range)
        self._z_dir       = 1       # preferred FIRST-probe direction — re-learned
                                    # from every search's winner, never configured
        self._z_disabled  = False   # set if the Z stepper doesn't respond (no HW)
        self._last_soft   = False   # last keeper still soft after escalation →
                                    # saved with a _soft filename tag
        self._nudge_scale = float(nudge_scale)   # defect-avoidance jump strength
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

    # Autofocus escalation: when the normal search's best frame is still below
    # the blur threshold, retry once with wider probes over the full roller
    # range (the Z axis is continuous — 10000 is a runaway guard, not a limit).
    ESCALATE_MULT  = 3
    ESCALATE_RANGE = 10000

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
                    frame = self._acquire_keeper()
                    if frame is not None:
                        img_num = row * cols + col + 1   # positional filename
                        # A field that stayed soft even after the escalated
                        # focus search is saved for the record but tagged, so
                        # analysis (which only accepts pure NNN names) skips it.
                        tag = "_soft" if self._last_soft else ""
                        path = os.path.join(self._out, f"{img_num:03d}{tag}.jpg")
                        if tag:
                            print(f"[capture] image {img_num:03d} still out of "
                                  f"focus after escalated search — saved as "
                                  f"{os.path.basename(path)} (excluded from analysis)")
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

    # ── Review-aware capture (focus / operator gate) ──────────────────────────

    # Minimum horizontal-scratch pixels (and mask contrast) before the scratch
    # mask is trustworthy focus evidence.  Below these, the mask is either
    # absent or JPEG-grain phantoms — the metric then falls back to judging
    # the DUST SPECS instead (present on every frame, same slide plane, and
    # they visibly fuzz exactly when the frame goes soft).  Calibrated on the
    # 20-set / 2,400-frame archive of old-system captures.
    _FOCUS_MIN_PIXELS   = 1500
    _FOCUS_MIN_CONTRAST = 28.0
    _FOCUS_MIN_SPOTS    = 400

    # Scale factors.  The E/c normalisation is empirical: across visually
    # sharp archive frames the scratch-edge energy E scales ~linearly with
    # scratch darkness c (verified from contrast 33 → 83), so E/c is what
    # stays flat for sharp frames — the old E/c² over-punished dark scratches.
    # _SPEC_SCALE aligns the spec-fallback range with the scratch-tier range
    # so ONE threshold serves both.  In-focus frames read ≈4000-8000, soft
    # ones ≲2100 — default GUI threshold 3000 (bench-tested; sits in the wide
    # gap between the soft and sharp score bands).
    _FOCUS_SCALE = 10.0
    _SPEC_SCALE  = 0.76

    @staticmethod
    def _horizontal_scratch_mask(gray):
        """
        Isolate straight HORIZONTAL scratch structures — the thing we actually
        want in focus — reusing the 'accurate' detector's logic: darkness below
        a local background, a contrast threshold, then keep only connected dark
        regions that are clearly WIDER THAN TALL (a real horizontal scratch).
        Round dust/blobs (width ≈ height) and vertical features are rejected by
        the aspect gate, so they can't drive the focus score.  Returns
        (mask uint8 0/255, darkness float image) — darkness is reused by the
        focus score for contrast normalisation (avoids a second morphology).
        """
        import cv2
        bg       = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
        darkness = cv2.subtract(bg, gray)
        thr      = max(8.0, float(darkness.mean()) + float(darkness.std()))
        dark     = (darkness > thr).astype(np.uint8)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        mask = np.zeros_like(dark)
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 40:           continue   # too small to be a scratch
            if w < 2 * h:        continue   # not clearly horizontal (rejects round dust)
            if w < 30:           continue   # too short to be a scratch
            mask[labels == i] = 255

        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return mask, darkness.astype(np.float64)

    @classmethod
    def _focus_score(cls, frame) -> float:
        """
        Two-tier focus score, higher = sharper.

        Tier 1 — scratches: when the frame has a substantial, genuinely dark
        horizontal-scratch mask, measure the vertical-gradient energy along
        the scratches over their darkness (E/c).  A defocused line keeps its
        darkness but loses edge steepness, so E/c drops hard; on sharp frames
        E grows ~linearly with c, so the score stays flat regardless of how
        deep the scratches are.

        Tier 2 — dust specs: when scratch evidence is weak (few mask pixels
        or low contrast → phantom JPEG-grain squiggles), judge the dust specs
        instead: they exist on every frame, sit on the same slide plane, and
        fuzz out exactly when the frame goes soft.  Same E/c form on the spec
        pixels, scaled to the scratch tier's range.

        Returns +inf only when there are neither scratches nor specs to judge
        (blank field → treated as in focus).

        Calibrated on the 2,400-frame old-system archive: in-focus frames
        read ≈4000-8000 in BOTH tiers, soft/degraded ones ≲2100.  Default
        threshold 2500.
        """
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mask, darkness = cls._horizontal_scratch_mask(gray)
        sel = mask > 0
        if int(sel.sum()) >= cls._FOCUS_MIN_PIXELS:
            contrast = float(darkness[sel].mean()) + 1e-6
            if contrast >= cls._FOCUS_MIN_CONTRAST:
                gy = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
                edge_energy = float((gy[sel] ** 2).mean())
                return cls._FOCUS_SCALE * edge_energy / contrast

        # Spec fallback — small dark spots against a smooth local background.
        bg = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        dk = cv2.subtract(bg, gray).astype(np.float64)
        thr = max(8.0, dk.mean() + 2.0 * dk.std())
        spots = dk > thr
        if int(spots.sum()) < cls._FOCUS_MIN_SPOTS:
            return float("inf")
        g64 = gray.astype(np.float64)
        gx = cv2.Sobel(g64, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g64, cv2.CV_64F, 0, 1, ksize=3)
        edge_energy = float(((gx ** 2 + gy ** 2)[spots]).mean())
        contrast = float(dk[spots].mean()) + 1e-6
        return cls._SPEC_SCALE * cls._FOCUS_SCALE * edge_energy / contrast

    def _acquire_keeper(self):
        """
        Grab the frame to save for this position.  The DEFECT-NUDGE always runs
        first (in every mode — it's the core job), producing the best
        defect-avoided frame; the review mode then layers FOCUS handling on top:
          none      — nothing further.
          auto      — if the frame is soft, hand it to the operator.
          manual    — always hand the frame to the operator.
          autofocus — if the frame is soft, drive Z to refocus and re-capture.
        """
        self._last_soft = False             # set again only if autofocus gives up
        frame = self._best_frame()          # defect-nudge — ALWAYS
        if frame is None:
            return None

        mode = self._review_mode
        if mode == "none":
            return frame

        if mode == "autofocus":
            # Focus is objective: a frame is either sharp enough or not.  Only
            # soft frames (rare) trigger a Z search, so this fires seldom.
            if self._z_disabled or self._focus_score(frame) >= self._blur_thresh:
                return frame            # in focus, or Z unavailable — keep as-is
            return self._autofocus(frame)

        if mode == "auto":
            if self._focus_score(frame) >= self._blur_thresh:
                return frame
            return self._operator_review(frame, "blurry")

        # manual: every frame goes to the operator
        return self._operator_review(frame, "manual")

    def _operator_review(self, frame, reason):
        """
        Show `frame` to the operator and act on Enter (good) / Space (bad).
        Enter keeps the frame; Space logs it as bad, then RE-captures the same
        position (the operator adjusts focus during the pause) and asks again.
        Loops until kept or the run stops.  Reviewed frames are copied to the
        training dataset (kept→good, rejected→bad).
        """
        while not self._stop_event.is_set():
            decision = self._review_fn(frame, reason)
            if self._stop_event.is_set():
                return frame
            if decision == "good":
                self._save_training(frame, self._good_dir)
                return frame
            # bad → log it and retake at the same spot (focus was adjusted).
            # Retake through _best_frame so the defect-nudge still runs.
            self._save_training(frame, self._bad_dir)
            retake = self._best_frame()
            if retake is None:
                return frame
            frame  = retake
            reason = "retake"
        return frame

    def _save_training(self, frame, dest_dir):
        """Copy a reviewed frame into the ML training dataset (best-effort)."""
        if not dest_dir:
            return
        try:
            os.makedirs(dest_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            ms    = int(time.time() * 1000) % 1000
            self._save(frame, os.path.join(dest_dir, f"{stamp}_{ms:03d}.jpg"))
        except Exception:
            pass   # never let dataset bookkeeping break a scan

    def _autofocus(self, soft_frame):
        """
        Run the Z focus search, but degrade gracefully if the Z stepper doesn't
        respond — e.g. the focus motor isn't wired yet, or the firmware predates
        MOVE Z.  On the first such failure we disable Z for the rest of the run
        and keep the (soft) frame we already have, so a missing focus axis can
        never abort a whole scan.
        """
        try:
            return self._autofocus_search(soft_frame)
        except RuntimeError as e:
            self._z_disabled = True
            print(f"[capture] autofocus disabled — Z stepper not responding ({e}). "
                  f"Keeping frames as captured for the rest of this run.")
            return soft_frame

    def _autofocus_search(self, soft_frame):
        """
        Drive the Z stepper to bring a soft field into focus, then re-capture.

        We don't know a priori which way is "into focus" (and there's coupling
        backlash), so we probe ONE step each way from the current height, take
        the uphill direction, and hill-climb until the focus score stops rising
        — i.e. we just passed the peak.  The search samples cheap RAW frames
        (focus only); the final keeper at the best height is taken through
        _best_frame so the defect-nudge still runs on the saved image.

        Robustness notes:
          • We keep the height with the best SCORE, and the saved image is taken
            there — so residual backlash in the final Z position doesn't hurt the
            image, it only sets the (nearby) start for the next field, which will
            re-focus itself anyway.
          • The winning probe direction is remembered (self._z_dir) so the next
            field's search tries the likely direction first — one probe saved
            per field once the rig's Z sense is known.
          • If the peak found is still below the blur threshold, the search
            ESCALATES once: wider probes (×3 step) over the full ±ESCALATE_RANGE.
            The Z axis is a continuous roller with no hard limit, so the range
            bound is only a runaway guard.
          • A field still soft after escalation is kept (best we have) but
            flagged via self._last_soft, which tags the saved filename so
            analysis skips it.
        """
        cur_z = 0                                   # net Z applied during this search
        s0    = self._focus_score(soft_frame)
        best_s, best_z = s0, 0

        def go_to(target_z):
            nonlocal cur_z
            d = target_z - cur_z
            if d != 0:
                self._motor.move("Z", d)
                cur_z = target_z
                time.sleep(self.SETTLE_S)           # let focus settle before sampling

        def score_at(target_z):
            go_to(target_z)
            f = self._wait_for_frame()
            return (float("-inf") if f is None else self._focus_score(f))

        def climb(step, bound, best_s, best_z):
            """Probe one step each way from best_z (learned direction first),
            then hill-climb the uphill way until a step stops improving.
            Remembers the winning direction for the next search."""
            up = best_z + self._z_dir * step
            s_up = score_at(up)
            if s_up > best_s:
                best_s, best_z, direction = s_up, up, self._z_dir
            else:
                dn = best_z - self._z_dir * step
                s_dn = score_at(dn)
                if s_dn > best_s:
                    best_s, best_z, direction = s_dn, dn, -self._z_dir
                else:
                    return best_s, best_z          # neither way improves
            self._z_dir = direction                # learned: probe here first next time
            while not self._stop_event.is_set():
                target = best_z + direction * step
                if abs(target) > bound:            # runaway guard
                    break
                s = score_at(target)
                if s > best_s:
                    best_s, best_z = s, target
                else:
                    break                          # passed the peak
            return best_s, best_z

        best_s, best_z = climb(self._z_step, self._z_range, best_s, best_z)

        # Still soft?  Escalate once: wider probes, full roller range.  Catches
        # fields whose focus peak lies beyond the normal search's reach.
        if best_s < self._blur_thresh and not self._stop_event.is_set():
            best_s, best_z = climb(self._z_step * self.ESCALATE_MULT,
                                   self.ESCALATE_RANGE, best_s, best_z)

        self._last_soft = best_s < self._blur_thresh
        # Settle at the best height and take the defect-nudged keeper there.
        # (best_z is 0 when nothing improved — that restores the start height.)
        go_to(best_z)
        return self._best_frame()

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
            frame, self._x_spacing, self._y_spacing, self._nudge_scale
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
                    y_spacing: int,
                    nudge_scale: float = 0.4) -> tuple[int, int]:
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

    nudge_scale is the "defect jump" strength from the GUI (fraction of one grid
    step per unit of off-centre offset).  Raise it if defects only move a little
    and stay on screen; the historical fixed value was 0.4.

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
    NUDGE_SCALE  = nudge_scale   # "defect jump" strength (GUI-tunable)

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


def focus_score(frame) -> float:
    """
    Module-level access to the two-tier focus metric (higher = sharper,
    +inf = blank field with nothing to judge).  Used by the GUI's
    Auto-calibrate tour, which scores frames without building a pipeline.
    """
    return CapturePipeline._focus_score(frame)
