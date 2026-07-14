"""
ML quality classifier — binary: good (1) vs bad (0).

THREE MODES
───────────
"rules"   — Pure shape analysis (no ML required).
            Three checks: row projection, blob contour detection, and FFT
            residual.  Passes frames where all dark features are horizontal
            (regardless of thickness).  Flags dust, blobs, watermarks.
            Fast, interpretable, works immediately with no training data.

"ml"      — Pure ML (MobileNetV3-Small fine-tuned on your labeled images).
            Runs inside inference_worker.py subprocess to avoid WinError 1114.
            Requires model.onnx (or model.pt) to be present.

"hybrid"  — Rules first.  If the rule-based confidence is high (≥ 0.75) the
            answer is used directly.  When the rules are uncertain (borderline
            frame) the ML model is also consulted and the scores are blended.
            Best of both worlds: rules handle the clear cases reliably and
            quickly; ML covers edge cases that are hard to express as geometry.

Select the mode via QualityClassifier(mode=...) or by changing .mode at
runtime (the labeling tool exposes this as a dropdown for testing).

"good"  = frame is acceptable for saving — only horizontal scratches present
"bad"   = dust, debris, watermarks, non-horizontal artefacts detected
"""

import math
import os
import sys
import base64
import atexit
import subprocess
import threading

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH  = os.path.join(_HERE, "model.onnx")
MODEL_PATH = os.path.join(_HERE, "model.pt")
_WORKER    = os.path.join(_HERE, "inference_worker.py")


# ── Rule-based classifier ─────────────────────────────────────────────────────

# Blob / defect detection uses SHAPE (aspect ratio), not column span.
# Physical invariant measured on real frames:
#   horizontal scratch  → elongated, aspect (w/h) 16–53  (1-D line)
#   defect (blob/fibre)  → 2-D extent, aspect 0.7–4.7
# Anything with aspect ≥ this is treated as a horizontal line and ignored,
# regardless of how wide or thick it is.  This is what lets a WIDE fibre be
# detected (it has low aspect) instead of being mistaken for a scratch.
_ASPECT_LINE_MIN = 8.0

# Scratch-erasure before blob hunting: a horizontal opening of this run length
# finds scratch RUNS pixel-by-pixel, which are then subtracted from the dark
# mask (after a small dilation so scratch edges go too).  This replaces the old
# whole-contour aspect test for merged structures: on dense scratch fields the
# individual lines fuse into one 2-D cluster whose bounding-box aspect is < 8,
# which the old test mistook for a giant blob (→ every heavy-abrasion frame
# read "bad" and the nudge fired uselessly on unavoidable structure).  Erasing
# the runs first leaves only genuinely compact objects for the blob scan, and
# still isolates a blob that sits ON a scratch line.
_SCRATCH_RUN     = 15   # px — min horizontal run treated as scratch structure
_SPECKLE_OPEN    = 5    # px — ellipse opening that drops leftover grain/speckle

# FFT residual check: after zeroing near-zero horizontal frequencies, the
# fraction of the remaining std vs the original std.
# Near 0  → frame is dominated by horizontal content (lines, scratches) → good
# Near 1  → frame has significant non-horizontal energy (blobs, watermarks) → bad
# LOCALISATION: uniform substrate texture (PET film) also leaves a large global
# residual, but a nudge can't move away from texture that covers the whole
# frame — so the residual must additionally be spatially CONCENTRATED
# (peak block std ≥ _FFT_LOC_MIN × median block std) before it means "defect".
_FFT_H_BAND_FRAC  = 0.05    # fraction of the kx frequency range treated as "horizontal"
_FFT_RESIDUAL_BAD = 0.45    # residual/original ratio above which is flagged as bad
_FFT_LOC_BLOCKS   = 8       # residual is judged on an 8×8 grid of blocks
_FFT_LOC_MIN      = 3.0     # peak/median block-std ratio: texture ≈1-2, defect ≳3

# Darkening gate (sensitivity-scaled at call time): a contour is only a defect
# if its interior is meaningfully darker than the background mean.
# Measured on real images:
#   grey halos (unavoidable focus artifacts): 17–21% darker than background
#   real defects (fibres, debris, solid blobs): 30–63% darker
# The gate runs from 0.30 (lenient) down to 0.20 (strict); the lenient end sits
# above the grey-halo band so halos are never flagged.


def _fft_residual_ratio(gray: np.ndarray):
    """
    2D real-FFT analysis: strip horizontal frequency content (low kx), return
    (residual_std / original_std, residual image) — the residual image feeds
    the localisation check.

    Horizontal scratches (regardless of thickness) concentrate their energy at
    kx ≈ 0.  After zeroing that band the residual is near zero.  Dust blobs and
    other localised defects spread energy across all kx values so their residual
    survives the stripping and the ratio stays high.

    This is the key discriminator: it is thickness-agnostic because even a thick
    horizontal line produces energy only at low kx.
    """
    std = float(gray.std())
    if std < 1.0:
        return 0.0, np.zeros_like(gray, np.float32)  # nearly uniform image

    gray_f = gray.astype(np.float32)
    F      = np.fft.rfft2(gray_f)          # shape (H, W//2+1)
    _, W_f = F.shape

    # Zero out near-zero kx columns (horizontal frequency band to remove)
    band       = max(2, int(W_f * _FFT_H_BAND_FRAC))
    F_residual = F.copy()
    F_residual[:, :band] = 0

    residual = np.fft.irfft2(F_residual, s=gray.shape)
    return float(residual.std() / (std + 1e-6)), residual


def _residual_localisation(residual: np.ndarray) -> float:
    """
    How spatially concentrated the non-horizontal residual is: the frame is
    split into an _FFT_LOC_BLOCKS² grid and each block's std measured.
    Returns peak/median block std — ≈1-2 for uniform substrate texture
    (energy everywhere), ≳3 when one region holds the defect.
    """
    H, W = residual.shape
    bh, bw = H // _FFT_LOC_BLOCKS, W // _FFT_LOC_BLOCKS
    if bh == 0 or bw == 0:
        return 1.0
    stds = [
        float(residual[r*bh:(r+1)*bh, c*bw:(c+1)*bw].std())
        for r in range(_FFT_LOC_BLOCKS) for c in range(_FFT_LOC_BLOCKS)
    ]
    stds.sort()
    med = stds[len(stds)//2]
    return stds[-1] / (med + 1e-6)


def _rule_predict(rgb_array: np.ndarray, sensitivity: float = 0.5) -> tuple[str, float]:
    """
    Two-check quality gate.  A frame is GOOD unless a check positively
    identifies a defect — absence of features is good (a clean slide).

    CHECK 1 — Blob / fibre detection (shape-based, scratch-erased):
      Work on DARKNESS below the local background (large morphological close),
      thresholded adaptively against the frame's own darkness statistics — on
      grainy substrates (PET film) the bar rises with the texture, so grain
      never reaches the blob scan.  Horizontal scratch RUNS are then erased
      from the mask pixel-by-pixel (they are the sample's own structure, at
      any density), a small opening drops leftover speckle, and whatever
      remains compact, large and genuinely dark is a nudgeable defect.
      Erasing runs (not whole contours) keeps working when dense scratches
      merge into 2-D clusters, and still isolates a blob sitting ON a line.

    CHECK 2 — FFT residual (diffuse non-horizontal signal), localised:
      Strip horizontal frequency content from the 2D FFT; horizontal lines —
      any thickness — leave almost no residual.  A large residual is only a
      defect if it is spatially CONCENTRATED (watermark, smudge, fibre);
      uniform substrate texture also leaves a big residual, but covers the
      whole frame — no stage nudge can avoid it, so it must read "good".

    Logic:
      • blob_bad                    → bad (localised dark non-line object)
      • fft_certain AND localised   → bad (diffuse but concentrated defect)
      • otherwise                   → good (clean, scratches, or texture)
    """
    import cv2

    gray   = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    mean_v = float(gray.mean())
    std_v  = float(gray.std())

    if std_v < 2.0:
        return "good", 1.0

    img_h, img_w = gray.shape
    img_pixels   = img_h * img_w

    # ── Sensitivity-scaled thresholds ────────────────────────────────────────
    # sensitivity 0.0 = lenient (only large, clearly-dark defects)
    # sensitivity 1.0 = strict  (flag smaller, fainter defects)
    s = max(0.0, min(1.0, sensitivity))
    blob_min_area = int(1600 - 1500 * s)   # lenient=1600 px²  strict=100 px²
    min_dark      = 0.30 - 0.10 * s        # lenient=0.30      strict=0.20
    fft_gate      = 0.75 - 0.25 * s        # lenient=0.75      strict=0.50

    # ── Check 1: blob / fibre detection ──────────────────────────────────────
    # Darkness below the local background, thresholded against the frame's own
    # statistics: clean glass → low bar (faint fibres caught); grainy PET →
    # bar above the grain, only genuinely dark objects survive.
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    darkness = cv2.subtract(bg, gray)
    d_mean, d_std = float(darkness.mean()), float(darkness.std())
    # Local term: dark vs the local background — catches thin fibres/edges even
    # under uneven lighting, with the bar riding above the substrate grain.
    # Global term: absolutely dark — a solid object WIDER than the close kernel
    # becomes its own "background" (local darkness reads 0 inside it), so big
    # debris is only visible to an absolute threshold.
    dark_mask = (
        (darkness > max(15.0, d_mean + 2.5 * d_std))
        | (gray < mean_v - 1.5 * std_v)
    ).astype(np.uint8) * 255

    def _elongation(cnt) -> float:
        """Rotated-rect elongation — unlike bbox aspect it also reads tilted
        line remnants as lines (a 213×27 bbox at 6° is elongation ~12)."""
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        return max(rw, rh) / max(min(rw, rh), 1.0)

    def _dark_enough(cnt_or_mask) -> bool:
        mean_inside = float(cv2.mean(gray, mask=cnt_or_mask)[0])
        return (mean_v - mean_inside) / (mean_v + 1e-6) >= min_dark

    worst_blob_frac = 0.0

    # Pass A — CURVED FIBRES: judge each connected component WHOLE by how much
    # of it lies in horizontal scratch RUNS.  Scratches — however dense, even
    # merged into 2-D clusters — are made of long runs (coverage ≈ 1); a curved
    # fibre only touches runs where its tangent goes horizontal (coverage ≲ 0.3).
    # Judging coverage instead of erasing runs keeps the fibre in one piece.
    run_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_SCRATCH_RUN, 1))
    scratch_runs = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, run_kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (dark_mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < blob_min_area:
            continue
        comp = (labels[y:y+h, x:x+w] == i)
        run_cov = float(scratch_runs[y:y+h, x:x+w][comp].mean()) / 255.0
        if run_cov >= 0.5:
            continue                     # scratch structure (line/cluster/smear)
        comp_mask = np.zeros(gray.shape, np.uint8)
        comp_mask[y:y+h, x:x+w][comp] = 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if _elongation(cnt) >= _ASPECT_LINE_MIN:
            continue                     # straight line at any angle → not debris
        if not _dark_enough(comp_mask):
            continue                     # soft grey halo → ignore
        worst_blob_frac = max(worst_blob_frac, a / img_pixels)

    def _line_continues(x, y, w, h) -> bool:
        """A thick scratch SEGMENT looks compact after opening, but the scratch
        it belongs to continues beyond the segment at similar thickness; debris
        ends where it ends (a blob ON a thin line only continues thinly).  Look
        40 px left and right of the bbox: if the dark rows there amount to at
        least half the candidate's thickness, this is scratch structure."""
        need = max(2, h // 2)
        for x0, x1 in ((max(0, x - 40), x), (x + w, min(img_w, x + w + 40))):
            if x1 - x0 < 15:
                continue                 # at the frame edge — can't judge this side
            strip = dark_mask[y:y + h, x0:x1]
            rows = int((strip.sum(axis=1) / 255 >= 8).sum())
            if rows >= need:
                return True
        return False

    # Pass B — SOLID BLOBS: an ellipse opening erases thin lines and fibres in
    # any direction but a compact chunk keeps its core.  The floor is a real
    # chunk size (≥1200 px² ≈ 20 px radius) so knots where thick scratches
    # overlap don't read as debris; elongated remnants drop via elongation; a
    # remnant whose structure continues past its ends is a scratch segment.
    blob_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blob_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, blob_kernel)
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)      # outer contour → holes don't shrink it
        if area < max(blob_min_area, 1200):
            continue
        if _elongation(cnt) >= _ASPECT_LINE_MIN:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if _line_continues(bx, by, bw, bh):
            continue                     # thick scratch segment → sample structure
        cnt_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        if not _dark_enough(cnt_mask):
            continue
        worst_blob_frac = max(worst_blob_frac, area / img_pixels)

    # Pass C — BIG DEBRIS RIDING ON A SCRATCH: when a chunk sits on a line of
    # comparable mask thickness, Pass B's opening reconnects them into one
    # elongated contour and the chunk is dropped as "line".  A hard erosion
    # with NO re-dilation kills any line ≤14 px thick outright, so only the
    # chunk's core is left standing, wherever it sits.
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    core_mask = cv2.erode(dark_mask, core_kernel)
    contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 250:   # core of a ≥~36 px chunk after the −7 px rim
            continue
        if _elongation(cnt) >= _ASPECT_LINE_MIN:
            continue                     # thick smear core → sample structure
        # measure the chunk at (roughly) its true extent: re-dilate just this core
        cm = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, -1)
        cm = cv2.dilate(cm, core_kernel)
        if not _dark_enough(cm):
            continue
        area = float((cm > 0).sum())
        worst_blob_frac = max(worst_blob_frac, area / img_pixels)

    blob_bad      = worst_blob_frac > 0.0005
    blob_bad_conf = round(min(worst_blob_frac / 0.005, 1.0), 4)

    # ── Check 2: FFT residual, localised ─────────────────────────────────────
    fft_ratio, residual = _fft_residual_ratio(gray)
    localised = _residual_localisation(residual) >= _FFT_LOC_MIN

    fft_certain  = fft_ratio > fft_gate and localised
    fft_bad      = fft_ratio > _FFT_RESIDUAL_BAD and localised
    fft_bad_conf = round(
        min((fft_ratio - _FFT_RESIDUAL_BAD) / (1.0 - _FFT_RESIDUAL_BAD), 1.0), 4
    ) if fft_bad else 0.0

    # ── Combine ───────────────────────────────────────────────────────────────
    # Localised dark non-line contour found (a blob/fibre/dust defect).
    if blob_bad:
        return "bad", blob_bad_conf

    # Strong non-horizontal FFT signal — catches diffuse defects that don't
    # produce a clear contour (watermarks, gradients, texture anomalies).
    if fft_certain:
        return "bad", fft_bad_conf

    # Good: a clean frame, or one whose only features are horizontal scratches.
    # (The row-projection check was removed: it punished clean/sparse frames for
    #  "lacking horizontal structure" and oscillated on the 0.40 boundary.
    #  Absence of structure is GOOD here — defects are caught by blob + FFT.)
    good_conf = round(1.0 - max(blob_bad_conf * 0.5, fft_bad_conf * 0.5), 4)
    return "good", max(good_conf, 0.55)


# ── ML worker process management ──────────────────────────────────────────────

_worker_proc = None
_worker_lock = threading.Lock()   # one inference request at a time


def _get_worker() -> subprocess.Popen:
    """
    Return the inference worker subprocess, spawning it if needed.
    The worker loads model.onnx (or model.pt as fallback) once and stays alive.
    All torch/onnxruntime DLLs are loaded in this separate process to avoid
    WinError 1114 in the PyQt5 GUI thread.
    """
    global _worker_proc
    with _worker_lock:
        if _worker_proc is not None and _worker_proc.poll() is None:
            return _worker_proc

        _worker_proc = subprocess.Popen(
            [sys.executable, _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        while True:
            line = _worker_proc.stdout.readline().strip()
            if line == "ready":
                break
            if line.startswith("loaded"):
                print(f"[inference_worker] {line}", flush=True)
            elif line.startswith("onnx_failed") or line.startswith("pt_failed"):
                print(f"[inference_worker] {line}", flush=True)
            elif not line:
                raise RuntimeError("inference_worker failed to start")

        return _worker_proc


def _shutdown_worker():
    """Send quit to worker on program exit so it closes cleanly."""
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        try:
            _worker_proc.stdin.write("quit\n")
            _worker_proc.stdin.flush()
            _worker_proc.wait(timeout=3)
        except Exception:
            _worker_proc.kill()


atexit.register(_shutdown_worker)


def _ml_predict(rgb_array: np.ndarray) -> tuple[str, float]:
    """
    Run ML inference via the persistent inference_worker subprocess.
    Raises on failure so callers can fall back gracefully.
    """
    worker = _get_worker()

    h, w, c = rgb_array.shape
    b64     = base64.b64encode(rgb_array.tobytes()).decode()
    msg     = f"{h} {w} {c} {b64}\n"

    with _worker_lock:
        worker.stdin.write(msg)
        worker.stdin.flush()
        response = worker.stdout.readline().strip()

    if response.startswith("error"):
        raise RuntimeError(response)

    label, conf_str = response.split()
    return label, float(conf_str)


# ── Hybrid classifier ─────────────────────────────────────────────────────────

# Rule confidence must exceed this to skip consulting the ML model
_RULE_CONFIDENCE_THRESHOLD = 0.75

# Weight given to rules vs ML when both are consulted (must sum to 1.0)
_RULE_WEIGHT = 0.6
_ML_WEIGHT   = 0.4

def _hybrid_predict(rgb_array: np.ndarray, sensitivity: float = 0.5) -> tuple[str, float]:
    """
    Combine rules and ML:
      - Run rule-based check first (fast, no subprocess).
      - If rule confidence ≥ 0.75 → use rule result directly.
      - Otherwise → also run ML, blend the signed confidence scores.

    Signed score convention: +conf means "good", −conf means "bad".
    The blend is rule-weighted (60/40) since rules are more physically
    meaningful for this specific task.
    """
    rule_label, rule_conf = _rule_predict(rgb_array, sensitivity)

    # Rules are highly confident — no need to call ML
    if rule_conf >= _RULE_CONFIDENCE_THRESHOLD:
        return rule_label, rule_conf

    # Rules uncertain — bring in ML opinion
    try:
        ml_label, ml_conf = _ml_predict(rgb_array)
    except Exception as e:
        print(f"[ml_inference] ML unavailable in hybrid mode ({e}), using rules only")
        return rule_label, rule_conf

    # Convert to signed scores (+good / −bad) for weighted average
    rule_score = rule_conf  if rule_label == "good" else -rule_conf
    ml_score   = ml_conf    if ml_label   == "good" else -ml_conf

    combined = _RULE_WEIGHT * rule_score + _ML_WEIGHT * ml_score

    label = "good" if combined >= 0 else "bad"
    conf  = round(min(abs(combined), 1.0), 4)
    return label, conf


# ── Public classifier class ───────────────────────────────────────────────────

MODES = ("hybrid", "rules", "ml")   # valid mode strings

class QualityClassifier:
    """
    Unified quality classifier supporting three modes.

    mode="hybrid"  (default) — rules first, ML for borderline cases
    mode="rules"             — shape-based only, no ML needed
    mode="ml"                — ML only, falls back to heuristic if model missing

    Change .mode at runtime to switch without recreating the object.
    The labeling tool exposes this as a dropdown so you can compare modes live.
    """

    def __init__(self, mode: str = "hybrid", sensitivity: float = 0.5):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        # 0.0 = lenient (flag only large obvious defects)
        # 1.0 = strict  (flag smaller, more subtle defects)
        self.sensitivity = sensitivity

    def predict(self, rgb_array: np.ndarray) -> tuple[str, float]:
        """
        Classify a frame.
        rgb_array : numpy uint8 array shaped (H, W, 3)
        Returns   : ("good"|"bad", confidence 0.0–1.0)
        """
        if self.mode == "rules":
            return _rule_predict(rgb_array, self.sensitivity)

        if self.mode == "ml":
            try:
                return _ml_predict(rgb_array)
            except Exception as e:
                print(f"[ml_inference] ML predict failed ({e}), using heuristic")
                return _heuristic(rgb_array)

        # Default: hybrid
        return _hybrid_predict(rgb_array, self.sensitivity)

    def is_good(self, rgb_array: np.ndarray) -> bool:
        label, _ = self.predict(rgb_array)
        return label == "good"

    def calibrate(self, frames: list) -> float:
        """
        Suggest a detection SENSITIVITY tuned to this slide.

        The operator points the camera at a clean, representative region, then we
        sweep sensitivity from lenient → strict and, at each level, measure what
        fraction of the sample frames still read as "good".  We pick the
        STRICTEST sensitivity at which the clean reference still passes (≥ 80% of
        frames good) — the tightest defect gate this slide tolerates without
        false-flagging its own normal texture.

        Why not the old algorithm: it took the 25th-percentile "good" confidence
        and backed off 15%.  But a clean frame's good-confidence is essentially
        always ~1.0 here (see _rule_predict), so that figure saturated at the 0.8
        clamp every run → the slider always landed on 8 → the readout was 0.79
        no matter what the slide looked like.  Sweeping the actual decision
        boundary makes the result genuinely slide-specific and varies as it
        should.

        Returns a sensitivity in [0.0, 1.0]; 0.5 if no frames are supplied.
        """
        if not frames:
            return 0.5

        STEPS     = 20          # sensitivity grid: 0.00, 0.05, … 0.95
        PASS_FRAC = 0.8         # clean reference must still read ≥80% good

        saved = self.sensitivity
        best  = 0.0
        try:
            for i in range(STEPS):
                s = i / STEPS
                self.sensitivity = s
                good = sum(1 for f in frames if self.predict(f)[0] == "good")
                if good >= PASS_FRAC * len(frames):
                    best = s        # still clean at this strictness — try stricter
                else:
                    break           # this level starts false-flagging — stop here
        finally:
            self.sensitivity = saved
        return round(best, 2)

    def load(self):
        """
        Pre-warm the ML worker subprocess so the first prediction has no delay.
        Safe to call even in rules-only mode (just skips the worker spawn).
        """
        if self.mode == "rules":
            print("[ml_inference] rules mode — no ML worker needed")
            return
        try:
            _get_worker()
            print("[ml_inference] inference worker ready")
        except Exception as e:
            print(f"[ml_inference] worker pre-warm failed: {e}")


# ── Laplacian heuristic fallback ──────────────────────────────────────────────

def _heuristic(rgb_array: np.ndarray) -> tuple[str, float]:
    """
    Simple sharpness check used when ML model files are missing.
    High Laplacian variance → sharp image → "good".
    Not dust-aware — just a last-resort fallback.
    """
    import cv2
    gray  = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if score >= 80.0:
        return "good", min(score / 200.0, 1.0)
    else:
        return "bad", 1.0 - score / 80.0
