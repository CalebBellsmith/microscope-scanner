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

# Row-projection: fraction of feature pixels that must fall on horizontal rows.
_HORIZONTAL_COVERAGE_GOOD = 0.40

# Blob / defect detection uses SHAPE (aspect ratio), not column span.
# Physical invariant measured on real frames:
#   horizontal scratch  → elongated, aspect (w/h) 16–53  (1-D line)
#   defect (blob/fibre)  → 2-D extent, aspect 0.7–4.7
# Anything with aspect ≥ this is treated as a horizontal line and ignored,
# regardless of how wide or thick it is.  This is what lets a WIDE fibre be
# detected (it has low aspect) instead of being mistaken for a scratch.
_ASPECT_LINE_MIN = 8.0

# FFT residual check: after zeroing near-zero horizontal frequencies, the
# fraction of the remaining std vs the original std.
# Near 0  → frame is dominated by horizontal content (lines, scratches) → good
# Near 1  → frame has significant non-horizontal energy (blobs, watermarks) → bad
_FFT_H_BAND_FRAC  = 0.05    # fraction of the kx frequency range treated as "horizontal"
_FFT_RESIDUAL_BAD = 0.45    # residual/original ratio above which is flagged as bad
_FFT_RESIDUAL_CERTAIN = 0.72  # standalone bad flag without needing blob confirmation

# Darkening gate (sensitivity-scaled at call time): a contour is only a defect
# if its interior is meaningfully darker than the background mean.
# Measured on real images:
#   grey halos (unavoidable focus artifacts): 17–21% darker than background
#   real defects (fibres, debris, solid blobs): 30–63% darker
# The gate runs from 0.30 (lenient) down to 0.20 (strict); the lenient end sits
# above the grey-halo band so halos are never flagged.


def _fft_residual_ratio(gray: np.ndarray) -> float:
    """
    2D real-FFT analysis: strip horizontal frequency content (low kx), return
    residual_std / original_std.

    Horizontal scratches (regardless of thickness) concentrate their energy at
    kx ≈ 0.  After zeroing that band the residual is near zero.  Dust blobs and
    other localised defects spread energy across all kx values so their residual
    survives the stripping and the ratio stays high.

    This is the key discriminator: it is thickness-agnostic because even a thick
    horizontal line produces energy only at low kx.
    """
    std = float(gray.std())
    if std < 1.0:
        return 0.0     # nearly uniform image — no structure to analyse

    gray_f = gray.astype(np.float32)
    F      = np.fft.rfft2(gray_f)          # shape (H, W//2+1)
    _, W_f = F.shape

    # Zero out near-zero kx columns (horizontal frequency band to remove)
    band       = max(2, int(W_f * _FFT_H_BAND_FRAC))
    F_residual = F.copy()
    F_residual[:, :band] = 0

    residual = np.fft.irfft2(F_residual, s=gray.shape)
    return float(residual.std() / (std + 1e-6))


def _rule_predict(rgb_array: np.ndarray, sensitivity: float = 0.5) -> tuple[str, float]:
    """
    Three-check quality gate:

    CHECK 1 — Row projection (catches frames with no horizontal features at all):
      Pixels that deviate significantly from the mean (in either direction) are
      "feature pixels".  If ≥ 40% of them fall on high-density rows the frame
      is considered horizontally structured.  Works in both camera modes
      (scratches appear dark when put_Negative is a no-op, bright otherwise).

    CHECK 2 — Blob contour detection (catches localised defects):
      Find all dark contours.  Flag any contour that spans < 15% of image width
      and is not extremely elongated.  A real scratch spans the full frame width;
      a dust spot is compact.

    CHECK 3 — FFT residual (thickness-agnostic horizontal test):
      Strip horizontal frequency content from the 2D FFT.  Measure how much
      signal remains.  Horizontal lines — regardless of thickness — leave almost
      no residual.  Blobs and non-horizontal artefacts leave a large residual.

      This is the primary fix for thick lines being over-flagged: even if their
      edge-fragments trigger the blob check, the FFT will disagree (low residual)
      so the two-check consensus won't fire.

    Logic:
      • row_bad                    → bad (no horizontal structure at all)
      • blob_bad                   → bad (localised dark contour found)
      • fft_ratio > 0.65           → bad (very strong non-horizontal signal;
                                         catches diffuse defects with no clear contour)
      • otherwise                  → good

    Note: thick lines do NOT trigger blob_bad because their bounding rect spans
    the full frame width (col_span > 0.15) — they are correctly excluded by
    the column-span test.  FFT confirmation is therefore NOT needed to protect
    thick lines from the blob gate.
    """
    import cv2

    gray   = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    mean_v = float(gray.mean())
    std_v  = float(gray.std())

    if std_v < 2.0:
        return "good", 1.0

    img_h, img_w = gray.shape
    img_pixels   = img_h * img_w

    # ── Check 1: row projection ───────────────────────────────────────────────
    deviation  = np.abs(gray.astype(np.int16) - int(mean_v))
    feature    = deviation > (1.5 * std_v)
    total_feat = int(feature.sum())

    horizontal_frac = 1.0   # default: assume good if barely any features
    if total_feat >= 50:
        row_density = feature.mean(axis=1)
        row_mean    = float(row_density.mean())
        row_std     = float(row_density.std())
        if row_std > 1e-6:
            h_rows          = row_density > (row_mean + 0.5 * row_std)
            horizontal_frac = int(feature[h_rows].sum()) / total_feat

    row_bad      = horizontal_frac < _HORIZONTAL_COVERAGE_GOOD
    row_bad_conf = round(1.0 - horizontal_frac, 4)

    # ── Sensitivity-scaled thresholds ────────────────────────────────────────
    # sensitivity 0.0 = lenient (only large, clearly-dark defects)
    # sensitivity 1.0 = strict  (flag smaller, fainter defects)
    s = max(0.0, min(1.0, sensitivity))
    blob_min_area = int(1600 - 1500 * s)   # lenient=1600 px²  strict=100 px²
    min_dark      = 0.30 - 0.10 * s        # lenient=0.30      strict=0.20
    fft_gate      = 0.75 - 0.25 * s        # lenient=0.75      strict=0.50

    # ── Check 2: blob / fibre detection (shape-based) ────────────────────────
    # Find dark contours, then keep only those that are NOT horizontal lines.
    # A scratch is rejected by its high aspect ratio (elongated), so a wide
    # fibre — which has a LOW aspect ratio despite being wide — is correctly
    # flagged instead of being mistaken for a scratch (the old col_span bug).
    thr = max(0, int(mean_v - 1.5 * std_v))
    _, dark_mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    contours, _  = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    worst_blob_frac = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < blob_min_area:
            continue
        _, _, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        # Elongated horizontal line (scratch) → ignore, whatever its thickness
        if aspect >= _ASPECT_LINE_MIN:
            continue

        # Darkening gate: grey halos (focus/lens artifacts) are only ~17–21%
        # darker than the background.  Real defects (fibres, debris) are 30%+
        # darker.  Skip anything not dark enough to be a genuine defect.
        cnt_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        mean_inside = float(cv2.mean(gray, mask=cnt_mask)[0])
        darkening   = (mean_v - mean_inside) / (mean_v + 1e-6)
        if darkening < min_dark:
            continue   # soft grey halo — not a flaggable defect

        worst_blob_frac = max(worst_blob_frac, area / img_pixels)

    blob_bad      = worst_blob_frac > 0.0005
    blob_bad_conf = round(min(worst_blob_frac / 0.005, 1.0), 4)

    # ── Check 3: FFT residual ─────────────────────────────────────────────────
    fft_ratio = _fft_residual_ratio(gray)

    fft_certain  = fft_ratio > fft_gate
    fft_bad      = fft_ratio > _FFT_RESIDUAL_BAD
    fft_bad_conf = round(
        min((fft_ratio - _FFT_RESIDUAL_BAD) / (1.0 - _FFT_RESIDUAL_BAD), 1.0), 4
    ) if fft_bad else 0.0

    # ── Combine ───────────────────────────────────────────────────────────────
    # No horizontal structure at all
    if row_bad:
        return "bad", row_bad_conf

    # Localised dark contour found (thick lines can't trigger this — they span
    # the full width and fail the col_span < 0.15 guard)
    if blob_bad:
        return "bad", blob_bad_conf

    # Strong non-horizontal FFT signal — catches diffuse defects that don't
    # produce a clear contour (watermarks, gradients, texture anomalies)
    if fft_certain:
        return "bad", fft_bad_conf

    # Good: all checks passed
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
        Given a list of RGB frames captured from the current slide, suggest a
        quality threshold tailored to this slide's typical appearance.

        Algorithm: classify each frame, collect confidence scores of frames that
        classify as "good", then return the 25th-percentile score minus a small
        safety margin.  This sets the bar just below the weakest good frame so
        that most frames on this slide pass while still flagging clear defects.

        Returns a threshold in [0.1, 0.8].  Returns 0.5 if no good frames found.
        """
        if not frames:
            return 0.5
        good_confs = []
        for frame in frames:
            label, conf = self.predict(frame)
            if label == "good":
                good_confs.append(conf)
        if not good_confs:
            return 0.5
        good_confs.sort()
        # 25th percentile of good-frame confidences, backed off 15%
        p25 = good_confs[max(0, len(good_confs) // 4)]
        suggested = round(max(0.1, min(0.8, p25 * 0.85)), 1)
        return suggested

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
