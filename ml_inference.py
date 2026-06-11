"""
ML quality classifier — binary: good (1) vs bad (0).

THREE MODES
───────────
"rules"   — Pure shape analysis (no ML required).
            Treats a frame as good if every dark feature is a horizontal
            scratch (low roundness, wider than tall).  Dust spots, blobs,
            and vertical lines fail the shape test → bad.
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

# What fraction of feature pixels must fall on horizontal rows to pass.
_HORIZONTAL_COVERAGE_GOOD = 0.60

def _rule_predict(rgb_array: np.ndarray) -> tuple[str, float]:
    """
    Row-projection quality check — mode-agnostic version.

    Previous versions looked only for DARK pixels, which broke in analysis
    mode because the camera applies a negative (inversion): horizontal
    scratches appear BRIGHT there, not dark.  Looking for dark pixels in
    an inverted image finds only the background, giving garbage results.

    Fix: use pixels that deviate significantly from the mean in EITHER
    direction — these are the "feature pixels" regardless of camera mode.
        analysis mode: scratches are bright  (above mean + σ)
        raw mode:      scratches are dark    (below mean − σ)
        both modes:    dust/blobs deviate locally from the background

    Row-projection logic (unchanged):
      Features that span many columns → raise entire rows above average density.
      Localised blobs → raise only a few isolated pixels, not whole rows.
      If ≥ 60% of feature pixels live on high-density rows → good.
    """
    import cv2

    gray   = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    mean_v = float(gray.mean())
    std_v  = float(gray.std())

    if std_v < 2.0:
        return "good", 1.0   # nearly uniform frame — nothing to flag

    # Feature pixels: significantly different from background in either direction
    deviation = np.abs(gray.astype(np.int16) - int(mean_v))
    feature   = deviation > (1.5 * std_v)   # shape (H, W) boolean

    total_feat = int(feature.sum())
    if total_feat < 50:
        return "good", 1.0   # barely any features — clean frame

    # Per-row density of feature pixels
    row_density = feature.mean(axis=1)   # shape (H,)
    row_mean    = float(row_density.mean())
    row_std     = float(row_density.std())

    if row_std < 1e-6:
        # All rows equally featureful — uniform noise, not a scratch
        return "good", 0.7

    # A "horizontal feature row" has significantly more feature pixels than average
    h_rows = row_density > (row_mean + 0.5 * row_std)

    # Fraction of feature pixels that live on those rows
    feat_on_h_rows  = int(feature[h_rows].sum())
    horizontal_frac = feat_on_h_rows / total_feat

    if horizontal_frac >= _HORIZONTAL_COVERAGE_GOOD:
        return "good", round(float(horizontal_frac), 4)
    else:
        return "bad", round(1.0 - float(horizontal_frac), 4)


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

def _hybrid_predict(rgb_array: np.ndarray) -> tuple[str, float]:
    """
    Combine rules and ML:
      - Run rule-based check first (fast, no subprocess).
      - If rule confidence ≥ 0.75 → use rule result directly.
      - Otherwise → also run ML, blend the signed confidence scores.

    Signed score convention: +conf means "good", −conf means "bad".
    The blend is rule-weighted (60/40) since rules are more physically
    meaningful for this specific task.
    """
    rule_label, rule_conf = _rule_predict(rgb_array)

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

    def __init__(self, mode: str = "hybrid"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    def predict(self, rgb_array: np.ndarray) -> tuple[str, float]:
        """
        Classify a frame.
        rgb_array : numpy uint8 array shaped (H, W, 3)
        Returns   : ("good"|"bad", confidence 0.0–1.0)
        """
        if self.mode == "rules":
            return _rule_predict(rgb_array)

        if self.mode == "ml":
            try:
                return _ml_predict(rgb_array)
            except Exception as e:
                print(f"[ml_inference] ML predict failed ({e}), using heuristic")
                return _heuristic(rgb_array)

        # Default: hybrid
        return _hybrid_predict(rgb_array)

    def is_good(self, rgb_array: np.ndarray) -> bool:
        label, _ = self.predict(rgb_array)
        return label == "good"

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
