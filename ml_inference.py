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

# Minimum contour area to consider — filters single-pixel noise and JPEG
# compression artefacts.  Anything above this that isn't a horizontal line
# is flagged immediately, regardless of size.
_MIN_NOISE_AREA = 25   # px²

# Once the total area of non-scratch features exceeds this fraction of the
# image the confidence saturates at 1.0.  Set very low so even a small blob
# scores high confidence.
_BAD_AREA_SATURATE = 0.0003   # 0.03% of image (~25px on a 1024×822 frame)

def _rule_predict(rgb_array: np.ndarray) -> tuple[str, float]:
    """
    Shape-based quality check — maximum sensitivity mode.

    Rule: ANYTHING that is not a horizontal line is flagged as bad.
    A feature is a horizontal line if and only if:
        roundness < 0.20   (elongated, not circular — matches analysis_pipeline.THRESHOLD)
        AND aspect > 1.1   (wider than tall)
    Every other dark feature — circles, blobs, spots, vertical lines,
    diagonal smears — is immediately treated as a defect regardless of size,
    as long as it is above the noise floor (_MIN_NOISE_AREA px²).

    Confidence is proportional to the total area of non-scratch features
    relative to _BAD_AREA_SATURATE.  Even a single small dust spot will
    saturate to near-1.0 confidence bad.
    """
    import cv2

    gray   = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    mean_v = float(gray.mean())
    std_v  = float(gray.std())

    # 1.5σ below mean catches mid-grey dust that a stricter threshold misses
    thr = max(0, int(mean_v - 1.5 * std_v))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_pixels    = rgb_array.shape[0] * rgb_array.shape[1]
    total_bad_frac = 0.0   # accumulated non-scratch area as fraction of image

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_NOISE_AREA:
            continue   # below noise floor — ignore

        _, _, w, h = cv2.boundingRect(cnt)
        aspect    = w / max(h, 1)   # > 1 = wider than tall

        # Horizontal scratch: width must be at least 2× the height.
        # Roundness is NOT used here — a thick scratch can have high roundness
        # even though it is clearly a horizontal line.  Aspect ratio alone
        # reliably separates lines (aspect >> 1) from blobs (aspect ≈ 1).
        if aspect >= 2.0:
            continue

        # Anything else: accumulate its area as a bad contribution
        total_bad_frac += area / img_pixels

    bad_conf = min(total_bad_frac / _BAD_AREA_SATURATE, 1.0)

    if bad_conf < 0.25:
        return "good", round(1.0 - bad_conf, 4)
    else:
        return "bad", round(min(bad_conf + 0.3, 1.0), 4)   # boost so bad reads clearly bad


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
