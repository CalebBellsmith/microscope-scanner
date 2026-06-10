"""
ML quality classifier — binary: good (1) vs bad (0).

HOW IT WORKS
────────────
Inference runs inside a persistent subprocess (inference_worker.py) so
that torch / onnxruntime DLLs are never loaded in the PyQt5 GUI thread,
which causes WinError 1114 on some Windows machines.

On the first call to predict(), this module spawns inference_worker.py
and keeps it alive.  Each prediction sends the image over stdin and
reads the result from stdout.  The worker is shut down on program exit.

"good"  = clean slide area (horizontal scratches are fine — that's what
          the analysis pipeline counts)
"bad"   = dust, debris, watermarks, or non-horizontal artefacts
"""

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


# ── Worker process management ─────────────────────────────────────────────────

_worker_proc  = None   # the subprocess.Popen object
_worker_lock  = threading.Lock()   # ensures only one request at a time


def _get_worker() -> subprocess.Popen:
    """
    Return the inference worker subprocess, spawning it if needed.
    The worker loads the model once and then stays alive.
    """
    global _worker_proc
    with _worker_lock:
        # Check if worker is alive (returncode is None while running)
        if _worker_proc is not None and _worker_proc.poll() is None:
            return _worker_proc

        # Spawn a new worker process using the same Python interpreter
        _worker_proc = subprocess.Popen(
            [sys.executable, _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,   # line-buffered so responses arrive immediately
        )

        # Wait for the worker to print "ready" (model loaded)
        while True:
            line = _worker_proc.stdout.readline().strip()
            if line == "ready":
                break
            if line.startswith("loaded"):
                # e.g. "loaded onnx" — informational, keep waiting for "ready"
                print(f"[inference_worker] {line}", flush=True)
            elif line.startswith("onnx_failed") or line.startswith("pt_failed"):
                print(f"[inference_worker] {line}", flush=True)
            elif not line:
                # Worker died before printing "ready"
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


atexit.register(_shutdown_worker)   # called automatically when Python exits


# ── Public classifier class ───────────────────────────────────────────────────

class QualityClassifier:
    """
    Wraps the inference worker subprocess.
    Call predict(rgb_array) to get (label, confidence).
    The worker is started lazily on the first prediction.
    """

    def predict(self, rgb_array: np.ndarray) -> tuple[str, float]:
        """
        Classify a frame.
        rgb_array : numpy uint8 array shaped (H, W, 3)
        Returns   : ("good"|"bad", confidence 0.0–1.0)
        """
        try:
            worker = _get_worker()

            # Encode the image as base64 so it fits on one stdin line
            h, w, c = rgb_array.shape
            b64     = base64.b64encode(rgb_array.tobytes()).decode()
            line    = f"{h} {w} {c} {b64}\n"

            # Send image → worker, read back prediction (thread-safe)
            with _worker_lock:
                worker.stdin.write(line)
                worker.stdin.flush()
                response = worker.stdout.readline().strip()

            if response.startswith("error"):
                raise RuntimeError(response)

            # Response format: "good 0.923456" or "bad 0.731200"
            label, conf_str = response.split()
            return label, float(conf_str)

        except Exception as e:
            # If the worker crashes, fall back to a simple heuristic
            print(f"[ml_inference] predict failed ({e}), using heuristic")
            return _heuristic(rgb_array)

    def is_good(self, rgb_array: np.ndarray) -> bool:
        """Convenience wrapper — returns True if label == 'good'."""
        label, _ = self.predict(rgb_array)
        return label == "good"

    def load(self):
        """
        Pre-warm the worker so the first real prediction has no startup delay.
        Called by main.py after the camera connects.
        """
        try:
            _get_worker()
            print("[ml_inference] inference worker ready")
        except Exception as e:
            print(f"[ml_inference] worker pre-warm failed: {e}")


# ── Heuristic fallback (no model) ─────────────────────────────────────────────

def _heuristic(rgb_array: np.ndarray) -> tuple[str, float]:
    """
    Simple Laplacian-variance sharpness check used when no model is
    available.  A sharp image scores high → "good"; blurry → "bad".
    Note: does NOT detect dust specifically — just a rough proxy.
    """
    import cv2
    gray  = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if score >= 80.0:
        return "good", min(score / 200.0, 1.0)
    else:
        return "bad", 1.0 - score / 80.0
