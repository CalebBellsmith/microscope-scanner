"""
Inference worker — runs as a standalone subprocess so that onnxruntime
and torch DLLs are never loaded inside the PyQt5 GUI thread (which causes
WinError 1114 on some Windows machines).

Protocol (stdin/stdout, line-based):
  Parent sends:  "<H> <W> <C> <base64-encoded uint8 pixel data>\n"
  Worker replies: "<label> <confidence>\n"   e.g. "good 0.923\n"

  Parent sends:  "quit\n"
  Worker exits cleanly.

The worker prints "ready\n" to stdout once the model is loaded so the
parent knows it can start sending images.
"""

import os
import sys
import base64

import numpy as np

# Must be set BEFORE any torch/onnxruntime import to prevent CUDA DLL crash
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── Paths ─────────────────────────────────────────────────────────────────────
# __file__ lives in the same folder as model.onnx / model.pt
_HERE     = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(_HERE, "model.onnx")
PT_PATH   = os.path.join(_HERE, "model.pt")
IMG_SIZE  = 224   # MobileNetV3 input size


# ── Pre-processing ────────────────────────────────────────────────────────────

def _preprocess(rgb_array: np.ndarray) -> np.ndarray:
    """
    Resize image to 224×224, normalise with ImageNet mean/std,
    return float32 array shaped (1, 3, 224, 224) ready for the model.
    """
    from PIL import Image
    img  = Image.fromarray(rgb_array).resize((IMG_SIZE, IMG_SIZE))
    arr  = np.array(img, dtype=np.float32) / 255.0   # scale 0-1
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std                         # ImageNet normalisation
    return arr.transpose(2, 0, 1)[np.newaxis]         # HWC → NCHW


# ── Load model ────────────────────────────────────────────────────────────────

session      = None   # onnxruntime session (preferred)
torch_model  = None   # torch model (fallback)
transform    = None   # torchvision transforms for torch path
use_heuristic = False

if os.path.exists(ONNX_PATH):
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(
            ONNX_PATH,
            providers=["CPUExecutionProvider"],   # CPU only, no CUDA DLLs
        )
        print("loaded onnx", flush=True)
    except Exception as e:
        print(f"onnx_failed {e}", flush=True)
        session = None

if session is None and os.path.exists(PT_PATH):
    try:
        import torch
        import torchvision.transforms as T
        torch_model = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        torch_model.eval()
        transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print("loaded pt", flush=True)
    except Exception as e:
        print(f"pt_failed {e}", flush=True)
        torch_model = None

if session is None and torch_model is None:
    use_heuristic = True
    print("loaded heuristic", flush=True)

# Signal to parent that we are ready to receive images
print("ready", flush=True)


# ── Inference helpers ─────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


def _predict_array(rgb_array: np.ndarray) -> tuple[str, float]:
    """Run inference on a numpy RGB uint8 image, return (label, confidence)."""
    if use_heuristic:
        # Laplacian variance as sharpness proxy (no model available)
        import cv2
        gray  = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score >= 80.0:
            return "good", min(score / 200.0, 1.0)
        else:
            return "bad", 1.0 - score / 80.0

    if session is not None:
        # onnxruntime path
        inp    = _preprocess(rgb_array)
        logits = session.run(None, {"input": inp})[0][0]   # shape (2,)
        probs  = _softmax(logits)
    else:
        # torch path
        from PIL import Image
        import torch
        img    = Image.fromarray(rgb_array)
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(torch_model(tensor), dim=1)[0].numpy()

    # Index 1 = good, index 0 = bad  (matches train.py label convention)
    good_conf = float(probs[1])
    if good_conf >= 0.5:
        return "good", good_conf
    else:
        return "bad", float(probs[0])


# ── Main loop — read images from stdin, write predictions to stdout ───────────

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if line == "quit":
        break

    try:
        # Format: "<H> <W> <C> <base64data>"
        h_str, w_str, c_str, b64 = line.split(" ", 3)
        h, w, c = int(h_str), int(w_str), int(c_str)
        raw     = base64.b64decode(b64)                       # bytes
        arr     = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, c)

        label, conf = _predict_array(arr)
        print(f"{label} {conf:.6f}", flush=True)   # reply to parent
    except Exception as e:
        # Send error back so parent doesn't hang waiting for a result
        print(f"error {e}", flush=True)
