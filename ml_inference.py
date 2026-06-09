"""
ML quality classifier — binary: good (1) vs bad (0).

Inference priority:
  1. model.onnx via onnxruntime  (no CUDA DLL issues, fast)
  2. model.pt  via torch         (fallback if onnx missing)
  3. Laplacian variance heuristic (fallback if no model at all)

"good"  = clean slide area suitable for scratch measurement
"bad"   = dust, debris, watermarks, or non-horizontal lines
"""
import os
import numpy as np
from PIL import Image

# Prevent CUDA DLL crash if torch is used as fallback
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE      = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH  = os.path.join(_HERE, "model.onnx")
MODEL_PATH = os.path.join(_HERE, "model.pt")
_IMG_SIZE  = 224
_SHARPNESS_THRESHOLD = 80.0


def _laplacian_variance(rgb_array):
    import cv2
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _preprocess(rgb_array) -> np.ndarray:
    """Resize, normalise, return float32 NCHW array ready for inference."""
    img = Image.fromarray(rgb_array).resize((_IMG_SIZE, _IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis]   # NCHW


class QualityClassifier:
    def __init__(self):
        self._session      = None   # onnxruntime InferenceSession
        self._torch_model  = None   # torch model (fallback)
        self._transform    = None
        self._use_heuristic = True
        self.load()

    def load(self):
        # ── Try onnxruntime first ─────────────────────────────────────────────
        if os.path.exists(ONNX_PATH):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(
                    ONNX_PATH,
                    providers=["CPUExecutionProvider"],
                )
                self._use_heuristic = False
                print("ML model loaded from model.onnx (onnxruntime)")
                return
            except Exception as e:
                print(f"onnxruntime load failed ({e})")

        # ── Try torch fallback ────────────────────────────────────────────────
        if os.path.exists(MODEL_PATH):
            try:
                import torch
                import torchvision.transforms as T
                self._torch_model = torch.load(
                    MODEL_PATH, map_location="cpu", weights_only=False
                )
                self._torch_model.eval()
                self._transform = T.Compose([
                    T.Resize((_IMG_SIZE, _IMG_SIZE)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                self._use_heuristic = False
                print("ML model loaded from model.pt (torch fallback)")
                return
            except Exception as e:
                print(f"model.pt load failed ({e})")

        print("No model found — using sharpness heuristic until model is trained")

    def predict(self, rgb_array) -> tuple[str, float]:
        """Return ("good"|"bad", confidence 0–1)."""
        if self._use_heuristic:
            score = _laplacian_variance(rgb_array)
            if score >= _SHARPNESS_THRESHOLD:
                return "good", min(score / 200.0, 1.0)
            else:
                return "bad", 1.0 - score / _SHARPNESS_THRESHOLD

        # ── onnxruntime inference ─────────────────────────────────────────────
        if self._session is not None:
            inp   = _preprocess(rgb_array)
            logits = self._session.run(None, {"input": inp})[0][0]
            # softmax
            e     = np.exp(logits - logits.max())
            probs = e / e.sum()
            good_conf = float(probs[1])
            if good_conf >= 0.5:
                return "good", good_conf
            else:
                return "bad", float(probs[0])

        # ── torch inference ───────────────────────────────────────────────────
        import torch
        img    = Image.fromarray(rgb_array)
        tensor = self._transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._torch_model(tensor), dim=1)[0]
        good_conf = float(probs[1])
        if good_conf >= 0.5:
            return "good", good_conf
        else:
            return "bad", float(probs[0])

    def is_good(self, rgb_array) -> bool:
        label, _ = self.predict(rgb_array)
        return label == "good"
