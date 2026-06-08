"""
ML quality classifier — binary: good / bad.

"good"  = clean slide area (horizontal scratches are what we're measuring; those are fine)
"bad"   = dust specks, debris, or other contamination that would corrupt measurements

Loads model.pt if present; otherwise uses a Laplacian variance heuristic
as a placeholder until the model is trained.
"""
import os
import numpy as np
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
_IMG_SIZE  = 224
_SHARPNESS_THRESHOLD = 80.0  # Laplacian variance below this → probably bad


def _laplacian_variance(rgb_array):
    import cv2
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class QualityClassifier:
    def __init__(self):
        self._model     = None
        self._transform = None
        self._use_heuristic = True
        self.load()

    def load(self):
        if not os.path.exists(MODEL_PATH):
            print("model.pt not found — using sharpness heuristic until model is trained")
            return
        try:
            import torch
            import torchvision.transforms as T
            self._model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            self._model.eval()
            self._transform = T.Compose([
                T.Resize((_IMG_SIZE, _IMG_SIZE)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self._use_heuristic = False
            print("ML model loaded from model.pt")
        except Exception as e:
            print(f"model.pt load failed ({e}) — falling back to heuristic")

    def predict(self, rgb_array) -> tuple[str, float]:
        """Return ("good"|"bad", confidence 0–1). Blocks briefly."""
        if self._use_heuristic:
            score = _laplacian_variance(rgb_array)
            if score >= _SHARPNESS_THRESHOLD:
                return "good", min(score / 200.0, 1.0)
            else:
                return "bad", 1.0 - score / _SHARPNESS_THRESHOLD

        import torch
        img    = Image.fromarray(rgb_array)
        tensor = self._transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)[0]
        # index 1 = good, index 0 = bad  (matches train.py label convention)
        good_conf = float(probs[1])
        if good_conf >= 0.5:
            return "good", good_conf
        else:
            return "bad", float(probs[0])

    def is_good(self, rgb_array) -> bool:
        label, _ = self.predict(rgb_array)
        return label == "good"
