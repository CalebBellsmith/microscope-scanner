"""
ML quality classifier.
Classes: good, watermark, blotch, vertical_scratch, debris
Only "good" (horizontal scratch) is the target pass class.
Loads model from model.pt if present; otherwise uses a Laplacian variance
sharpness heuristic as a placeholder until the model is trained.
"""
import os
import numpy as np
from PIL import Image

CLASSES = ["good", "watermark", "blotch", "vertical_scratch", "debris"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
_IMG_SIZE = 224
_SHARPNESS_THRESHOLD = 80.0  # Laplacian variance below this = blurry


def _laplacian_variance(rgb_array):
    import cv2
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class QualityClassifier:
    def __init__(self):
        self._model = None
        self._transform = None
        self._use_heuristic = True

    def load(self):
        if not os.path.exists(MODEL_PATH):
            print("model.pt not found — using sharpness heuristic until model is trained")
            return
        import torch
        import torchvision.transforms as T
        self._model = torch.load(MODEL_PATH, map_location="cpu")
        self._model.eval()
        self._transform = T.Compose([
            T.Resize((_IMG_SIZE, _IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._use_heuristic = False
        print("ML model loaded from model.pt")

    def predict(self, rgb_array) -> tuple[str, float]:
        """Return (class_name, confidence). Blocks briefly."""
        if self._use_heuristic:
            score = _laplacian_variance(rgb_array)
            if score >= _SHARPNESS_THRESHOLD:
                return "good", min(score / 200.0, 1.0)
            else:
                return "blotch", 1.0 - score / _SHARPNESS_THRESHOLD

        import torch
        img = Image.fromarray(rgb_array)
        tensor = self._transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        return CLASSES[idx], float(probs[idx])

    def is_good(self, rgb_array) -> bool:
        label, _ = self.predict(rgb_array)
        return label == "good"
