"""
Labeling tool for building the ML training dataset.
Run: python labeling_tool.py

Controls:
  G = good          W = watermark
  B = blotch        V = vertical_scratch
  D = debris        S = skip / discard
  Q = quit
"""
import os
import sys
import cv2
import time
from camera import open_camera

CLASSES = {
    ord('g'): "good",
    ord('w'): "watermark",
    ord('b'): "blotch",
    ord('v'): "vertical_scratch",
    ord('d'): "debris",
}
SKIP_KEY = ord('s')
QUIT_KEY = ord('q')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "training_data")


def main():
    cam = open_camera()
    for cls in CLASSES.values():
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    print("Labeling tool started. Keys: G/W/B/V/D=label  S=skip  Q=quit")
    counts = {c: 0 for c in CLASSES.values()}

    while True:
        frame = None
        deadline = time.time() + 2.0
        while frame is None and time.time() < deadline:
            frame = cam.grab()
            time.sleep(0.03)

        if frame is None:
            print("No frame received — retrying")
            continue

        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(display, "G=good W=watermark B=blotch V=vert_scratch D=debris S=skip Q=quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.imshow("Labeling Tool", display)
        key = cv2.waitKey(0) & 0xFF

        if key == QUIT_KEY:
            break
        elif key == SKIP_KEY:
            continue
        elif key in CLASSES:
            cls = CLASSES[key]
            counts[cls] += 1
            fname = f"{cls}_{counts[cls]:05d}.png"
            path = os.path.join(OUTPUT_DIR, cls, fname)
            from PIL import Image
            Image.fromarray(frame).save(path)
            print(f"Saved {path}")
        else:
            print(f"Unknown key {key} — ignored")

    cam.close()
    cv2.destroyAllWindows()
    print("Counts:", counts)


if __name__ == "__main__":
    main()
