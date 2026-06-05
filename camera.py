"""
Camera abstraction: ToupTek SDK → OpenCV UVC → mss screen capture.
Each backend exposes the same interface: open(), grab() -> np.ndarray, close().
"""
import numpy as np

TARGET_W, TARGET_H = 1024, 822


class ToupTekCamera:
    def __init__(self):
        import toupcam
        self._sdk = toupcam
        self._cam = None
        self._frame = None

    def open(self):
        import toupcam
        arr = self._sdk.Toupcam.EnumV2()
        if not arr:
            raise RuntimeError("No ToupTek camera found")
        self._cam = self._sdk.Toupcam.Open(arr[0].id)

        # Resolution & format (same in both modes)
        self._cam.put_Size(TARGET_W, TARGET_H)
        self._cam.put_Option(toupcam.TOUPCAM_OPTION_RGB, 0)  # RGB24

        # Start in analysis mode by default
        self._analysis_mode = True
        self._negative_fallback = False
        self._apply_settings(analysis=True)

        self._cam.StartPullModeWithCallback(self._on_event, None)

    def _apply_settings(self, analysis: bool):
        """Switch between analysis settings and raw camera defaults."""
        import toupcam
        self._analysis_mode = analysis
        if analysis:
            # Fixed settings required by analysis pipeline
            self._cam.put_AutoExpoEnable(False)
            self._cam.put_ExpoAGain(300)       # 3x gain
            self._cam.put_ExpoTime(1300000)    # 1300ms
            try:
                self._cam.put_Negative(True)
                self._negative_fallback = False
            except Exception:
                self._negative_fallback = True
        else:
            # Raw mode — auto exposure, no inversion, default gain
            self._cam.put_AutoExpoEnable(True)
            self._cam.put_ExpoAGain(100)       # 1x gain
            try:
                self._cam.put_Negative(False)
                self._negative_fallback = False
            except Exception:
                self._negative_fallback = False  # no inversion needed in raw mode

    def set_analysis_mode(self, enabled: bool):
        """Called from GUI toggle."""
        if self._cam:
            self._apply_settings(analysis=enabled)

    def _on_event(self, event, ctx):
        import toupcam
        if event == toupcam.TOUPCAM_EVENT_IMAGE:
            buf = bytes(TARGET_W * TARGET_H * 3)
            self._cam.PullImageV2(buf, 24, None)
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(TARGET_H, TARGET_W, 3).copy()
            # Software inversion fallback — only apply in analysis mode
            if self._negative_fallback and self._analysis_mode:
                frame = 255 - frame
            self._frame = frame

    def grab(self):
        return self._frame

    def close(self):
        if self._cam:
            self._cam.Close()
            self._cam = None


class OpenCVCamera:
    def __init__(self, index=0):
        self._index = index
        self._cap = None

    def open(self):
        import cv2
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV camera index {self._index} not available")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    def grab(self):
        import cv2
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None


class MSSCamera:
    """Fallback: captures the primary monitor and crops/resizes to target."""
    def __init__(self):
        self._sct = None

    def open(self):
        import mss
        self._sct = mss.mss()

    def grab(self):
        import mss, cv2
        mon = self._sct.monitors[1]
        img = np.array(self._sct.grab(mon))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (TARGET_W, TARGET_H))
        return img

    def close(self):
        if self._sct:
            self._sct.close()
            self._sct = None


def open_camera():
    """Try each backend in priority order, return the first that works."""
    for Cls, args in [(ToupTekCamera, []), (OpenCVCamera, [0]), (MSSCamera, [])]:
        try:
            cam = Cls(*args)
            cam.open()
            print(f"Camera opened: {Cls.__name__}")
            return cam
        except Exception as e:
            print(f"{Cls.__name__} failed: {e}")
    raise RuntimeError("No camera backend could be opened")
