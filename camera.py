"""
Camera abstraction layer.

Tries three backends in order:
  1. ToupTekCamera  — uses the manufacturer SDK (toupcam.py + toupcam.dll)
                      Required for the Olympus U-PMTVC microscope camera.
  2. OpenCVCamera   — generic USB/webcam via OpenCV VideoCapture.
  3. MSSCamera      — screen capture fallback for testing without hardware.

All three expose the same interface:
    open()           — connect to camera, start streaming
    grab()           — return latest frame as numpy RGB uint8 array (fast)
    grab_fresh()     — wait for a NEW frame at capture exposure (ToupTek only)
    close()          — stop streaming, release device
    set_analysis_mode(bool) — switch between analysis and raw camera settings
"""
import numpy as np

# Target resolution — must match ToupView live/snap settings
TARGET_W, TARGET_H = 1024, 822


# ── ToupTek SDK backend ───────────────────────────────────────────────────────

class ToupTekCamera:
    """
    Interfaces with the camera via the ToupTek SDK (pull-mode callback).
    Every time the camera delivers a frame, _on_event() is called by the SDK,
    which stores the frame in self._frame for grab() to return.
    """

    # Two exposure times are used:
    #   PREVIEW  — short (100ms) keeps the live GUI feed smooth at ~10fps
    #   CAPTURE  — long  (1300ms) gives full quality for saved images
    PREVIEW_EXPOSURE_US = 100_000     # 100ms  in microseconds
    CAPTURE_EXPOSURE_US = 1_300_000   # 1300ms in microseconds

    def __init__(self):
        import toupcam
        self._sdk = toupcam      # the SDK module (toupcam.py in project folder)
        self._cam = None         # SDK camera handle, set in open()
        self._frame = None       # latest decoded frame (numpy RGB uint8)

    def open(self):
        import toupcam

        # Find all connected ToupTek cameras and open the first one
        arr = self._sdk.Toupcam.EnumV2()
        if not arr:
            raise RuntimeError("No ToupTek camera found")
        self._cam = self._sdk.Toupcam.Open(arr[0].id)

        # Set live resolution and colour format (RGB24 = 3 bytes per pixel)
        self._cam.put_Size(TARGET_W, TARGET_H)
        self._cam.put_Option(toupcam.TOUPCAM_OPTION_RGB, 0)

        # Internal state flags
        self._analysis_mode     = True   # True = analysis settings active
        self._negative_fallback = False  # True = invert in software (SDK put_Negative failed)
        self._frame_count       = 0      # incremented each time a new frame arrives
        self._preview_expo_us   = self.PREVIEW_EXPOSURE_US  # tracks current live exposure

        # Apply analysis settings (exposure, gain, negative, white balance)
        self._apply_settings(analysis=True)

        # Neutralise warm/orange tint by setting colour temperature to daylight
        try:
            self._cam.put_TempTint(6503, 0)
        except Exception:
            pass

        # Start the camera streaming; _on_event will fire for each new frame
        self._cam.StartPullModeWithCallback(self._on_event, None)

    def _apply_settings(self, analysis: bool):
        """
        Switch the camera between two modes:
          analysis=True  — fixed settings that match the MATLAB pipeline:
                           no auto-exposure, 1300ms, 3x gain, negative on
          analysis=False — auto-exposure, 1x gain, negative off (raw view)
        """
        self._analysis_mode = analysis
        if analysis:
            self._cam.put_AutoExpoEnable(False)    # disable auto-exposure
            self._cam.put_ExpoAGain(300)           # 3x analogue gain
            self._cam.put_ExpoTime(self.PREVIEW_EXPOSURE_US)  # 100ms for smooth live feed
            try:
                self._cam.put_Negative(True)       # invert colours (SDK)
                self._negative_fallback = False
            except Exception:
                # SDK doesn't support put_Negative — do it in software instead
                self._negative_fallback = True
        else:
            self._cam.put_AutoExpoEnable(True)     # let camera choose exposure
            self._cam.put_ExpoAGain(100)           # 1x gain (no amplification)
            try:
                self._cam.put_Negative(False)
            except Exception:
                pass
            self._negative_fallback = False

    def set_analysis_mode(self, enabled: bool):
        """Called from the GUI toggle buttons."""
        if self._cam:
            self._apply_settings(analysis=enabled)

    def _on_event(self, event, ctx):
        """
        SDK callback — fires every time the camera delivers a frame.
        Pulls the raw pixel buffer, converts to numpy, applies software
        negative if the SDK doesn't support put_Negative, then stores
        the frame and increments _frame_count so grab_fresh() can detect it.
        """
        import toupcam
        if event == toupcam.TOUPCAM_EVENT_IMAGE:
            # Allocate a buffer large enough for one RGB24 frame
            buf = bytes(TARGET_W * TARGET_H * 3)
            self._cam.PullImageV2(buf, 24, None)   # 24 = 24-bit RGB

            # Convert raw bytes → numpy array shaped (H, W, 3)
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(TARGET_H, TARGET_W, 3).copy()

            # Software negative: invert pixel values (255 - value) if SDK can't do it
            if self._negative_fallback and self._analysis_mode:
                frame = 255 - frame

            # Convert to greyscale in analysis mode so preview matches old MATLAB output.
            # Still stored as RGB (all 3 channels equal) so downstream code is unchanged.
            if self._analysis_mode:
                grey  = (0.299 * frame[:, :, 0] +
                         0.587 * frame[:, :, 1] +
                         0.114 * frame[:, :, 2]).astype(np.uint8)
                frame = np.stack([grey, grey, grey], axis=2)

            self._frame = frame
            self._frame_count += 1   # signals grab_fresh() that a new frame arrived

    def grab(self):
        """Return the most recently received frame (used by the live preview timer)."""
        return self._frame

    def grab_fresh(self, timeout=5.0):
        """
        Capture a full-quality frame for saving to disk:
          1. Switch to long capture exposure (1300ms)
          2. Wait until _frame_count increases (new frame at correct exposure)
          3. Restore the user's chosen preview exposure
          4. Return the fresh frame

        Used by capture_pipeline instead of grab() so saved images are
        taken at the correct exposure even while the live preview runs faster.
        """
        import time
        if self._cam is None:
            return None

        # Switch to long exposure for the saved image
        self._cam.put_ExpoTime(self.CAPTURE_EXPOSURE_US)
        count_before = self._frame_count

        # Wait for the camera to deliver one new frame at the new exposure
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._frame_count > count_before:
                frame = self._frame
                # Restore preview exposure (user's chosen value, not hardcoded)
                self._cam.put_ExpoTime(self._preview_expo_us)
                return frame
            time.sleep(0.05)

        # Timed out — restore exposure and return whatever frame we have
        self._cam.put_ExpoTime(self._preview_expo_us)
        return self._frame

    def close(self):
        if self._cam:
            self._cam.Close()
            self._cam = None


# ── OpenCV backend (generic USB camera) ──────────────────────────────────────

class OpenCVCamera:
    """Uses OpenCV VideoCapture — works with any standard USB/UVC webcam."""

    def __init__(self, index=0):
        self._index = index   # camera index (0 = first camera, 1 = second, etc.)
        self._cap   = None

    def open(self):
        import cv2
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV camera index {self._index} not available")
        # Request target resolution (camera may ignore this)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    def grab(self):
        import cv2
        ret, frame = self._cap.read()
        if not ret:
            return None
        # OpenCV returns BGR; convert to RGB so all backends return the same format
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None


# ── Screen-capture fallback (for testing without any camera) ─────────────────

class MSSCamera:
    """Captures the primary monitor and resizes to target resolution."""

    def __init__(self):
        self._sct = None

    def open(self):
        import mss
        self._sct = mss.mss()

    def grab(self):
        import mss, cv2
        mon = self._sct.monitors[1]                      # monitor 1 = primary display
        img = np.array(self._sct.grab(mon))              # BGRA screenshot
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)      # → RGB
        img = cv2.resize(img, (TARGET_W, TARGET_H))      # resize to target
        return img

    def close(self):
        if self._sct:
            self._sct.close()
            self._sct = None


# ── Factory function ──────────────────────────────────────────────────────────

def open_camera():
    """
    Try each backend in priority order and return the first one that opens.
    ToupTek → OpenCV → screen capture.
    """
    for Cls, args in [(ToupTekCamera, []), (OpenCVCamera, [0]), (MSSCamera, [])]:
        try:
            cam = Cls(*args)
            cam.open()
            print(f"Camera opened: {Cls.__name__}")
            return cam
        except Exception as e:
            print(f"{Cls.__name__} failed: {e}")
    raise RuntimeError("No camera backend could be opened")
