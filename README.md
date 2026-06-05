# Automated Microscope Slide Scanner

## Quick start

1. Flash `firmware/firmware.ino` to the ESP32 (Arduino IDE, board: "ESP32 Dev Module", install `ESP32Servo` library).
2. On the Windows laptop: `pip install -r requirements.txt`
3. Double-click `run.bat`.

## Workflow

### First use — build the ML model
1. Connect camera + ESP32, click **Connect** in the GUI.
2. In a separate terminal: `python labeling_tool.py`
   - Live camera feed appears. Press keys to label frames: `G` good · `W` watermark · `B` blotch · `V` vertical scratch · `D` debris · `S` skip · `Q` quit.
   - Aim for ~50+ examples per class.
3. `python train.py` — fine-tunes MobileNetV3-Small, saves `model.pt`.
4. Until `model.pt` exists the app falls back to Laplacian sharpness heuristic.

### Scanning
1. Set rows, columns, X spacing (stepper steps), Y spacing (servo units).
2. Set output directory.
3. Click **Go**. The stage scans in a boustrophedon (snake) pattern. At each position the ML classifier checks quality; if bad it spirals outward nudging the stage until a good frame is found (or max attempts reached). The best frame is saved.
4. The analysis pipeline runs concurrently, watching the output folder and processing images as they arrive.
5. Results are written to `<output>/<timestamp>/results.jsonl`.

### Adding your analysis code
Edit `analysis_pipeline.py` → replace the body of `_analyze_image(self, image_path)` with your logic. The method receives an absolute path to a PNG (RGB, 1024×822) and must return a JSON-serialisable value.

## File overview

| File | Purpose |
|------|---------|
| `main.py` | PyQt5 GUI, wires everything together |
| `camera.py` | ToupTek → OpenCV → mss fallback |
| `motor.py` | Serial to ESP32 |
| `capture_pipeline.py` | Grid scan + nudge search |
| `ml_inference.py` | Quality classifier |
| `analysis_pipeline.py` | Concurrent analysis — **edit this** |
| `labeling_tool.py` | Build training dataset |
| `train.py` | Fine-tune MobileNetV3-Small |
| `firmware/firmware.ino` | ESP32 motor controller |

## Tuning parameters

| Parameter | Where | Notes |
|-----------|-------|-------|
| X/Y spacing | GUI spinboxes | Calibrate by measuring steps per mm |
| Nudge offsets | `capture_pipeline.py` `_NUDGE_SPIRAL` | Adjust step sizes for your optics |
| Sharpness threshold | `ml_inference.py` `_SHARPNESS_THRESHOLD` | Tune before training model |
| Servo pulse widths | `firmware.ino` `SERVO_CW_US / SERVO_CCW_US` | Tune for your specific servo |
| Step delay | `firmware.ino` `STEP_DELAY_US` | Slower = more torque, less noise |

## Hardware wiring (ESP32)

```
28BYJ-48 stepper via ULN2003:
  IN1 → GPIO 16
  IN2 → GPIO 17
  IN3 → GPIO 18
  IN4 → GPIO 19
  VCC → 4×AA pack positive
  GND → common ground (ESP32 GND + battery pack GND)

Continuous rotation servo:
  Signal → GPIO 21
  VCC    → 4×AA pack positive
  GND    → common ground

ESP32:
  USB → laptop (power + serial)
```
