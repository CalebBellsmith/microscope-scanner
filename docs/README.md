# Automated Microscope Slide Scanner

## Quick start

1. Flash `firmware/firmware.ino` to the ESP32 (Arduino IDE, board: "ESP32 Dev Module", install `ESP32Servo` library).
2. Double-click **`run.bat`** — the only file in the top folder. First run installs
   the dependencies itself (`app/requirements-core.txt`); after that it just opens.

Everything else lives in subfolders so the operator sees one obvious thing to click:

| Folder | Holds |
|--------|-------|
| `app/` | all Python code, the summary template, the requirements files |
| `scripts/` | `train.bat`, the launcher for the ML labelling tool |
| `firmware/` | the ESP32 sketch |
| `docs/` | this file and `ALGORITHMS.md` |

The app opens its folder picker at **`Abrasion` on the Desktop** (any capitalisation),
creating it if it is not there yet.

## Workflow

### First use — build the ML model
1. Connect camera + ESP32, click **Connect** in the GUI.
2. Double-click `scripts/train.bat` (or: `python app/labeling_tool.py`)
   - Live camera feed appears. Press keys to label frames: `G` good · `W` watermark · `B` blotch · `V` vertical scratch · `D` debris · `S` skip · `Q` quit.
   - Aim for ~50+ examples per class.
3. `python app/train.py` — fine-tunes MobileNetV3-Small, saves `model.pt`.
4. Until `model.pt` exists the app falls back to Laplacian sharpness heuristic.

### Scanning
1. Set rows, columns, X spacing (stepper steps), Y spacing (servo units).
2. Set output directory.
3. Click **Go**. The stage scans in a boustrophedon (snake) pattern. At each position the ML classifier checks quality; if bad it spirals outward nudging the stage until a good frame is found (or max attempts reached). The best frame is saved.
4. The analysis pipeline runs concurrently, watching the output folder and processing images as they arrive.
5. Results are written to `<output>/<timestamp>/results.jsonl`.

### Adding your analysis code
Edit `app/analysis_pipeline.py` → replace the body of `_analyze_image(self, image_path)` with your logic. The method receives an absolute path to a PNG (RGB, 1024×822) and must return a JSON-serialisable value.

## File overview

| File | Purpose |
|------|---------|
| `app/main.py` | PyQt5 GUI, wires everything together |
| `app/camera.py` | ToupTek → OpenCV → mss fallback |
| `app/motor.py` | Serial to ESP32 |
| `app/capture_pipeline.py` | Grid scan + nudge search |
| `app/ml_inference.py` | Quality classifier |
| `app/analysis_pipeline.py` | Concurrent analysis — **edit this** |
| `app/labeling_tool.py` | Build training dataset |
| `app/train.py` | Fine-tune MobileNetV3-Small |
| `firmware/firmware.ino` | ESP32 motor controller |

## Wireless (optional)

USB serial is the default and always works. To run the motion link over WiFi:

**Rig hosts its own network (recommended).** `firmware.ino` ships with
`WIFI_AP_MODE 1`, so the ESP32 broadcasts `AutoScope-Rig` / `autoscope2026` on
channel 1 and always answers at the fixed address **192.168.4.1**. In the GUI,
tick **Wireless (WiFi)** and leave **Join the rig's WiFi automatically** ticked —
Connect switches this PC onto the rig's network, and Disconnect/exit puts it back
on the one it came from. While joined to the rig the PC has no internet unless it
is also on Ethernet.

This mode exists because joining the house network was unreliable on site: the
mesh kept handing the board between nodes on a band with 7 visible networks, and
a stationary board lost ~50% of pings. Hosting our own link removes the mesh, the
congestion and the DHCP lease, and needs no credentials at a demo venue.

**Rig joins an existing network.** Set `WIFI_AP_MODE 0` and fill in `WIFI_SSID` /
`WIFI_PASS`. The board prints its address as `WIFI <ip>` at boot — type that into
the GUI's ESP32 IP field and untick auto-switch. It also prints `WIFI mac=…`, for
a DHCP reservation so the address stops drifting.

Auto-switching is Windows-only (it shells out to `netsh`). Elsewhere, or on a
managed laptop that blocks scripted joins, untick it and switch networks by hand.

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
