#!/bin/bash
# ============================================================================
#  Microscope Slide Scanner — macOS launcher
#  Double-click this file in Finder (or run ./launch_mac.command in Terminal).
#
#  What it does:
#    1. If your system python3 already has every core dependency, it just runs
#       main.py immediately (no install, no venv).
#    2. Otherwise it builds an isolated .venv next to this script, installs the
#       core deps from requirements-core.txt (NO torch — rules mode needs no
#       ML), and launches from there.  The .venv is reused on later runs.
#
#  Notes for Mac:
#    - No ToupTek SDK on Mac, so the camera falls back to your built-in/USB
#      webcam (OpenCV) and then to screen-capture (mss).  macOS will ask for
#      Camera / Screen-Recording permission the first time — click Allow.
#    - The ESP32 motor controller is optional: if it isn't plugged in the app
#      still opens with the live camera feed (motor moves are skipped).
# ============================================================================
set -e
cd "$(dirname "$0")"

# The set of modules main.py needs to launch (rules mode; torch excluded).
CORE_IMPORTS='import PyQt5, cv2, numpy, scipy, skimage, serial, mss, PIL, openpyxl'

# ── Fast path: system python3 already has everything ────────────────────────
if command -v python3 >/dev/null 2>&1 && python3 -c "$CORE_IMPORTS" >/dev/null 2>&1; then
    echo "▶ All dependencies present — launching with system python3…"
    exec python3 main.py
fi

# ── Fallback: isolated virtual environment ──────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ python3 not found."
    echo "   Install it from https://www.python.org/downloads/macos/  (or: brew install python)"
    read -r -p "Press Enter to close…"
    exit 1
fi

VENV=".venv"
if [ ! -d "$VENV" ]; then
    echo "▶ Creating virtual environment (.venv)…"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [ ! -f "$VENV/.core_deps_ok" ]; then
    echo "▶ Installing core dependencies (first run only — a couple of minutes)…"
    python -m pip install --upgrade pip
    python -m pip install -r requirements-core.txt
    touch "$VENV/.core_deps_ok"
fi

echo "▶ Launching scanner…"
python main.py
status=$?
if [ "$status" -ne 0 ]; then
    echo "⚠ main.py exited with status $status"
    read -r -p "Press Enter to close…"
fi
