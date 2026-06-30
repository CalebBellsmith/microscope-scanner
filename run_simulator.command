#!/bin/bash
# Double-click launcher (macOS) for the hardware-free Scan Path Simulator.
# Opens the dry-run GUI so you can verify the capture motion before running
# the real rig on Windows.
cd "$(dirname "$0")"
exec python3 scan_simulator.py
