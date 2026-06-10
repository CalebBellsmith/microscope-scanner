"""
Serial communication to the ESP32 motor controller.

The ESP32 runs firmware/firmware.ino which listens for plain-text commands
over USB serial at 115200 baud and replies with OK or ERR.

Supported commands:
    MOVE X <steps>   — move X stepper <steps> steps (negative = reverse)
    MOVE Y <steps>   — move Y stepper <steps> steps
    HOME             — move both axes to their home position

A threading.Lock prevents two threads sending commands simultaneously,
which would corrupt the serial stream.
"""
import threading
import serial
import serial.tools.list_ports

_BAUD    = 115200   # must match Serial.begin() in firmware.ino
_TIMEOUT = 5.0      # seconds to wait for a response before giving up


class MotorController:

    def __init__(self, port=None, baud=_BAUD):
        # If no port given, scan USB devices and pick the first likely ESP32
        self._port = port or self._auto_detect()
        self._baud = baud
        self._ser  = None                  # serial.Serial object, set in open()
        self._lock = threading.Lock()      # one command at a time

    @staticmethod
    def _auto_detect():
        """
        Scan all COM ports and return the first one that looks like an ESP32.
        Common USB-serial chips: CP2102 (CP210x), CH340, CH341.
        """
        for p in serial.tools.list_ports.comports():
            if ("CP210" in p.description or
                    "CH340" in p.description or
                    "USB"   in p.description.upper()):
                return p.device
        raise RuntimeError("ESP32 serial port not found. Connect device and retry.")

    def open(self):
        """Open the serial port and flush any startup noise from the ESP32."""
        self._ser = serial.Serial(self._port, self._baud, timeout=_TIMEOUT)
        import time
        time.sleep(0.5)                    # wait for ESP32 to finish booting
        self._ser.reset_input_buffer()     # discard any startup messages

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def move(self, axis: str, amount: int):
        """
        Send a MOVE command and block until the ESP32 replies OK.
        axis   : "X" or "Y"
        amount : steps to move (positive = forward, negative = reverse)
        Raises RuntimeError if the firmware replies with an error.
        """
        axis = axis.upper()
        assert axis in ("X", "Y"), f"Unknown axis {axis}"

        cmd = f"MOVE {axis} {amount}\n"    # newline terminates the command

        with self._lock:                   # prevent concurrent commands
            self._ser.write(cmd.encode())
            resp = self._ser.readline().decode().strip()   # wait for "OK" or "ERR ..."

        if resp != "OK":
            raise RuntimeError(f"Motor error on MOVE {axis} {amount}: {resp!r}")

    def home(self):
        """Send HOME command — moves both axes to their origin position."""
        with self._lock:
            self._ser.write(b"HOME\n")
            resp = self._ser.readline().decode().strip()
        if resp != "OK":
            raise RuntimeError(f"HOME failed: {resp!r}")
