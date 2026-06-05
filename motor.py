"""
Serial communication to ESP32 motor controller.
Commands: MOVE X <steps>  |  MOVE Y <units>
Responses: OK  |  ERR <msg>
"""
import threading
import serial
import serial.tools.list_ports

_BAUD = 115200
_TIMEOUT = 5.0


class MotorController:
    def __init__(self, port=None, baud=_BAUD):
        self._port = port or self._auto_detect()
        self._baud = baud
        self._ser = None
        self._lock = threading.Lock()

    @staticmethod
    def _auto_detect():
        for p in serial.tools.list_ports.comports():
            if "CP210" in p.description or "CH340" in p.description or "USB" in p.description.upper():
                return p.device
        raise RuntimeError("ESP32 serial port not found. Connect device and retry.")

    def open(self):
        self._ser = serial.Serial(self._port, self._baud, timeout=_TIMEOUT)
        # flush startup noise
        import time; time.sleep(0.5)
        self._ser.reset_input_buffer()

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def move(self, axis: str, amount: int):
        """Send MOVE X|Y <amount> and block until OK."""
        axis = axis.upper()
        assert axis in ("X", "Y"), f"Unknown axis {axis}"
        cmd = f"MOVE {axis} {amount}\n"
        with self._lock:
            self._ser.write(cmd.encode())
            resp = self._ser.readline().decode().strip()
        if resp != "OK":
            raise RuntimeError(f"Motor error on MOVE {axis} {amount}: {resp!r}")

    def home(self):
        """Send HOME command if firmware supports it."""
        with self._lock:
            self._ser.write(b"HOME\n")
            resp = self._ser.readline().decode().strip()
        if resp != "OK":
            raise RuntimeError(f"HOME failed: {resp!r}")
