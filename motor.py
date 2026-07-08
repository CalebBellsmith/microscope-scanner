"""
Serial communication to the ESP32 motor controller.

The ESP32 runs firmware/firmware.ino which listens for plain-text commands
over USB serial at 115200 baud and replies with OK or ERR.

Supported commands:
    MOVE X <steps>            — move X stepper <steps> half-steps (negative = reverse)
    MOVE Y <steps>            — move Y stepper <steps> half-steps
    MOVE XY <xsteps> <ysteps> — move both steppers concurrently (interleaved)
    HOME                      — reset the logical home position

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

    def _command(self, cmd: str, expected_steps: int = 0) -> str:
        """
        Send one command and return the firmware's reply line.

        The firmware blocks for the WHOLE move before replying OK, so a large
        move can take far longer than the default 5 s.  We therefore scale the
        read timeout to the move size (budget 3 ms / half-step + margin) — a
        fixed 5 s timeout was cutting long sweeps/returns short and reporting a
        false "Motor error", which then left the late OK in the buffer and
        desynced every subsequent command (Python ran one move ahead of the
        stage → captures fired mid-motion → blurry photos).

        We also flush the input buffer before writing, so any stale/late reply
        from a previous hiccup can't be mistaken for this command's response.
        """
        read_timeout = 5.0 + abs(int(expected_steps)) * 0.003

        with self._lock:                       # one command at a time
            self._ser.reset_input_buffer()     # resync: drop any stale reply
            self._ser.timeout = read_timeout
            self._ser.write(cmd.encode())      # newline already in cmd
            return self._ser.readline().decode(errors="ignore").strip()

    def move(self, axis: str, amount: int):
        """
        Send a MOVE command and block until the ESP32 replies OK.
        axis   : "X", "Y", or "Z" (Z = focus stepper)
        amount : steps to move (positive = forward, negative = reverse)
        Raises RuntimeError if the firmware replies with an error.
        """
        axis = axis.upper()
        assert axis in ("X", "Y", "Z"), f"Unknown axis {axis}"

        resp = self._command(f"MOVE {axis} {amount}\n", expected_steps=amount)
        if resp != "OK":
            raise RuntimeError(f"Motor error on MOVE {axis} {amount}: {resp!r}")

    def move_xy(self, x_amount: int, y_amount: int):
        """
        Move BOTH axes concurrently via the firmware's interleaved MOVE XY.
        x_amount / y_amount : half-steps (positive = forward, negative = reverse).
        The stage travels a straight diagonal and both axes finish together,
        which is faster than two sequential moves for the rung repositioning.
        Raises RuntimeError if the firmware replies with an error.
        """
        x_amount = int(x_amount)
        y_amount = int(y_amount)
        resp = self._command(
            f"MOVE XY {x_amount} {y_amount}\n",
            expected_steps=max(abs(x_amount), abs(y_amount)),
        )
        if resp != "OK":
            raise RuntimeError(
                f"Motor error on MOVE XY {x_amount} {y_amount}: {resp!r}"
            )

    def home(self):
        """Send HOME command — moves both axes to their origin position."""
        resp = self._command(b"HOME\n".decode())
        if resp != "OK":
            raise RuntimeError(f"HOME failed: {resp!r}")
