"""
Communication with the ESP32 motor controller.

The ESP32 runs firmware/firmware.ino which listens for plain-text commands and
replies with OK or ERR.  Two transports are supported, selected at construction:

  • WIRED (default) — USB serial at 115200 baud.  Rock-solid; this is what the
    rig uses for real scans.
  • WIRELESS        — a TCP socket to the ESP32's WiFi server (firmware built
    with USE_WIFI 1).  Same line-oriented protocol, just carried over WiFi.
    Optional; intended for remote/demo operation, off by default.

The command protocol is transport-agnostic — every command is one text line and
the reply is one text line — so the wired/wireless choice lives entirely in the
_Link object; the MOVE/HOME logic below is identical either way.

Supported commands:
    MOVE X <steps>            — move X stepper <steps> half-steps (negative = reverse)
    MOVE Y <steps>            — move Y stepper <steps> half-steps
    MOVE XY <xsteps> <ysteps> — move both steppers concurrently (interleaved)
    HOME                      — reset the logical home position

A threading.Lock prevents two threads sending commands simultaneously,
which would corrupt the stream.
"""
import time
import socket
import threading
import serial
import serial.tools.list_ports

_BAUD     = 115200   # must match Serial.begin() in firmware.ino
_TIMEOUT  = 5.0      # seconds to wait for a response before giving up
_TCP_PORT = 3232     # must match WIFI_PORT in firmware.ino (wireless mode)


def _auto_detect_port():
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


class _SerialLink:
    """Wired USB-serial transport (the default)."""

    def __init__(self, port, baud):
        self._port = port or _auto_detect_port()
        self._baud = baud
        self._ser  = None

    def open(self):
        self._ser = serial.Serial(self._port, self._baud, timeout=_TIMEOUT)
        time.sleep(0.5)                    # wait for ESP32 to finish booting
        self._ser.reset_input_buffer()     # discard any startup messages

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def request(self, data: bytes, read_timeout: float) -> str:
        self._ser.reset_input_buffer()     # resync: drop any stale reply
        self._ser.timeout = read_timeout
        self._ser.write(data)              # newline already in data
        return self._ser.readline().decode(errors="ignore").strip()

    def describe(self):
        return f"serial {self._port} @ {self._baud}"


class _TcpLink:
    """Wireless transport — a TCP socket to the ESP32's WiFi server.

    Mirrors the serial link's request/response semantics: drain any stale bytes,
    send the command line, read one reply line (returning "" on timeout, which
    the caller treats as an error exactly as it would a serial timeout).  A
    manual line buffer is used instead of socket.makefile so the pre-write drain
    and the buffered reader can't fight over the same bytes.
    """

    def __init__(self, host, port=_TCP_PORT):
        self._host = host
        self._port = int(port)
        self._sock = None
        self._buf  = b""

    def open(self):
        self._sock = socket.create_connection((self._host, self._port),
                                               timeout=_TIMEOUT)
        time.sleep(0.5)                    # let the ESP32 settle / send READY
        self._drain()                      # discard the READY banner

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _drain(self):
        """Discard anything already waiting (stale reply / READY banner)."""
        self._buf = b""
        self._sock.setblocking(False)
        try:
            while self._sock.recv(4096):
                pass
        except (BlockingIOError, OSError):
            pass
        finally:
            self._sock.setblocking(True)

    def request(self, data: bytes, read_timeout: float) -> str:
        self._drain()                      # resync: drop any stale reply
        self._sock.settimeout(read_timeout)
        self._sock.sendall(data)           # newline already in data
        return self._readline().decode(errors="ignore").strip()

    def _readline(self) -> bytes:
        while b"\n" not in self._buf:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                return b""                 # timeout → empty, like serial readline
            if not chunk:                  # peer closed the connection
                return b""
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def describe(self):
        return f"tcp {self._host}:{self._port}"


class MotorController:

    def __init__(self, port=None, baud=_BAUD, host=None, tcp_port=_TCP_PORT):
        # host set → wireless (TCP over WiFi); otherwise wired USB serial.
        if host:
            self._link = _TcpLink(host, tcp_port)
        else:
            self._link = _SerialLink(port, baud)
        self._lock = threading.Lock()      # one command at a time

    @property
    def transport(self):
        """Human-readable description of the active link (for logs/status)."""
        return self._link.describe()

    def open(self):
        """Open the link and flush any startup noise from the ESP32."""
        self._link.open()

    def close(self):
        self._link.close()

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

        The link flushes its input before writing, so any stale/late reply from
        a previous hiccup can't be mistaken for this command's response.
        """
        read_timeout = 5.0 + abs(int(expected_steps)) * 0.003

        with self._lock:                       # one command at a time
            return self._link.request(cmd.encode(), read_timeout)

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
