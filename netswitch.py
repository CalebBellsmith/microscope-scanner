"""
Switch this PC's WiFi between the house network and the rig's own access point.

The ESP32 can host its own network (firmware `WIFI_AP_MODE 1`), which removes
the congested mesh, the roaming and the DHCP lease from the wireless link.  The
cost is that the PC has to leave whatever network it is on and join the rig's,
then come back afterwards.  Doing that by hand every session is tedious, so this
module drives it from the Connect button.

Windows only — it shells out to `netsh`, which is the only interface Windows
offers for this.  On any other platform `supported()` returns False and the GUI
falls back to asking the user to switch networks manually; nothing raises.

The rig's AP password is a shared secret printed in the firmware, not a personal
credential, so writing it into a WLAN profile is no worse than typing it into
the Windows WiFi menu.  The temporary profile XML is deleted immediately either
way, and the profile is created in `manual` connection mode so Windows never
auto-joins the rig behind your back.
"""

import os
import platform
import re
import subprocess
import tempfile
import time

# Suppress the console window that would otherwise flash on every netsh call.
_NO_WINDOW = 0x08000000 if platform.system() == "Windows" else 0


def supported() -> bool:
    """True if automatic switching is possible on this machine."""
    return platform.system() == "Windows"


def _netsh(*args, timeout=20) -> str:
    """Run a netsh command and return its combined output (never raises on a
    non-zero exit — callers judge success from the state, not the exit code,
    because netsh reports plenty of soft failures with status 0)."""
    proc = subprocess.run(
        ["netsh", *args],
        capture_output=True, text=True, timeout=timeout,
        creationflags=_NO_WINDOW,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def current_ssid():
    """The SSID this PC is presently joined to, or None if not on WiFi.

    Parsed out of `netsh wlan show interfaces`.  The label is localised on
    non-English Windows, so match the line shape (a key ending in SSID that is
    not BSSID) rather than the literal English word.
    """
    if not supported():
        return None
    try:
        out = _netsh("wlan", "show", "interfaces")
    except Exception:
        return None
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key.upper().endswith("SSID") and not key.upper().endswith("BSSID"):
            value = value.strip()
            return value or None
    return None


def _profile_names():
    try:
        out = _netsh("wlan", "show", "profiles")
    except Exception:
        return set()
    # Same localisation problem: take whatever follows the last colon on any
    # line that mentions a profile, which is the SSID on every locale.
    names = set()
    for line in out.splitlines():
        if ":" in line and re.search(r"\bprofile\b", line, re.IGNORECASE):
            name = line.rpartition(":")[2].strip()
            if name:
                names.add(name)
    return names


def _profile_xml(ssid: str, password: str) -> str:
    return f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{ssid}</name>
  <SSIDConfig><SSID><name>{ssid}</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>manual</connectionMode>
  <MSM><security>
    <authEncryption>
      <authentication>WPA2PSK</authentication>
      <encryption>AES</encryption>
      <useOneX>false</useOneX>
    </authEncryption>
    <sharedKey>
      <keyType>passPhrase</keyType>
      <protected>false</protected>
      <keyMaterial>{password}</keyMaterial>
    </sharedKey>
  </security></MSM>
</WLANProfile>
"""


def ensure_profile(ssid: str, password: str):
    """Create a WLAN profile for the rig if Windows doesn't already have one, so
    the very first connect works without a manual join."""
    if ssid in _profile_names():
        return
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(_profile_xml(ssid, password))
            path = fh.name
        _netsh("wlan", "add", "profile", f"filename={path}", "user=all")
    finally:
        if path:
            try:
                os.unlink(path)          # don't leave the key on disk
            except OSError:
                pass


def connect(ssid: str, password: str = "", timeout: float = 25.0):
    """Join `ssid`, blocking until Windows reports it as the active network.

    Raises RuntimeError with a message worth showing the user if the join does
    not complete inside `timeout`.
    """
    if not supported():
        raise RuntimeError(
            "Automatic WiFi switching is Windows-only. Join the rig's network "
            "from your WiFi menu, then click Connect again.")
    if current_ssid() == ssid:
        return
    if password:
        ensure_profile(ssid, password)

    _netsh("wlan", "connect", f"name={ssid}", f"ssid={ssid}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        if current_ssid() == ssid:
            time.sleep(1.5)              # let DHCP finish before anyone dials
            return
        time.sleep(0.5)

    raise RuntimeError(
        f"Could not join '{ssid}' within {timeout:.0f}s.\n\n"
        f"Check the rig is powered and broadcasting (its serial monitor prints "
        f"'WIFI AP up'), and that the SSID and password match the firmware. "
        f"Some managed laptops block scripted network changes — if so, join "
        f"'{ssid}' from the Windows WiFi menu and untick auto-switch.")
