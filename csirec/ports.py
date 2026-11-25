"""Lists serial ports and finds the csi_rx board."""

from __future__ import annotations

import sys
import time

import serial
from serial.tools import list_ports

from .csi_serial import FrameReader, open_quiet

SKIP_WORDS = ("bluetooth", "debug-console", "wlan-debug")


def all_ports() -> list[tuple[str, str]]:
    return [(p.device, p.description or "") for p in sorted(list_ports.comports())]


def candidates() -> list[str]:
    devs = [d for d, desc in all_ports() if not any(w in (d + desc).lower() for w in SKIP_WORDS)]
    if sys.platform == "darwin":  # each port shows up as /dev/cu.* and /dev/tty.*; use cu.*
        devs = [d for d in devs if not d.startswith("/dev/tty.")]
    return devs


def probe(port: str, baud: int, seconds: float = 3.0) -> str | None:
    """Returns a short description if the port streams csi_rx data, else None."""
    try:
        s = open_quiet(port, baud)
    except (serial.SerialException, OSError):
        return None
    reader = FrameReader()
    n = 0
    try:
        t_end = time.time() + seconds
        while time.time() < t_end:
            frames, lines = reader.feed(s.read(4096))
            n += len(frames)
            if n >= 5:
                return f"{n} CSI frames"
            for line in lines:
                if "fw=csi_rx" in line or line.startswith("# csi_rx"):
                    return line
    except (serial.SerialException, OSError):
        return None
    finally:
        s.close()
    return None


def find_rx(baud: int) -> str | None:
    for dev in candidates():
        print(f"  ลอง {dev} ...", end=" ", flush=True)
        found = probe(dev, baud)
        print(found or "ไม่ใช่", flush=True)
        if found:
            return dev
    return None
