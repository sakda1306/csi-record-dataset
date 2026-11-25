#!/usr/bin/env python3
"""Records the CSI dataset.

    python record.py                     uses config.toml next to this file
    python record.py --config other.toml
    python record.py --list-ports        shows the serial ports of this computer
    python record.py --port COM10        overrides serial.rx_port for this run

Open the printed /m address on a phone in the same Wi-Fi to control the recording.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

if sys.version_info < (3, 11):  # noqa: UP036  (friends may run an older Python)
    sys.exit("ต้องใช้ Python 3.11 ขึ้นไป (needs Python 3.11+ for tomllib)")

from csirec import config as config_mod
from csirec.engine import Engine
from csirec.ports import all_ports, find_rx
from csirec.server import lan_addresses, make_server

HERE = Path(__file__).resolve().parent

for _stream in (sys.stdout, sys.stderr):  # Thai text in a Windows console
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="CSI dataset recorder")
    ap.add_argument("--config", default=str(HERE / "config.toml"))
    ap.add_argument("--port", help="serial port of the Rx, overrides the config")
    ap.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    if a.list_ports:
        for dev, desc in all_ports():
            print(f"{dev:28s} {desc}")
        return 0

    try:
        cfg = config_mod.load(a.config)
    except config_mod.ConfigError as e:
        print(f"config ผิด: {e}")
        return 2

    session_dir = cfg.data_dir / cfg.session
    try:  # fail now, not after the first take, if the folder can't be written
        session_dir.mkdir(parents=True, exist_ok=True)
        probe = session_dir / ".write-test"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as e:
        print(f"เขียนโฟลเดอร์เก็บข้อมูลไม่ได้: {session_dir}\n  ({e})")
        print("แก้ [storage] data_dir ใน config.toml (ไดรฟ์นั้นมีอยู่ไหม? มีสิทธิ์เขียนไหม?)")
        return 2

    # Take the web port before touching any serial port: a second copy stops here.
    engine = Engine(cfg, port="")
    try:
        srv = make_server(engine)
    except OSError as e:
        print(
            f"เปิดเว็บที่พอร์ต {cfg.port} ไม่ได้ ({e}) — มีโปรแกรมนี้เปิดอยู่แล้วหรือเปล่า? "
            "หรือเปลี่ยน server.port ใน config.toml"
        )
        return 1

    port = a.port or cfg.rx_port
    if port == "auto":
        print("กำลังหาบอร์ด Rx ...")
        port = find_rx(cfg.baud)
        if not port:
            print("\nไม่เจอบอร์ด Rx ที่ส่ง CSI อยู่ พอร์ตที่มี:")
            for dev, desc in all_ports():
                print(f"  {dev:28s} {desc}")
            print("เช็กว่าเสียบ Rx แล้วและ Tx เปิดอยู่ หรือใส่ชื่อพอร์ตใน config.toml (serial.rx_port)")
            srv.server_close()
            return 1
    print(f"Rx: {port}")
    engine.port = port
    engine.start()
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    local = f"http://127.0.0.1:{cfg.port}/"
    print("\n" + "=" * 60)
    print(f"จอคอม (สัญญาณ):  {local}")
    if cfg.host in ("0.0.0.0", "::"):
        for ip in lan_addresses():
            print(f"มือถือ (รีโมต):   http://{ip}:{cfg.port}/m")
        print("  มือถือต้องต่อ Wi-Fi วงเดียวกับคอมนี้" + (f" — PIN {cfg.pin}" if cfg.pin else ""))
    else:
        print(f'server.host = {cfg.host}: มือถือเข้าไม่ได้ (ตั้งเป็น "0.0.0.0" ถ้าจะใช้รีโมต)')
    print(f"ข้อมูลเก็บที่:    {engine.store.session_dir}")
    print("กด Ctrl+C เพื่อปิด")
    print("=" * 60 + "\n")
    if cfg.open_browser and not a.no_browser:
        webbrowser.open(local)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nกำลังปิด ...")
    finally:
        engine.stop()
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
