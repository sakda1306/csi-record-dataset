"""Small HTTP server: the two web pages, a live event stream and a command endpoint.

GET  /            monitor page (the computer next to the Rx)
GET  /m           remote page (phone)
GET  /api/config  settings the pages need (never the PIN)
GET  /api/stream  server-sent events, 10 per second (?view=monitor adds waterfall rows)
POST /api/cmd     {"pin": "...", "action": "start|stop|keep|redo|discard", "args": {...}}
"""

from __future__ import annotations

import hmac
import json
import socket
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .engine import CommandError, Engine

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC = {
    "/": ("monitor.html", "text/html; charset=utf-8"),
    "/m": ("remote.html", "text/html; charset=utf-8"),
    "/common.js": ("common.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
STREAM_PERIOD_S = 0.1
MAX_BODY = 4096


def lan_addresses() -> list[str]:
    ips: list[str] = []
    try:  # the address of the interface that has the default route; no packet is sent
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass
    return ips


def make_server(engine: Engine) -> ThreadingHTTPServer:
    cfg = engine.cfg
    public = cfg.public()

    class Handler(BaseHTTPRequestHandler):
        server_version = "csirec"

        def log_message(self, fmt, *args):  # keep the terminal for recording messages
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in STATIC:
                name, ctype = STATIC[url.path]
                self._send(200, (WEB_DIR / name).read_bytes(), ctype)
            elif url.path == "/api/config":
                port = cfg.port
                urls = [f"http://{ip}:{port}/m" for ip in lan_addresses()]
                self._json(200, {**public, "remote_urls": urls})
            elif url.path == "/api/stream":
                view = parse_qs(url.query).get("view", ["remote"])[0]
                self._stream(view == "monitor")
            else:
                self._send(404, b"not found", "text/plain")

        def _stream(self, with_rows: bool) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            cursor = 0
            try:
                while not engine.stop_evt.is_set():
                    msg = {"status": engine.snapshot()}
                    if with_rows:
                        msg["rows"], cursor = engine.rows_since(cursor)
                    data = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(STREAM_PERIOD_S)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

        def do_POST(self):
            if urlparse(self.path).path != "/api/cmd":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > MAX_BODY:
                    return self._json(413, {"ok": False, "error": "request too large"})
                req = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                return self._json(400, {"ok": False, "error": "bad request"})
            local = self.client_address[0] in ("127.0.0.1", "::1")
            if cfg.pin and not local and not hmac.compare_digest(str(req.get("pin", "")), cfg.pin):
                return self._json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "PIN ไม่ถูกต้อง"})
            args = req.get("args") if isinstance(req.get("args"), dict) else {}
            who = "คอม" if local else self.client_address[0]
            try:
                self._json(200, engine.command(str(req.get("action", "")), args, who))
            except CommandError as e:
                self._json(409, {"ok": False, "error": str(e)})

    srv = _Server((cfg.host, cfg.port), Handler)
    srv.daemon_threads = True
    return srv


class _Server(ThreadingHTTPServer):
    """Refuses a port another program already holds. On Windows, SO_REUSEADDR (the
    http.server default) lets a second copy bind the same port without an error."""

    allow_reuse_address = sys.platform != "win32"

    def server_bind(self):
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()
