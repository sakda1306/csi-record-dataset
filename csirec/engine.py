"""Serial capture, live link statistics and the recording state machine.

Phases:  idle -> countdown -> recording -> (review) -> rest -> countdown ... -> idle
Everything that changes state holds self.lock. Times are host time.time() seconds.
Frame times come from the receiver's microsecond clock, anchored to host time, so
they are accurate to the frame even though serial reads arrive in ~20 ms chunks.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections import Counter, deque

import numpy as np
import serial

from .config import NAME_RE, Config
from .csi_serial import Frame, FrameReader, LinkStats, open_quiet
from .quality import DATA_SC, amplitudes, dedupe, evaluate
from .storage import NO_VOLUNTEER, Storage

FINISH_GRACE_S = 0.25  # wait for the last frames still in the serial pipe
LIVE_STATS_S = 5.0
REANCHOR_S = 0.25
ROW_STEP = 2  # monitor waterfall gets every 2nd frame (50 Hz)


class CommandError(Exception):
    pass


class Engine:
    def __init__(self, cfg: Config, port: str):
        self.cfg = cfg
        self.port = port
        self.store = Storage(cfg)
        self.lock = threading.RLock()
        self.stop_evt = threading.Event()

        # capture
        self.port_ok = False
        self.port_error = ""
        self.device: dict = {}
        self.device_stat = ""
        self.checksum_errors = 0
        self._anchor: tuple[float, int] | None = None
        maxlen = int(cfg.rate_hz * max(cfg.window_s, LIVE_STATS_S) * 1.5) + 50
        self.recent: deque[tuple[float, Frame]] = deque(maxlen=maxlen)
        self.rows: deque[tuple[int, float, list[int]]] = deque(maxlen=maxlen // ROW_STEP + 10)
        self.amp_recent: deque[np.ndarray] = deque(maxlen=int(cfg.rate_hz * 0.5))
        self.n_frames = 0

        # session
        self.phase = "idle"
        self.phase_end: float | None = None
        self.batch: dict | None = None
        self.take: dict | None = None
        self.take_frames: list[tuple[float, Frame]] = []
        self.take_ck_start = 0
        self.take_seq = 0
        self.pending: dict | None = None  # finished take waiting for keep/redo
        self.last_result: dict | None = None
        self.counts: Counter = self.store.load_counts()
        self.volunteers = list(cfg.volunteers)
        for v, _ in self.counts:
            if v not in self.volunteers and v != NO_VOLUNTEER:
                self.volunteers.append(v)
        self.log: deque[str] = deque(maxlen=30)
        self._snapshot: dict = {}
        self.note(f"session {cfg.session} -> {self.store.session_dir}")

    # ================================================================== threads
    def start(self) -> None:
        threading.Thread(target=self._capture_loop, name="capture", daemon=True).start()
        threading.Thread(target=self._tick_loop, name="tick", daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            if self.pending:
                self.store.discard(self.pending["paths"])
                self.note("เทคที่ยังไม่ได้ตัดสินถูกย้ายไป _discarded")
                self.pending = None
        self.stop_evt.set()

    def note(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        self.log.append(line)
        print(line, flush=True)

    # ================================================================== capture
    def _capture_loop(self) -> None:
        while not self.stop_evt.is_set():
            try:
                ser = open_quiet(self.port, self.cfg.baud, timeout_s=0.02)
            except (serial.SerialException, OSError) as e:
                with self.lock:
                    if self.port_ok or not self.port_error:
                        self.note(f"เปิดพอร์ต {self.port} ไม่ได้: {e}")
                    self.port_ok, self.port_error = False, str(e)
                self.stop_evt.wait(2.0)
                continue
            reader = FrameReader()
            with self.lock:
                self.port_ok, self.port_error = True, ""
                self._anchor = None
            self.note(f"เปิดพอร์ต {self.port} แล้ว")
            try:
                while not self.stop_evt.is_set():
                    data = ser.read(8192)
                    if not data:
                        continue
                    ck_before = reader.checksum_errors
                    frames, lines = reader.feed(data)
                    now = time.time()
                    with self.lock:
                        self.checksum_errors += reader.checksum_errors - ck_before
                        for line in lines:
                            self._on_line(line)
                        for f in frames:
                            self._on_frame(now, f)
            except (serial.SerialException, OSError) as e:
                with self.lock:
                    self.port_ok, self.port_error = False, str(e)
                self.note(f"พอร์ต {self.port} หลุด: {e}")
            finally:
                ser.close()
            self.stop_evt.wait(1.0)

    def _frame_time(self, now: float, t_us: int) -> float:
        """Host time of a frame from the receiver clock. A frame can't be later than its
        arrival, so the anchor follows the least-delayed frame; a big jump (receiver
        reboot, clock wrap) starts a new anchor."""
        if self._anchor is not None:
            h0, u0 = self._anchor
            d = (t_us - u0) % 2**32
            if d >= 2**31:
                d -= 2**32
            t = h0 + d / 1e6
            if now - REANCHOR_S <= t <= now:
                return t
        self._anchor = (now, t_us)
        return now

    def _on_line(self, line: str) -> None:
        if line.startswith("#INFO"):
            for part in line.split(",")[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    self.device[k.strip()] = v.strip()
        elif line.startswith("#STAT"):
            self.device_stat = line
        elif line.startswith("#") and line.strip("# "):
            self.note(f"Rx: {line.lstrip('# ')}")

    def _on_frame(self, now: float, f: Frame) -> None:
        t = self._frame_time(now, f.t_us)
        self.n_frames += 1
        self.recent.append((t, f))
        amp = amplitudes(f)[DATA_SC]
        self.amp_recent.append(amp)
        if self.n_frames % ROW_STEP == 0:
            self.rows.append((self.n_frames, t, amp.round().astype(int).tolist()))
        tk = self.take
        if tk is not None and self.phase in ("countdown", "recording"):
            if tk["t_start"] <= t < tk["t_end"]:
                self.take_frames.append((t - tk["t_start"], f))

    # ================================================================== state machine
    def _tick_loop(self) -> None:
        last_snap = 0.0
        while not self.stop_evt.is_set():
            now = time.time()
            with self.lock:
                try:
                    self._advance(now)
                except Exception as e:  # never let the state machine thread die
                    self.note(f"ERROR {type(e).__name__}: {e}")
                    self.phase, self.take, self.batch = "idle", None, None
                if now - last_snap >= 0.1:
                    self._snapshot = self._build_snapshot(now)
                    last_snap = now
            self.stop_evt.wait(0.02)

    def _advance(self, now: float) -> None:
        if self.phase == "countdown" and now >= self.take["t_start"]:
            self.phase = "recording"
            self.phase_end = self.take["t_end"]
        elif self.phase == "recording" and now >= self.take["t_end"] + FINISH_GRACE_S:
            self._finish_take()
        elif self.phase == "rest" and now >= self.phase_end:
            self._begin_countdown(now)

    def _begin_countdown(self, now: float) -> None:
        b = self.batch
        act = self.cfg.activity(b["activity"])
        self.take_seq += 1
        t_start = now + act.countdown_s
        self.take = {
            "seq": self.take_seq,
            "activity": act.id,
            "name": act.name,
            "label": act.label,
            "cue_text": act.cue_text,
            "volunteer": b["volunteer"],
            "position": b["position"],
            "rep": b["done"] + 1,
            "reps": b["reps"],
            "t_start": t_start,
            "t_end": t_start + act.duration_s,
            "cue_at": t_start + act.cue_s if act.cue_s else None,
            "duration_s": act.duration_s,
        }
        self.take_frames = []
        self.take_ck_start = self.checksum_errors
        self.phase = "countdown"
        self.phase_end = t_start

    def _finish_take(self) -> None:
        tk, b = self.take, self.batch
        act = self.cfg.activity(tk["activity"])
        frames = sorted(self.take_frames, key=lambda x: x[0])
        cue_off = act.cue_s if act.cue_s else None
        q = evaluate(
            frames,
            act.duration_s,
            cue_off,
            act.expect_motion,
            self.cfg,
            self.checksum_errors - self.take_ck_start,
        )
        kept, _ = dedupe(frames, self.cfg.rate_hz)
        meta = {
            "schema": 1,
            "frame_format": "csi_rx serial frame v2 (see firmware/csi_rx/csi_rx.ino)",
            "session": self.cfg.session,
            "location": self.cfg.location,
            "recorded_at": dt.datetime.fromtimestamp(tk["t_start"])
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "t_start": tk["t_start"],
            "volunteer": tk["volunteer"],
            "position": tk["position"],
            "activity": act.id,
            "activity_name": act.name,
            "label": act.label,
            "note": b.get("note", ""),
            "duration_s": act.duration_s,
            "countdown_s": act.countdown_s,
            "cue": None,
            "rate_hz_target": self.cfg.rate_hz,
            "device": dict(self.device),
            "quality": q,
        }
        if cue_off is not None:
            times = np.array([t for t, _ in kept]) if kept else np.zeros(0)
            meta["cue"] = {
                "offset_s": cue_off,
                "timestamp": tk["cue_at"],
                "csv_row": int(np.searchsorted(times, cue_off)) if times.size else None,
                "note": "sound/flash at this time; the movement starts right after it",
            }
        paths = self.store.write_take(meta, frames, kept)
        self.pending = {"meta": meta, "paths": paths}
        self.last_result = {
            "seq": tk["seq"],
            "take_id": meta["take_id"],
            "activity": act.id,
            "name": act.name,
            "volunteer": tk["volunteer"],
            "rep": tk["rep"],
            "reps": tk["reps"],
            "status": "pending",
            **{
                k: q.get(k)
                for k in (
                    "verdict",
                    "reasons",
                    "warnings",
                    "rate_hz",
                    "air_loss_pct",
                    "gap_max_ms",
                    "rssi_median",
                    "motion_ratio",
                    "frames",
                )
            },
        }
        self.note(
            f"{meta['take_id']}: {q['verdict'].upper()} {q['rate_hz']} Hz "
            f"loss {q['air_loss_pct']} % {' '.join(q['reasons'] + q['warnings'])}"
        )
        if q["verdict"] == "pass" and self.cfg.auto_keep_pass:
            self._keep("auto")
        else:
            self.phase, self.phase_end = "review", None

    def _keep(self, by: str) -> None:
        p = self.pending
        self.store.keep(p["meta"], by)
        self.counts[(p["meta"]["volunteer"], p["meta"]["activity"])] += 1
        self.last_result["status"] = "kept"
        self.pending = None
        self.batch["done"] += 1
        self._next_or_idle()

    def _next_or_idle(self) -> None:
        b = self.batch
        if b["done"] >= b["reps"]:
            self.note(f"ครบ {b['reps']} เทค: {b['activity']} / {b['volunteer']}")
            self.phase, self.phase_end, self.take, self.batch = "idle", None, None, None
        else:
            act = self.cfg.activity(b["activity"])
            self.phase, self.phase_end, self.take = "rest", time.time() + act.rest_s, None

    # ================================================================== commands
    def command(self, action: str, args: dict, who: str = "remote") -> dict:
        with self.lock:
            fn = {
                "start": self._cmd_start,
                "stop": self._cmd_stop,
                "keep": self._cmd_keep,
                "redo": self._cmd_redo,
                "discard": self._cmd_discard,
            }.get(action)
            if fn is None:
                raise CommandError(f"unknown action {action!r}")
            fn(args, who)
            self._snapshot = self._build_snapshot(time.time())
            return {"ok": True}

    def _cmd_start(self, a: dict, who: str) -> None:
        if self.phase != "idle":
            raise CommandError("กำลังทำงานอยู่ กดหยุดก่อน")
        try:
            act = self.cfg.activity(str(a.get("activity", "")))
        except KeyError:
            raise CommandError("ไม่รู้จักท่านี้") from None
        vol = NO_VOLUNTEER if act.no_volunteer else str(a.get("volunteer", "")).strip()
        pos = str(a.get("position", "")).strip() or (self.cfg.positions or ["p1"])[0]
        if not NAME_RE.match(vol):
            raise CommandError("ชื่ออาสาสมัครใช้ได้แค่ A-Z a-z 0-9 _ -")
        if not NAME_RE.match(pos):
            raise CommandError("ชื่อตำแหน่งใช้ได้แค่ A-Z a-z 0-9 _ -")
        try:
            reps = int(a.get("reps", 1))
        except (TypeError, ValueError):
            raise CommandError("จำนวนเทคต้องเป็นตัวเลข") from None
        if not 1 <= reps <= 200:
            raise CommandError("จำนวนเทค 1-200")
        if not self.recent or time.time() - self.recent[-1][0] > 2.0:
            raise CommandError("ยังไม่มีสัญญาณจาก Rx — เช็ก Tx/Rx ก่อน")
        if vol != NO_VOLUNTEER and vol not in self.volunteers:
            self.volunteers.append(vol)
        self.batch = {
            "activity": act.id,
            "volunteer": vol,
            "position": pos,
            "reps": reps,
            "done": 0,
            "note": str(a.get("note", ""))[:200],
        }
        self.note(f"เริ่ม {act.id} x{reps} / {vol} / {pos} (สั่งจาก {who})")
        self._begin_countdown(time.time())

    def _cmd_stop(self, a: dict, who: str) -> None:
        if self.phase == "review":
            raise CommandError("เลือก เก็บ / บันทึกใหม่ / ทิ้ง ก่อน")
        if self.phase != "idle":
            self.note(f"หยุด (สั่งจาก {who}) — เทคที่กำลังบันทึกไม่ถูกเก็บ")
        self.phase, self.phase_end, self.take, self.batch = "idle", None, None, None

    def _need_review(self) -> None:
        if self.phase != "review" or not self.pending:
            raise CommandError("ไม่มีเทคที่รอตัดสิน")

    def _cmd_keep(self, a: dict, who: str) -> None:
        self._need_review()
        self.note(f"เก็บ {self.pending['meta']['take_id']} (สั่งจาก {who})")
        self._keep(who)

    def _cmd_redo(self, a: dict, who: str) -> None:
        self._need_review()
        self.store.discard(self.pending["paths"])
        self.note(f"ทิ้ง {self.pending['meta']['take_id']} แล้วบันทึกใหม่ (สั่งจาก {who})")
        self.last_result["status"] = "discarded"
        self.pending = None
        self._begin_countdown(time.time())

    def _cmd_discard(self, a: dict, who: str) -> None:
        self._need_review()
        self.store.discard(self.pending["paths"])
        self.note(f"ทิ้ง {self.pending['meta']['take_id']} และหยุด (สั่งจาก {who})")
        self.last_result["status"] = "discarded"
        self.pending = None
        self.phase, self.phase_end, self.take, self.batch = "idle", None, None, None

    # ================================================================== snapshots
    def _live(self, now: float) -> dict:
        frames = [(t, f) for t, f in self.recent if now - t <= LIVE_STATS_S]
        last_age = now - self.recent[-1][0] if self.recent else None
        live = {
            "port": self.port,
            "port_ok": self.port_ok,
            "port_error": self.port_error,
            "last_frame_age_s": None if last_age is None else round(last_age, 2),
            "rate_hz": 0,
            "rssi": None,
            "air_loss_pct": None,
            "gap_max_ms": None,
            "uart_lost": 0,
            "checksum_errors": self.checksum_errors,
            "motion": None,
            "status": "none",
        }
        if len(frames) >= 3:
            st = LinkStats(rate_hz=self.cfg.rate_hz)
            kept = [(t, f) for t, f in frames if st.add(f)]
            last1 = [f for t, f in kept if now - t <= 1.0]
            t = np.array([x for x, _ in kept])
            gaps = np.diff(np.append(t, now)) * 1000
            live.update(
                rate_hz=len(last1),
                rssi=int(np.median([f.rssi for f in last1])) if last1 else None,
                air_loss_pct=round(st.air_loss_pct, 2),
                gap_max_ms=round(float(gaps.max()), 0),
                uart_lost=st.uart_lost,
            )
            q = self.cfg
            if (
                live["rate_hz"] >= q.min_rate_hz
                and st.air_loss_pct <= q.max_air_loss_pct
                and gaps.max() <= q.max_gap_ms
            ):
                live["status"] = "ok"
            elif live["rate_hz"] >= 0.8 * q.min_rate_hz:
                live["status"] = "warn"
            else:
                live["status"] = "bad"
        if len(self.amp_recent) >= 10 and last_age is not None and last_age < 1.0:
            live["motion"] = round(float(np.std(np.array(self.amp_recent), axis=0).mean()), 3)
        return live

    def _build_snapshot(self, now: float) -> dict:
        counts: dict[str, dict[str, int]] = {}
        for (v, a), n in self.counts.items():
            counts.setdefault(v, {})[a] = n
        return {
            "server_time": now,
            "phase": self.phase,
            "phase_end": self.phase_end,
            "take": dict(self.take) if self.take else None,
            "batch": dict(self.batch) if self.batch else None,
            "last_result": dict(self.last_result) if self.last_result else None,
            "link": self._live(now),
            "device": dict(self.device),
            "counts": counts,
            "volunteers": list(self.volunteers),
            "log": list(self.log)[-8:],
            "session_dir": str(self.store.session_dir),
        }

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot

    def rows_since(self, cursor: int) -> tuple[list, int]:
        with self.lock:
            out = [[round(t, 3), *amp] for n, t, amp in self.rows if n > cursor]
            return out, (self.rows[-1][0] if self.rows else cursor)
