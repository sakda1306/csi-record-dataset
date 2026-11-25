"""Loads and checks config.toml."""

from __future__ import annotations

import datetime as dt
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ID_RE = re.compile(r"^[a-z0-9_]+$")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")  # volunteer / position / session names in paths


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Activity:
    id: str
    name: str
    label: str
    target: int
    countdown_s: float
    duration_s: float
    cue_s: float  # 0 = no cue
    rest_s: float
    cue_text: str
    expect_motion: bool
    no_volunteer: bool

    def public(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Config:
    path: Path
    rx_port: str
    baud: int
    rate_hz: float
    host: str
    port: int
    pin: str
    open_browser: bool
    data_dir: Path
    session: str
    location: str
    write_csv: bool
    keep_discarded: bool
    min_rate_hz: float
    max_air_loss_pct: float
    max_gap_ms: float
    min_motion_ratio: float
    auto_keep_pass: bool
    cue_sound: bool
    cue_speech: bool
    cue_vibrate: bool
    cue_flash_s: float
    window_s: float
    volunteers: list[str]
    positions: list[str]
    activities: list[Activity] = field(default_factory=list)

    def activity(self, activity_id: str) -> Activity:
        for a in self.activities:
            if a.id == activity_id:
                return a
        raise KeyError(activity_id)

    def public(self) -> dict:
        """What the web pages may see (no PIN)."""
        return {
            "rate_hz": self.rate_hz,
            "session": self.session,
            "location": self.location,
            "data_dir": str(self.data_dir / self.session),
            "quality": {
                "min_rate_hz": self.min_rate_hz,
                "max_air_loss_pct": self.max_air_loss_pct,
                "max_gap_ms": self.max_gap_ms,
                "min_motion_ratio": self.min_motion_ratio,
            },
            "cue": {
                "sound": self.cue_sound,
                "speech": self.cue_speech,
                "vibrate": self.cue_vibrate,
                "flash_s": self.cue_flash_s,
            },
            "window_s": self.window_s,
            "pin_required": bool(self.pin),
            "volunteers": self.volunteers,
            "positions": self.positions,
            "activities": [a.public() for a in self.activities],
        }


def _get(d: dict, key: str, typ, default=None, where: str = ""):
    if key not in d:
        if default is None:
            raise ConfigError(f"{where}: missing '{key}'")
        return default
    v = d[key]
    if typ is float and isinstance(v, int) and not isinstance(v, bool):
        v = float(v)
    if not isinstance(v, typ) or (typ is int and isinstance(v, bool)):
        raise ConfigError(f"{where}.{key}: expected {typ.__name__}, got {v!r}")
    return v


KNOWN_KEYS = {
    "serial": {"rx_port", "baud", "rate_hz"},
    "server": {"host", "port", "pin", "open_browser"},
    "storage": {"data_dir", "session", "location", "write_csv", "keep_discarded"},
    "timing": {"countdown_s", "duration_s", "cue_s", "rest_s"},
    "quality": {
        "min_rate_hz",
        "max_air_loss_pct",
        "max_gap_ms",
        "min_motion_ratio",
        "auto_keep_pass",
    },
    "cue": {"sound", "speech", "vibrate", "flash_s"},
    "display": {"window_s"},
    "people": {"volunteers", "positions"},
}
ACTIVITY_KEYS = {
    "id",
    "name",
    "label",
    "target",
    "countdown_s",
    "duration_s",
    "cue_s",
    "rest_s",
    "cue_text",
    "expect_motion",
    "no_volunteer",
}


def _check_keys(raw: dict) -> None:
    """A typo, or a key written under the wrong [table], is an error instead of being ignored."""
    for table, value in raw.items():
        if table == "activities":
            for i, a in enumerate(value):
                bad = set(a) - ACTIVITY_KEYS
                if bad:
                    raise ConfigError(f"activities[{i}]: unknown key(s) {', '.join(sorted(bad))}")
            continue
        if table not in KNOWN_KEYS:
            raise ConfigError(f"unknown section or key '{table}'")
        bad = set(value) - KNOWN_KEYS[table]
        if bad:
            raise ConfigError(
                f"[{table}]: unknown key(s) {', '.join(sorted(bad))} — "
                "check the spelling, or whether it sits under the right [section]"
            )


def load(path: str | Path) -> Config:
    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        hint = ""
        if "escape" in str(e).lower():
            hint = (
                ' — Windows path ใน "..." ต้องใช้ / เช่น "D:/Dataset" '
                "หรือใช้ 'D:\\Dataset' (เครื่องหมาย ')"
            )
        raise ConfigError(f"{path.name}: {e}{hint}") from None

    _check_keys(raw)
    s = raw.get("serial", {})
    sv = raw.get("server", {})
    st = raw.get("storage", {})
    tm = raw.get("timing", {})
    q = raw.get("quality", {})
    cu = raw.get("cue", {})
    dp = raw.get("display", {})
    pp = raw.get("people", {})

    defaults = {
        "countdown_s": _get(tm, "countdown_s", float, 5.0, "timing"),
        "duration_s": _get(tm, "duration_s", float, 10.0, "timing"),
        "cue_s": _get(tm, "cue_s", float, 4.0, "timing"),
        "rest_s": _get(tm, "rest_s", float, 5.0, "timing"),
    }

    acts: list[Activity] = []
    seen = set()
    for i, a in enumerate(raw.get("activities", [])):
        where = f"activities[{i}]"
        aid = _get(a, "id", str, where=where)
        if not ID_RE.match(aid):
            raise ConfigError(f"{where}.id '{aid}': use only a-z, 0-9 and _")
        if aid in seen:
            raise ConfigError(f"{where}.id '{aid}' is used twice")
        seen.add(aid)
        label = _get(a, "label", str, "normal", where)
        if label not in ("fall", "normal"):
            raise ConfigError(f'{where}.label must be "fall" or "normal", got {label!r}')
        act = Activity(
            id=aid,
            name=_get(a, "name", str, aid, where),
            label=label,
            target=_get(a, "target", int, 10, where),
            countdown_s=_get(a, "countdown_s", float, defaults["countdown_s"], where),
            duration_s=_get(a, "duration_s", float, defaults["duration_s"], where),
            cue_s=_get(a, "cue_s", float, defaults["cue_s"], where),
            rest_s=_get(a, "rest_s", float, defaults["rest_s"], where),
            cue_text=_get(a, "cue_text", str, "ทำท่า!", where),
            expect_motion=_get(a, "expect_motion", bool, True, where),
            no_volunteer=_get(a, "no_volunteer", bool, False, where),
        )
        if act.duration_s <= 0 or act.countdown_s < 0 or act.rest_s < 0:
            raise ConfigError(f"{where}: times must be positive")
        if act.cue_s < 0 or (act.cue_s and act.cue_s >= act.duration_s):
            raise ConfigError(
                f"{where} ({aid}): cue_s {act.cue_s:g} must be 0 or less than "
                f"duration_s {act.duration_s:g}"
            )
        acts.append(act)
    if not acts:
        raise ConfigError("no [[activities]] defined")

    volunteers = _get(pp, "volunteers", list, [], "people")
    positions = _get(pp, "positions", list, ["p1"], "people")
    if not positions:
        raise ConfigError("people.positions needs at least one position")
    for name in [*volunteers, *positions]:
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise ConfigError(f"volunteer/position name {name!r}: use only A-Z a-z 0-9 _ -")

    session = _get(st, "session", str, "auto", "storage")
    if session == "auto":
        session = dt.date.today().isoformat()
    if not NAME_RE.match(session):
        raise ConfigError(f"storage.session {session!r}: use only A-Z a-z 0-9 _ -")

    data_dir = Path(_get(st, "data_dir", str, "data", "storage")).expanduser()
    if data_dir.drive and not data_dir.root:  # "D:Dataset" = relative to D:'s current folder
        raise ConfigError(
            f'storage.data_dir {data_dir.as_posix()!r}: ต้องมี / หลังชื่อไดรฟ์ เช่น "D:/Dataset"'
        )
    if not data_dir.is_absolute():
        data_dir = path.parent / data_dir

    port = _get(sv, "port", int, 8765, "server")
    if not 1 <= port <= 65535:
        raise ConfigError(f"server.port {port} is out of range")

    return Config(
        path=path,
        rx_port=_get(s, "rx_port", str, "auto", "serial"),
        baud=_get(s, "baud", int, 921600, "serial"),
        rate_hz=_get(s, "rate_hz", float, 100.0, "serial"),
        host=_get(sv, "host", str, "0.0.0.0", "server"),
        port=port,
        pin=str(_get(sv, "pin", str, "", "server")),
        open_browser=_get(sv, "open_browser", bool, True, "server"),
        data_dir=data_dir,
        session=session,
        location=_get(st, "location", str, "", "storage"),
        write_csv=_get(st, "write_csv", bool, True, "storage"),
        keep_discarded=_get(st, "keep_discarded", bool, True, "storage"),
        min_rate_hz=_get(q, "min_rate_hz", float, 95.0, "quality"),
        max_air_loss_pct=_get(q, "max_air_loss_pct", float, 5.0, "quality"),
        max_gap_ms=_get(q, "max_gap_ms", float, 200.0, "quality"),
        min_motion_ratio=_get(q, "min_motion_ratio", float, 2.0, "quality"),
        auto_keep_pass=_get(q, "auto_keep_pass", bool, True, "quality"),
        cue_sound=_get(cu, "sound", bool, True, "cue"),
        cue_speech=_get(cu, "speech", bool, True, "cue"),
        cue_vibrate=_get(cu, "vibrate", bool, True, "cue"),
        cue_flash_s=_get(cu, "flash_s", float, 1.5, "cue"),
        window_s=_get(dp, "window_s", float, 10.0, "display"),
        volunteers=volunteers,
        positions=positions,
        activities=acts,
    )
