"""Loads and checks config.toml."""

from __future__ import annotations

import copy
import datetime as dt
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ID_RE = re.compile(r"^[a-z0-9_]+$")  # activity and room ids (part of file names)
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")  # volunteer / session names in paths
LABELS = ("fall", "normal", "none")  # "none" = not a class, e.g. the empty-room baseline


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Activity:
    id: str
    name: str
    desc: str  # how to do it, shown on both screens
    label: str
    target: int  # takes per round (room + volunteer)
    countdown_s: float
    duration_s: float
    cue_s: float  # 0 = no cue
    rest_s: float
    cue_text: str
    expect_motion: bool
    ends_still: bool  # the person holds the last pose, so the movement must end before the take
    no_volunteer: bool  # nobody in the room (baseline)
    train: bool  # false = recorded for evaluation only (free movement)
    extra: bool = False  # added from the web page, outside the PDF plan
    hidden: bool = False  # extra item kept for its data but not offered any more

    def public(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    volunteers: list[str]
    use: str
    setup: str  # measured room size and board positions, saved into every take
    test_only: bool
    extra: bool = False
    hidden: bool = False

    def public(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Step:
    """One line of the round plan. Several activities in one step take turns."""

    name: str
    acts: tuple[str, ...]
    reps: int
    break_min: float  # planned break after this step (shown only)
    per: dict  # activity -> takes in this step
    cum: dict  # activity -> takes needed up to and including this step
    extra: bool = False  # after the PDF plan; its takes are counted outside the plan

    def public(self) -> dict:
        return {**self.__dict__, "acts": list(self.acts)}


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
    review_warnings: bool
    cue_sound: bool
    cue_speech: bool
    cue_vibrate: bool
    cue_flash_s: float
    window_s: float
    rooms: list[Room] = field(default_factory=list)
    activities: list[Activity] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    base_raw: dict = field(default_factory=dict)  # config.toml as read: the PDF plan
    plan_extra: dict = field(default_factory=dict)  # applied plan-extra.json, {} = none

    def activity(self, activity_id: str) -> Activity:
        for a in self.activities:
            if a.id == activity_id:
                return a
        raise KeyError(activity_id)

    def room(self, room_id: str) -> Room:
        for r in self.rooms:
            if r.id == room_id:
                return r
        raise KeyError(room_id)

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
            "rooms": [r.public() for r in self.rooms],
            "activities": [a.public() for a in self.activities],
            "steps": [s.public() for s in self.steps],
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
        "review_warnings",
    },
    "cue": {"sound", "speech", "vibrate", "flash_s"},
    "display": {"window_s"},
}
LIST_KEYS = {
    "activities": {
        "id",
        "name",
        "desc",
        "label",
        "target",
        "countdown_s",
        "duration_s",
        "cue_s",
        "rest_s",
        "cue_text",
        "expect_motion",
        "ends_still",
        "no_volunteer",
        "train",
    },
    "rooms": {"id", "name", "volunteers", "use", "setup", "test_only"},
    "steps": {"activity", "reps", "name", "break_min"},
}


def _check_keys(raw: dict) -> None:
    """A typo, or a key written under the wrong [table], is an error instead of being ignored."""
    for table, value in raw.items():
        if table in LIST_KEYS:
            if not isinstance(value, list):
                raise ConfigError(f"'{table}' must be written as [[{table}]]")
            for i, a in enumerate(value):
                bad = set(a) - LIST_KEYS[table]
                if bad:
                    raise ConfigError(f"{table}[{i}]: unknown key(s) {', '.join(sorted(bad))}")
            continue
        if table not in KNOWN_KEYS:
            raise ConfigError(f"unknown section or key '{table}'")
        bad = set(value) - KNOWN_KEYS[table]
        if bad:
            raise ConfigError(
                f"[{table}]: unknown key(s) {', '.join(sorted(bad))} — "
                "check the spelling, or whether it sits under the right [section]"
            )


def _load_rooms(
    raw: list, prefix: str = "rooms", extra: bool = False, taken: tuple[str, ...] = ()
) -> list[Room]:
    rooms: list[Room] = []
    for i, r in enumerate(raw):
        where = f"{prefix}[{i}]"
        rid = _get(r, "id", str, where=where)
        if not ID_RE.match(rid):
            raise ConfigError(f"{where}.id '{rid}': use only a-z, 0-9 and _")
        if rid in taken:
            raise ConfigError(f"{where}: ห้อง '{rid}' มีในแผน PDF แล้ว ใช้รหัสอื่น")
        if any(x.id == rid for x in rooms):
            raise ConfigError(f"{where}.id '{rid}' is used twice")
        vols = _get(r, "volunteers", list, [], where)
        for v in vols:
            if not isinstance(v, str) or not NAME_RE.match(v):
                raise ConfigError(f"{where}.volunteers {v!r}: use only A-Z a-z 0-9 _ -")
        rooms.append(
            Room(
                id=rid,
                name=_get(r, "name", str, rid.upper(), where),
                volunteers=vols,
                use=_get(r, "use", str, "", where),
                setup=_get(r, "setup", str, "", where),
                test_only=_get(r, "test_only", bool, False, where),
                extra=extra,
                hidden=_get(r, "hidden", bool, False, where),
            )
        )
    if not rooms and not extra:  # an old config without [[rooms]]: one room
        rooms.append(Room("a", "A", [], "", "", False))
    return rooms


def _load_steps(
    raw: list,
    acts: dict[str, dict],
    cum: dict[str, int],
    prefix: str = "steps",
    extra: bool = False,
) -> list[Step]:
    """cum: takes needed per activity so far; updated in place, so extra steps continue it."""
    steps: list[Step] = []
    for i, s in enumerate(raw):
        where = f"{prefix}[{i}]"
        a = s.get("activity")
        ids = [a] if isinstance(a, str) else a
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            raise ConfigError(f'{where}.activity: an activity id, or a list like ["a", "b"]')
        for x in ids:
            if x not in acts:
                raise ConfigError(f"{where}.activity '{x}': no [[activities]] with this id")
        if len(set(ids)) != len(ids):
            raise ConfigError(f"{where}.activity lists the same activity twice")
        reps = _get(s, "reps", int, where=where)
        if reps < 1:
            raise ConfigError(f"{where}.reps must be 1 or more")
        per = {x: reps // len(ids) + (1 if k < reps % len(ids) else 0) for k, x in enumerate(ids)}
        for x, n in per.items():
            cum[x] = cum.get(x, 0) + n
        default_name = " / ".join(str(acts[x].get("name", x)) for x in ids)
        steps.append(
            Step(
                name=_get(s, "name", str, default_name, where),
                acts=tuple(ids),
                reps=reps,
                break_min=_get(s, "break_min", float, 0.0, where),
                per=per,
                cum={x: cum[x] for x in ids},
                extra=extra,
            )
        )
    return steps


def _make_activity(
    a: dict, where: str, target: int, defaults: dict, extra: bool = False
) -> Activity:
    aid = a["id"]
    label = _get(a, "label", str, "normal", where)
    if label not in LABELS:
        raise ConfigError(f'{where}.label must be "fall", "normal" or "none", got {label!r}')
    act = Activity(
        id=aid,
        name=_get(a, "name", str, aid, where),
        desc=_get(a, "desc", str, "", where),
        label=label,
        target=target,
        countdown_s=_get(a, "countdown_s", float, defaults["countdown_s"], where),
        duration_s=_get(a, "duration_s", float, defaults["duration_s"], where),
        cue_s=_get(a, "cue_s", float, defaults["cue_s"], where),
        rest_s=_get(a, "rest_s", float, defaults["rest_s"], where),
        cue_text=_get(a, "cue_text", str, "ทำท่า!", where),
        expect_motion=_get(a, "expect_motion", bool, True, where),
        ends_still=_get(a, "ends_still", bool, True, where),
        no_volunteer=_get(a, "no_volunteer", bool, False, where),
        train=_get(a, "train", bool, True, where),
        extra=extra,
        hidden=_get(a, "hidden", bool, False, where),
    )
    if act.duration_s <= 0 or act.countdown_s < 0 or act.rest_s < 0:
        raise ConfigError(f"{where}: times must be positive")
    if act.cue_s < 0 or (act.cue_s and act.cue_s >= act.duration_s):
        raise ConfigError(
            f"{where} ({aid}): cue_s {act.cue_s:g} must be 0 or less than "
            f"duration_s {act.duration_s:g}"
        )
    return act


# plan-extra.json: what may be added to the PDF plan from the web page
EXTRA_TOP_KEYS = {"version", "updated", "rooms", "activities", "steps", "timing"}
EXTRA_LIST_KEYS = {
    "rooms": LIST_KEYS["rooms"] | {"hidden"},
    "activities": (LIST_KEYS["activities"] - {"target"}) | {"hidden"},
    "steps": LIST_KEYS["steps"],
}
EXTRA_TIMING_KEYS = {"countdown_s", "cue_s", "rest_s"}  # duration_s of PDF activities is locked


def _check_extra(extra: dict) -> None:
    if not isinstance(extra, dict):
        raise ConfigError("extra: must be an object")
    bad = set(extra) - EXTRA_TOP_KEYS
    if bad:
        raise ConfigError(f"extra: unknown key(s) {', '.join(sorted(bad))}")
    for table, keys in EXTRA_LIST_KEYS.items():
        items = extra.get(table, [])
        if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
            raise ConfigError(f"extra.{table}: must be a list of objects")
        for i, item in enumerate(items):
            bad = set(item) - keys
            if bad:
                raise ConfigError(f"extra.{table}[{i}]: unknown key(s) {', '.join(sorted(bad))}")
    if not isinstance(extra.get("timing", {}), dict):
        raise ConfigError("extra.timing: must be an object")


def _apply_timing(raw_acts: list[dict], timing: dict) -> None:
    """Timing changes of PDF activities, written into their raw dicts."""
    for aid, t in timing.items():
        where = f"extra.timing.{aid}"
        act = next((a for a in raw_acts if a.get("id") == aid), None)
        if act is None:
            raise ConfigError(f"extra.timing: ไม่มีท่า '{aid}' ในแผน PDF")
        if not isinstance(t, dict):
            raise ConfigError(f"{where}: must be an object")
        if "duration_s" in t:
            raise ConfigError(f"{where}: ท่าในแผน PDF แก้ duration_s ไม่ได้ (ล็อกตามแผน)")
        bad = set(t) - EXTRA_TIMING_KEYS
        if bad:
            raise ConfigError(f"{where}: unknown key(s) {', '.join(sorted(bad))}")
        act.update(t)


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
    return from_raw(raw, path)


def from_raw(raw: dict, path: Path, extra: dict | None = None) -> Config:
    """Config from the parsed config.toml (the PDF plan) plus an optional plan-extra dict.
    Extras never change the PDF targets: Activity.target comes from the PDF steps only."""
    base_raw = raw
    raw = copy.deepcopy(raw)
    extra = copy.deepcopy(extra or {})
    _check_keys(raw)
    _check_extra(extra)
    s = raw.get("serial", {})
    sv = raw.get("server", {})
    st = raw.get("storage", {})
    tm = raw.get("timing", {})
    q = raw.get("quality", {})
    cu = raw.get("cue", {})
    dp = raw.get("display", {})

    defaults = {
        "countdown_s": _get(tm, "countdown_s", float, 5.0, "timing"),
        "duration_s": _get(tm, "duration_s", float, 10.0, "timing"),
        "cue_s": _get(tm, "cue_s", float, 3.0, "timing"),
        "rest_s": _get(tm, "rest_s", float, 15.0, "timing"),
    }

    raw_acts = raw.get("activities", [])
    by_id: dict[str, dict] = {}
    for i, a in enumerate(raw_acts):
        where = f"activities[{i}]"
        aid = _get(a, "id", str, where=where)
        if not ID_RE.match(aid):
            raise ConfigError(f"{where}.id '{aid}': use only a-z, 0-9 and _")
        if aid in by_id:
            raise ConfigError(f"{where}.id '{aid}' is used twice")
        by_id[aid] = a
    if not by_id:
        raise ConfigError("no [[activities]] defined")
    _apply_timing(raw_acts, extra.get("timing", {}))

    cum: dict[str, int] = {}
    steps = _load_steps(raw.get("steps", []), by_id, cum)
    in_steps = dict(cum)

    acts: list[Activity] = []
    for i, a in enumerate(raw_acts):
        where = f"activities[{i}]"
        aid = a["id"]
        if "target" in a and aid in in_steps:
            raise ConfigError(
                f"{where} ({aid}): remove 'target', the takes come from [[steps]] ({in_steps[aid]})"
            )
        target = in_steps.get(aid) or _get(a, "target", int, 0 if steps else 10, where)
        acts.append(_make_activity(a, where, target, defaults))
    if not steps:  # no plan written: one step per activity, with its target
        steps = _load_steps(
            [{"activity": a.id, "reps": a.target} for a in acts if a.target > 0], by_id, cum
        )

    # ---- extras: after the PDF plan, counted outside its targets
    all_acts = dict(by_id)
    for i, a in enumerate(extra.get("activities", [])):
        where = f"extra.activities[{i}]"
        aid = _get(a, "id", str, where=where)
        if not ID_RE.match(aid):
            raise ConfigError(f"{where}.id '{aid}': use only a-z, 0-9 and _")
        if aid in by_id:
            raise ConfigError(f"{where}: ท่า '{aid}' มีในแผน PDF แล้ว ใช้รหัสอื่น")
        if aid in all_acts:
            raise ConfigError(f"{where}.id '{aid}' is used twice")
        all_acts[aid] = a
        acts.append(_make_activity(a, where, 0, defaults, extra=True))
    steps += _load_steps(extra.get("steps", []), all_acts, cum, "extra.steps", extra=True)
    rooms = _load_rooms(raw.get("rooms", []))
    rooms += _load_rooms(
        extra.get("rooms", []), "extra.rooms", extra=True, taken=tuple(r.id for r in rooms)
    )

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
        min_rate_hz=_get(q, "min_rate_hz", float, 80.0, "quality"),
        max_air_loss_pct=_get(q, "max_air_loss_pct", float, 20.0, "quality"),
        max_gap_ms=_get(q, "max_gap_ms", float, 200.0, "quality"),
        min_motion_ratio=_get(q, "min_motion_ratio", float, 2.0, "quality"),
        auto_keep_pass=_get(q, "auto_keep_pass", bool, True, "quality"),
        review_warnings=_get(q, "review_warnings", bool, True, "quality"),
        cue_sound=_get(cu, "sound", bool, True, "cue"),
        cue_speech=_get(cu, "speech", bool, True, "cue"),
        cue_vibrate=_get(cu, "vibrate", bool, True, "cue"),
        cue_flash_s=_get(cu, "flash_s", float, 1.5, "cue"),
        window_s=_get(dp, "window_s", float, 10.0, "display"),
        rooms=rooms,
        activities=acts,
        steps=steps,
        base_raw=base_raw,
        plan_extra=extra,
    )
