import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from csirec import config as config_mod
from csirec.csi_serial import Frame, FrameReader, encode_frame
from csirec.engine import CommandError, Engine
from csirec.quality import evaluate

ROOT = Path(__file__).resolve().parent.parent

MINI = """
[storage]
data_dir = "{data}"
session = "test"
write_csv = true
[timing]
countdown_s = 1
duration_s = 4
cue_s = 2
rest_s = 1
[quality]
auto_keep_pass = {auto}
[people]
volunteers = ["v01"]
positions = ["p1"]
[[activities]]
id = "fall_forward"
name = "fall"
label = "fall"
target = 2
[[activities]]
id = "empty_room"
name = "empty"
cue_s = 0
expect_motion = false
no_volunteer = true
"""


def make_cfg(tmp_path, auto="true"):
    p = tmp_path / "c.toml"
    p.write_text(MINI.format(data=(tmp_path / "data").as_posix(), auto=auto), encoding="utf-8")
    return config_mod.load(p)


def frame(i, t_us, amp=10, motion=0.0, rng=None):
    vals = np.full(128, amp, dtype=np.int8)
    if motion and rng is not None:
        vals = np.clip(amp + rng.normal(0, motion, 128), -127, 127).astype(np.int8)
    return Frame(0, i % 65536, i % 4096, t_us % 2**32, -40, 6, vals.tobytes())


def test_shipped_config_loads():
    cfg = config_mod.load(ROOT / "config.toml")
    assert cfg.activities and cfg.rx_port
    assert all(a.cue_s == 0 or a.cue_s < a.duration_s for a in cfg.activities)


def test_shipped_config_reads_people_lists():
    cfg = config_mod.load(ROOT / "config.toml")
    assert cfg.positions == ["p1", "p2", "p3"] and len(cfg.volunteers) == 5


def test_config_rejects_key_under_wrong_table(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[display]\nwindow_s = 10\npositions = ["p1"]\n[[activities]]\nid="x"\n', encoding="utf-8"
    )
    with pytest.raises(config_mod.ConfigError, match="positions"):
        config_mod.load(p)


ACT = '[[activities]]\nid = "x"\n'
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="drive letters are Windows-only")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ('data_dir = "data"', None),  # next to the config file
        pytest.param('data_dir = "D:/Dataset/CSI"', "D:/Dataset/CSI", marks=WINDOWS_ONLY),
        pytest.param(r"data_dir = 'D:\Dataset\CSI'", "D:/Dataset/CSI", marks=WINDOWS_ONLY),
        ('data_dir = "~/csi"', (Path.home() / "csi").as_posix()),
    ],
)
def test_config_data_dir_forms(tmp_path, line, expected):
    p = tmp_path / "c.toml"
    p.write_text(f"[storage]\n{line}\n{ACT}", encoding="utf-8")
    got = config_mod.load(p).data_dir.as_posix()
    assert got == (expected or (p.resolve().parent / "data").as_posix())


@WINDOWS_ONLY
def test_config_rejects_drive_relative_path(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[storage]\ndata_dir = "D:Dataset"\n' + ACT, encoding="utf-8")
    with pytest.raises(config_mod.ConfigError, match="D:/Dataset"):
        config_mod.load(p)


def test_second_server_on_same_port_fails(tmp_path):
    from csirec.server import make_server

    cfg = make_cfg(tmp_path)
    object.__setattr__(cfg, "host", "127.0.0.1")
    object.__setattr__(cfg, "port", 18765)
    first = make_server(Engine(cfg, "TEST"))
    try:
        with pytest.raises(OSError):
            make_server(Engine(cfg, "TEST"))
    finally:
        first.server_close()


def test_config_backslash_in_double_quotes_gets_a_hint(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[storage]\ndata_dir = "D:\\Dataset"\n' + ACT, encoding="utf-8")
    with pytest.raises(config_mod.ConfigError, match="D:/Dataset"):
        config_mod.load(p)


def test_config_rejects_cue_after_end(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[activities]]\nid="x"\nduration_s=5\ncue_s=6\n', encoding="utf-8")
    with pytest.raises(config_mod.ConfigError, match="cue_s"):
        config_mod.load(p)


def test_config_rejects_bad_id(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[activities]]\nid="Fall Forward"\n', encoding="utf-8")
    with pytest.raises(config_mod.ConfigError, match="a-z"):
        config_mod.load(p)


def test_encode_frame_round_trip():
    f = Frame(3, 65535, 4095, 2**32 - 1, -77, 11, bytes(range(128)))
    frames, lines = FrameReader().feed(b"#INFO,x=1\n" + encode_frame(f) * 2)
    assert frames == [f, f] and lines == ["#INFO,x=1"]


def test_quality_pass_and_motion(tmp_path):
    cfg = make_cfg(tmp_path)
    rng = np.random.default_rng(0)
    fr = [
        (i / 100, frame(i, i * 10_000, motion=8 if 2.1 < i / 100 < 3 else 0.3, rng=rng))
        for i in range(400)
    ]
    q = evaluate(fr, 4.0, 2.0, True, cfg)
    assert q["verdict"] == "pass", q
    assert q["rate_hz"] == 100.0 and q["air_loss_pct"] == 0
    assert q["motion_ratio"] > 5


def test_quality_fails_on_loss_and_gap(tmp_path):
    cfg = make_cfg(tmp_path)
    fr = [(i / 100, frame(i, i * 10_000)) for i in range(400) if not 100 <= i < 160]
    q = evaluate(fr, 4.0, None, False, cfg)
    assert q["verdict"] == "fail"
    assert q["air_lost"] == 60
    assert any("ขาดช่วง" in r for r in q["reasons"])


def test_quality_no_data(tmp_path):
    q = evaluate([], 4.0, 2.0, True, make_cfg(tmp_path))
    assert q["verdict"] == "fail" and q["frames"] == 0


def feed(engine, t_from, t_to, i0):
    """Feeds frames at 100 Hz between two host times; returns next index."""
    i = i0
    t = t_from
    while t < t_to:
        engine._on_frame(t, frame(i, int(t * 1e6)))
        i += 1
        t += 0.01
    return i


def run_take(engine, i):
    tk = engine.take
    i = feed(engine, tk["t_start"] - 0.5, tk["t_end"] + 0.1, i)
    engine._advance(tk["t_start"] + 0.01)
    assert engine.phase == "recording"
    engine._advance(tk["t_end"] + 1.0)
    return i


def test_engine_full_batch(tmp_path):
    cfg = make_cfg(tmp_path)
    e = Engine(cfg, "TEST")
    now = time.time()
    i = feed(e, now - 1, now, 0)
    with pytest.raises(CommandError):
        e.command("keep", {})
    e.command(
        "start", {"volunteer": "v02", "position": "p1", "activity": "fall_forward", "reps": 2}
    )
    assert e.phase == "countdown"
    i = run_take(e, i)
    assert e.phase == "rest"  # passed, auto kept, one more to go
    e._advance(e.phase_end + 0.01)
    assert e.phase == "countdown"
    run_take(e, i)
    assert e.phase == "idle"
    sess = cfg.data_dir / "test"
    rows = list(csv.DictReader((sess / "manifest.csv").open(encoding="utf-8")))
    assert [r["take_id"] for r in rows] == ["v02/fall_forward_p1_001", "v02/fall_forward_p1_002"]
    meta = json.loads((sess / "v02" / "fall_forward_p1_001.json").read_text(encoding="utf-8"))
    assert meta["label"] == "fall" and meta["cue"]["offset_s"] == 2.0
    assert 195 <= meta["cue"]["csv_row"] <= 205
    data = (sess / "v02" / "fall_forward_p1_001.bin").read_bytes()
    frames, _ = FrameReader().feed(data)
    assert 395 <= len(frames) <= 401
    with (sess / "v02" / "fall_forward_p1_001.csv").open(encoding="utf-8") as fh:
        assert len(fh.readlines()) == len(frames) + 1
    assert "v02" in e.volunteers and e.counts[("v02", "fall_forward")] == 2


def test_engine_review_redo_and_discard(tmp_path):
    cfg = make_cfg(tmp_path, auto="false")
    e = Engine(cfg, "TEST")
    now = time.time()
    i = feed(e, now - 1, now, 0)
    e.command("start", {"volunteer": "v01", "activity": "fall_forward", "reps": 1})
    i = run_take(e, i)
    assert e.phase == "review"
    with pytest.raises(CommandError):
        e.command("stop", {})
    e.command("redo", {})
    assert e.phase == "countdown"
    assert list((cfg.data_dir / "test" / "_discarded").glob("*.bin"))
    i = run_take(e, i)
    e.command("discard", {})
    assert e.phase == "idle" and not (cfg.data_dir / "test" / "manifest.csv").exists()


def test_engine_empty_room_uses_none_volunteer(tmp_path):
    cfg = make_cfg(tmp_path)
    e = Engine(cfg, "TEST")
    now = time.time()
    i = feed(e, now - 1, now, 0)
    e.command("start", {"volunteer": "v01", "activity": "empty_room"})
    assert e.take["volunteer"] == "none" and e.take["cue_at"] is None
    run_take(e, i)
    assert (cfg.data_dir / "test" / "none" / "empty_room_p1_001.json").exists()


def test_engine_start_needs_signal_and_valid_names(tmp_path):
    e = Engine(make_cfg(tmp_path), "TEST")
    with pytest.raises(CommandError, match="สัญญาณ"):
        e.command("start", {"volunteer": "v01", "activity": "fall_forward"})
    feed(e, time.time() - 1, time.time(), 0)
    with pytest.raises(CommandError):
        e.command("start", {"volunteer": "../x", "activity": "fall_forward"})
    with pytest.raises(CommandError):
        e.command("start", {"volunteer": "v01", "activity": "nope"})


def test_stop_during_recording_writes_nothing(tmp_path):
    cfg = make_cfg(tmp_path)
    e = Engine(cfg, "TEST")
    i = feed(e, time.time() - 1, time.time(), 0)
    e.command("start", {"volunteer": "v01", "activity": "fall_forward"})
    feed(e, e.take["t_start"], e.take["t_start"] + 1, i)
    e._advance(e.take["t_start"] + 0.01)
    e.command("stop", {})
    assert e.phase == "idle"
    assert not list((cfg.data_dir / "test").rglob("*.bin"))
