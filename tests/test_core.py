import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from csirec import config as config_mod
from csirec import plan as plan_mod
from csirec.csi_serial import Frame, FrameReader, encode_frame
from csirec.engine import CommandError, Engine
from csirec.quality import evaluate
from csirec.storage import LABEL_FIELDS

ROOT = Path(__file__).resolve().parent.parent

MINI = """
[storage]
data_dir = "{data}"
session = "{session}"
write_csv = true
[timing]
countdown_s = 1
duration_s = 4
cue_s = 2
rest_s = 1
[quality]
auto_keep_pass = {auto}
review_warnings = {review}
[[rooms]]
id = "a"
volunteers = ["p01"]
[[rooms]]
id = "c"
volunteers = ["p06"]
test_only = true
[[activities]]
id = "baseline"
label = "none"
cue_s = 0
expect_motion = false
no_volunteer = true
[[activities]]
id = "fall"
name = "fall"
label = "fall"
[[activities]]
id = "slow_sit"
[[activities]]
id = "slow_lie"
[[steps]]
activity = "baseline"
reps = 1
[[steps]]
activity = "fall"
reps = 2
[[steps]]
activity = ["slow_sit", "slow_lie"]
reps = 4
"""
STEP_BASELINE, STEP_FALL, STEP_SLOW = 0, 1, 2


def make_cfg(tmp_path, auto="true", review="false", session="test"):
    p = tmp_path / f"{session}.toml"
    data = (tmp_path / "data").as_posix()
    p.write_text(
        MINI.format(data=data, auto=auto, review=review, session=session), encoding="utf-8"
    )
    return config_mod.load(p)


def frame(i, t_us, amp=10, motion=0.0, rng=None):
    vals = np.full(128, amp, dtype=np.int8)
    if motion and rng is not None:
        vals = np.clip(amp + rng.normal(0, motion, 128), -127, 127).astype(np.int8)
    return Frame(0, i % 65536, i % 4096, t_us % 2**32, -40, 6, vals.tobytes())


def read_labels(cfg):
    with (cfg.data_dir / "labels.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- shipped config = the data plan
def test_shipped_config_follows_the_data_plan():
    cfg = config_mod.load(ROOT / "config.toml")
    assert [r.id for r in cfg.rooms] == ["a", "b", "c"]
    assert cfg.room("a").volunteers == cfg.room("b").volunteers == [f"p0{i}" for i in range(1, 6)]
    assert cfg.room("c").volunteers == ["p06", "p07"] and cfg.room("c").test_only
    used = [a for a in cfg.activities if a.train and a.label in ("fall", "normal")]
    assert sum(a.target for a in used) == 65  # one round
    assert sum(a.target for a in used if a.label == "fall") == 20
    assert sum(a.target for a in used if a.label == "normal") == 45
    assert {a.id: a.target for a in used} == {
        "fall": 20,
        "sit_stand": 10,
        "pick_up": 10,
        "walk_stop": 10,
        "slow_sit": 5,
        "slow_lie": 5,
        "shower": 5,
    }
    for a in used:  # 0-3 s still, cue at 3 s, 10 s per take
        assert a.duration_s == 10 and a.cue_s == 3
    free = cfg.activity("free_move")
    assert not free.train and free.duration_s * free.target == 20 * 60
    assert cfg.activity("baseline").no_volunteer
    assert cfg.min_rate_hz == 80 and cfg.max_air_loss_pct == 20  # ~1,000 samples, <= 20 % lost


def test_shipped_config_round_order():
    cfg = config_mod.load(ROOT / "config.toml")
    assert [s.acts for s in cfg.steps] == [
        ("baseline",),
        ("fall",),
        ("sit_stand",),
        ("pick_up",),
        ("fall",),
        ("walk_stop",),
        ("slow_sit", "slow_lie"),
        ("shower",),
        ("free_move",),
    ]
    assert cfg.steps[4].cum == {"fall": 20}  # the second fall set comes after the first 10
    assert cfg.steps[6].per == {"slow_sit": 5, "slow_lie": 5}


# ---------------------------------------------------------------- config checks
def test_config_rejects_key_under_wrong_table(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[display]\nwindow_s = 10\nvolunteers = ["p1"]\n[[activities]]\nid="x"\n', encoding="utf-8"
    )
    with pytest.raises(config_mod.ConfigError, match="volunteers"):
        config_mod.load(p)


def test_config_rejects_unknown_step_activity(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[activities]]\nid="x"\n[[steps]]\nactivity="y"\nreps=2\n', encoding="utf-8")
    with pytest.raises(config_mod.ConfigError, match="'y'"):
        config_mod.load(p)


def test_config_rejects_target_on_planned_activity(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[[activities]]\nid="x"\ntarget=3\n[[steps]]\nactivity="x"\nreps=2\n', encoding="utf-8"
    )
    with pytest.raises(config_mod.ConfigError, match="target"):
        config_mod.load(p)


def test_config_without_steps_makes_one_per_activity(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[activities]]\nid="x"\ntarget=3\n', encoding="utf-8")
    cfg = config_mod.load(p)
    assert [(s.acts, s.reps) for s in cfg.steps] == [(("x",), 3)]
    assert [r.id for r in cfg.rooms] == ["a"]


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


# ------------------------------------------------------------- plan extras (outside the PDF plan)
EXTRA = {
    "rooms": [{"id": "d", "name": "ห้องน้ำ D", "volunteers": ["p08"]}],
    "activities": [{"id": "fall_back", "name": "ล้มไปข้างหลัง", "label": "fall"}],
    "steps": [{"activity": "fall_back", "reps": 10, "break_min": 3}],
}


def with_extra(tmp_path, extra):
    cfg = make_cfg(tmp_path)
    return config_mod.from_raw(cfg.base_raw, cfg.path, extra)


def test_extra_adds_room_activity_step(tmp_path):
    base = make_cfg(tmp_path)
    cfg = with_extra(tmp_path, EXTRA)
    assert cfg.room("d").extra and not cfg.room("a").extra
    assert cfg.activity("fall_back").extra and not cfg.activity("fall").extra
    assert cfg.steps[-1].extra and cfg.steps[-1].acts == ("fall_back",)
    assert not any(s.extra for s in cfg.steps[:-1])
    assert cfg.activity("fall_back").target == 0
    assert [a.target for a in base.activities] == [a.target for a in cfg.activities[:-1]]
    assert cfg.plan_extra == EXTRA and base.plan_extra == {}


def test_extra_step_on_plan_activity_continues_cum(tmp_path):
    cfg = with_extra(tmp_path, {"steps": [{"activity": "fall", "reps": 5}]})
    assert cfg.activity("fall").target == 2  # PDF target stays
    assert cfg.steps[-1].cum == {"fall": 7}


def test_extra_timing_overrides_plan_activity(tmp_path):
    cfg = with_extra(tmp_path, {"timing": {"fall": {"rest_s": 20}}})
    assert cfg.activity("fall").rest_s == 20
    with pytest.raises(config_mod.ConfigError, match="duration_s"):
        with_extra(tmp_path, {"timing": {"fall": {"duration_s": 5}}})
    with pytest.raises(config_mod.ConfigError, match="nope"):
        with_extra(tmp_path, {"timing": {"nope": {"rest_s": 5}}})


def test_extra_id_clash_and_bad_keys(tmp_path):
    with pytest.raises(config_mod.ConfigError, match="มีในแผน PDF แล้ว"):
        with_extra(tmp_path, {"rooms": [{"id": "a"}]})
    with pytest.raises(config_mod.ConfigError, match="มีในแผน PDF แล้ว"):
        with_extra(tmp_path, {"activities": [{"id": "fall"}]})
    with pytest.raises(config_mod.ConfigError, match="color"):
        with_extra(tmp_path, {"color": "red"})
    with pytest.raises(config_mod.ConfigError, match="target"):
        with_extra(tmp_path, {"activities": [{"id": "x", "target": 3}]})


def test_extra_hidden_flags(tmp_path):
    extra = {
        "rooms": [{"id": "d", "hidden": True}],
        "activities": [{"id": "fall_back", "label": "fall", "hidden": True}],
    }
    cfg = with_extra(tmp_path, extra)
    assert cfg.room("d").hidden and cfg.activity("fall_back").hidden
    assert not cfg.room("a").hidden


def test_plan_file_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path)
    saved = plan_mod.save_extra(cfg, EXTRA)
    assert plan_mod.extra_path(cfg).exists() and saved.room("d").extra
    assert json.loads(plan_mod.extra_path(cfg).read_text(encoding="utf-8"))["version"] == 1
    loaded, msg = plan_mod.with_saved_extra(make_cfg(tmp_path))
    assert msg is None and loaded.room("d").extra and loaded.steps[-1].acts == ("fall_back",)


def test_plan_build_keeps_running_settings(tmp_path):
    cfg = make_cfg(tmp_path)
    object.__setattr__(cfg, "port", 18799)
    object.__setattr__(cfg, "data_dir", tmp_path / "other")
    new = plan_mod.build(cfg, EXTRA)
    assert new.port == 18799 and new.data_dir == tmp_path / "other" and new.room("d").extra


def test_plan_file_missing_is_the_pdf_plan(tmp_path):
    cfg = make_cfg(tmp_path)
    loaded, msg = plan_mod.with_saved_extra(cfg)
    assert msg is None and loaded is cfg


def test_plan_file_broken_falls_back(tmp_path):
    cfg = make_cfg(tmp_path)
    plan_mod.extra_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    plan_mod.extra_path(cfg).write_text("{", encoding="utf-8")
    loaded, msg = plan_mod.with_saved_extra(cfg)
    assert "plan-extra.json" in msg and not any(s.extra for s in loaded.steps)


def test_plan_reset_moves_file(tmp_path):
    cfg = plan_mod.save_extra(make_cfg(tmp_path), EXTRA)
    fresh = plan_mod.reset_extra(cfg)
    assert not plan_mod.extra_path(cfg).exists()
    assert len(list(cfg.data_dir.glob("plan-extra_old_*.json"))) == 1
    assert not any(s.extra for s in fresh.steps) and fresh.plan_extra == {}


def test_encode_frame_round_trip():
    f = Frame(3, 65535, 4095, 2**32 - 1, -77, 11, bytes(range(128)))
    frames, lines = FrameReader().feed(b"#INFO,x=1\n" + encode_frame(f) * 2)
    assert frames == [f, f] and lines == ["#INFO,x=1"]


# ---------------------------------------------------------------- quality
def moving(duration_s, start_s, end_s, seed=0):
    """Frames at 100 Hz, still except for strong motion between start_s and end_s."""
    rng = np.random.default_rng(seed)
    return [
        (i / 100, frame(i, i * 10_000, motion=8 if start_s <= i / 100 < end_s else 0.3, rng=rng))
        for i in range(int(duration_s * 100))
    ]


def test_quality_pass_and_motion(tmp_path):
    cfg = make_cfg(tmp_path)
    q = evaluate(moving(4, 2.1, 3), 4.0, 2.0, True, cfg)
    assert q["verdict"] == "pass", q
    assert q["rate_hz"] == 100.0 and q["air_loss_pct"] == 0
    assert q["frames"] == q["frames_expected"] == 400
    assert q["motion_ratio"] > 5


def test_quality_estimates_when_the_movement_happens(tmp_path):
    cfg = make_cfg(tmp_path)
    q = evaluate(moving(10, 3.3, 5.6), 10.0, 3.0, True, cfg)
    assert abs(q["action_start_s"] - 3.3) <= 0.3 and abs(q["action_end_s"] - 5.6) <= 0.3, q
    assert q["warnings"] == []


def test_quality_warns_when_the_movement_is_not_finished(tmp_path):
    cfg = make_cfg(tmp_path)
    q = evaluate(moving(10, 3.3, 10), 10.0, 3.0, True, cfg)
    assert any("ยังไม่จบ" in w for w in q["warnings"])
    q = evaluate(moving(10, 3.3, 10), 10.0, 3.0, True, cfg, ends_still=False)
    assert not any("ยังไม่จบ" in w for w in q["warnings"])


def test_quality_warns_when_moving_before_the_cue(tmp_path):
    cfg = make_cfg(tmp_path)
    q = evaluate(moving(10, 2.0, 4.5), 10.0, 3.0, True, cfg)
    assert any("ก่อนสัญญาณ" in w for w in q["warnings"])


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


# ---------------------------------------------------------------- engine
def feed(engine, t_from, t_to, i0):
    """Feeds frames at 100 Hz between two host times; returns next index."""
    i = i0
    t = t_from
    while t < t_to:
        engine._on_frame(t, frame(i, int(t * 1e6)))
        i += 1
        t += 0.01
    return i


def test_live_reports_rx_channel(tmp_path):
    e = Engine(make_cfg(tmp_path), "TEST")
    assert e._live(100.0)["rx_ch"] is None
    feed(e, 99.0, 100.0, 0)
    assert e._live(100.0)["rx_ch"] == 6  # frame() is captured on channel 6


def run_take(engine, i):
    tk = engine.take
    i = feed(engine, tk["t_start"] - 0.5, tk["t_end"] + 0.1, i)
    engine._advance(tk["t_start"] + 0.01)
    assert engine.phase == "recording"
    engine._advance(tk["t_end"] + 1.0)
    return i


def run_batch(engine, i):
    """Runs takes until the batch ends (auto keep); returns (next index, activities taken)."""
    acts = []
    while True:
        acts.append(engine.take["activity"])
        i = run_take(engine, i)
        if engine.phase == "idle":
            return i, acts
        assert engine.phase == "rest", engine.phase
        engine._advance(engine.phase_end + 0.01)


def started(cfg):
    e = Engine(cfg, "TEST")
    now = time.time()
    return e, feed(e, now - 1, now, 0)


def test_engine_full_batch(tmp_path):
    cfg = make_cfg(tmp_path)
    e, i = started(cfg)
    with pytest.raises(CommandError):
        e.command("keep", {})
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_FALL})
    assert e.phase == "countdown" and e.batch["reps"] == 2  # what the step still needs
    i, acts = run_batch(e, i)
    assert acts == ["fall", "fall"]
    sess = cfg.data_dir / "test"
    rows = list(csv.DictReader((sess / "manifest.csv").open(encoding="utf-8")))
    assert [r["take_id"] for r in rows] == ["p01/a_p01_fall_01", "p01/a_p01_fall_02"]
    meta = json.loads((sess / "p01" / "a_p01_fall_01.json").read_text(encoding="utf-8"))
    assert meta["label"] == "fall" and meta["cue"]["offset_s"] == 2.0
    assert meta["room"] == "a" and meta["trial"] == 1 and meta["people_in_room"] == 1
    assert 195 <= meta["cue"]["csv_row"] <= 205
    data = (sess / "p01" / "a_p01_fall_01.bin").read_bytes()
    frames, _ = FrameReader().feed(data)
    assert 395 <= len(frames) <= 401
    with (sess / "p01" / "a_p01_fall_01.csv").open(encoding="utf-8") as fh:
        assert len(fh.readlines()) == len(frames) + 1
    assert e.counts[("a", "p01", "fall")] == 2

    labels = read_labels(cfg)
    assert list(labels[0]) == LABEL_FIELDS
    assert [(r["file"], r["room"], r["action"], r["label"], r["trial"]) for r in labels] == [
        ("test/p01/a_p01_fall_01.bin", "a", "fall", "fall", "1"),
        ("test/p01/a_p01_fall_02.bin", "a", "fall", "fall", "2"),
    ]
    assert labels[0]["baseline_file"] == ""  # no baseline recorded for this round yet


def test_set_plan_applies_live(tmp_path):
    e, _ = started(make_cfg(tmp_path))
    rev = e.plan_rev
    e.set_plan(EXTRA)
    assert e.plan_rev == rev + 1 and plan_mod.extra_path(e.cfg).exists()
    assert e.store.cfg is e.cfg and "p08" in e.volunteers
    step = len(e.cfg.steps) - 1
    e.command("start", {"room": "d", "volunteer": "p08", "step": step, "reps": 1})
    assert e.take["activity"] == "fall_back" and e.batch["step"] == step


def test_set_plan_refused_while_recording(tmp_path):
    e, _ = started(make_cfg(tmp_path))
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_FALL})
    with pytest.raises(CommandError, match="หยุดบันทึกก่อนแก้แผน"):
        e.set_plan(EXTRA)
    with pytest.raises(CommandError, match="หยุดบันทึกก่อนแก้แผน"):
        e.reset_plan()
    assert not plan_mod.extra_path(e.cfg).exists()


def test_set_plan_reports_bad_plan(tmp_path):
    e, _ = started(make_cfg(tmp_path))
    with pytest.raises(CommandError, match="มีในแผน PDF แล้ว"):
        e.set_plan({"rooms": [{"id": "a"}]})


def test_set_plan_refuses_removing_used_activity(tmp_path):
    e, i = started(make_cfg(tmp_path))
    e.set_plan(EXTRA)
    e.command("start", {"room": "a", "volunteer": "p01", "step": len(e.cfg.steps) - 1, "reps": 1})
    i, _ = run_batch(e, i)
    assert e.plan_info()["used"]["activities"]["fall_back"] == 1
    with pytest.raises(CommandError, match="ซ่อน"):
        e.set_plan({**EXTRA, "activities": [], "steps": []})
    hidden = {**EXTRA, "activities": [{**EXTRA["activities"][0], "hidden": True}]}
    e.set_plan(hidden)  # hiding keeps the data and is allowed
    e.reset_plan()  # going back to the PDF plan is allowed too; the data stays
    assert e.counts[("a", "p01", "fall_back")] == 1


def test_take_outside_plan_marked(tmp_path):
    e, i = started(make_cfg(tmp_path))
    e.set_plan(EXTRA)
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_FALL, "reps": 3})
    i, _ = run_batch(e, i)  # PDF target of fall is 2: the 3rd take is outside the plan
    e.command("start", {"room": "d", "volunteer": "p08", "step": STEP_FALL, "reps": 1})
    i, _ = run_batch(e, i)  # every take in an extra room is outside the plan
    sess = e.cfg.data_dir / "test"
    metas = [
        json.loads((sess / v / f"{n}.json").read_text(encoding="utf-8"))
        for v, n in [("p01", "a_p01_fall_01"), ("p01", "a_p01_fall_03"), ("p08", "d_p08_fall_01")]
    ]
    assert [m["in_plan"] for m in metas] == [True, False, False]
    assert metas[0]["plan_extra_updated"] == e.cfg.plan_extra["updated"]
    notes = [r["note"] for r in read_labels(e.cfg)]
    assert ["นอกแผน" in n for n in notes] == [False, False, True, True]


def test_start_refuses_hidden_room_and_step(tmp_path):
    e, _ = started(make_cfg(tmp_path))
    e.set_plan(
        {
            "rooms": [{"id": "d", "volunteers": ["p08"], "hidden": True}],
            "activities": [{"id": "fall_back", "label": "fall", "hidden": True}],
            "steps": [{"activity": "fall_back", "reps": 2}],
        }
    )
    with pytest.raises(CommandError, match="ห้องนี้ถูกซ่อน"):
        e.command("start", {"room": "d", "volunteer": "p08", "step": STEP_FALL})
    with pytest.raises(CommandError, match="ขั้นนี้ถูกซ่อน"):
        e.command("start", {"room": "a", "volunteer": "p01", "step": len(e.cfg.steps) - 1})


def test_engine_starts_with_saved_plan(tmp_path):
    plan_mod.save_extra(make_cfg(tmp_path), EXTRA)
    e = Engine(make_cfg(tmp_path), "TEST")
    assert e.cfg.room("d").extra and e.store.cfg is e.cfg


def test_engine_baseline_is_linked_to_the_round(tmp_path):
    cfg = make_cfg(tmp_path)
    e, i = started(cfg)
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_BASELINE})
    assert e.take["cue_at"] is None
    i, _ = run_batch(e, i)
    meta = json.loads(
        (cfg.data_dir / "test" / "p01" / "a_p01_baseline_01.json").read_text(encoding="utf-8")
    )
    assert meta["people_in_room"] == 0 and meta["label"] == "none"
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_FALL, "reps": 1})
    run_batch(e, i)
    labels = read_labels(cfg)
    assert labels[0]["note"] == "ห้องว่าง"
    assert labels[1]["baseline_file"] == "test/p01/a_p01_baseline_01.bin"
    assert e._build_snapshot(time.time())["baselines"] == ["a/p01"]


def test_engine_step_with_two_activities_takes_turns(tmp_path):
    cfg = make_cfg(tmp_path)
    e, i = started(cfg)
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_SLOW, "reps": 3})
    i, acts = run_batch(e, i)
    assert acts == ["slow_sit", "slow_lie", "slow_sit"]
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_SLOW})
    assert e.batch["reps"] == 1  # 4 in the step, 3 done
    _, acts = run_batch(e, i)
    assert acts == ["slow_lie"]


def test_engine_trial_numbers_continue_in_a_new_session(tmp_path):
    cfg = make_cfg(tmp_path)
    e, i = started(cfg)
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_FALL, "reps": 1})
    run_batch(e, i)
    cfg2 = make_cfg(tmp_path, session="test2")  # e.g. a backup day, same dataset folder
    e2, i = started(cfg2)
    assert e2.counts[("a", "p01", "fall")] == 1
    e2.command("start", {"room": "a", "volunteer": "p01", "step": STEP_FALL})
    assert e2.batch["reps"] == 1
    run_batch(e2, i)
    assert (cfg.data_dir / "test2" / "p01" / "a_p01_fall_02.bin").exists()
    assert [r["file"] for r in read_labels(cfg)] == [
        "test/p01/a_p01_fall_01.bin",
        "test2/p01/a_p01_fall_02.bin",
    ]


def test_engine_review_redo_and_discard(tmp_path):
    cfg = make_cfg(tmp_path, auto="false")
    e, i = started(cfg)
    e.command("start", {"room": "a", "volunteer": "p01", "activity": "fall", "reps": 1})
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
    assert not (cfg.data_dir / "labels.csv").exists()


def test_engine_holds_a_take_with_warnings_for_review(tmp_path):
    cfg = make_cfg(tmp_path, review="true")
    e, i = started(cfg)
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_FALL, "reps": 1})
    run_take(e, i)  # no movement after the cue -> warning
    assert e.last_result["verdict"] == "pass" and e.last_result["warnings"]
    assert e.phase == "review"
    e.command("keep", {})
    assert e.counts[("a", "p01", "fall")] == 1


def test_engine_moves_an_old_manifest_aside(tmp_path):
    cfg = make_cfg(tmp_path)
    sess = cfg.data_dir / "test"
    sess.mkdir(parents=True)
    (sess / "manifest.csv").write_text("take_id,position\nv01/x,p1\n", encoding="utf-8")
    e, i = started(cfg)
    e.command("start", {"room": "a", "volunteer": "p01", "step": STEP_FALL, "reps": 1})
    run_batch(e, i)
    old = list(sess.glob("manifest_old_*.csv"))
    assert len(old) == 1 and "position" in old[0].read_text(encoding="utf-8")
    rows = list(csv.DictReader((sess / "manifest.csv").open(encoding="utf-8")))
    assert [r["take_id"] for r in rows] == ["p01/a_p01_fall_01"]


def test_engine_start_needs_signal_and_valid_names(tmp_path):
    e = Engine(make_cfg(tmp_path), "TEST")
    with pytest.raises(CommandError, match="สัญญาณ"):
        e.command("start", {"room": "a", "volunteer": "p01", "activity": "fall"})
    feed(e, time.time() - 1, time.time(), 0)
    with pytest.raises(CommandError):
        e.command("start", {"room": "a", "volunteer": "../x", "activity": "fall"})
    with pytest.raises(CommandError):
        e.command("start", {"room": "a", "volunteer": "p01", "activity": "nope"})
    with pytest.raises(CommandError, match="ห้อง"):
        e.command("start", {"room": "z", "volunteer": "p01", "activity": "fall"})
    with pytest.raises(CommandError, match="ขั้น"):
        e.command("start", {"room": "a", "volunteer": "p01", "step": 9})


def test_stop_during_recording_writes_nothing(tmp_path):
    cfg = make_cfg(tmp_path)
    e, i = started(cfg)
    e.command("start", {"room": "a", "volunteer": "p01", "activity": "fall"})
    feed(e, e.take["t_start"], e.take["t_start"] + 1, i)
    e._advance(e.take["t_start"] + 0.01)
    e.command("stop", {})
    assert e.phase == "idle"
    assert not list((cfg.data_dir / "test").rglob("*.bin"))


def test_remote_target_with_and_without_pin():
    from csirec.server import remote_target

    ips = ["192.168.1.45", "172.18.96.1"]
    assert remote_target("1234", 8765, ips) == "http://192.168.1.45:8765/m#pin=1234"
    assert remote_target("", 8765, ips) == "http://192.168.1.45:8765/m"


def test_remote_target_quotes_pin():
    from csirec.server import remote_target

    assert (
        remote_target("a#b&c d", 8765, ["10.0.0.2"]) == "http://10.0.0.2:8765/m#pin=a%23b%26c%20d"
    )


def test_remote_target_no_network():
    from csirec.server import remote_target

    assert remote_target("1234", 8765, []) is None


def test_is_local():
    from csirec.server import _is_local

    assert _is_local("127.0.0.1")
    assert _is_local("::1")
    assert _is_local("::ffff:127.0.0.1")
    assert not _is_local("192.168.1.20")


def test_qr_route_serves_svg_only_to_this_computer(tmp_path, monkeypatch):
    import threading
    import urllib.request

    from csirec import server as server_mod

    monkeypatch.setattr(server_mod, "lan_addresses", lambda: ["192.168.1.45"])
    cfg = make_cfg(tmp_path)
    object.__setattr__(cfg, "host", "127.0.0.1")
    object.__setattr__(cfg, "port", 18766)
    srv = server_mod.make_server(Engine(cfg, "TEST"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen("http://127.0.0.1:18766/api/qr.svg", timeout=5) as r:
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("image/svg+xml")
            assert r.read().startswith(b"<svg")
    finally:
        srv.shutdown()
        srv.server_close()


def test_api_plan_roundtrip(tmp_path):
    import threading
    import urllib.error
    import urllib.request

    from csirec import server as server_mod

    cfg = make_cfg(tmp_path)
    object.__setattr__(cfg, "host", "127.0.0.1")
    object.__setattr__(cfg, "port", 18768)
    srv = server_mod.make_server(Engine(cfg, "TEST"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:18768"

    def call(path, body=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body))
        req = urllib.request.Request(
            base + path, data=data.encode() if isinstance(data, str) else data
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        assert call("/api/plan", {"extra": EXTRA})[0] == 200
        rooms = json.loads(call("/api/config")[1])["rooms"]
        assert [r["id"] for r in rooms] == ["a", "c", "d"] and rooms[-1]["extra"]
        info = json.loads(call("/api/plan")[1])
        assert info["extra"]["rooms"][0]["id"] == "d" and info["used"]["rooms"] == {}
        code, body = call("/api/plan", {"extra": {"rooms": [{"id": "a"}]}})
        assert code == 409 and "มีในแผน PDF แล้ว" in json.loads(body)["error"]
        assert call("/api/plan", b"{")[0] == 400
        assert call("/api/plan", {"reset": True})[0] == 200
        assert [r["id"] for r in json.loads(call("/api/config")[1])["rooms"]] == ["a", "c"]
        code, body = call("/plan")
        assert code == 200 and b"<html" in body
    finally:
        srv.shutdown()
        srv.server_close()


def _get_qr(tmp_path, monkeypatch, ips, local):
    import threading
    import urllib.error
    import urllib.request

    from csirec import server as server_mod

    monkeypatch.setattr(server_mod, "lan_addresses", lambda: ips)
    monkeypatch.setattr(server_mod, "_is_local", lambda addr: local)
    cfg = make_cfg(tmp_path)
    object.__setattr__(cfg, "host", "127.0.0.1")
    object.__setattr__(cfg, "port", 18767)
    srv = server_mod.make_server(Engine(cfg, "TEST"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen("http://127.0.0.1:18767/api/qr.svg", timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    finally:
        srv.shutdown()
        srv.server_close()


def test_qr_route_refuses_other_devices(tmp_path, monkeypatch):
    status, body = _get_qr(tmp_path, monkeypatch, ["192.168.1.45"], local=False)
    assert status == 403
    assert b"<svg" not in body


def test_qr_route_without_network_is_404(tmp_path, monkeypatch):
    status, body = _get_qr(tmp_path, monkeypatch, [], local=True)
    assert status == 404
    assert b"<svg" not in body


def test_server_serves_waterfall_script(tmp_path):
    import threading
    import urllib.request

    from csirec import server as server_mod

    cfg = make_cfg(tmp_path)
    object.__setattr__(cfg, "host", "127.0.0.1")
    object.__setattr__(cfg, "port", 18768)
    srv = server_mod.make_server(Engine(cfg, "TEST"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen("http://127.0.0.1:18768/waterfall.js", timeout=5) as r:
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("text/javascript")
            assert b"function ampRange" in r.read()
    finally:
        srv.shutdown()
        srv.server_close()
