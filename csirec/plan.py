"""Additions to the PDF plan made from the web page, kept in <data_dir>/plan-extra.json.

config.toml stays the PDF plan and is never written. The extra file lives next to the
dataset, so every computer that records into the same (synced) folder uses the same plan.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from .config import Config, ConfigError, from_raw

EXTRA_FILE = "plan-extra.json"


def extra_path(cfg: Config) -> Path:
    return cfg.data_dir / EXTRA_FILE


def build(cfg: Config, extra: dict) -> Config:
    """cfg with its plan replaced by the PDF plan plus extra. Everything that is not the plan
    (port, data folder, ...) stays as it is now. Raises ConfigError; writes nothing."""
    new = from_raw(cfg.base_raw, cfg.path, extra)
    return replace(
        cfg,
        rooms=new.rooms,
        activities=new.activities,
        steps=new.steps,
        plan_extra=new.plan_extra,
    )


def with_saved_extra(cfg: Config) -> tuple[Config, str | None]:
    """cfg with the saved extra applied. A missing file is the PDF plan; a file that can't be
    used is ignored (the recorder still starts) and the message says why."""
    p = extra_path(cfg)
    if not p.exists():
        return cfg, None
    try:
        return build(cfg, json.loads(p.read_text(encoding="utf-8"))), None
    except (OSError, ValueError, ConfigError) as e:
        return build(cfg, {}), f"อ่าน {EXTRA_FILE} ไม่ได้ ใช้แผน PDF อย่างเดียว: {e}"


def save_extra(cfg: Config, extra: dict) -> Config:
    """Checks extra, writes it (temp file + rename, so a crash never leaves half a file) and
    returns the new config."""
    extra = {**extra, "version": 1, "updated": dt.datetime.now().isoformat(timespec="seconds")}
    new = build(cfg, extra)
    p = extra_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return new


def reset_extra(cfg: Config) -> Config:
    """Back to the PDF plan. The file is moved aside, not deleted."""
    p = extra_path(cfg)
    if p.exists():
        p.rename(p.with_name(f"plan-extra_old_{time.strftime('%Y%m%d%H%M%S')}.json"))
    return build(cfg, {})
