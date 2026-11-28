"""Writes takes to disk and keeps the session manifest and the dataset labels.

Layout (the data plan §9, recordings/<session>/<volunteer>/):
    <data_dir>/labels.csv                       one row per kept take, the whole dataset
    <data_dir>/<session>/manifest.csv           one row per kept take of this session + quality
    <data_dir>/<session>/<volunteer>/<room>_<volunteer>_<action>_<NN>.bin   raw frames (csi_rx v2)
                                                                 ....json  metadata + quality
                                                                 ....csv   amplitude (exp7/exp8)
    <data_dir>/<session>/_discarded/            takes thrown away (if keep_discarded)
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from .csi_serial import Frame, encode_frame
from .quality import N_SC, amplitudes

LABEL_FIELDS = [  # the columns the data plan asks for (§9)
    "file",
    "room",
    "volunteer",
    "action",
    "label",
    "trial",
    "action_start_s",
    "action_end_s",
    "baseline_file",
    "note",
]
MANIFEST_FIELDS = [
    "take_id",
    "recorded_at",
    "room",
    "volunteer",
    "activity",
    "label",
    "trial",
    "duration_s",
    "cue_offset_s",
    "cue_row",
    "action_start_s",
    "action_end_s",
    "frames",
    "rate_hz",
    "air_loss_pct",
    "rssi_median",
    "motion_ratio",
    "verdict",
    "kept_by",
    "note",
    "bin",
    "csv",
    "json",
]


def _blank(v) -> object:
    return "" if v is None else v


class Storage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session_dir: Path = cfg.data_dir / cfg.session
        self.manifest: Path = self.session_dir / "manifest.csv"
        self.labels: Path = cfg.data_dir / "labels.csv"

    # ------------------------------------------------------------------ counts
    def load_counts(self) -> tuple[Counter, dict]:
        """Kept takes per (room, volunteer, action) over the whole dataset, and the latest
        baseline file of each round (room, volunteer)."""
        counts: Counter = Counter()
        baselines: dict[tuple[str, str], str] = {}
        if self.labels.exists():
            with self.labels.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    key = (row["room"], row["volunteer"])
                    counts[(*key, row["action"])] += 1
                    if self._is_baseline(row["action"]):
                        baselines[key] = row["file"]
        return counts, baselines

    def _is_baseline(self, action: str) -> bool:
        try:
            return self.cfg.activity(action).no_volunteer
        except KeyError:
            return False

    def _next_index(self, folder: Path, stem_prefix: str) -> int:
        pat = re.compile(re.escape(stem_prefix) + r"_(\d{2,})\.json$")
        used = [
            int(m.group(1))
            for p in folder.glob(stem_prefix + "_*.json")
            if (m := pat.match(p.name))
        ]
        return max(used, default=0) + 1

    # ------------------------------------------------------------------ write
    def write_take(
        self,
        take: dict,
        frames: list[tuple[float, Frame]],
        kept: list[tuple[float, Frame]],
        min_trial: int = 1,
    ) -> dict:
        """Writes .bin (all frames), .csv (retries removed) and .json. Returns file paths.
        min_trial: trial numbers continue from takes kept in earlier sessions."""
        folder = self.session_dir / take["volunteer"]
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{take['room']}_{take['volunteer']}_{take['activity']}"
        idx = max(self._next_index(folder, prefix), min_trial)
        stem = f"{prefix}_{idx:02d}"
        paths = {k: folder / f"{stem}.{k}" for k in ("bin", "csv", "json")}

        with paths["bin"].open("wb") as fh:
            for _, f in frames:
                fh.write(encode_frame(f))

        t0 = take["t_start"]
        if self.cfg.write_csv:
            with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["Timestamp", "RSSI"] + [f"Sub_{i}" for i in range(N_SC)])
                for t, f in kept:
                    amp = amplitudes(f)
                    w.writerow([f"{t0 + t:.6f}", f.rssi] + [f"{a:.4f}" for a in amp])
        else:
            paths.pop("csv")

        take["trial"] = idx
        take["take_id"] = f"{take['volunteer']}/{stem}"
        take["file"] = f"{self.cfg.session}/{take['volunteer']}/{stem}.bin"  # from data_dir
        take["files"] = {k: p.name for k, p in paths.items()}
        paths["json"].write_text(json.dumps(take, ensure_ascii=False, indent=2), encoding="utf-8")
        return {k: str(p) for k, p in paths.items()}

    # ------------------------------------------------------------------ keep / discard
    def _append(self, path: Path, fields: list[str], row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with path.open(encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh), None)
            if header and header != fields:  # written by an older version: keep it aside
                old = path.with_name(f"{path.stem}_old_{time.strftime('%Y%m%d%H%M%S')}.csv")
                path.rename(old)
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)

    def keep(self, take: dict, kept_by: str) -> None:
        q = take["quality"]
        cue = take.get("cue") or {}
        vol = take["volunteer"]
        self._append(
            self.manifest,
            MANIFEST_FIELDS,
            {
                "take_id": take["take_id"],
                "recorded_at": take["recorded_at"],
                "room": take["room"],
                "volunteer": vol,
                "activity": take["activity"],
                "label": take["label"],
                "trial": take["trial"],
                "duration_s": take["duration_s"],
                "cue_offset_s": cue.get("offset_s", ""),
                "cue_row": _blank(cue.get("csv_row")),
                "action_start_s": _blank(take.get("action_start_s")),
                "action_end_s": _blank(take.get("action_end_s")),
                "frames": q.get("frames", ""),
                "rate_hz": q.get("rate_hz", ""),
                "air_loss_pct": q.get("air_loss_pct", ""),
                "rssi_median": q.get("rssi_median", ""),
                "motion_ratio": _blank(q.get("motion_ratio")),
                "verdict": q["verdict"],
                "kept_by": kept_by,
                "note": take.get("note", ""),
                "bin": f"{vol}/{take['files']['bin']}",
                "csv": f"{vol}/{take['files']['csv']}" if "csv" in take["files"] else "",
                "json": f"{vol}/{take['files']['json']}",
            },
        )
        notes = [take.get("note", "")]
        if not take.get("train", True):
            notes.append("ไม่ใช้ฝึก")
        if take.get("label") == "none":
            notes.append("ห้องว่าง")
        if take.get("in_plan") is False:
            notes.append("นอกแผน")
        self._append(
            self.labels,
            LABEL_FIELDS,
            {
                "file": take["file"],
                "room": take["room"],
                "volunteer": vol,
                "action": take["activity"],
                "label": take["label"],
                "trial": take["trial"],
                "action_start_s": _blank(take.get("action_start_s")),
                "action_end_s": _blank(take.get("action_end_s")),
                "baseline_file": take.get("baseline_file") or "",
                "note": "; ".join(n for n in notes if n),
            },
        )

    def discard(self, paths: dict) -> None:
        if not self.cfg.keep_discarded:
            for p in paths.values():
                Path(p).unlink(missing_ok=True)
            return
        dest = self.session_dir / "_discarded"
        dest.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        for p in map(Path, paths.values()):
            if p.exists():
                target = dest / f"{p.parent.name}__{p.stem}__{stamp}{p.suffix}"
                shutil.move(str(p), target)
