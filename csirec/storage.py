"""Writes takes to disk and keeps the session manifest.

Layout (follows docs/Design.md §7, recordings/<session>/<volunteer>/):
    <data_dir>/<session>/manifest.csv                  one row per kept take
    <data_dir>/<session>/<volunteer>/<activity>_<position>_<NNN>.bin   raw frames (csi_rx v2)
                                                    ....json  metadata + quality
                                                    ....csv   amplitude, same columns as exp7/exp8
    <data_dir>/<session>/_discarded/                   takes thrown away (if keep_discarded)
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

MANIFEST_FIELDS = [
    "take_id",
    "recorded_at",
    "volunteer",
    "position",
    "activity",
    "label",
    "duration_s",
    "cue_offset_s",
    "cue_row",
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
NO_VOLUNTEER = "none"


class Storage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session_dir: Path = cfg.data_dir / cfg.session
        self.manifest: Path = self.session_dir / "manifest.csv"

    # ------------------------------------------------------------------ counts
    def load_counts(self) -> Counter:
        counts: Counter = Counter()
        if self.manifest.exists():
            with self.manifest.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    counts[(row["volunteer"], row["activity"])] += 1
        return counts

    def _next_index(self, folder: Path, stem_prefix: str) -> int:
        pat = re.compile(re.escape(stem_prefix) + r"_(\d{3,})\.json$")
        used = [
            int(m.group(1))
            for p in folder.glob(stem_prefix + "_*.json")
            if (m := pat.match(p.name))
        ]
        return max(used, default=0) + 1

    # ------------------------------------------------------------------ write
    def write_take(
        self, take: dict, frames: list[tuple[float, Frame]], kept: list[tuple[float, Frame]]
    ) -> dict:
        """Writes .bin (all frames), .csv (retries removed) and .json. Returns file paths."""
        folder = self.session_dir / take["volunteer"]
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{take['activity']}_{take['position']}"
        idx = self._next_index(folder, prefix)
        stem = f"{prefix}_{idx:03d}"
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

        take["take_id"] = f"{take['volunteer']}/{stem}"
        take["files"] = {k: p.name for k, p in paths.items()}
        paths["json"].write_text(json.dumps(take, ensure_ascii=False, indent=2), encoding="utf-8")
        return {k: str(p) for k, p in paths.items()}

    # ------------------------------------------------------------------ keep / discard
    def keep(self, take: dict, kept_by: str) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        new = not self.manifest.exists()
        q = take["quality"]
        cue = take.get("cue") or {}
        row = {
            "take_id": take["take_id"],
            "recorded_at": take["recorded_at"],
            "volunteer": take["volunteer"],
            "position": take["position"],
            "activity": take["activity"],
            "label": take["label"],
            "duration_s": take["duration_s"],
            "cue_offset_s": cue.get("offset_s", ""),
            "cue_row": cue.get("csv_row", ""),
            "frames": q.get("frames", ""),
            "rate_hz": q.get("rate_hz", ""),
            "air_loss_pct": q.get("air_loss_pct", ""),
            "rssi_median": q.get("rssi_median", ""),
            "motion_ratio": q.get("motion_ratio", "") if q.get("motion_ratio") is not None else "",
            "verdict": q["verdict"],
            "kept_by": kept_by,
            "note": take.get("note", ""),
            "bin": f"{take['volunteer']}/{take['files']['bin']}",
            "csv": f"{take['volunteer']}/{take['files']['csv']}" if "csv" in take["files"] else "",
            "json": f"{take['volunteer']}/{take['files']['json']}",
        }
        with self.manifest.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

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
