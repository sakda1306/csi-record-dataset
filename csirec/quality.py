"""CSI amplitude helpers and the per-take quality check."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .csi_serial import Frame, LinkStats

# Raw CSI order from the ESP32 is subcarrier 0..31 then -32..-1.
# The 52 data subcarriers of HT20 L-LTF are -26..-1 and 1..26.
DATA_SC = np.r_[38:64, 1:27]
N_SC = 64
MOTION_BLOCK_S = 0.25


def amplitudes(f: Frame) -> np.ndarray:
    """|H| of all 64 subcarriers in raw order (float32)."""
    iq = np.frombuffer(f.csi[: 2 * N_SC], dtype=np.int8).astype(np.float32)
    amp = np.hypot(iq[0::2], iq[1::2])
    if amp.size < N_SC:
        amp = np.pad(amp, (0, N_SC - amp.size))
    return amp


def motion_blocks(amp_data: np.ndarray, rate_hz: float) -> np.ndarray:
    """Motion per 0.25 s block: std over time of each data subcarrier, averaged."""
    n = max(2, int(rate_hz * MOTION_BLOCK_S))
    k = amp_data.shape[0] // n
    if k == 0:
        return np.zeros(0)
    blocks = amp_data[: k * n].reshape(k, n, -1)
    return blocks.std(axis=1).mean(axis=1)


def dedupe(frames: Sequence[tuple[float, Frame]], rate_hz: float):
    """Drops retransmissions; returns (kept frames, LinkStats)."""
    st = LinkStats(rate_hz=rate_hz)
    kept = [(t, f) for t, f in frames if st.add(f)]
    return kept, st


def evaluate(
    frames: Sequence[tuple[float, Frame]],
    duration_s: float,
    cue_offset_s: float | None,
    expect_motion: bool,
    cfg,
    checksum_errors: int = 0,
) -> dict:
    """Quality of one take. frames are (host time relative to take start, Frame)."""
    kept, st = dedupe(frames, cfg.rate_hz)
    q: dict = {
        "frames": len(kept),
        "duplicates": st.duplicates,
        "air_lost": st.air_lost,
        "air_loss_pct": round(st.air_loss_pct, 2),
        "uart_lost": st.uart_lost,
        "resyncs": st.resyncs,
        "checksum_errors": checksum_errors,
        "first_word_invalid": st.first_word_invalid,
        "rate_hz": round(len(kept) / duration_s, 1),
    }
    reasons: list[str] = []
    warnings: list[str] = []

    if len(kept) < 3:
        q.update(verdict="fail", reasons=["ไม่มีข้อมูลจาก Rx"], warnings=[])
        return q

    # gaps by host time, including the start and end of the take
    t = np.array([x for x, _ in kept])
    edges = np.concatenate(([0.0], t, [duration_s]))
    gaps_ms = np.diff(edges) * 1000
    inner_ms = np.diff(t) * 1000
    rssi = np.array([f.rssi for _, f in kept])
    q.update(
        gap_max_ms=round(float(gaps_ms.max()), 1),
        gaps_over_50ms=int((inner_ms > 50).sum()),
        rssi_median=int(np.median(rssi)),
        rssi_min=int(rssi.min()),
    )

    if q["rate_hz"] < cfg.min_rate_hz:
        reasons.append(f"อัตรา {q['rate_hz']} Hz ต่ำกว่า {cfg.min_rate_hz:g} Hz")
    if q["air_loss_pct"] > cfg.max_air_loss_pct:
        reasons.append(f"เฟรมหาย {q['air_loss_pct']} % เกิน {cfg.max_air_loss_pct:g} %")
    if q["gap_max_ms"] > cfg.max_gap_ms:
        reasons.append(f"สัญญาณขาดช่วง {q['gap_max_ms']:.0f} ms เกิน {cfg.max_gap_ms:g} ms")
    if checksum_errors:
        reasons.append(f"checksum ผิด {checksum_errors} เฟรม (สาย USB?)")
    if st.uart_lost:
        warnings.append(f"เฟรมหายที่ serial {st.uart_lost} เฟรม")
    if st.resyncs:
        warnings.append("Tx รีบูตหรือลำดับเฟรมกระโดดระหว่างบันทึก")

    q["motion_ratio"] = None
    if cue_offset_s is not None:
        amp = np.array([amplitudes(f)[DATA_SC] for _, f in kept])
        mb = motion_blocks(amp, cfg.rate_hz)
        if mb.size:
            bt = np.arange(mb.size) * MOTION_BLOCK_S
            before = mb[(bt >= 0.5) & (bt + MOTION_BLOCK_S <= cue_offset_s)]
            after = mb[(bt >= cue_offset_s) & (bt < cue_offset_s + 4.0)]
            if before.size and after.size:
                ratio = float(after.max() / max(float(np.median(before)), 1e-6))
                q["motion_ratio"] = round(ratio, 2)
                if expect_motion and ratio < cfg.min_motion_ratio:
                    warnings.append(f"แทบไม่เห็นการเคลื่อนไหวหลังสัญญาณ (x{ratio:.1f}) ทำท่าทันไหม?")

    q["verdict"] = "fail" if reasons else "pass"
    q["reasons"] = reasons
    q["warnings"] = warnings
    return q
