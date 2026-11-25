"""Parser for the csi_rx serial stream (frame version 2) and link statistics.

Copied from the project repo, firmware/tools/csi_serial.py, plus encode_frame().
The frame layout is documented at the top of firmware/csi_rx/csi_rx.ino.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import serial

SYNC = b"\xa5\x5a"
FRAME_VERSION = 2
HDR_LEN = 17
HDR_FMT = "<BBHHIbBH"  # version, flags, uart_seq, rx_seq, t_us, rssi, channel, len
CSI_MAX_LEN = 384
FLAG_FIRST_WORD_INVALID = 0x01
FLAG_RETRY = 0x02
WIFI_SEQ_MOD = 4096  # 802.11 sequence numbers are 12 bits
UART_SEQ_MOD = 65536


@dataclass(frozen=True)
class Frame:
    """One CSI frame as the receiver sent it."""

    flags: int
    uart_seq: int
    rx_seq: int
    t_us: int
    rssi: int
    channel: int
    csi: bytes

    @property
    def is_retry(self) -> bool:
        return bool(self.flags & FLAG_RETRY)

    @property
    def first_word_invalid(self) -> bool:
        return bool(self.flags & FLAG_FIRST_WORD_INVALID)


@dataclass
class FrameReader:
    """Splits a byte stream into Frames and '#' text lines; counts bad checksums."""

    checksum_errors: int = 0
    skipped_bytes: int = 0
    _buf: bytearray = field(default_factory=bytearray)

    def feed(self, data: bytes) -> tuple[list[Frame], list[str]]:
        self._buf += data
        frames: list[Frame] = []
        lines: list[str] = []
        buf = self._buf
        while len(buf) >= 2:
            if buf[0] == SYNC[0] and buf[1] == SYNC[1]:
                if len(buf) < HDR_LEN:
                    break
                ver, flags, uart_seq, rx_seq, t_us, rssi, ch, ln = struct.unpack_from(
                    HDR_FMT, buf, 2
                )
                if ver != FRAME_VERSION or ln > CSI_MAX_LEN:
                    del buf[0]
                    self.skipped_bytes += 1
                    continue
                if len(buf) < HDR_LEN + ln:
                    break
                ck = 0
                for b in buf[2 : HDR_LEN - 1]:
                    ck ^= b
                payload = bytes(buf[HDR_LEN : HDR_LEN + ln])
                for b in payload:
                    ck ^= b
                if ck != buf[HDR_LEN - 1]:
                    self.checksum_errors += 1
                    del buf[0]
                    continue
                frames.append(Frame(flags, uart_seq, rx_seq, t_us, rssi, ch, payload))
                del buf[: HDR_LEN + ln]
            elif buf[0] == ord("#"):
                nl = buf.find(b"\n")
                sync = buf.find(SYNC, 1)
                # A '#' inside binary garbage: a frame starts before the line would end
                if sync >= 0 and (nl < 0 or sync < nl):
                    self.skipped_bytes += sync
                    del buf[:sync]
                    continue
                if nl < 0:
                    if len(buf) > 512:  # a '#' that never ends is noise
                        del buf[0]
                        self.skipped_bytes += 1
                        continue
                    break
                lines.append(buf[:nl].decode("utf-8", "replace").strip())
                del buf[: nl + 1]
            else:
                del buf[0]
                self.skipped_bytes += 1
        return frames, lines


@dataclass
class LinkStats:
    """Counts loss at the two places it can happen.

    air loss:  gaps in the 802.11 sequence number (frames the radio never got)
    uart loss: gaps in the receiver's uart_seq (frames lost in its queue or the serial link)
    Retransmissions repeat the sequence number of a frame we already have; they are
    counted as duplicates, not as received frames.

    A sequence jump bigger than the time since the last frame allows (at rate_hz,
    with slack) is a restart of the counter, e.g. a Tx reboot, and is counted as a
    resync instead of loss: the sequence number alone can't tell the two apart.
    """

    rate_hz: float = 100.0
    frames: int = 0
    duplicates: int = 0
    air_lost: int = 0
    uart_lost: int = 0
    resyncs: int = 0
    first_word_invalid: int = 0
    _last_rx_seq: int | None = None
    _last_uart_seq: int | None = None
    _last_t_us: int | None = None

    def add(self, f: Frame) -> bool:
        """Adds a frame; returns False when it is a duplicate of the previous one."""
        if f.first_word_invalid:
            self.first_word_invalid += 1
        if self._last_uart_seq is not None:
            self.uart_lost += (f.uart_seq - self._last_uart_seq - 1) % UART_SEQ_MOD
        self._last_uart_seq = f.uart_seq

        if self._last_rx_seq is not None and self._last_t_us is not None:
            step = (f.rx_seq - self._last_rx_seq) % WIFI_SEQ_MOD
            if step == 0:
                self.duplicates += 1
                return False
            dt_s = ((f.t_us - self._last_t_us) % 2**32) / 1e6
            max_plausible = dt_s * self.rate_hz * 1.5 + 10
            if step - 1 <= max_plausible:
                self.air_lost += step - 1
            else:
                self.resyncs += 1
        self._last_rx_seq = f.rx_seq
        self._last_t_us = f.t_us
        self.frames += 1
        return True

    @property
    def air_loss_pct(self) -> float:
        total = self.frames + self.air_lost
        return 100.0 * self.air_lost / total if total else 0.0


def encode_frame(f: Frame) -> bytes:
    """Builds the exact bytes the receiver sent for this frame (inverse of FrameReader)."""
    hdr = struct.pack(
        HDR_FMT,
        FRAME_VERSION,
        f.flags,
        f.uart_seq,
        f.rx_seq,
        f.t_us,
        f.rssi,
        f.channel,
        len(f.csi),
    )
    ck = 0
    for b in hdr:
        ck ^= b
    for b in f.csi:
        ck ^= b
    return SYNC + hdr + bytes([ck]) + f.csi


def open_quiet(port: str, baud: int, timeout_s: float = 0.05) -> serial.Serial:
    """Opens a port without toggling DTR/RTS, so the board is not reset."""
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, timeout_s
    s.dtr = False
    s.rts = False
    s.open()
    return s
