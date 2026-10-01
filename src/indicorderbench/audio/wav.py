"""Minimal 16-bit PCM WAV I/O using only the standard library."""

from __future__ import annotations

import struct
import wave
from collections.abc import Sequence
from pathlib import Path


def read_wav(path: Path) -> tuple[list[float], int]:
    """Read a 16-bit PCM WAV as mono floats in [-1, 1]; stereo is averaged."""
    with wave.open(str(path), "rb") as w:
        width = w.getsampwidth()
        if width != 2:
            raise ValueError(
                f"{path}: only 16-bit PCM WAV is supported, got {width * 8}-bit samples"
            )
        channels = w.getnchannels()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    count = len(raw) // 2
    ints = struct.unpack(f"<{count}h", raw[: count * 2])
    if channels == 1:
        return [v / 32768.0 for v in ints], sr
    frames = count // channels
    return [
        sum(ints[i * channels : (i + 1) * channels]) / channels / 32768.0 for i in range(frames)
    ], sr


def write_wav(path: Path, samples: Sequence[float], sr: int) -> None:
    """Write mono 16-bit PCM; samples are clipped to [-1, 1]."""
    ints = [max(-32768, min(32767, round(s * 32768.0))) for s in samples]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{len(ints)}h", *ints))
