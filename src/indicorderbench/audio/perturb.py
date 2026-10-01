"""Audio perturbations on mono float arrays (numpy, imported lazily).

Pink noise is made by shaping white noise with a 1/sqrt(f) filter in the FFT domain.
SNR is the ratio of whole-clip RMS of signal to noise.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# numpy's own stubs use 3.12 syntax, which mypy rejects with python_version = "3.11", so
# arrays are typed loosely instead of importing numpy for type checking.
Array = Any


class AudioExtraMissing(RuntimeError):
    """Raised when numpy is not installed."""


def _np() -> Any:
    try:
        numpy = importlib.import_module("numpy")
    except ImportError as e:
        raise AudioExtraMissing("install indicorderbench[audio]") from e
    return numpy


def _rms(x: Array) -> float:
    np = _np()
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def _pink(n: int, rng: Any) -> Array:
    np = _np()
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0.0
    out: Array = np.fft.irfft(spec * scale, n)
    return out


def add_noise(
    samples: Array,
    sr: int,
    snr_db: float,
    kind: Literal["white", "pink"] = "white",
    seed: int | None = None,
) -> Array:
    np = _np()
    rng = np.random.default_rng(seed)
    n = len(samples)
    noise = rng.standard_normal(n) if kind == "white" else _pink(n, rng)
    sig_rms, noise_rms = _rms(samples), _rms(noise)
    if noise_rms == 0.0:
        return np.asarray(samples, dtype=float).copy()
    target = sig_rms / (10 ** (snr_db / 20.0))
    out: Array = samples + noise * (target / noise_rms)
    return out


def gain(samples: Array, db: float) -> Array:
    out: Array = samples * (10 ** (db / 20.0))
    return out


def _bandpass(x: Array, sr: int, lo: float, hi: float) -> Array:
    np = _np()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    spec[(freqs < lo) | (freqs > hi)] = 0
    out: Array = np.fft.irfft(spec, len(x))
    return out


def telephone(samples: Array, sr: int) -> Array:
    """Band-pass 300-3400 Hz, round-trip through 8 kHz with linear interpolation, re-filter."""
    np = _np()
    n = len(samples)
    if n == 0:
        return np.asarray(samples, dtype=float).copy()
    x = _bandpass(np.asarray(samples, dtype=float), sr, 300.0, 3400.0)
    tel_sr = 8000
    t = np.arange(n) / sr
    m = max(1, round(n * tel_sr / sr))
    t8 = np.arange(m) / tel_sr
    low = np.interp(t8, t, x)
    back = np.interp(t, t8, low)
    return _bandpass(back, sr, 300.0, 3400.0)


def normalise_peak(samples: Array, peak: float = 0.95) -> Array:
    np = _np()
    m = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if m == 0.0:
        return np.asarray(samples, dtype=float).copy()
    out: Array = samples * (peak / m)
    return out


@dataclass
class PerturbSpec:
    snr_db: float | None = None
    noise: Literal["white", "pink"] = "white"
    gain_db: float = 0.0
    telephone: bool = False
    seed: int | None = None


def apply(samples: Array, sr: int, spec: PerturbSpec) -> Array:
    """Apply telephone, gain, then noise (so SNR is relative to the final signal level)."""
    out = samples
    if spec.telephone:
        out = telephone(out, sr)
    if spec.gain_db:
        out = gain(out, spec.gain_db)
    if spec.snr_db is not None:
        out = add_noise(out, sr, spec.snr_db, kind=spec.noise, seed=spec.seed)
    return out


def perturb_dir(src: Path, dst: Path, spec: PerturbSpec) -> list[Path]:
    """Perturb every WAV under ``src`` into ``dst`` (same relative paths)."""
    from indicorderbench.audio.wav import read_wav, write_wav

    np = _np()
    written: list[Path] = []
    for wav in sorted(src.rglob("*.wav")):
        samples, sr = read_wav(wav)
        out = np.clip(apply(np.asarray(samples, dtype=float), sr, spec), -1.0, 1.0)
        target = dst / wav.relative_to(src)
        write_wav(target, out.tolist(), sr)
        written.append(target)
    manifest: dict[str, Any] = {"version": 1, "clips": []}
    src_manifest = src / "manifest.json"
    if src_manifest.exists():
        manifest = json.loads(src_manifest.read_text(encoding="utf-8"))
    manifest["derived_from"] = str(src)
    manifest["perturbation"] = asdict(spec)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return written
