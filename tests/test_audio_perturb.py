import json
import math
from pathlib import Path

import numpy as np
import pytest

from indicorderbench.audio.perturb import (
    PerturbSpec,
    add_noise,
    apply,
    gain,
    normalise_peak,
    perturb_dir,
    telephone,
)
from indicorderbench.audio.wav import read_wav, write_wav

SR = 16000


def sine(freq: float, secs: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * secs)) / SR
    return amp * np.sin(2 * math.pi * freq * t)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def band_energy_db(x: np.ndarray, freq: float) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    idx = int(np.argmin(np.abs(freqs - freq)))
    return 20 * math.log10(float(spec[max(idx - 2, 0) : idx + 3].max()) + 1e-12)


def test_wav_roundtrip(tmp_path: Path) -> None:
    x = sine(440, 0.2)
    p = tmp_path / "a.wav"
    write_wav(p, x.tolist(), SR)
    back, sr = read_wav(p)
    assert sr == SR
    assert len(back) == len(x)
    assert np.max(np.abs(np.array(back) - x)) < 1e-3


def test_wav_stereo_averaged(tmp_path: Path) -> None:
    import wave

    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(np.array([16000, 0, 16000, 0], dtype="<i2").tobytes())
    samples, sr = read_wav(p)
    assert sr == 8000
    assert len(samples) == 2
    assert samples[0] == pytest.approx(8000 / 32768, abs=1e-4)


def test_wav_rejects_8bit(tmp_path: Path) -> None:
    import wave

    p = tmp_path / "e.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(b"\x80\x80")
    with pytest.raises(ValueError, match="16-bit"):
        read_wav(p)


@pytest.mark.parametrize("kind", ["white", "pink"])
def test_add_noise_snr(kind: str) -> None:
    x = sine(1000, 2.0)
    y = add_noise(x, SR, 10.0, kind=kind, seed=1)  # type: ignore[arg-type]
    noise = y - x
    snr = 20 * math.log10(rms(x) / rms(noise))
    assert snr == pytest.approx(10.0, abs=0.5)


def test_add_noise_seed_deterministic() -> None:
    x = sine(1000, 0.5)
    assert np.array_equal(add_noise(x, SR, 5, seed=3), add_noise(x, SR, 5, seed=3))
    assert not np.array_equal(add_noise(x, SR, 5, seed=3), add_noise(x, SR, 5, seed=4))


def test_pink_noise_has_more_low_than_high() -> None:
    x = np.zeros(SR * 4)
    n = add_noise(x + 1e-9, SR, -20.0, kind="pink", seed=0)
    spec = np.abs(np.fft.rfft(n)) ** 2
    freqs = np.fft.rfftfreq(len(n), 1 / SR)
    assert (
        spec[(freqs > 100) & (freqs < 400)].mean()
        > 4 * spec[(freqs > 4000) & (freqs < 7000)].mean()
    )


def test_gain_minus_6db_halves_rms() -> None:
    x = sine(300)
    assert rms(gain(x, -6.0)) / rms(x) == pytest.approx(0.5, rel=0.01)


def test_telephone_band_limits() -> None:
    mix = sine(100) + sine(1000) + sine(6000)
    y = telephone(mix, SR)
    assert len(y) == len(mix)
    ref = band_energy_db(y, 1000)
    assert ref - band_energy_db(y, 100) >= 20
    assert ref - band_energy_db(y, 6000) >= 20


def test_normalise_peak() -> None:
    y = normalise_peak(sine(300, amp=0.2), 0.9)
    assert float(np.max(np.abs(y))) == pytest.approx(0.9, abs=1e-6)
    assert np.array_equal(normalise_peak(np.zeros(10)), np.zeros(10))


def test_apply_composes_and_defaults_are_identity() -> None:
    x = sine(500, 0.5)
    assert np.allclose(apply(x, SR, PerturbSpec()), x)
    y = apply(x, SR, PerturbSpec(snr_db=15, gain_db=-3, telephone=True, seed=1))
    assert y.shape == x.shape


def test_perturb_dir_mirrors_tree(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "s1").mkdir(parents=True)
    (src / "_defaults" / "hi-en").mkdir(parents=True)
    write_wav(src / "s1" / "t1.wav", sine(400, 0.1).tolist(), SR)
    write_wav(src / "_defaults" / "hi-en" / "closing.wav", sine(400, 0.1).tolist(), SR)
    (src / "manifest.json").write_text(json.dumps({"version": 1, "clips": []}))
    dst = tmp_path / "dst"
    out = perturb_dir(src, dst, PerturbSpec(snr_db=10, seed=2))
    rel = sorted(p.relative_to(dst).as_posix() for p in out)
    assert rel == ["_defaults/hi-en/closing.wav", "s1/t1.wav"]
    m = json.loads((dst / "manifest.json").read_text())
    assert m["derived_from"] == str(src)
    assert m["perturbation"]["snr_db"] == 10
    samples, sr = read_wav(dst / "s1" / "t1.wav")
    assert sr == SR and len(samples) == 1600
