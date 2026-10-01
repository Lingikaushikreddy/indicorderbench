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


def test_apply_defaults_are_identity() -> None:
    x = sine(500, 0.5)
    assert np.allclose(apply(x, SR, PerturbSpec()), x)


def test_apply_each_option_takes_effect() -> None:
    x = sine(300, 1.0)
    assert rms(apply(x, SR, PerturbSpec(gain_db=-6.0))) / rms(x) == pytest.approx(0.5, rel=0.01)
    low = sine(100) + sine(1000)
    y = apply(low, SR, PerturbSpec(telephone=True))
    assert band_energy_db(y, 1000) - band_energy_db(y, 100) >= 20
    n = apply(x, SR, PerturbSpec(snr_db=10, seed=1)) - x
    assert 20 * math.log10(rms(x) / rms(n)) == pytest.approx(10.0, abs=0.5)


def test_apply_order_is_telephone_gain_noise() -> None:
    x = sine(1000) + sine(100)
    spec = PerturbSpec(snr_db=20, gain_db=-6, telephone=True, seed=5)
    expected = add_noise(gain(telephone(x, SR), -6), SR, 20, seed=5)
    assert np.allclose(apply(x, SR, spec), expected)


def test_add_noise_pink_empty_array() -> None:
    out = add_noise(np.zeros(0), SR, 10.0, kind="pink", seed=1)
    assert out.shape == (0,)


def test_perturb_dir_mirrors_tree(tmp_path: Path) -> None:
    src = tmp_path / "src"
    x = sine(400, 0.1)
    write_wav(src / "s1" / "t1.wav", x.tolist(), SR)
    write_wav(src / "_defaults" / "hi-en" / "closing.wav", x.tolist(), SR)
    write_manifest(src, [("s1/t1.wav", "hello")])
    dst = tmp_path / "dst"
    out = perturb_dir(src, dst, PerturbSpec(snr_db=10, seed=2))
    rel = sorted(p.relative_to(dst).as_posix() for p in out)
    assert rel == ["_defaults/hi-en/closing.wav", "s1/t1.wav"]
    samples, sr = read_wav(dst / "s1" / "t1.wav")
    assert sr == SR and len(samples) == 1600
    noise = np.array(samples) - x
    assert 20 * math.log10(rms(x) / rms(noise)) == pytest.approx(10.0, abs=0.7)


def write_manifest(root: Path, items: list[tuple[str, str]], **extra: object) -> None:
    from indicorderbench.audio.manifest import sha256_file

    clips = [
        {
            "path": f"clips/{rel}",
            "sha256": sha256_file(root / rel),
            "scenario_id": "s1",
            "turn_id": "t1",
            "text": text,
            "language": "hi-en",
            "provider": "p",
            "voice": "v",
            "model": "m",
            "created_at": "2026-10-01T00:00:00Z",
        }
        for rel, text in items
    ]
    (root / "manifest.json").write_text(json.dumps({"version": 1, "clips": clips, **extra}))


def test_perturb_dir_rehashes_manifest_and_records_derivation(tmp_path: Path) -> None:
    from indicorderbench.audio.manifest import load_manifest, sha256_file
    from indicorderbench.audio.transcribe import OracleTranscriber

    src = tmp_path / "src"
    write_wav(src / "s1" / "t1.wav", sine(400, 0.1).tolist(), SR)
    write_manifest(src, [("s1/t1.wav", "hello")])
    dst = tmp_path / "dst"
    perturb_dir(src, dst, PerturbSpec(snr_db=10, seed=2))
    m = load_manifest(dst / "manifest.json")
    assert m.derived_from == str(src)
    assert m.perturbation is not None and m.perturbation["snr_db"] == 10
    assert m.clips[0].sha256 == sha256_file(dst / "s1" / "t1.wav")
    assert m.clips[0].sha256 != sha256_file(src / "s1" / "t1.wav")
    assert OracleTranscriber(m).transcribe((dst / "s1" / "t1.wav").read_bytes(), "hi-en") == "hello"
    # chained perturbation keeps the earlier derivation
    dst2 = tmp_path / "dst2"
    perturb_dir(dst, dst2, PerturbSpec(gain_db=-3))
    m2 = load_manifest(dst2 / "manifest.json")
    assert m2.derived_from == str(dst)
    assert m2.perturbation is not None
    prev = m2.perturbation["previous"]
    assert prev["derived_from"] == str(src) and prev["perturbation"]["snr_db"] == 10
    assert m2.clips[0].sha256 == sha256_file(dst2 / "s1" / "t1.wav")
