import base64
import json
import shutil
from pathlib import Path

import httpx
import pytest

from indicorderbench.audio.manifest import (
    ClipEntry,
    ClipManifest,
    load_manifest,
    save_manifest,
    sha256_file,
)
from indicorderbench.audio.tts import (
    AudioProviderError,
    SarvamTTS,
    language_code,
    synth_pack,
)
from indicorderbench.audio.wav import write_wav
from indicorderbench.packs import load_pack

MINIPACK = Path(__file__).parent / "fixtures" / "minipack"


def wav_bytes(tmp_path: Path, n: int = 160) -> bytes:
    p = tmp_path / "x.wav"
    write_wav(p, [0.1] * n, 16000)
    return p.read_bytes()


def test_language_code() -> None:
    assert language_code("hi-en") == "hi-IN"
    assert language_code("en-IN") == "en-IN"
    with pytest.raises(ValueError):
        language_code("fr")


def test_sarvam_tts_request_shape(tmp_path: Path) -> None:
    audio = wav_bytes(tmp_path)
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"audios": [base64.b64encode(audio).decode()]})

    tts = SarvamTTS("k123", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert tts.synthesize("Do paneer wrap", "hi-en", "anushka") == audio
    req = seen[0]
    assert req.method == "POST"
    assert str(req.url) == "https://api.sarvam.ai/text-to-speech"
    assert req.headers["API-Subscription-Key"] == "k123"
    assert json.loads(req.content) == {
        "inputs": ["Do paneer wrap"],
        "target_language_code": "hi-IN",
        "speaker": "anushka",
        "model": "bulbul:v3",
    }


@pytest.mark.parametrize(
    "response",
    [httpx.Response(429, text="slow down"), httpx.Response(200, json={"nope": 1})],
)
def test_sarvam_tts_errors(response: httpx.Response) -> None:
    tts = SarvamTTS("k", client=httpx.Client(transport=httpx.MockTransport(lambda r: response)))
    with pytest.raises(AudioProviderError) as ei:
        tts.synthesize("hi", "en-IN", "v")
    assert ei.value.status == response.status_code
    assert ei.value.detail


def test_sarvam_tts_network_error() -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    tts = SarvamTTS("k", client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(AudioProviderError) as ei:
        tts.synthesize("hi", "en-IN", "v")
    assert ei.value.status is None


def test_manifest_roundtrip(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    e = ClipEntry(
        path="clips/a/t1.wav",
        sha256="abc",
        scenario_id="a",
        turn_id="t1",
        text="hello",
        language="en-IN",
        provider="sarvam",
        voice="v",
        model="m",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    m = ClipManifest(clips=[e])
    p = tmp_path / "clips" / "manifest.json"
    save_manifest(m, p)
    assert load_manifest(p) == m
    assert m.by_hash()["abc"] is e
    assert not list(p.parent.glob("*.tmp"))


class FakeTTS:
    name = "fake"
    model = "fake-1"

    def __init__(self, audio: bytes, fail_on: str | None = None) -> None:
        self.audio = audio
        self.fail_on = fail_on
        self.calls: list[tuple[str, str, str]] = []

    def synthesize(self, text: str, language: str, voice: str) -> bytes:
        self.calls.append((text, language, voice))
        if self.fail_on and self.fail_on in text:
            raise AudioProviderError("boom", status=500)
        return self.audio


def pack_copy(tmp_path: Path):  # type: ignore[no-untyped-def]
    dst = tmp_path / "pack"
    shutil.copytree(MINIPACK, dst)
    return load_pack(dst)


def test_synth_pack_writes_expected_paths_and_manifest(tmp_path: Path) -> None:
    pack = pack_copy(tmp_path)
    tts = FakeTTS(wav_bytes(tmp_path))
    rep = synth_pack(pack, tts, "anushka")
    assert not rep.failed and not rep.skipped
    rels = {p.relative_to(pack.root).as_posix() for p in rep.written}
    assert "clips/en_quantity_01/t1.wav" in rels
    for lang in ("en-IN",):
        for d in ("closing", "confirm", "fallback", "nudge"):
            assert f"clips/_defaults/{lang}/{d}.wav" in rels
    # every scripted turn resolves through Pack.clip_path
    for s in pack.scenarios:
        for t in s.caller.turns:
            assert pack.clip_path(s.id, t, s.language) is not None
        d = pack.defaults_for(s.language)
        assert pack.clip_path(s.id, d.closing, s.language) is not None
    m = load_manifest(pack.root / "clips" / "manifest.json")
    by_path = {c.path: c for c in m.clips}
    assert set(by_path) == rels
    c = by_path["clips/en_quantity_01/t1.wav"]
    assert c.text == "Two paneer wraps please."
    assert c.sha256 == sha256_file(pack.root / c.path)
    assert (c.provider, c.voice, c.model, c.language) == ("fake", "anushka", "fake-1", "en-IN")
    assert "\\" not in c.path


def test_synth_pack_resumable(tmp_path: Path) -> None:
    pack = pack_copy(tmp_path)
    audio = wav_bytes(tmp_path)
    first = FakeTTS(audio)
    synth_pack(pack, first, "v")
    second = FakeTTS(audio)
    rep = synth_pack(pack, second, "v")
    assert second.calls == []
    assert rep.skipped and not rep.written
    # corrupt one file: hash mismatch forces re-synthesis of exactly that clip
    target = pack.root / "clips" / "en_quantity_01" / "t1.wav"
    target.write_bytes(b"junk")
    third = FakeTTS(audio)
    rep = synth_pack(pack, third, "v")
    assert rep.written == [target]
    assert len(third.calls) == 1
    # only_missing=False redoes everything
    fourth = FakeTTS(audio)
    synth_pack(pack, fourth, "v", only_missing=False)
    assert len(fourth.calls) == len(rep.skipped) + 1


def test_synth_pack_collects_failures(tmp_path: Path) -> None:
    pack = pack_copy(tmp_path)
    tts = FakeTTS(wav_bytes(tmp_path), fail_on="paneer")
    rep = synth_pack(pack, tts, "v")
    assert rep.failed
    assert all("AudioProviderError" in msg for _, msg in rep.failed)
    assert rep.written  # other clips still produced
    m = load_manifest(pack.root / "clips" / "manifest.json")
    failed_paths = {p for p, _ in rep.failed}
    assert failed_paths.isdisjoint({c.path for c in m.clips})


def test_synth_pack_scenario_filter_and_clarifications(tmp_path: Path) -> None:
    pack = pack_copy(tmp_path)
    scen = pack.scenario("en_quantity_01")
    from indicorderbench.schemas.scenario import CallerTurn, ClarificationRule

    scen.caller.clarifications.append(
        ClarificationRule(id="size", match=["size"], reply=CallerTurn(text="Regular."))
    )
    pack.manifest.defaults.caller[scen.language].clarifications.append(
        ClarificationRule(id="spice", match=["spice"], reply=CallerTurn(text="Medium."))
    )
    rep = synth_pack(pack, FakeTTS(wav_bytes(tmp_path)), "v", scenario_ids=["en_quantity_01"])
    rels = {p.relative_to(pack.root).as_posix() for p in rep.written}
    assert "clips/en_quantity_01/c_size.wav" in rels
    assert "clips/_defaults/en-IN/c_spice.wav" in rels
    assert not any("en_cancellation_01" in r for r in rels)


def test_synth_pack_resynthesizes_when_job_changed(tmp_path: Path) -> None:
    pack = pack_copy(tmp_path)
    audio = wav_bytes(tmp_path)
    synth_pack(pack, FakeTTS(audio), "v")
    # same inputs: nothing to do
    assert not synth_pack(pack, FakeTTS(audio), "v").written
    # voice change
    assert synth_pack(pack, FakeTTS(audio), "other").written
    # text change
    pack.scenario("en_quantity_01").caller.turns[0].text = "Three paneer wraps."
    tts = FakeTTS(audio)
    rep = synth_pack(pack, tts, "other")
    assert [c[0] for c in tts.calls] == ["Three paneer wraps."]
    assert len(rep.written) == 1
    # model change
    tts2 = FakeTTS(audio)
    tts2.model = "fake-2"
    assert len(synth_pack(pack, tts2, "other").written) > 0
