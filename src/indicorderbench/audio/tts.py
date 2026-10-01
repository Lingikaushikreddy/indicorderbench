"""Text-to-speech providers and pack clip synthesis."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx

from indicorderbench.audio.manifest import (
    ClipEntry,
    ClipManifest,
    load_manifest,
    save_manifest,
    sha256_file,
)

# Re-exported so existing ``from indicorderbench.audio.tts import ...`` imports keep working.
from indicorderbench.audio.provider import SARVAM_BASE as SARVAM_BASE
from indicorderbench.audio.provider import AudioProviderError as AudioProviderError
from indicorderbench.audio.provider import error_from_response as error_from_response
from indicorderbench.audio.provider import language_code as language_code
from indicorderbench.packs import DEFAULT_TURN_IDS, Pack
from indicorderbench.schemas.scenario import CallerTurn, Language

WAV_HEADER_BYTES = 44  # a canonical RIFF/WAVE header; anything shorter holds no audio


class TTSProvider(Protocol):
    name: str
    model: str

    def synthesize(self, text: str, language: str, voice: str) -> bytes: ...


class PlaceholderTTS:
    """Offline provider for pipeline tests: a short, near-silent clip keyed to the text.

    Every text gets distinct bytes (a deterministic low-amplitude pattern seeded by the
    text), so manifest hashes stay unique and the OracleTranscriber can map a clip back to
    its text. It makes ``--modality audio`` runnable without any API key; results produced
    with it must say so.
    """

    name = "silence"
    model = "placeholder"

    def __init__(self, seconds: float = 0.5, sample_rate: int = 16000) -> None:
        self.seconds = seconds
        self.sample_rate = sample_rate

    def synthesize(self, text: str, language: str, voice: str) -> bytes:
        import hashlib
        import io
        import struct
        import wave

        digest = hashlib.sha256(f"{language}\x1f{voice}\x1f{text}".encode()).digest()
        n = int(self.seconds * self.sample_rate)
        # amplitude 1/32768: inaudible, but unique per text
        samples = [(digest[i % len(digest)] % 3) - 1 for i in range(n)]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(struct.pack(f"<{n}h", *samples))
        return buf.getvalue()


class SarvamTTS:
    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        model: str = "bulbul:v3",
        timeout: float = 25.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def synthesize(self, text: str, language: str, voice: str) -> bytes:
        try:
            resp = self._client.post(
                f"{SARVAM_BASE}/text-to-speech",
                headers={"API-Subscription-Key": self.api_key},
                json={
                    "inputs": [text],
                    "target_language_code": language_code(language),
                    "speaker": voice,
                    "model": self.model,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise AudioProviderError(f"request failed: {e}") from e
        if not resp.is_success:
            raise error_from_response(resp)
        try:
            audio = base64.b64decode(resp.json()["audios"][0], validate=True)
        except (ValueError, KeyError, IndexError, TypeError, binascii.Error) as e:
            raise AudioProviderError(
                f"malformed response: {resp.text[:300]}", status=resp.status_code
            ) from e
        if len(audio) < WAV_HEADER_BYTES:
            raise AudioProviderError("empty audio payload", status=resp.status_code)
        return audio


@dataclass
class SynthReport:
    written: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _Job:
    rel: str  # path relative to pack root, forward slashes
    scenario_id: str
    turn_id: str
    text: str
    language: str


def _target(turn: CallerTurn, scenario_id: str, language: Language, default: bool) -> str | None:
    """Pack-relative clip path for a turn, mirroring ``Pack.clip_path``."""
    assert turn.id is not None
    if turn.audio:
        return Path(turn.audio).as_posix()
    if default:
        return f"clips/_defaults/{language.value}/{turn.id}.wav"
    if turn.id in DEFAULT_TURN_IDS:
        return None  # indistinguishable from the language default; Pack.clip_path resolves it there
    return f"clips/{scenario_id}/{turn.id}.wav"


def _jobs(pack: Pack, scenario_ids: list[str] | None) -> list[_Job]:
    jobs: dict[str, _Job] = {}

    def add(turn: CallerTurn, scenario_id: str, language: Language, default: bool) -> None:
        rel = _target(turn, scenario_id, language, default)
        if rel is not None and rel not in jobs:
            assert turn.id is not None
            jobs[rel] = _Job(rel, scenario_id, turn.id, turn.text, language.value)

    scenarios = pack.filter(ids=scenario_ids) if scenario_ids else pack.scenarios
    languages: list[Language] = []
    for s in scenarios:
        if s.language not in languages:
            languages.append(s.language)
        script = s.caller
        for t in script.turns:
            add(t, s.id, s.language, False)
        for rule in script.clarifications:
            add(rule.reply, s.id, s.language, False)
        for t2 in (script.closing, script.confirm, script.fallback, script.nudge):
            if t2 is not None:
                add(t2, s.id, s.language, False)
    for lang in languages:
        d = pack.defaults_for(lang)
        for t3 in (d.closing, d.confirm, d.fallback, d.nudge):
            add(t3, "_defaults", lang, True)
        for rule in d.clarifications:
            add(rule.reply, "_defaults", lang, True)
    return list(jobs.values())


def synth_pack(
    pack: Pack,
    provider: TTSProvider,
    voice: str,
    only_missing: bool = True,
    scenario_ids: list[str] | None = None,
) -> SynthReport:
    """Synthesize every caller clip in the pack and update ``clips/manifest.json``.

    Scenario-level overrides of closing/confirm/fallback/nudge that keep the default id and
    have no explicit ``audio`` path are not synthesized separately: ``Pack.clip_path`` would
    resolve them to the language default clip.
    """
    report = SynthReport()
    manifest_path = pack.root / "clips" / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else ClipManifest()
    entries = {c.path: c for c in manifest.clips}

    for job in _jobs(pack, scenario_ids):
        path = pack.root / job.rel
        existing = entries.get(job.rel)
        if (
            only_missing
            and path.exists()
            and existing is not None
            and existing.sha256 == sha256_file(path)
            and existing.text == job.text
            and existing.language == job.language
            and existing.voice == voice
            and existing.model == provider.model
        ):
            report.skipped.append(path)
            continue
        try:
            audio = provider.synthesize(job.text, job.language, voice)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
        except Exception as e:
            report.failed.append((job.rel, f"{type(e).__name__}: {e}"))
            continue
        entries[job.rel] = ClipEntry(
            path=job.rel,
            sha256=sha256_file(path),
            scenario_id=job.scenario_id,
            turn_id=job.turn_id,
            text=job.text,
            language=job.language,
            provider=provider.name,
            voice=voice,
            model=provider.model,
            created_at=datetime.now(UTC),
        )
        report.written.append(path)

    manifest.clips = sorted(entries.values(), key=lambda c: c.path)
    save_manifest(manifest, manifest_path)
    return report
