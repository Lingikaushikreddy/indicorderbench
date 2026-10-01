"""Speech-to-text: the Transcriber protocol, an oracle, and the Sarvam provider."""

from __future__ import annotations

import contextlib
import hashlib
from typing import Protocol, runtime_checkable

import httpx

from indicorderbench.audio.manifest import ClipManifest
from indicorderbench.audio.provider import (
    SARVAM_BASE,
    AudioProviderError,
    error_from_response,
    language_code,
)


@runtime_checkable
class Transcriber(Protocol):
    """Turns caller audio into text. The reference agent and the audio providers share it."""

    def transcribe(self, audio: bytes, language: str) -> str: ...


class OracleTranscriber:
    """Returns the manifest text for a clip by hash, so the audio path works without STT."""

    def __init__(self, manifest: ClipManifest) -> None:
        self._by_hash = manifest.by_hash()

    def transcribe(self, audio: bytes, language: str) -> str:
        try:
            return self._by_hash[hashlib.sha256(audio).hexdigest()].text
        except KeyError:
            raise AudioProviderError("clip not in manifest") from None


class SarvamTranscriber:
    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        model: str = "saaras:v3",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def transcribe(self, audio: bytes, language: str) -> str:
        data = {"model": self.model}
        with contextlib.suppress(ValueError):  # unknown language: let the service auto-detect
            data["language_code"] = language_code(language)
        try:
            resp = self._client.post(
                f"{SARVAM_BASE}/speech-to-text",
                headers={"API-Subscription-Key": self.api_key},
                files={"file": ("audio.wav", audio, "audio/wav")},
                data=data,
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise AudioProviderError(f"request failed: {e}") from e
        if not resp.is_success:
            raise error_from_response(resp)
        try:
            transcript = resp.json()["transcript"]
            if not isinstance(transcript, str):
                raise TypeError("transcript is not a string")
            return transcript
        except (ValueError, KeyError, TypeError) as e:
            raise AudioProviderError(
                f"malformed response: {resp.text[:300]}", status=resp.status_code
            ) from e
