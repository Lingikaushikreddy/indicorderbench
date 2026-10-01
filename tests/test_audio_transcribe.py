import hashlib
from datetime import UTC, datetime

import httpx
import pytest

from indicorderbench.audio.manifest import ClipEntry, ClipManifest
from indicorderbench.audio.transcribe import OracleTranscriber, SarvamTranscriber
from indicorderbench.audio.tts import AudioProviderError


def test_oracle_lookup_and_miss() -> None:
    audio = b"RIFFfake"
    entry = ClipEntry(
        path="clips/a/t1.wav",
        sha256=hashlib.sha256(audio).hexdigest(),
        scenario_id="a",
        turn_id="t1",
        text="do paneer wrap",
        language="hi-en",
        provider="p",
        voice="v",
        model="m",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    t = OracleTranscriber(ClipManifest(clips=[entry]))
    assert t.transcribe(audio, "hi-en") == "do paneer wrap"
    with pytest.raises(AudioProviderError, match="clip not in manifest"):
        t.transcribe(b"other", "hi-en")


def _client(handler):  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sarvam_stt_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"transcript": "do wrap"})

    stt = SarvamTranscriber("k9", client=_client(handler))
    assert stt.transcribe(b"WAVDATA", "hi-en") == "do wrap"
    req = seen[0]
    assert str(req.url) == "https://api.sarvam.ai/speech-to-text"
    assert req.headers["API-Subscription-Key"] == "k9"
    assert req.headers["content-type"].startswith("multipart/form-data")
    body = req.content
    assert b'name="file"; filename="audio.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert b"WAVDATA" in body
    assert b'name="model"' in body and b"saaras:v3" in body
    assert b'name="language_code"' in body and b"hi-IN" in body


def test_sarvam_stt_unknown_language_omits_code() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"transcript": "x"})

    SarvamTranscriber("k", client=_client(handler)).transcribe(b"a", "xx")
    assert b"language_code" not in seen[0].content


@pytest.mark.parametrize(
    "response",
    [httpx.Response(401, text="bad key"), httpx.Response(200, json={"other": 1})],
)
def test_sarvam_stt_errors(response: httpx.Response) -> None:
    stt = SarvamTranscriber("k", client=_client(lambda r: response))
    with pytest.raises(AudioProviderError) as ei:
        stt.transcribe(b"a", "en-IN")
    assert ei.value.status == response.status_code
