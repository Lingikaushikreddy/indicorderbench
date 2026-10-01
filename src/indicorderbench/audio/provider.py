"""Pieces shared by the Sarvam TTS and STT providers.

Kept free of pack imports so speech-to-text does not pull in the pack loader.
"""

from __future__ import annotations

import httpx

SARVAM_BASE = "https://api.sarvam.ai"


class AudioProviderError(RuntimeError):
    def __init__(self, detail: str, status: int | None = None) -> None:
        super().__init__(f"{detail} (status {status})" if status is not None else detail)
        self.status = status
        self.detail = detail


def language_code(language: str) -> str:
    """Map a benchmark language to a Sarvam language code (Hinglish uses hi-IN)."""
    if language == "hi-en":
        return "hi-IN"
    if language == "en-IN":
        return "en-IN"
    raise ValueError(f"unknown language {language!r}")


def error_from_response(resp: httpx.Response) -> AudioProviderError:
    return AudioProviderError(resp.text[:300], status=resp.status_code)
