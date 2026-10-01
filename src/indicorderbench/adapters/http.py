"""Adapter for agents that run as a separate HTTP service speaking the turn protocol."""

from __future__ import annotations

import base64
import binascii
from typing import Any

import httpx

from indicorderbench.adapters.protocol import AgentReply, CallerUtterance, SessionInfo


class AdapterError(RuntimeError):
    """The agent service failed, timed out, or sent a reply that breaks the protocol."""


class HttpTurnAdapter:
    def __init__(
        self, url: str, timeout_s: float = 30.0, client: httpx.AsyncClient | None = None
    ) -> None:
        self._url = url
        self._timeout = timeout_s
        self._client = client
        self._owns_client = client is None
        self._session_id: str | None = None

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        if self._client is None:
            raise AdapterError("adapter not started")
        event = payload["event"]
        try:
            resp = await self._client.post(self._url, json=payload, timeout=self._timeout)
        except httpx.HTTPError as e:
            raise AdapterError(f"{event} request failed: {type(e).__name__}: {e}") from e
        if not resp.is_success:
            raise AdapterError(
                f"{event} request returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp

    async def start(self, session: SessionInfo) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient()
            self._owns_client = True
        await self._post(
            {
                "event": "start",
                "session_id": session.session_id,
                "scenario_id": session.scenario_id,
                "backend_url": session.backend_url,
                "language": session.language,
                "modality": session.modality.value,
            }
        )
        self._session_id = session.session_id

    async def respond(self, utterance: CallerUtterance) -> AgentReply:
        if self._session_id is None:
            raise AdapterError("adapter not started")
        audio = utterance.audio_bytes
        resp = await self._post(
            {
                "event": "turn",
                "session_id": self._session_id,
                "turn_id": utterance.turn_id,
                "text": utterance.text,
                "audio_b64": base64.b64encode(audio).decode("ascii") if audio else None,
                "audio_format": "wav",
                "language": utterance.language,
            }
        )
        try:
            data = resp.json()
        except ValueError as e:
            raise AdapterError("turn reply is not valid JSON") from e
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise AdapterError("turn reply must be a JSON object with a string 'text'")
        meta = data.get("meta")
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise AdapterError("turn reply 'meta' must be an object")
        audio_b64 = data.get("audio_b64")
        audio_out: bytes | None = None
        if audio_b64 is not None:
            if not isinstance(audio_b64, str):
                raise AdapterError("turn reply 'audio_b64' must be a string")
            try:
                audio_out = base64.b64decode(audio_b64, validate=True)
            except (binascii.Error, ValueError) as e:
                raise AdapterError("turn reply 'audio_b64' is not valid base64") from e
        return AgentReply(text=data["text"], audio_bytes=audio_out, meta=meta)

    async def stop(self) -> None:
        client, session_id = self._client, self._session_id
        self._session_id = None
        try:
            if client is not None and session_id is not None:
                try:
                    await self._post({"event": "stop", "session_id": session_id})
                except AdapterError as e:
                    raise AdapterError(f"stop failed: {e}") from e
        finally:
            if self._owns_client and client is not None:
                self._client = None
                await client.aclose()
