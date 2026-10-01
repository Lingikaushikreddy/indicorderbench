"""Adapter contracts between the runner and an agent under test."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from indicorderbench.backend.state import OrderBackend


class Modality(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    BOTH = "both"


@dataclass
class CallerUtterance:
    turn_id: str
    text: str | None
    audio_path: Path | None
    audio_bytes: bytes | None
    language: str


@dataclass
class AgentReply:
    text: str
    audio_bytes: bytes | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionInfo:
    scenario_id: str
    trial: int
    language: str
    modality: Modality
    backend: OrderBackend
    session_id: str
    backend_url: str | None = None


@runtime_checkable
class AgentAdapter(Protocol):
    """Transport between the benchmark and one agent session."""

    async def start(self, session: SessionInfo) -> None: ...

    async def respond(self, utterance: CallerUtterance) -> AgentReply: ...

    async def stop(self) -> None: ...


class Agent(Protocol):
    """An agent that runs inside the benchmark process."""

    def handle(self, utterance: CallerUtterance) -> AgentReply | Awaitable[AgentReply]: ...


AgentFactory = Callable[[OrderBackend, SessionInfo], Agent]
