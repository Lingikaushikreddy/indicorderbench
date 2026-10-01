"""Adapter for agents that run inside the benchmark process."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable

from indicorderbench.adapters.protocol import (
    Agent,
    AgentFactory,
    AgentReply,
    CallerUtterance,
    SessionInfo,
)


class InProcessAdapter:
    """Builds a fresh agent per session from ``agent_factory(backend, session)``.

    An ``async def handle`` is awaited on the event loop. A plain ``def handle`` runs in a
    worker thread (``asyncio.to_thread``) so the runner's per-turn timeout can abandon a slow
    call. Python cannot interrupt a thread: a sync handler that times out keeps running in
    its worker thread until it returns, after the trial has already been scored
    ``infra_error``.
    """

    def __init__(self, agent_factory: AgentFactory) -> None:
        self._factory = agent_factory
        self._agent: Agent | None = None

    async def start(self, session: SessionInfo) -> None:
        self._agent = self._factory(session.backend, session)

    async def respond(self, utterance: CallerUtterance) -> AgentReply:
        if self._agent is None:
            raise RuntimeError("adapter not started")
        handle = self._agent.handle
        is_async = inspect.iscoroutinefunction(handle)  # a bool, so mypy keeps handle's type
        result: AgentReply | Awaitable[AgentReply]
        if is_async:
            result = handle(utterance)
        else:
            result = await asyncio.to_thread(handle, utterance)
        if inspect.isawaitable(result):
            return await result
        return result

    async def stop(self) -> None:
        agent, self._agent = self._agent, None
        close = getattr(agent, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
