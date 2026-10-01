"""Adapter for agents that run inside the benchmark process."""

from __future__ import annotations

import inspect

from indicorderbench.adapters.protocol import (
    Agent,
    AgentFactory,
    AgentReply,
    CallerUtterance,
    SessionInfo,
)


class InProcessAdapter:
    """Builds a fresh agent per session from ``agent_factory(backend, session)``."""

    def __init__(self, agent_factory: AgentFactory) -> None:
        self._factory = agent_factory
        self._agent: Agent | None = None

    async def start(self, session: SessionInfo) -> None:
        self._agent = self._factory(session.backend, session)

    async def respond(self, utterance: CallerUtterance) -> AgentReply:
        if self._agent is None:
            raise RuntimeError("adapter not started")
        result = self._agent.handle(utterance)
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
