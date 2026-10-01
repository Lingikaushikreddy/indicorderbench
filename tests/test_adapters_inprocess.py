import asyncio

from indicorderbench.adapters.inprocess import InProcessAdapter
from indicorderbench.adapters.protocol import (
    AgentAdapter,
    AgentReply,
    CallerUtterance,
    Modality,
    SessionInfo,
)
from indicorderbench.backend.state import OrderBackend
from tests.test_schemas import make_menu


class SyncAgent:
    def __init__(self, backend: OrderBackend) -> None:
        self.backend = backend
        self.closed = False

    def handle(self, u: CallerUtterance) -> AgentReply:
        self.backend.add_item("mango_lassi")
        return AgentReply(text=f"got {u.text}")

    def close(self) -> None:
        self.closed = True


class AsyncAgent:
    def __init__(self, backend: OrderBackend) -> None:
        self.backend = backend
        self.closed = False

    async def handle(self, u: CallerUtterance) -> AgentReply:
        await asyncio.sleep(0)
        return AgentReply(text="async", meta={"cost_usd": 0.01})

    async def close(self) -> None:
        self.closed = True


def session(backend: OrderBackend) -> SessionInfo:
    return SessionInfo(
        scenario_id="s",
        trial=1,
        language="en-IN",
        modality=Modality.TEXT,
        backend=backend,
        session_id="sess",
    )


def utterance(text: str) -> CallerUtterance:
    return CallerUtterance(
        turn_id="t1", text=text, audio_path=None, audio_bytes=None, language="en-IN"
    )


async def test_sync_agent_roundtrip_and_close():
    backend = OrderBackend(make_menu())
    agents: list[SyncAgent] = []

    def factory(b: OrderBackend, s: SessionInfo) -> SyncAgent:
        a = SyncAgent(b)
        agents.append(a)
        return a

    adapter = InProcessAdapter(factory)
    assert isinstance(adapter, AgentAdapter)
    await adapter.start(session(backend))
    reply = await adapter.respond(utterance("hi"))
    assert reply.text == "got hi" and len(backend.get_cart()) == 1
    await adapter.stop()
    assert agents[0].closed


async def test_async_agent_and_async_close():
    created: list[AsyncAgent] = []

    def factory(b: OrderBackend, s: SessionInfo) -> AsyncAgent:
        a = AsyncAgent(b)
        created.append(a)
        return a

    adapter = InProcessAdapter(factory)
    await adapter.start(session(OrderBackend(make_menu())))
    reply = await adapter.respond(utterance("hi"))
    assert reply.text == "async" and reply.meta["cost_usd"] == 0.01
    await adapter.stop()
    assert created[0].closed


async def test_respond_before_start_raises():
    adapter = InProcessAdapter(lambda b, s: SyncAgent(b))
    try:
        await adapter.respond(utterance("hi"))
    except RuntimeError as e:
        assert "not started" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


async def test_fresh_agent_per_start():
    count = {"n": 0}

    def factory(b: OrderBackend, s: SessionInfo) -> SyncAgent:
        count["n"] += 1
        return SyncAgent(b)

    adapter = InProcessAdapter(factory)
    await adapter.start(session(OrderBackend(make_menu())))
    await adapter.stop()
    await adapter.start(session(OrderBackend(make_menu())))
    assert count["n"] == 2
