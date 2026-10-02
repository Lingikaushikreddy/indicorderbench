import asyncio
import threading
import time
from pathlib import Path
from typing import Any

from indicorderbench.adapters.inprocess import InProcessAdapter
from indicorderbench.adapters.protocol import (
    AgentReply,
    CallerUtterance,
    Modality,
    SessionInfo,
)
from indicorderbench.backend.state import OrderBackend
from indicorderbench.packs import load_pack
from indicorderbench.runner.runner import RunConfig, SessionRegistry, run_suite, run_trial
from indicorderbench.schemas.results import Outcome
from indicorderbench.schemas.scenario import CallerTurn, ClarificationRule

FIX = Path(__file__).parent / "fixtures" / "minipack"


class ScriptedAgent:
    """Adapter and agent in one: replies from a list; optional backend actions per turn."""

    def __init__(
        self,
        replies: list[str],
        actions: dict[int, str] | None = None,
        hang_turn: int | None = None,
        raise_turn: int | None = None,
        stop_raises: bool = False,
    ) -> None:
        self.replies = replies
        self.actions = actions or {}
        self.hang_turn = hang_turn
        self.raise_turn = raise_turn
        self.stop_raises = stop_raises
        self.n = 0
        self.seen: list[CallerUtterance] = []
        self.session: SessionInfo | None = None
        self.backend: OrderBackend | None = None

    async def start(self, session: SessionInfo) -> None:
        self.session = session
        self.backend = session.backend

    async def respond(self, u: CallerUtterance) -> AgentReply:
        assert self.backend is not None
        self.seen.append(u)
        self.n += 1
        if self.n == self.hang_turn:
            await asyncio.sleep(10)
        if self.n == self.raise_turn:
            raise RuntimeError("agent exploded")
        action = self.actions.get(self.n)
        if action == "add2":
            self.backend.add_item("paneer_wrap", 2)
        elif action == "add1":
            self.backend.add_item("paneer_wrap", 1)
        elif action == "submit":
            self.backend.submit_order()
        elif action == "lassi":
            self.backend.add_item("mango_lassi", 1)
        elif action == "cancel":
            for o in self.backend.active_orders():
                self.backend.cancel_order(o.order_id)
        return AgentReply(
            text=self.replies[min(self.n - 1, len(self.replies) - 1)], meta={"turns": self.n}
        )

    async def stop(self) -> None:
        if self.stop_raises:
            raise RuntimeError("stop failed")


def cfg(**kw: Any) -> RunConfig:
    return RunConfig(**{"timeout_turn_s": 0.2, "timeout_trial_s": 2.0, **kw})


async def test_pass_trial():
    pack = load_pack(FIX)
    agent = ScriptedAgent(
        ["Added two paneer wraps. Anything else?", "Order placed, thank you!"],
        {1: "add2", 2: "submit"},
    )
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.PASS and r.check is not None and r.check.passed
    assert [t.source for t in r.transcript if t.speaker == "caller"] == ["script", "closing"]
    assert [t.speaker for t in r.transcript] == ["caller", "agent", "caller", "agent"]
    assert r.transcript[1].latency_ms is not None and r.snapshot is not None
    assert len(r.snapshot.trace) == 2 and r.caller_valid and r.error is None
    assert r.agent_meta == {"turns": 2} and r.duration_ms > 0
    assert agent.session is not None and agent.session.scenario_id == "en_quantity_01"
    assert agent.seen[0].text == "Two paneer wraps please." and agent.seen[0].audio_bytes is None


async def test_fail_trial_with_field_checks():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["Added. Anything else?", "Order placed!"], {1: "add1", 2: "submit"})
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.FAIL and r.check is not None
    assert any(f.field == "Paneer Wrap quantity" and not f.passed for f in r.check.field_checks)


async def test_cancellation_scenario_passes_when_order_cancelled():
    pack = load_pack(FIX)
    agent = ScriptedAgent(
        ["Added a lassi. Anything else?", "Cancelled. Anything else?", "Okay, bye!"],
        {1: "lassi", 2: "cancel"},
    )
    agent.actions[1] = "lassi"
    # submit right after adding so there is something to cancel

    async def respond(u: CallerUtterance) -> AgentReply:
        reply = await ScriptedAgent.respond(agent, u)
        if agent.n == 1:
            assert agent.backend is not None
            agent.backend.submit_order()
        return reply

    agent.respond = respond  # type: ignore[method-assign]
    r = await run_trial(pack, pack.scenario("en_cancellation_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.PASS


async def test_simulator_invalid_when_agent_keeps_asking():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["Which branch?"] * 10)
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.SIMULATOR_INVALID and not r.caller_valid and r.check is None
    assert r.caller_invalid_reason and "question" in r.caller_invalid_reason


async def test_no_submission_after_nudges_is_fail():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["Okay."] * 10, {1: "add2"})
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.FAIL and r.caller_valid and r.check is not None
    first = r.check.field_checks[0]
    assert first.field == "Submitted orders" and first.actual == "0"
    assert [t.source for t in r.transcript if t.speaker == "caller"] == [
        "script",
        "closing",
        "nudge",
        "nudge",
    ]


async def test_timeout_and_exception_are_infra_errors():
    pack = load_pack(FIX)
    hang = ScriptedAgent(["x"], hang_turn=1)
    r = await run_trial(pack, pack.scenario("en_quantity_01"), hang, cfg(), trial=1)
    assert r.outcome is Outcome.INFRA_ERROR and "timed out" in (r.error or "")
    boom = ScriptedAgent(["x"], raise_turn=1)
    r = await run_trial(pack, pack.scenario("en_quantity_01"), boom, cfg(), trial=1)
    assert r.outcome is Outcome.INFRA_ERROR and "agent exploded" in (r.error or "")
    assert r.snapshot is not None and r.transcript[0].speaker == "caller"


async def test_slow_sync_inprocess_agent_hits_the_turn_timeout():
    """A blocking sync ``handle`` must not stall the event loop past ``timeout_turn_s``."""
    pack = load_pack(FIX)
    release = threading.Event()

    class SlowSyncAgent:
        def handle(self, u: CallerUtterance) -> AgentReply:
            release.wait(3)  # a 3 s blocking call; released at the end so the thread exits
            return AgentReply(text="Order placed!")

    adapter = InProcessAdapter(lambda backend, session: SlowSyncAgent())
    t0 = time.perf_counter()
    try:
        r = await run_trial(
            pack,
            pack.scenario("en_quantity_01"),
            adapter,
            RunConfig(timeout_turn_s=0.5, timeout_trial_s=5),
            trial=1,
        )
        elapsed = time.perf_counter() - t0
    finally:
        release.set()
    assert r.outcome is Outcome.INFRA_ERROR and "timed out" in (r.error or "")
    assert elapsed < 2.0, f"run_trial took {elapsed:.2f}s"


async def test_stop_failure_is_infra_error():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["Added. Order placed!"], {1: "add2"}, stop_raises=True)
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.INFRA_ERROR and "stop failed" in (r.error or "")


async def test_max_turns_hit_is_fail_not_invalid():
    pack = load_pack(FIX)
    scen = pack.scenario("en_quantity_01").model_copy(deep=True)
    scen.caller.clarifications = [
        ClarificationRule(id="loop", match=["."], reply=CallerTurn(text="two"), max_uses=99)
    ]
    agent = ScriptedAgent(["hmm?"] * 50)
    r = await run_trial(pack, scen, agent, cfg(max_turns=6), trial=1)
    assert r.outcome is Outcome.FAIL and r.max_turns_hit and r.caller_valid
    assert sum(1 for t in r.transcript if t.speaker == "agent") == 6


async def test_audio_modality_missing_clip_is_infra_error():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["x"])
    r = await run_trial(
        pack, pack.scenario("en_quantity_01"), agent, cfg(modality=Modality.AUDIO), trial=1
    )
    assert r.outcome is Outcome.INFRA_ERROR and "clip" in (r.error or "")


async def test_audio_modality_sends_bytes_and_strips_text(tmp_path: Path):
    import shutil

    shutil.copytree(FIX, tmp_path / "p")
    clip = tmp_path / "p" / "clips" / "en_quantity_01" / "t1.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFFfake")
    closing = tmp_path / "p" / "clips" / "_defaults" / "en-IN" / "closing.wav"
    closing.parent.mkdir(parents=True)
    closing.write_bytes(b"RIFFclose")
    pack = load_pack(tmp_path / "p")
    agent = ScriptedAgent(["Added. Anything else?", "Order placed!"], {1: "add2", 2: "submit"})
    r = await run_trial(
        pack, pack.scenario("en_quantity_01"), agent, cfg(modality=Modality.AUDIO), trial=1
    )
    assert r.outcome is Outcome.PASS
    assert agent.seen[0].text is None and agent.seen[0].audio_bytes == b"RIFFfake"
    assert agent.seen[1].audio_bytes == b"RIFFclose"
    assert r.transcript[0].audio_path == str(clip) and r.transcript[0].text is not None
    both = ScriptedAgent(["Added. Anything else?", "Order placed!"], {1: "add2", 2: "submit"})
    await run_trial(
        pack, pack.scenario("en_quantity_01"), both, cfg(modality=Modality.BOTH), trial=1
    )
    assert both.seen[0].text == "Two paneer wraps please." and both.seen[0].audio_bytes


async def test_registry_registers_session_for_the_trial():
    pack = load_pack(FIX)
    registry = SessionRegistry()
    seen: dict[str, OrderBackend] = {}

    class Peek(ScriptedAgent):
        async def start(self, session: SessionInfo) -> None:
            await super().start(session)
            assert session.backend_url == "http://127.0.0.1:1"
            seen["backend"] = registry.get(session.session_id)

    agent = Peek(["Added. Anything else?", "Order placed!"], {1: "add2", 2: "submit"})
    r = await run_trial(
        pack,
        pack.scenario("en_quantity_01"),
        agent,
        cfg(),
        trial=1,
        backend_url="http://127.0.0.1:1",
        registry=registry,
    )
    assert seen["backend"] is agent.backend and agent.session is not None
    try:
        registry.get(agent.session.session_id)
    except KeyError:
        pass
    else:
        raise AssertionError("session should be unregistered after the trial")
    assert r.outcome is Outcome.PASS


async def test_run_suite_continues_after_infra_error_and_computes_metrics():
    pack = load_pack(FIX)
    calls = {"n": 0}

    def factory() -> ScriptedAgent:
        calls["n"] += 1
        if calls["n"] == 1:
            return ScriptedAgent(["x"], raise_turn=1)
        return ScriptedAgent(["Cancelled. Anything else?", "Order placed"], {1: "lassi"})

    seen: list[str] = []
    suite = await run_suite(
        pack,
        pack.scenarios,
        factory,
        cfg(trials=1, agent_label="t", agent_spec="test"),
        on_scenario=lambda sr: seen.append(sr.scenario_id),
    )
    assert suite.metrics.n_infra_error == 1 and suite.metrics.n_scenarios == 2
    assert suite.pack.id == "minipack" and suite.run.trials == 1
    assert suite.agent.label == "t" and suite.run.modality == "text"
    assert suite.scenarios[0].trials[0].outcome is Outcome.INFRA_ERROR
    assert seen == ["en_cancellation_01", "en_quantity_01"]
    assert suite.run.benchmark_version and suite.run.finished_at >= suite.run.started_at


def test_run_suite_sync_runs_multiple_trials():
    from indicorderbench.runner.runner import run_suite_sync

    pack = load_pack(FIX)
    suite = run_suite_sync(
        pack,
        pack.filter(ids=["en_quantity_01"]),
        lambda: ScriptedAgent(["Added. Anything else?", "Order placed!"], {1: "add2", 2: "submit"}),
        cfg(trials=3),
    )
    assert [t.trial for t in suite.scenarios[0].trials] == [1, 2, 3]
    assert suite.metrics.pass_k == {1: 1.0, 2: 1.0, 3: 1.0}


async def test_text_modality_records_no_audio_path_even_when_a_clip_exists(tmp_path: Path):
    """The transcript records what the agent was sent; a text run sends no audio, so the
    report must not show (or copy) clips the agent never heard."""
    import shutil

    shutil.copytree(FIX, tmp_path / "p")
    clip = tmp_path / "p" / "clips" / "en_quantity_01" / "t1.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFFfake")
    pack = load_pack(tmp_path / "p")
    agent = ScriptedAgent(["Added. Anything else?", "Order placed!"], {1: "add2", 2: "submit"})
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.PASS
    assert agent.seen[0].audio_bytes is None and agent.seen[0].audio_path is None
    assert all(t.audio_path is None for t in r.transcript)
