"""Shared test helpers.

``make_suite`` builds a realistic :class:`SuiteResult` without running an agent, so report,
compare and CLI tests can exercise every outcome. Each trial carries what the runner would
record for that outcome:

- PASS: a four-turn transcript (caller/agent with latencies), a backend snapshot with one
  submitted order and its tool-call trace, and a passing ``CheckResult``.
- FAIL: the same conversation, but the agent committed two Paneer Wraps; the ``CheckResult``
  holds the failing field check ``Paneer Wrap quantity`` (expected ``1``, actual ``2``).
- SIMULATOR_INVALID: the agent asked something the script cannot answer; ``caller_valid`` is
  False, ``caller_invalid_reason`` is set and there is no check.
- INFRA_ERROR: only the first caller turn (the agent never replied), an ``error`` string and
  no check.

Scenario language and category come from the id: ``<prefix>_<category>_<nn>`` where the
prefix is ``en`` (en-IN) or ``hien`` (hi-en) and the category may itself contain underscores
(``hien_duplicate_submission_01`` -> ``duplicate_submission``). Metrics come from the real
``compute_metrics``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from indicorderbench import __version__
from indicorderbench.runner.stats import compute_metrics
from indicorderbench.schemas.results import (
    AgentInfo,
    BackendSnapshot,
    CartLine,
    CheckResult,
    FieldCheck,
    Outcome,
    PackInfo,
    RunInfo,
    ScenarioResult,
    SubmittedOrder,
    SuiteResult,
    ToolCall,
    TrialResult,
    TurnRecord,
)

BASE_TIME = datetime(2026, 10, 1, 9, 30, tzinfo=UTC)
LANGUAGE_BY_PREFIX = {"en": "en-IN", "hien": "hi-en"}
PACK_HASH = "3f2a9c1d7be04e65a8c1f0d2b6e9a4c7d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6"
INFRA_ERROR_TEXT = "RuntimeError: agent timed out after 30.0s on turn t1"
INVALID_REASON = "agent asked a question the script cannot answer: 'Which size?'"


def scenario_language_and_category(scenario_id: str) -> tuple[str, str]:
    """Derive (language, category) from an id such as ``en_quantity_01``."""
    prefix, *middle, _number = scenario_id.split("_")
    if prefix not in LANGUAGE_BY_PREFIX or not middle:
        raise ValueError(f"scenario id {scenario_id!r} is not <en|hien>_<category>_<nn>")
    return LANGUAGE_BY_PREFIX[prefix], "_".join(middle)


def _turn(index: int, speaker: str, text: str, t_ms: float, **kw: object) -> TurnRecord:
    return TurnRecord.model_validate(
        {"index": index, "speaker": speaker, "text": text, "t_ms": t_ms, **kw}
    )


def _check(quantity: int) -> CheckResult:
    checks = [
        FieldCheck(field="Submitted orders", expected="1", actual="1", passed=True),
        FieldCheck(
            field="Paneer Wrap quantity", expected="1", actual=str(quantity), passed=quantity == 1
        ),
        FieldCheck(
            field="Paneer Wrap · Onion", expected="No onion", actual="No onion", passed=True
        ),
    ]
    if quantity == 1:
        return CheckResult(
            passed=True, best_end_state_id="e1", field_checks=checks, summary="matched end state e1"
        )
    return CheckResult(
        passed=False,
        best_end_state_id="e1",
        field_checks=checks,
        summary=f"1 field(s) wrong vs e1: Paneer Wrap quantity expected 1 got {quantity}",
    )


def _ordered_snapshot(quantity: int) -> BackendSnapshot:
    line = CartLine(line_id="l1", item_id="paneer_wrap", quantity=quantity, modifiers={"no_onion"})
    return BackendSnapshot(
        orders=[SubmittedOrder(order_id="o1", lines=[line], submitted_at_ms=2310.0)],
        trace=[
            ToolCall(
                seq=1,
                t_ms=420.5,
                name="add_item",
                args={"item_id": "paneer_wrap", "quantity": quantity, "modifiers": ["no_onion"]},
                result={"line_id": "l1"},
            ),
            ToolCall(seq=2, t_ms=2310.0, name="submit_order", args={}, result={"order_id": "o1"}),
        ],
    )


def make_trial(
    scenario_id: str, trial: int, outcome: Outcome, with_clip: Path | None = None
) -> TrialResult:
    """One trial with the transcript, snapshot and check the runner records for ``outcome``."""
    started = BASE_TIME + timedelta(seconds=10 * trial)
    clip = {"audio_path": str(with_clip)} if with_clip is not None else {}
    first = _turn(
        0,
        "caller",
        "One paneer wrap, no onion, please.",
        0.0,
        turn_id="t1",
        source="script",
        **clip,
    )
    latency = 140.0 + 10 * trial
    if outcome is Outcome.INFRA_ERROR:
        return TrialResult(
            scenario_id=scenario_id,
            trial=trial,
            outcome=outcome,
            transcript=[first],
            snapshot=BackendSnapshot(),
            error=INFRA_ERROR_TEXT,
            started_at=started,
            duration_ms=30_000.0,
        )
    if outcome is Outcome.SIMULATOR_INVALID:
        reply = _turn(1, "agent", "Which size would you like?", 160.0, latency_ms=latency)
        return TrialResult(
            scenario_id=scenario_id,
            trial=trial,
            outcome=outcome,
            transcript=[first, reply],
            snapshot=BackendSnapshot(),
            caller_valid=False,
            caller_invalid_reason=INVALID_REASON,
            started_at=started,
            duration_ms=900.0,
        )
    quantity = 1 if outcome is Outcome.PASS else 2
    wraps = "One Paneer Wrap" if quantity == 1 else "Two Paneer Wraps"
    transcript = [
        first,
        _turn(1, "agent", f"{wraps}, no onion. Anything else?", 480.0, latency_ms=latency),
        _turn(
            2, "caller", "That's all, please place the order.", 900.0, turn_id="t2", source="script"
        ),
        _turn(3, "agent", "Done, your order is placed.", 2400.0, latency_ms=latency + 260.0),
    ]
    return TrialResult(
        scenario_id=scenario_id,
        trial=trial,
        outcome=outcome,
        check=_check(quantity),
        transcript=transcript,
        snapshot=_ordered_snapshot(quantity),
        started_at=started,
        duration_ms=2450.0,
        agent_meta={"model": "rule_based"},
    )


def make_suite(
    outcomes_by_scenario: dict[str, list[Outcome]],
    pack_id: str = "starter",
    agent_label: str = "agent",
    with_clip: Path | None = None,
) -> SuiteResult:
    """A SuiteResult with one scenario per key and one trial per listed outcome.

    ``with_clip`` sets ``audio_path`` on every trial's first caller turn (turn id ``t1``).
    """
    scenarios = []
    for sid, outcomes in outcomes_by_scenario.items():
        language, category = scenario_language_and_category(sid)
        number = sid.rsplit("_", 1)[1]
        scenarios.append(
            ScenarioResult(
                scenario_id=sid,
                title=f"{category.replace('_', ' ').capitalize()} {number} ({language})",
                language=language,
                category=category,
                tags=["smoke"],
                trials=[
                    make_trial(sid, n, outcome, with_clip)
                    for n, outcome in enumerate(outcomes, start=1)
                ],
            )
        )
    trials = max((len(o) for o in outcomes_by_scenario.values()), default=1)
    return SuiteResult(
        run=RunInfo(
            started_at=BASE_TIME,
            finished_at=BASE_TIME + timedelta(minutes=2, seconds=5),
            trials=trials,
            modality="text",
            seed=7,
            host="ci-runner",
            benchmark_version=__version__,
            python="3.11.9",
            max_turns=20,
            timeout_turn_s=30.0,
            timeout_trial_s=300.0,
        ),
        pack=PackInfo(id=pack_id, version="0.1.0", content_hash=PACK_HASH, path=f"packs/{pack_id}"),
        agent=AgentInfo(label=agent_label, spec="builtin:correct"),
        scenarios=scenarios,
        metrics=compute_metrics(scenarios, trials),
    )


# ---------------------------------------------------------------------------------------
# Tiny in-process agents for the fixture mini pack, used by CLI tests through the
# ``python:tests.helpers:<factory>`` agent spec.
# ---------------------------------------------------------------------------------------


class MiniPackAgent:
    """Handles the two minipack scenarios with keyword rules; ``wrap_quantity`` injects a bug."""

    def __init__(self, backend: object, wrap_quantity: int = 2) -> None:
        from indicorderbench.backend.state import OrderBackend

        assert isinstance(backend, OrderBackend)
        self.backend = backend
        self.wrap_quantity = wrap_quantity

    def handle(self, utterance: object) -> object:
        from indicorderbench.adapters.protocol import AgentReply, CallerUtterance

        assert isinstance(utterance, CallerUtterance)
        text = (utterance.text or "").lower()
        if "cancel" in text:
            for order in self.backend.active_orders():
                self.backend.cancel_order(order.order_id)
            self.backend.clear_cart()
            return AgentReply(text="Cancelled. Anything else?")
        if "paneer" in text:
            self.backend.add_item("paneer_wrap", self.wrap_quantity)
            return AgentReply(text="Added paneer wraps. Anything else?")
        if "lassi" in text:
            self.backend.add_item("mango_lassi", 1)
            self.backend.submit_order()
            return AgentReply(text="Added a lassi. Anything else?")
        if "all" in text or "place" in text:
            if self.backend.get_cart():
                self.backend.submit_order()
                return AgentReply(text="Order placed, thank you!")
            if self.backend.active_orders():
                return AgentReply(text="Your order is already placed.")
            return AgentReply(text="Nothing to place. Anything else?")
        return AgentReply(text="Anything else?")


def make_minipack_agent(backend: object, session: object) -> MiniPackAgent:
    return MiniPackAgent(backend)


def make_minipack_buggy_agent(backend: object, session: object) -> MiniPackAgent:
    return MiniPackAgent(backend, wrap_quantity=1)


def make_minipack_agent_broken_on_cancellation(backend: object, session: object) -> MiniPackAgent:
    """Works on en_quantity_01; raises while starting en_cancellation_01 (an infra error)."""
    from indicorderbench.adapters.protocol import SessionInfo

    assert isinstance(session, SessionInfo)
    if session.scenario_id == "en_cancellation_01":
        raise RuntimeError("agent could not start a session for en_cancellation_01")
    return MiniPackAgent(backend)
