"""Trial and suite execution.

A trial is one conversation between the scripted caller and the agent under test, scored
by the checker. A suite runs every selected scenario for the configured number of trials
and aggregates the metrics.
"""

from __future__ import annotations

import asyncio
import platform
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indicorderbench import __version__
from indicorderbench.adapters.protocol import (
    AgentAdapter,
    AgentReply,
    CallerUtterance,
    Modality,
    SessionInfo,
)
from indicorderbench.backend.state import OrderBackend
from indicorderbench.caller.scripted import ScriptedCaller
from indicorderbench.checker.checker import check
from indicorderbench.packs import Pack
from indicorderbench.runner.stats import compute_metrics
from indicorderbench.schemas.results import (
    AgentInfo,
    Outcome,
    PackInfo,
    RunInfo,
    ScenarioResult,
    SuiteResult,
    TrialResult,
    TurnRecord,
)
from indicorderbench.schemas.scenario import Scenario


@dataclass
class RunConfig:
    trials: int = 1
    modality: Modality = Modality.TEXT
    seed: int | None = None
    timeout_turn_s: float = 30.0
    timeout_trial_s: float = 300.0
    max_turns: int = 20
    agent_label: str = "agent"
    agent_spec: str = ""


AdapterFactory = Callable[[], AgentAdapter]


class SessionRegistry:
    """Thread-safe map from session id to backend, shared with the HTTP backend server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, OrderBackend] = {}

    def register(self, session_id: str, backend: OrderBackend) -> None:
        with self._lock:
            self._sessions[session_id] = backend

    def get(self, session_id: str) -> OrderBackend:
        with self._lock:
            return self._sessions[session_id]

    def unregister(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


class MissingClipError(Exception):
    pass


def _build_utterance(
    scenario: Scenario, turn_id: str, text: str, clip: Path | None, modality: Modality
) -> CallerUtterance:
    if modality is not Modality.TEXT and clip is None:
        raise MissingClipError(
            f"no audio clip for {scenario.id}/{turn_id}; run `iob synth` or use --modality text"
        )
    send_audio = modality is not Modality.TEXT and clip is not None
    return CallerUtterance(
        turn_id=turn_id,
        text=None if modality is Modality.AUDIO else text,
        audio_path=clip if send_audio else None,
        audio_bytes=clip.read_bytes() if (send_audio and clip is not None) else None,
        language=scenario.language.value,
    )


async def run_trial(
    pack: Pack,
    scenario: Scenario,
    adapter: AgentAdapter,
    config: RunConfig,
    trial: int,
    backend_url: str | None = None,
    registry: SessionRegistry | None = None,
) -> TrialResult:
    started = datetime.now(UTC)
    t0 = time.perf_counter()

    def clock() -> float:
        return (time.perf_counter() - t0) * 1000.0

    backend = OrderBackend(pack.menu, clock=clock)
    session_id = f"{scenario.id}-{trial}-{uuid.uuid4().hex[:8]}"
    if registry is not None:
        registry.register(session_id, backend)
    session = SessionInfo(
        scenario_id=scenario.id,
        trial=trial,
        language=scenario.language.value,
        modality=config.modality,
        backend=backend,
        session_id=session_id,
        backend_url=backend_url,
    )
    caller = ScriptedCaller(scenario.caller.resolved(pack.defaults_for(scenario.language)))
    transcript: list[TurnRecord] = []
    error: str | None = None
    max_turns_hit = False
    agent_meta: dict[str, Any] = {}

    async def converse() -> None:
        nonlocal max_turns_hit
        await adapter.start(session)
        move = caller.first_move()
        turns = 0
        while True:
            turn = move.turn
            turn_id = turn.id or "turn"
            clip = pack.clip_path(scenario.id, turn, scenario.language)
            utterance = _build_utterance(scenario, turn_id, turn.text, clip, config.modality)
            transcript.append(
                TurnRecord(
                    index=len(transcript),
                    speaker="caller",
                    text=turn.text,
                    audio_path=str(utterance.audio_path) if utterance.audio_path else None,
                    t_ms=clock(),
                    turn_id=turn_id,
                    source=move.source,
                )
            )
            sent = time.perf_counter()
            try:
                reply: AgentReply = await asyncio.wait_for(
                    adapter.respond(utterance), timeout=config.timeout_turn_s
                )
            except TimeoutError as e:
                raise RuntimeError(
                    f"agent timed out after {config.timeout_turn_s}s on turn {turn_id}"
                ) from e
            latency = (time.perf_counter() - sent) * 1000.0
            transcript.append(
                TurnRecord(
                    index=len(transcript),
                    speaker="agent",
                    text=reply.text,
                    t_ms=clock(),
                    latency_ms=latency,
                )
            )
            if reply.meta:
                agent_meta.update(reply.meta)
            turns += 1
            if caller.should_end(reply.text, bool(backend.active_orders())):
                return
            if turns >= config.max_turns:
                max_turns_hit = True
                return
            nxt = caller.next_move(reply.text)
            if nxt is None:
                return
            move = nxt

    try:
        try:
            await asyncio.wait_for(converse(), timeout=config.timeout_trial_s)
        except TimeoutError:
            error = f"trial timed out after {config.timeout_trial_s}s"
        except MissingClipError as e:
            error = str(e)
        except Exception as e:
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    finally:
        try:
            await asyncio.wait_for(adapter.stop(), timeout=config.timeout_turn_s)
        except Exception as e:
            error = error or f"adapter.stop failed: {type(e).__name__}: {e}"
        if registry is not None:
            registry.unregister(session_id)

    snapshot = backend.snapshot()
    duration = (time.perf_counter() - t0) * 1000.0
    if error is not None:
        outcome, result = Outcome.INFRA_ERROR, None
    elif caller.invalid:
        outcome, result = Outcome.SIMULATOR_INVALID, None
    else:
        result = check(scenario.expected, snapshot, pack.menu)
        outcome = Outcome.PASS if result.passed else Outcome.FAIL
    return TrialResult(
        scenario_id=scenario.id,
        trial=trial,
        outcome=outcome,
        check=result,
        transcript=transcript,
        snapshot=snapshot,
        caller_valid=not caller.invalid,
        caller_invalid_reason=caller.invalid_reason,
        error=error,
        max_turns_hit=max_turns_hit,
        started_at=started,
        duration_ms=duration,
        agent_meta=dict(agent_meta),
    )


async def run_suite(
    pack: Pack,
    scenarios: list[Scenario],
    adapter_factory: AdapterFactory,
    config: RunConfig,
    backend_url: str | None = None,
    registry: SessionRegistry | None = None,
    on_scenario: Callable[[ScenarioResult], None] | None = None,
) -> SuiteResult:
    started = datetime.now(UTC)
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        trials = []
        for n in range(1, config.trials + 1):
            adapter = adapter_factory()
            trials.append(
                await run_trial(pack, scenario, adapter, config, n, backend_url, registry)
            )
        sr = ScenarioResult(
            scenario_id=scenario.id,
            title=scenario.title,
            language=scenario.language.value,
            category=scenario.category.value,
            tags=list(scenario.tags),
            trials=trials,
        )
        results.append(sr)
        if on_scenario:
            on_scenario(sr)
    finished = datetime.now(UTC)
    return SuiteResult(
        run=RunInfo(
            started_at=started,
            finished_at=finished,
            trials=config.trials,
            modality=config.modality.value,
            seed=config.seed,
            host=socket.gethostname(),
            benchmark_version=__version__,
            python=platform.python_version(),
            max_turns=config.max_turns,
            timeout_turn_s=config.timeout_turn_s,
            timeout_trial_s=config.timeout_trial_s,
        ),
        pack=PackInfo(
            id=pack.manifest.id,
            version=pack.manifest.version,
            content_hash=pack.content_hash(),
            path=str(pack.root),
            clips=str(pack.clips_dir) if pack.clips_dir is not None else None,
        ),
        agent=AgentInfo(label=config.agent_label, spec=config.agent_spec),
        scenarios=results,
        metrics=compute_metrics(results, config.trials),
    )


def run_suite_sync(
    pack: Pack,
    scenarios: list[Scenario],
    adapter_factory: AdapterFactory,
    config: RunConfig,
    backend_url: str | None = None,
    registry: SessionRegistry | None = None,
    on_scenario: Callable[[ScenarioResult], None] | None = None,
) -> SuiteResult:
    return asyncio.run(
        run_suite(pack, scenarios, adapter_factory, config, backend_url, registry, on_scenario)
    )
