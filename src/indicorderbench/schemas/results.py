"""Result schemas: backend state, checks, trials, suites."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SIMULATOR_INVALID = "simulator_invalid"
    INFRA_ERROR = "infra_error"


class CartLine(BaseModel):
    line_id: str
    item_id: str
    quantity: int = Field(ge=0)
    modifiers: set[str] = Field(default_factory=set)
    note: str | None = None

    @field_serializer("modifiers")
    def _sorted_modifiers(self, value: set[str]) -> list[str]:
        """Sets iterate in hash-seed order; sort so results.json is byte-stable."""
        return sorted(value)


class SubmittedOrder(BaseModel):
    order_id: str
    lines: list[CartLine]
    submitted_at_ms: float
    status: Literal["submitted", "cancelled"] = "submitted"


class ToolCall(BaseModel):
    seq: int
    t_ms: float
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class BackendSnapshot(BaseModel):
    cart: list[CartLine] = Field(default_factory=list)
    orders: list[SubmittedOrder] = Field(default_factory=list)
    trace: list[ToolCall] = Field(default_factory=list)

    def active_orders(self) -> list[SubmittedOrder]:
        return [o for o in self.orders if o.status == "submitted"]


class FieldCheck(BaseModel):
    field: str
    expected: str
    actual: str
    passed: bool


class CheckResult(BaseModel):
    passed: bool
    best_end_state_id: str
    field_checks: list[FieldCheck]
    summary: str


CallerSource = Literal["script", "clarification", "confirm", "closing", "fallback", "nudge"]


class TurnRecord(BaseModel):
    index: int
    speaker: Literal["caller", "agent"]
    text: str | None
    audio_path: str | None = None
    t_ms: float
    latency_ms: float | None = None
    turn_id: str | None = None
    source: CallerSource | None = None


class TrialResult(BaseModel):
    scenario_id: str
    trial: int
    outcome: Outcome
    check: CheckResult | None = None
    transcript: list[TurnRecord] = Field(default_factory=list)
    snapshot: BackendSnapshot | None = None
    caller_valid: bool = True
    caller_invalid_reason: str | None = None
    error: str | None = None
    max_turns_hit: bool = False
    started_at: datetime
    duration_ms: float
    agent_meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.outcome in (Outcome.PASS, Outcome.FAIL)

    @property
    def latencies_ms(self) -> list[float]:
        return [
            t.latency_ms
            for t in self.transcript
            if t.speaker == "agent" and t.latency_ms is not None
        ]


class ScenarioResult(BaseModel):
    scenario_id: str
    title: str
    language: str
    category: str
    tags: list[str] = Field(default_factory=list)
    trials: list[TrialResult]

    @property
    def n_valid(self) -> int:
        return sum(1 for t in self.trials if t.is_valid)

    @property
    def n_pass(self) -> int:
        return sum(1 for t in self.trials if t.outcome is Outcome.PASS)

    @property
    def pass_rate(self) -> float | None:
        return None if self.n_valid == 0 else self.n_pass / self.n_valid

    @property
    def worst_outcome(self) -> Outcome:
        order = [Outcome.INFRA_ERROR, Outcome.FAIL, Outcome.SIMULATOR_INVALID, Outcome.PASS]
        present = {t.outcome for t in self.trials}
        for o in order:
            if o in present:
                return o
        return Outcome.INFRA_ERROR


class GroupMetric(BaseModel):
    key: str
    n_trials: int
    n_pass: int
    pass_rate: float | None
    ci_low: float | None
    ci_high: float | None


class Metrics(BaseModel):
    n_scenarios: int
    n_trials_total: int
    n_valid: int
    n_pass: int
    n_fail: int
    n_simulator_invalid: int
    n_infra_error: int
    n_max_turns: int
    pass_rate: float | None
    pass_k: dict[int, float | None] = Field(default_factory=dict)
    by_language: list[GroupMetric] = Field(default_factory=list)
    by_category: list[GroupMetric] = Field(default_factory=list)
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None


class RunInfo(BaseModel):
    started_at: datetime
    finished_at: datetime
    trials: int
    modality: str
    seed: int | None
    host: str
    benchmark_version: str
    python: str
    max_turns: int
    timeout_turn_s: float
    timeout_trial_s: float


class PackInfo(BaseModel):
    id: str
    version: str
    content_hash: str
    path: str
    clips: str | None = None  # an alternate clip directory (``--clips``), when one was used


class AgentInfo(BaseModel):
    label: str
    spec: str


class SuiteResult(BaseModel):
    schema_version: int = 1
    run: RunInfo
    pack: PackInfo
    agent: AgentInfo
    scenarios: list[ScenarioResult]
    metrics: Metrics
