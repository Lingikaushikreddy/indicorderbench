"""View models for the HTML report: plain values the template only loops over."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Card:
    id: str
    label: str
    value: str
    sub: str
    tone: str = "neutral"  # neutral | warn | bad
    sup: str = ""


@dataclass(frozen=True)
class GroupRow:
    key: str
    trials: int
    passed: int
    rate: str
    ci: str
    band_style: str  # empty when there is no interval to draw
    dot_style: str


@dataclass(frozen=True)
class DeltaRow:
    id: str
    title: str
    baseline: str
    current: str
    delta: str
    status: str
    linked: bool


@dataclass(frozen=True)
class ComparisonView:
    verdict: str
    verdict_cls: str  # bad | ok
    headline: str
    tolerance: str
    baseline_label: str
    counts: str
    rows: list[DeltaRow]
    problems: list[str]


@dataclass(frozen=True)
class CheckRow:
    field: str
    expected: str
    actual: str
    cls: str  # pass | fail
    mark: str


@dataclass(frozen=True)
class TurnView:
    speaker: str  # caller | agent
    who: str
    tag: str
    text: str
    latency: str
    audio: str | None


@dataclass(frozen=True)
class CallView:
    seq: int
    t: str
    name: str
    args: str
    result: str
    title: str  # the full result when the cell is truncated
    error: bool


@dataclass(frozen=True)
class TrialView:
    number: int
    cls: str
    label: str
    duration: str
    summary: str
    error: str
    invalid_reason: str
    max_turns_hit: bool
    orders: list[str]
    checks: list[CheckRow]
    turns: list[TurnView]
    calls: list[CallView]


@dataclass(frozen=True)
class ScenarioView:
    id: str
    title: str
    meta: str
    tags: list[str]
    rate: str
    counts: str
    cls: str
    open: bool
    chips: list[str]
    trials: list[TrialView]


@dataclass(frozen=True)
class PageView:
    title: str
    pack: str
    agent: str
    facts: list[tuple[str, str]]
    cards: list[Card]
    by_language: list[GroupRow]
    by_category: list[GroupRow]
    comparison: ComparisonView | None
    scenarios: list[ScenarioView]
    footer: list[str] = field(default_factory=list)
