"""Single-file HTML report (plus ``assets/`` for copied audio clips).

Everything the template shows is pre-computed here into the view models in
:mod:`indicorderbench.report.html_views`, so the Jinja template only loops and branches.
Autoescaping is on; the page uses no external URLs and works from ``file://``.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined

from indicorderbench.report.compare import Comparison, compare
from indicorderbench.report.formatting import DASH, ms, pct, signed_pct
from indicorderbench.report.html_views import (
    CallView,
    Card,
    CheckRow,
    ComparisonView,
    DeltaRow,
    GroupRow,
    PageView,
    ScenarioView,
    TrialView,
    TurnView,
)
from indicorderbench.schemas.results import (
    GroupMetric,
    Outcome,
    ScenarioResult,
    SuiteResult,
    TrialResult,
    TurnRecord,
)

OUTCOME_CLASS = {
    Outcome.PASS: "pass",
    Outcome.FAIL: "fail",
    Outcome.SIMULATOR_INVALID: "invalid",
    Outcome.INFRA_ERROR: "infra",
}
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")
_MAX_CELL = 160


class _ClipCopier:
    """Copies clips to ``<out>/assets/<scenario>/<turn>.wav`` and returns relative paths."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.by_source: dict[Path, str] = {}
        self.taken: set[str] = set()

    def copy(self, source: str, scenario_id: str, name: str) -> str | None:
        src = Path(source)
        if not src.is_file():
            return None
        key = src.resolve()
        if key in self.by_source:
            return self.by_source[key]
        folder = _UNSAFE_NAME.sub("_", scenario_id) or "scenario"
        stem = _UNSAFE_NAME.sub("_", name) or "clip"
        rel = f"assets/{folder}/{stem}.wav"
        n = 2
        while rel in self.taken:
            rel = f"assets/{folder}/{stem}-{n}.wav"
            n += 1
        dest = self.out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        self.taken.add(rel)
        self.by_source[key] = rel
        return rel


def _stamp(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _when(dt: datetime) -> str:
    zone = dt.tzname()
    return dt.strftime("%Y-%m-%d %H:%M:%S") + (f" {zone}" if zone else "")


def _span(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes} min {secs} s" if minutes else f"{seconds:.1f} s"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _short(text: str) -> str:
    return text if len(text) <= _MAX_CELL else text[: _MAX_CELL - 1] + "…"


def _cards(suite: SuiteResult) -> list[Card]:
    m, trials = suite.metrics, suite.run.trials
    p50, p95 = m.latency_p50_ms, m.latency_p95_ms
    # pass^k is None once any scenario has fewer than k valid trials: show the highest k
    # that has a value, and say why the run's own k is missing.
    available = [k for k, v in m.pass_k.items() if v is not None]
    k = max(available, default=trials)
    pass_k = m.pass_k.get(k)
    pass_k_sub = "chance a single trial passes" if k == 1 else f"chance all {k} trials pass"
    if k < trials:
        pass_k_sub += f" · pass^{trials} needs {trials} valid trials in every scenario"
    return [
        Card(
            "card-pass-rate",
            "Pass rate",
            pct(m.pass_rate),
            f"{m.n_pass} of {m.n_valid} valid trials passed · mean over scenarios",
        ),
        Card("card-pass-k", "pass", pct(pass_k), pass_k_sub, sup=str(k)),
        Card(
            "card-invalid",
            "Simulator invalid",
            str(m.n_simulator_invalid),
            "caller script could not continue · excluded",
            "warn" if m.n_simulator_invalid else "neutral",
        ),
        Card(
            "card-infra",
            "Infra errors",
            str(m.n_infra_error),
            f"of {m.n_trials_total} trials · excluded",
            "bad" if m.n_infra_error else "neutral",
        ),
        Card(
            "card-latency",
            "Latency p50 / p95",
            DASH if p50 is None or p95 is None else f"{p50:.0f} / {p95:.0f}",
            "milliseconds per agent reply",
        ),
    ]


def _group_rows(groups: list[GroupMetric]) -> list[GroupRow]:
    rows = []
    for g in groups:
        has_ci = g.ci_low is not None and g.ci_high is not None and g.pass_rate is not None
        lo, hi, rate = g.ci_low or 0.0, g.ci_high or 0.0, g.pass_rate or 0.0
        rows.append(
            GroupRow(
                key=g.key.replace("_", " "),
                trials=g.n_trials,
                passed=g.n_pass,
                rate=pct(g.pass_rate),
                ci=f"{pct(g.ci_low)} – {pct(g.ci_high)}" if has_ci else DASH,
                band_style=f"left:{lo * 100:.1f}%;width:{(hi - lo) * 100:.1f}%" if has_ci else "",
                dot_style=f"left:{rate * 100:.1f}%",
            )
        )
    return rows


def _comparison(
    c: Comparison, baseline: SuiteResult | None, current_ids: set[str]
) -> ComparisonView:
    statuses = [d.status for d in c.scenarios]
    counts = " · ".join(
        f"{statuses.count(s)} {s}"
        for s in ("regressed", "improved", "same", "new", "removed", "unknown")
        if s in statuses
    )
    label = ""
    if baseline is not None:
        label = (
            f"{baseline.agent.label} ({baseline.agent.spec}), run {_when(baseline.run.started_at)}"
        )
    return ComparisonView(
        verdict="Regressed" if c.regressed else "No regression",
        verdict_cls="bad" if c.regressed else "ok",
        headline=f"{pct(c.baseline_rate)} → {pct(c.current_rate)} ({signed_pct(c.delta)})",
        tolerance=pct(c.max_regression),
        baseline_label=label,
        counts=counts,
        rows=[
            DeltaRow(
                id=d.scenario_id,
                title=d.title,
                baseline=pct(d.baseline_rate),
                current=pct(d.current_rate),
                delta=signed_pct(d.delta),
                status=d.status,
                linked=d.scenario_id in current_ids,
            )
            for d in c.scenarios
        ],
        problems=list(c.problems),
    )


def _turn(turn: TurnRecord, scenario_id: str, clips: _ClipCopier | None) -> TurnView:
    audio = None
    if clips is not None and turn.audio_path:
        audio = clips.copy(
            turn.audio_path, scenario_id, turn.turn_id or f"{turn.speaker}_{turn.index}"
        )
    if turn.speaker == "caller":
        who, tag = "Caller", " · ".join(x for x in (turn.turn_id, turn.source) if x)
    else:
        who, tag = "Agent", ""
    return TurnView(
        speaker=turn.speaker,
        who=who,
        tag=tag,
        text=turn.text if turn.text is not None else "",
        latency=ms(turn.latency_ms) if turn.latency_ms is not None else "",
        audio=audio,
    )


def _trial(t: TrialResult, scenario_id: str, clips: _ClipCopier | None) -> TrialView:
    snapshot = t.snapshot
    orders = [
        f"{o.order_id} ({o.status}): "
        + ", ".join(
            f"{line.quantity}× {line.item_id}"
            + (f" [{', '.join(sorted(line.modifiers))}]" if line.modifiers else "")
            for line in o.lines
        )
        for o in (snapshot.orders if snapshot else [])
    ]
    calls = []
    for c in snapshot.trace if snapshot else []:
        full = f"error: {c.error}" if c.error else ("" if c.result is None else _json(c.result))
        calls.append(
            CallView(
                seq=c.seq,
                t=f"{c.t_ms / 1000:.2f} s",
                name=c.name,
                args=_short(_json(c.args)),
                result=_short(full),
                title=full if len(full) > _MAX_CELL else "",
                error=c.error is not None,
            )
        )
    return TrialView(
        number=t.trial,
        cls=OUTCOME_CLASS[t.outcome],
        label=t.outcome.value.replace("_", " "),
        duration=f"{t.duration_ms / 1000:.2f} s",
        summary=t.check.summary if t.check else "",
        error=t.error or "",
        invalid_reason=t.caller_invalid_reason or "",
        max_turns_hit=t.max_turns_hit,
        orders=orders,
        checks=[
            CheckRow(f.field, f.expected, f.actual, *(("pass", "✓") if f.passed else ("fail", "✗")))
            for f in (t.check.field_checks if t.check else [])
        ],
        turns=[_turn(turn, scenario_id, clips) for turn in t.transcript],
        calls=calls,
    )


def _scenario(s: ScenarioResult, clips: _ClipCopier | None) -> ScenarioView:
    worst = OUTCOME_CLASS[s.worst_outcome] if s.trials else "infra"
    return ScenarioView(
        id=s.scenario_id,
        title=s.title,
        meta=f"{s.language} · {s.category.replace('_', ' ')}",
        tags=list(s.tags),
        rate=pct(s.pass_rate),
        counts=f"{s.n_pass}/{s.n_valid} valid",
        cls=worst,
        open=worst != "pass",
        chips=[OUTCOME_CLASS[t.outcome] for t in s.trials],
        trials=[_trial(t, s.scenario_id, clips) for t in s.trials],
    )


def _build_page(
    suite: SuiteResult,
    comparison: Comparison | None = None,
    baseline: SuiteResult | None = None,
    clips: _ClipCopier | None = None,
) -> PageView:
    run, pack, agent = suite.run, suite.pack, suite.agent
    facts = [
        ("Agent spec", agent.spec or DASH),
        ("Started", _when(run.started_at)),
        ("Duration", _span((run.finished_at - run.started_at).total_seconds())),
        ("Scenarios", str(suite.metrics.n_scenarios)),
        ("Trials per scenario", str(run.trials)),
        ("Modality", run.modality),
        *([("Clip set", pack.clips)] if pack.clips else []),
        ("Seed", DASH if run.seed is None else str(run.seed)),
        ("Host", run.host),
    ]
    current_ids = {s.scenario_id for s in suite.scenarios}
    return PageView(
        title=f"IndicOrderBench · {pack.id} · {agent.label}",
        pack=f"{pack.id} {pack.version}",
        agent=agent.label,
        facts=facts,
        cards=_cards(suite),
        by_language=_group_rows(suite.metrics.by_language),
        by_category=_group_rows(suite.metrics.by_category),
        comparison=_comparison(comparison, baseline, current_ids) if comparison else None,
        scenarios=[_scenario(s, clips) for s in suite.scenarios],
        footer=[
            f"IndicOrderBench {run.benchmark_version}",
            f"pack {pack.id} {pack.version} · content hash {pack.content_hash}",
            f"run {_stamp(run.started_at)} → {_stamp(run.finished_at)}",
            f"Python {run.python} on {run.host} · max {run.max_turns} turns, "
            f"{run.timeout_turn_s:g} s per turn, {run.timeout_trial_s:g} s per trial",
        ],
    )


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=PackageLoader("indicorderbench.report", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def write_html(
    suite: SuiteResult,
    out_dir: Path,
    baseline: SuiteResult | None = None,
    comparison: Comparison | None = None,
    copy_audio: bool = True,
) -> Path:
    """Render ``out_dir/report.html``; clips are copied to ``out_dir/assets`` when present.

    A ``baseline`` without a ``comparison`` is compared with default tolerance.
    """
    if comparison is None and baseline is not None:
        comparison = compare(baseline, suite)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = _ClipCopier(out_dir) if copy_audio else None
    page = _build_page(suite, comparison, baseline, clips)
    html = _environment().get_template("report.html.j2").render(page=page)
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
