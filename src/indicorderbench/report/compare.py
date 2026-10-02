"""Baseline comparison: per-scenario pass-rate deltas and the regression gate.

A scenario's status is ``regressed`` or ``improved`` when its pass rate moved by more than
``max_regression`` (``same`` otherwise), ``new``/``removed`` when it exists in only one run,
and ``unknown`` when either run has no valid trial for it. The comparison as a whole has
``regressed`` set when the suite pass rate dropped by more than ``max_regression``, or when
any scenario that passed every valid trial in the baseline fell below
``1.0 - max_regression``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from indicorderbench.report.formatting import pct, signed_pct
from indicorderbench.report.json_report import ReportError
from indicorderbench.schemas.results import ScenarioResult, SuiteResult

Status = Literal["regressed", "improved", "same", "new", "removed", "unknown"]

_EPS = 1e-9  # pass rates are means of fractions; ignore float noise when comparing


@dataclass
class ScenarioDelta:
    scenario_id: str
    title: str
    baseline_rate: float | None
    current_rate: float | None
    delta: float | None
    status: Status


@dataclass
class Comparison:
    baseline_rate: float | None
    current_rate: float | None
    delta: float | None
    max_regression: float
    regressed: bool
    scenarios: list[ScenarioDelta] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _diff(baseline: float | None, current: float | None) -> float | None:
    return None if baseline is None or current is None else current - baseline


def _scenario_delta(
    base: ScenarioResult, cur: ScenarioResult, max_regression: float
) -> ScenarioDelta:
    delta = _diff(base.pass_rate, cur.pass_rate)
    status: Status
    if delta is None:
        status = "unknown"
    elif delta < -max_regression - _EPS:
        status = "regressed"
    elif delta > max_regression + _EPS:
        status = "improved"
    else:
        status = "same"
    return ScenarioDelta(cur.scenario_id, cur.title, base.pass_rate, cur.pass_rate, delta, status)


def _run_problems(baseline: SuiteResult, current: SuiteResult) -> list[str]:
    problems = []
    b, c = baseline.pack, current.pack
    if b.version != c.version:
        problems.append(f"pack version differs (baseline {b.version}, current {c.version})")
    if b.content_hash != c.content_hash:
        problems.append(
            f"pack content hash differs (baseline {b.content_hash[:12]}, current "
            f"{c.content_hash[:12]}): scenarios may have changed between runs"
        )
    if b.clips != c.clips:
        own = "pack clips"
        problems.append(f"clip set differs (baseline {b.clips or own}, current {c.clips or own})")
    for name in ("trials", "modality"):
        bv, cv = getattr(baseline.run, name), getattr(current.run, name)
        if bv != cv:
            problems.append(f"{name} differs (baseline {bv}, current {cv})")
    return problems


def compare(baseline: SuiteResult, current: SuiteResult, max_regression: float = 0.0) -> Comparison:
    """Compare ``current`` against ``baseline``; both must come from the same pack."""
    if not 0.0 <= max_regression <= 1.0:
        raise ValueError(f"max_regression must be between 0 and 1, got {max_regression}")
    if baseline.schema_version != current.schema_version:
        raise ReportError(
            f"cannot compare results with different schema versions: baseline "
            f"{baseline.schema_version}, current {current.schema_version}"
        )
    if baseline.pack.id != current.pack.id:
        raise ReportError(
            f"cannot compare runs of different packs: baseline {baseline.pack.id!r}, "
            f"current {current.pack.id!r}"
        )
    problems = _run_problems(baseline, current)
    base_by_id = {s.scenario_id: s for s in baseline.scenarios}
    current_ids = {s.scenario_id for s in current.scenarios}
    deltas: list[ScenarioDelta] = []
    for cur in current.scenarios:
        base = base_by_id.get(cur.scenario_id)
        if base is None:
            deltas.append(
                ScenarioDelta(cur.scenario_id, cur.title, None, cur.pass_rate, None, "new")
            )
        else:
            deltas.append(_scenario_delta(base, cur, max_regression))
    for base in baseline.scenarios:
        if base.scenario_id not in current_ids:
            deltas.append(
                ScenarioDelta(base.scenario_id, base.title, base.pass_rate, None, None, "removed")
            )
    n_new = sum(1 for d in deltas if d.status == "new")
    n_removed = sum(1 for d in deltas if d.status == "removed")
    if n_new or n_removed:
        problems.append(
            f"{n_new} new and {n_removed} removed scenario(s): the suite pass rates cover "
            "different scenario sets"
        )
    for d in deltas:
        if d.status == "unknown":
            runs = [
                name
                for name, rate in (("baseline", d.baseline_rate), ("current", d.current_rate))
                if rate is None
            ]
            where = f"{' and '.join(runs)} run{'s' if len(runs) > 1 else ''}"
            problems.append(f"{d.scenario_id}: no valid trials in the {where}, not compared")
    delta = _diff(baseline.metrics.pass_rate, current.metrics.pass_rate)
    suite_dropped = delta is not None and delta < -max_regression - _EPS
    perfect_dropped = any(
        d.baseline_rate is not None
        and d.baseline_rate >= 1.0 - _EPS
        and d.current_rate is not None
        and d.current_rate < 1.0 - max_regression - _EPS
        for d in deltas
    )
    return Comparison(
        baseline_rate=baseline.metrics.pass_rate,
        current_rate=current.metrics.pass_rate,
        delta=delta,
        max_regression=max_regression,
        regressed=suite_dropped or perfect_dropped,
        scenarios=deltas,
        problems=problems,
    )


def render_comparison_text(c: Comparison) -> str:
    """A fixed-width table of per-scenario deltas, the suite line, then any problems."""
    rows = [("scenario", "baseline", "current", "delta", "status")]
    rows += [
        (d.scenario_id, pct(d.baseline_rate), pct(d.current_rate), signed_pct(d.delta), d.status)
        for d in c.scenarios
    ]
    verdict = "REGRESSED" if c.regressed else "ok"
    rows.append(
        (
            "suite",
            pct(c.baseline_rate),
            pct(c.current_rate),
            signed_pct(c.delta),
            f"{verdict} (max regression {pct(c.max_regression)})",
        )
    )
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = [
        f"{r[0]:<{widths[0]}}  {r[1]:>{widths[1]}}  {r[2]:>{widths[2]}}  "
        f"{r[3]:>{widths[3]}}  {r[4]}"
        for r in rows
    ]
    lines += [f"note: {p}" for p in c.problems]
    return "\n".join(lines)
