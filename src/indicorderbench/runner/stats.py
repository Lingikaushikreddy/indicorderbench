"""Suite statistics: pass rate, pass^k, Wilson intervals, latency percentiles."""

from __future__ import annotations

import math
from collections.abc import Callable

from indicorderbench.schemas.results import GroupMetric, Metrics, Outcome, ScenarioResult


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile; None for an empty list."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def pass_k(scenarios: list[ScenarioResult], k: int) -> float | None:
    """τ-bench pass^k: mean over scenarios of C(c, k) / C(n, k) over valid trials.

    None when any scenario has fewer than k valid trials, or when no scenario has a
    valid trial.
    """
    vals: list[float] = []
    for s in scenarios:
        n, c = s.n_valid, s.n_pass
        if n == 0:
            continue
        if k > n:
            return None
        vals.append(math.comb(c, k) / math.comb(n, k))
    return sum(vals) / len(vals) if vals else None


def group_metrics(
    scenarios: list[ScenarioResult], key: Callable[[ScenarioResult], str]
) -> list[GroupMetric]:
    buckets: dict[str, tuple[int, int]] = {}
    for s in scenarios:
        n, c = buckets.get(key(s), (0, 0))
        buckets[key(s)] = (n + s.n_valid, c + s.n_pass)
    out = []
    for k in sorted(buckets):
        n, c = buckets[k]
        lo, hi = wilson_interval(c, n)
        out.append(
            GroupMetric(
                key=k,
                n_trials=n,
                n_pass=c,
                pass_rate=(c / n if n else None),
                ci_low=(lo if n else None),
                ci_high=(hi if n else None),
            )
        )
    return out


def compute_metrics(scenarios: list[ScenarioResult], trials: int) -> Metrics:
    all_trials = [t for s in scenarios for t in s.trials]
    rates = [s.pass_rate for s in scenarios if s.pass_rate is not None]
    latencies = [x for t in all_trials for x in t.latencies_ms]
    return Metrics(
        n_scenarios=len(scenarios),
        n_trials_total=len(all_trials),
        n_valid=sum(1 for t in all_trials if t.is_valid),
        n_pass=sum(1 for t in all_trials if t.outcome is Outcome.PASS),
        n_fail=sum(1 for t in all_trials if t.outcome is Outcome.FAIL),
        n_simulator_invalid=sum(1 for t in all_trials if t.outcome is Outcome.SIMULATOR_INVALID),
        n_infra_error=sum(1 for t in all_trials if t.outcome is Outcome.INFRA_ERROR),
        n_max_turns=sum(1 for t in all_trials if t.max_turns_hit),
        pass_rate=(sum(rates) / len(rates) if rates else None),
        pass_k={k: pass_k(scenarios, k) for k in range(1, max(1, trials) + 1)},
        by_language=group_metrics(scenarios, lambda s: s.language),
        by_category=group_metrics(scenarios, lambda s: s.category),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
    )
