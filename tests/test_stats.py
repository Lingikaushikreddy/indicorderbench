from datetime import datetime

import pytest

from indicorderbench.runner.stats import compute_metrics, pass_k, percentile, wilson_interval
from indicorderbench.schemas.results import Outcome, ScenarioResult, TrialResult, TurnRecord


def trial(sid: str, n: int, outcome: Outcome, latency: float = 100.0) -> TrialResult:
    return TrialResult(
        scenario_id=sid,
        trial=n,
        outcome=outcome,
        started_at=datetime(2026, 1, 1),
        duration_ms=1.0,
        max_turns_hit=outcome is Outcome.FAIL,
        transcript=[
            TurnRecord(index=0, speaker="caller", text="hi", t_ms=0.0),
            TurnRecord(index=1, speaker="agent", text="x", t_ms=0.0, latency_ms=latency),
        ],
    )


def scen(sid: str, lang: str, cat: str, outcomes: list[Outcome]) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        title=sid,
        language=lang,
        category=cat,
        trials=[trial(sid, i + 1, o) for i, o in enumerate(outcomes)],
    )


def test_wilson_interval_bounds():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(9, 10)
    assert 0.59 < lo < 0.60 and 0.98 < hi < 0.99
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0)


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)
    assert percentile([7], 95) == 7.0


def test_pass_k_tau_bench_estimator():
    s = [
        scen("a", "en-IN", "quantity", [Outcome.PASS, Outcome.PASS, Outcome.FAIL]),
        scen("b", "en-IN", "quantity", [Outcome.PASS, Outcome.PASS, Outcome.PASS]),
    ]
    assert pass_k(s, 1) == pytest.approx((2 / 3 + 1) / 2)
    assert pass_k(s, 2) == pytest.approx((1 / 3 + 1) / 2)
    assert pass_k(s, 3) == pytest.approx((0 + 1) / 2)
    assert pass_k(s, 4) is None
    assert pass_k([], 1) is None


def test_compute_metrics_excludes_invalid_and_groups():
    s = [
        scen("a", "en-IN", "quantity", [Outcome.PASS, Outcome.SIMULATOR_INVALID]),
        scen("b", "hi-en", "modifier", [Outcome.FAIL, Outcome.INFRA_ERROR]),
        scen("c", "hi-en", "modifier", [Outcome.SIMULATOR_INVALID, Outcome.SIMULATOR_INVALID]),
    ]
    m = compute_metrics(s, trials=2)
    assert (m.n_scenarios, m.n_trials_total) == (3, 6)
    assert (m.n_valid, m.n_pass, m.n_fail, m.n_simulator_invalid, m.n_infra_error) == (
        2,
        1,
        1,
        3,
        1,
    )
    assert m.n_max_turns == 1
    assert m.pass_rate == pytest.approx(0.5)  # scenario-weighted over a and b
    by_lang = {g.key: g for g in m.by_language}
    assert by_lang["en-IN"].n_trials == 1 and by_lang["en-IN"].pass_rate == 1.0
    assert by_lang["hi-en"].n_trials == 1 and by_lang["hi-en"].pass_rate == 0.0
    assert by_lang["hi-en"].ci_low is not None and by_lang["hi-en"].ci_high is not None
    by_cat = {g.key: g for g in m.by_category}
    assert by_cat["modifier"].n_trials == 1
    assert m.latency_p50_ms == 100.0 and m.latency_p95_ms == 100.0
    assert m.pass_k[1] == pytest.approx(0.5) and m.pass_k[2] is None


def test_compute_metrics_with_no_valid_trials():
    s = [scen("a", "en-IN", "quantity", [Outcome.INFRA_ERROR])]
    m = compute_metrics(s, trials=1)
    assert m.pass_rate is None and m.pass_k == {1: None}
    assert m.by_language[0].pass_rate is None and m.by_language[0].ci_low is None
