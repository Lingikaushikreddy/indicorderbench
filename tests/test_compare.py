import pytest

from indicorderbench.report.compare import compare, render_comparison_text
from indicorderbench.report.json_report import ReportError
from indicorderbench.schemas.results import Outcome
from tests.helpers import make_suite

P, F, INF = Outcome.PASS, Outcome.FAIL, Outcome.INFRA_ERROR


def test_detects_a_regression_and_new_and_removed_scenarios() -> None:
    baseline = make_suite({"en_quantity_01": [P, P], "en_modifier_01": [P, P]})
    current = make_suite({"en_quantity_01": [P, F], "hien_correction_01": [P, P]})
    c = compare(baseline, current)
    by_id = {s.scenario_id: s for s in c.scenarios}
    assert [s.scenario_id for s in c.scenarios] == [
        "en_quantity_01",
        "hien_correction_01",
        "en_modifier_01",
    ]
    q = by_id["en_quantity_01"]
    assert (q.baseline_rate, q.current_rate, q.status) == (1.0, 0.5, "regressed")
    assert q.delta == pytest.approx(-0.5) and q.title == "Quantity 01 (en-IN)"
    new, removed = by_id["hien_correction_01"], by_id["en_modifier_01"]
    assert (new.status, new.baseline_rate, new.current_rate, new.delta) == ("new", None, 1.0, None)
    assert (removed.status, removed.current_rate, removed.delta) == ("removed", None, None)
    assert c.baseline_rate == 1.0 and c.current_rate == pytest.approx(0.75)
    assert c.delta == pytest.approx(-0.25) and c.regressed and c.max_regression == 0.0
    assert any("1 new and 1 removed" in p for p in c.problems)


def test_same_and_improved_are_not_regressions() -> None:
    baseline = make_suite({"en_quantity_01": [P, F], "en_modifier_01": [P, P]})
    current = make_suite({"en_quantity_01": [P, P], "en_modifier_01": [P, P]})
    c = compare(baseline, current)
    assert [s.status for s in c.scenarios] == ["improved", "same"]
    assert c.delta == pytest.approx(0.25) and not c.regressed
    assert c.problems == []


def test_max_regression_tolerates_small_drops() -> None:
    baseline = make_suite({"en_quantity_01": [P, P, P, P]})
    current = make_suite({"en_quantity_01": [P, P, P, F]})  # 1.0 -> 0.75
    loose = compare(baseline, current, max_regression=0.3)
    assert not loose.regressed and loose.scenarios[0].status == "same"
    tight = compare(baseline, current, max_regression=0.2)
    assert tight.regressed and tight.scenarios[0].status == "regressed"


def test_scenario_falling_from_perfect_regresses_even_if_suite_rate_holds() -> None:
    baseline = make_suite({"en_quantity_01": [P, P], "en_modifier_01": [F, F]})
    current = make_suite({"en_quantity_01": [P, F], "en_modifier_01": [P, F]})
    c = compare(baseline, current)
    assert c.delta == pytest.approx(0.0)
    assert c.regressed


def test_suite_drop_regresses_without_any_perfect_scenario() -> None:
    baseline = make_suite({"en_quantity_01": [P, F]})
    current = make_suite({"en_quantity_01": [F, F]})
    assert compare(baseline, current).regressed
    assert not compare(baseline, current, max_regression=0.5).regressed


def test_scenarios_without_valid_trials_are_unknown() -> None:
    baseline = make_suite({"en_quantity_01": [P], "en_modifier_01": [P]})
    current = make_suite({"en_quantity_01": [INF], "en_modifier_01": [P]})
    c = compare(baseline, current)
    assert c.scenarios[0].status == "unknown" and c.scenarios[0].delta is None
    assert not c.regressed
    assert any("en_quantity_01" in p and "no valid trials" in p for p in c.problems)


def test_unknown_problem_names_each_run_without_valid_trials() -> None:
    baseline = make_suite(
        {"en_quantity_01": [INF], "en_modifier_01": [INF], "hien_correction_01": [P]}
    )
    current = make_suite(
        {"en_quantity_01": [INF], "en_modifier_01": [P], "hien_correction_01": [INF]}
    )
    c = compare(baseline, current)
    assert [s.status for s in c.scenarios] == ["unknown", "unknown", "unknown"]
    assert c.problems == [
        "en_quantity_01: no valid trials in the baseline and current runs, not compared",
        "en_modifier_01: no valid trials in the baseline run, not compared",
        "hien_correction_01: no valid trials in the current run, not compared",
    ]


def test_refuses_different_pack_ids() -> None:
    with pytest.raises(ReportError) as e:
        compare(
            make_suite({"en_quantity_01": [P]}, pack_id="starter"),
            make_suite({"en_quantity_01": [P]}, pack_id="other"),
        )
    assert "starter" in str(e.value) and "other" in str(e.value)


def test_refuses_different_schema_versions() -> None:
    baseline = make_suite({"en_quantity_01": [P]})
    baseline.schema_version = 2
    with pytest.raises(ReportError) as e:
        compare(baseline, make_suite({"en_quantity_01": [P]}))
    assert "2" in str(e.value) and "1" in str(e.value)


def test_reports_pack_content_changes_as_problems() -> None:
    baseline = make_suite({"en_quantity_01": [P]})
    current = make_suite({"en_quantity_01": [P]})
    current.pack.content_hash = "0" * 64
    c = compare(baseline, current)
    assert any("content hash" in p for p in c.problems)


def test_rejects_out_of_range_max_regression() -> None:
    s = make_suite({"en_quantity_01": [P]})
    with pytest.raises(ValueError, match="max_regression"):
        compare(s, s, max_regression=-0.1)


def test_render_comparison_text() -> None:
    baseline = make_suite({"en_quantity_01": [P, P], "en_modifier_01": [P, P]})
    current = make_suite({"en_quantity_01": [P, F], "hien_correction_01": [P, P]})
    text = render_comparison_text(compare(baseline, current))
    lines = text.splitlines()
    assert lines[0].split() == ["scenario", "baseline", "current", "delta", "status"]
    assert lines[1].split() == ["en_quantity_01", "100.0%", "50.0%", "-50.0%", "regressed"]
    assert lines[2].split() == ["hien_correction_01", "—", "100.0%", "—", "new"]
    assert lines[3].split() == ["en_modifier_01", "100.0%", "—", "—", "removed"]
    assert lines[4].split()[:4] == ["suite", "100.0%", "75.0%", "-25.0%"]
    assert "REGRESSED" in lines[4]
    words = ["status", "regressed", "new", "removed", "REGRESSED"]
    assert len({line.index(w) for line, w in zip(lines, words, strict=False)}) == 1  # aligned
    assert "1 new and 1 removed" in text


def test_reports_different_clip_sets_as_a_problem() -> None:
    baseline = make_suite({"en_quantity_01": [P]})
    current = make_suite({"en_quantity_01": [P]})
    current.pack.clips = "/tmp/noisy"
    c = compare(baseline, current)
    assert not c.regressed
    assert any("clip set differs" in p and "/tmp/noisy" in p for p in c.problems)
    assert not any(
        "clip" in p for p in compare(baseline, make_suite({"en_quantity_01": [P]})).problems
    )
