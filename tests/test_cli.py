import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from indicorderbench.cli import _gate, app
from indicorderbench.report.json_report import read_json
from indicorderbench.schemas.results import Outcome
from tests.helpers import make_suite

FIX = Path(__file__).parent / "fixtures" / "minipack"
GOOD = "python:tests.helpers:make_minipack_agent"
BUGGY = "python:tests.helpers:make_minipack_buggy_agent"
HALF_BROKEN = "python:tests.helpers:make_minipack_agent_broken_on_cancellation"

runner = CliRunner()


def run_cli(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args), catch_exceptions=False)
    return result.exit_code, result.output


def test_validate_ok_and_failing(tmp_path: Path):
    code, out = run_cli("validate", str(FIX))
    assert code == 0 and "valid" in out.lower()
    shutil.copytree(FIX, tmp_path / "p")
    bad = tmp_path / "p" / "scenarios" / "bad.yaml"
    bad.write_text("id: bad\ntitle: t\nlanguage: xx\n")
    code, out = run_cli("validate", str(tmp_path / "p"))
    assert code == 1 and "bad.yaml" in out


def test_ls_lists_and_filters():
    code, out = run_cli("ls", str(FIX))
    assert code == 0 and "en_quantity_01" in out and "en_cancellation_01" in out
    code, out = run_cli("ls", str(FIX), "--category", "quantity")
    assert code == 0 and "en_quantity_01" in out and "en_cancellation_01" not in out
    code, out = run_cli("ls", str(FIX), "--tag", "nope")
    assert code == 1 and "no scenarios" in out.lower()


def test_run_writes_three_files_and_passes(tmp_path: Path):
    out_dir = tmp_path / "out"
    code, out = run_cli("run", str(FIX), "--agent", GOOD, "--out", str(out_dir))
    assert code == 0, out
    assert (out_dir / "results.json").exists()
    assert (out_dir / "junit.xml").exists()
    assert (out_dir / "report.html").exists()
    suite = read_json(out_dir / "results.json")
    assert suite.metrics.pass_rate == 1.0 and suite.agent.spec == GOOD
    assert "pass rate" in out.lower() and "en_quantity_01" in out


def test_run_fail_under_exit_2(tmp_path: Path):
    code, out = run_cli(
        "run", str(FIX), "--agent", BUGGY, "--out", str(tmp_path / "o"), "--fail-under", "0.9"
    )
    assert code == 2 and "below" in out.lower()


def test_run_with_baseline_regression_exit_2(tmp_path: Path):
    base = tmp_path / "base"
    assert run_cli("run", str(FIX), "--agent", GOOD, "--out", str(base))[0] == 0
    code, out = run_cli(
        "run",
        str(FIX),
        "--agent",
        BUGGY,
        "--out",
        str(tmp_path / "cur"),
        "--baseline",
        str(base / "results.json"),
    )
    assert code == 2 and "regress" in out.lower() and "en_quantity_01" in out
    html = (tmp_path / "cur" / "report.html").read_text()
    assert 'id="comparison"' in html


def test_run_filters_and_trials(tmp_path: Path):
    code, out = run_cli(
        "run",
        str(FIX),
        "--agent",
        GOOD,
        "--out",
        str(tmp_path / "o"),
        "--id",
        "en_quantity_01",
        "--trials",
        "2",
    )
    assert code == 0
    suite = read_json(tmp_path / "o" / "results.json")
    assert [s.scenario_id for s in suite.scenarios] == ["en_quantity_01"]
    assert len(suite.scenarios[0].trials) == 2 and suite.metrics.pass_k == {1: 1.0, 2: 1.0}
    code, out = run_cli(
        "run", str(FIX), "--agent", GOOD, "--out", str(tmp_path / "o"), "--tag", "x"
    )
    assert code == 1 and "no scenarios" in out.lower()


def test_run_rejects_bad_agent_spec(tmp_path: Path):
    code, out = run_cli("run", str(FIX), "--agent", "bogus:thing", "--out", str(tmp_path / "o"))
    assert code == 2 and "builtin:" in out and "http:" in out and "python:" in out
    code, out = run_cli(
        "run", str(FIX), "--agent", "python:tests.helpers:missing", "--out", str(tmp_path / "o")
    )
    assert code == 2 and "missing" in out


def test_report_regenerates_html(tmp_path: Path):
    out = tmp_path / "o"
    assert run_cli("run", str(FIX), "--agent", GOOD, "--out", str(out))[0] == 0
    (out / "report.html").unlink()
    code, _ = run_cli("report", str(out / "results.json"), "--out", str(tmp_path / "again"))
    assert code == 0 and (tmp_path / "again" / "report.html").exists()


def test_compare_detects_regression_and_rejects_other_pack(tmp_path: Path):
    base, cur = tmp_path / "b", tmp_path / "c"
    assert run_cli("run", str(FIX), "--agent", GOOD, "--out", str(base))[0] == 0
    assert run_cli("run", str(FIX), "--agent", BUGGY, "--out", str(cur))[0] == 0
    code, out = run_cli("compare", str(base / "results.json"), str(cur / "results.json"))
    assert code == 2 and "en_quantity_01" in out
    code, out = run_cli(
        "compare", str(base / "results.json"), str(cur / "results.json"), "--max-regression", "1.0"
    )
    assert code == 0
    doc = json.loads((cur / "results.json").read_text())
    doc["pack"]["id"] = "otherpack"
    (cur / "results.json").write_text(json.dumps(doc))
    code, out = run_cli("compare", str(base / "results.json"), str(cur / "results.json"))
    assert code == 1 and "otherpack" in out


def test_synth_silence_provider_then_audio_run(tmp_path: Path):
    shutil.copytree(FIX, tmp_path / "p")
    pack = tmp_path / "p"
    code, out = run_cli("synth", str(pack), "--provider", "silence")
    assert code == 0, out
    assert (pack / "clips" / "en_quantity_01" / "t1.wav").exists()
    assert (pack / "clips" / "_defaults" / "en-IN" / "closing.wav").exists()
    manifest = json.loads((pack / "clips" / "manifest.json").read_text())
    assert any(c["provider"] == "silence" for c in manifest["clips"])
    # a second run writes nothing new
    code, out = run_cli("synth", str(pack), "--provider", "silence")
    assert code == 0 and "written 0" in out
    # audio modality now has every clip it needs; the python agent ignores audio and uses no
    # text, so it fails the check, but the run itself is valid (no infra errors)
    code, out = run_cli(
        "run", str(pack), "--agent", GOOD, "--out", str(tmp_path / "o"), "--modality", "audio"
    )
    suite = read_json(tmp_path / "o" / "results.json")
    assert suite.metrics.n_infra_error == 0 and suite.run.modality == "audio"


def test_synth_sarvam_needs_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    shutil.copytree(FIX, tmp_path / "p")
    code, out = run_cli("synth", str(tmp_path / "p"), "--provider", "sarvam")
    assert code == 1 and "SARVAM_API_KEY" in out


def test_perturb_writes_derived_clips(tmp_path: Path):
    shutil.copytree(FIX, tmp_path / "p")
    pack = tmp_path / "p"
    assert run_cli("synth", str(pack), "--provider", "silence")[0] == 0
    code, out = run_cli(
        "perturb", str(pack), "--out", str(tmp_path / "noisy"), "--snr", "10", "--telephone"
    )
    assert code == 0, out
    assert (tmp_path / "noisy" / "en_quantity_01" / "t1.wav").exists()
    manifest = json.loads((tmp_path / "noisy" / "manifest.json").read_text())
    assert manifest["perturbation"]["snr_db"] == 10 and manifest["perturbation"]["telephone"]


def test_run_audio_modality_without_clips_is_infra_error(tmp_path: Path):
    code, out = run_cli(
        "run", str(FIX), "--agent", GOOD, "--out", str(tmp_path / "o"), "--modality", "audio"
    )
    assert code == 1 and "infra" in out.lower()


def test_partial_infra_errors_exit_1_unless_allowed(tmp_path: Path):
    code, out = run_cli("run", str(FIX), "--agent", HALF_BROKEN, "--out", str(tmp_path / "a"))
    assert code == 1, out
    assert "1 trial(s) hit infra errors (allowed: 0); fix the agent connection" in out
    suite = read_json(tmp_path / "a" / "results.json")
    assert suite.metrics.n_infra_error == 1 and suite.metrics.n_pass == 1
    code, out = run_cli(
        "run", str(FIX), "--agent", HALF_BROKEN, "--out", str(tmp_path / "b"), "--allow-infra", "0"
    )
    assert code == 1, out
    code, out = run_cli(
        "run", str(FIX), "--agent", HALF_BROKEN, "--out", str(tmp_path / "c"), "--allow-infra", "1"
    )
    assert code == 0, out
    assert "infra errors (allowed" not in out


def test_infra_gate_takes_precedence_over_threshold_but_both_print(
    capsys: pytest.CaptureFixture[str],
):
    suite = make_suite({"en_quantity_01": [Outcome.FAIL], "en_modifier_01": [Outcome.INFRA_ERROR]})
    assert _gate(suite, None, 0.9, allow_infra=0) == 1
    out = capsys.readouterr().out
    assert "below --fail-under" in out and "1 trial(s) hit infra errors (allowed: 0)" in out
    assert _gate(suite, None, 0.9, allow_infra=1) == 2
    assert "infra errors" not in capsys.readouterr().out
    all_infra = make_suite({"en_quantity_01": [Outcome.INFRA_ERROR]})
    assert _gate(all_infra, None, None, allow_infra=5) == 1
    assert "every trial was an infra error" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------
# built-in reference agents, bundled pack name, demo
# ---------------------------------------------------------------------------------------


def test_pack_name_resolves_to_repo_pack():
    code, out = run_cli("validate", "starter")
    assert code == 0 and "42 scenarios" in out
    code, out = run_cli("validate", "nonexistent-pack")
    assert code == 1 and "neither" in out


def test_builtin_correct_passes_smoke(tmp_path: Path):
    code, out = run_cli(
        "run",
        "starter",
        "--agent",
        "builtin:correct",
        "--tag",
        "smoke",
        "--out",
        str(tmp_path / "o"),
        "--fail-under",
        "1.0",
    )
    assert code == 0, out
    suite = read_json(tmp_path / "o" / "results.json")
    assert suite.metrics.n_scenarios == 10 and suite.metrics.pass_rate == 1.0
    assert suite.metrics.n_simulator_invalid == 0 and suite.metrics.n_infra_error == 0


def test_builtin_buggy_fails_threshold_and_names_bug(tmp_path: Path):
    code, out = run_cli(
        "run",
        "starter",
        "--agent",
        "builtin:buggy:ignore_corrections",
        "--tag",
        "smoke",
        "--out",
        str(tmp_path / "o"),
        "--fail-under",
        "1.0",
    )
    assert code == 2
    suite = read_json(tmp_path / "o" / "results.json")
    failed = {s.scenario_id for s in suite.scenarios if s.trials[0].outcome.value == "fail"}
    assert failed == {"en_correction_01", "hien_correction_01"}
    code, out = run_cli(
        "run", "starter", "--agent", "builtin:buggy:nope", "--out", str(tmp_path / "x")
    )
    assert code == 2 and "nope" in out and "ignore_corrections" in out


def test_demo_prints_table_and_writes_reports(tmp_path: Path):
    code, out = run_cli("demo", "--out", str(tmp_path / "demo"))
    assert code == 0, out
    assert "Paneer Wrap quantity" in out and "Fail" in out
    for label in ("buggy", "fixed"):
        for name in ("results.json", "junit.xml", "report.html"):
            assert (tmp_path / "demo" / label / name).exists()
    assert (tmp_path / "demo" / "README.md").exists()
    fixed = read_json(tmp_path / "demo" / "fixed" / "results.json")
    assert fixed.metrics.pass_rate == 1.0
    assert 'id="comparison"' in (tmp_path / "demo" / "fixed" / "report.html").read_text()


def test_audio_modality_with_placeholder_clips_and_oracle(tmp_path: Path):
    """Offline audio pipeline: synth placeholder clips, run audio-only, oracle transcribes."""
    shutil.copytree(Path(__file__).resolve().parents[1] / "packs" / "starter", tmp_path / "p")
    assert run_cli("synth", str(tmp_path / "p"), "--provider", "silence")[0] == 0
    code, out = run_cli(
        "run",
        str(tmp_path / "p"),
        "--agent",
        "builtin:correct",
        "--tag",
        "smoke",
        "--modality",
        "audio",
        "--out",
        str(tmp_path / "o"),
        "--fail-under",
        "1.0",
    )
    assert code == 0, out
    suite = read_json(tmp_path / "o" / "results.json")
    assert suite.run.modality == "audio" and suite.metrics.pass_rate == 1.0
    first_caller_turn = suite.scenarios[0].trials[0].transcript[0]
    assert first_caller_turn.audio_path and first_caller_turn.audio_path.endswith(".wav")
    assert (tmp_path / "o" / "assets").is_dir()
