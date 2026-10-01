"""JUnit XML: one testcase per scenario, so CI dashboards show which orders went wrong.

A scenario is a failure if any valid trial failed, an error if any trial hit an
infrastructure error and none failed, skipped only if every trial was simulator-invalid,
and passing otherwise.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from indicorderbench.schemas.results import Outcome, ScenarioResult, SuiteResult, TrialResult

CaseKind = Literal["pass", "failure", "error", "skipped"]

# Characters XML 1.0 cannot represent (control codes such as ESC or NUL in tracebacks).
_XML_ILLEGAL = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")


def _clean(text: str) -> str:
    return _XML_ILLEGAL.sub("", text)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def classify(scenario: ScenarioResult) -> CaseKind:
    outcomes = {t.outcome for t in scenario.trials}
    if Outcome.FAIL in outcomes:
        return "failure"
    if Outcome.INFRA_ERROR in outcomes:
        return "error"
    if not outcomes or outcomes == {Outcome.SIMULATOR_INVALID}:
        return "skipped"
    return "pass"


def _failing_checks(trial: TrialResult) -> list[str]:
    if trial.check is None:
        return []
    return [
        f"{f.field}: expected {f.expected}, actual {f.actual}"
        for f in trial.check.field_checks
        if not f.passed
    ]


def _failure(scenario: ScenarioResult) -> tuple[str, str]:
    failed = [t for t in scenario.trials if t.outcome is Outcome.FAIL]
    first = failed[0]
    detail = "; ".join(_failing_checks(first)) or (
        first.check.summary if first.check else "check failed"
    )
    message = f"{len(failed)} of {_plural(scenario.n_valid, 'valid trial')} failed: {detail}"
    body: list[str] = []
    for t in failed:
        checks = _failing_checks(t) or [t.check.summary if t.check else "check failed"]
        body.extend(f"trial {t.trial}: {c}" for c in checks)
    return message, "\n".join(body)


def _error(scenario: ScenarioResult) -> tuple[str, str]:
    infra = [t for t in scenario.trials if t.outcome is Outcome.INFRA_ERROR]
    first_line = (infra[0].error or "infrastructure error").splitlines()[0]
    message = (
        f"{len(infra)} of {_plural(len(scenario.trials), 'trial')} hit an "
        f"infrastructure error: {first_line}"
    )
    body = "\n\n".join(f"trial {t.trial}:\n{t.error or 'infrastructure error'}" for t in infra)
    return message, body


def _skipped(scenario: ScenarioResult) -> str:
    if not scenario.trials:
        return "no trials were run"
    reason = scenario.trials[0].caller_invalid_reason or "caller script could not continue"
    n = len(scenario.trials)
    which = "the only trial was" if n == 1 else f"all {n} trials were"
    return f"{which} simulator-invalid: {reason}"


def _transcript(trial: TrialResult) -> str:
    lines = [
        f"--- trial {trial.trial}: {trial.outcome.value} ({trial.duration_ms / 1000:.2f} s) ---"
    ]
    for turn in trial.transcript:
        text = turn.text if turn.text is not None else "(no text)"
        if turn.speaker == "caller":
            who = f"caller {turn.turn_id}" if turn.turn_id else "caller"
        else:
            who = f"agent {turn.latency_ms:.0f} ms" if turn.latency_ms is not None else "agent"
        lines.append(f"[{who}] {text}")
    if trial.check is not None:
        lines.append(f"check: {trial.check.summary}")
    if trial.caller_invalid_reason:
        lines.append(f"caller invalid: {trial.caller_invalid_reason}")
    if trial.error:
        lines.append(f"error: {trial.error}")
    return "\n".join(lines)


def _testcase(parent: ET.Element, scenario: ScenarioResult) -> tuple[CaseKind, float]:
    seconds = sum(t.duration_ms for t in scenario.trials) / 1000.0
    case = ET.SubElement(
        parent,
        "testcase",
        name=scenario.scenario_id,
        classname=f"{scenario.language}.{scenario.category}",
        time=f"{seconds:.3f}",
    )
    kind = classify(scenario)
    if kind == "failure":
        message, body = _failure(scenario)
        el = ET.SubElement(case, "failure", message=_clean(message), type="WrongOrder")
        el.text = _clean(body)
    elif kind == "error":
        message, body = _error(scenario)
        el = ET.SubElement(case, "error", message=_clean(message), type="InfraError")
        el.text = _clean(body)
    elif kind == "skipped":
        ET.SubElement(case, "skipped", message=_clean(_skipped(scenario)))
    if kind != "pass":
        out = [_transcript(t) for t in scenario.trials if t.outcome is not Outcome.PASS]
        ET.SubElement(case, "system-out").text = _clean("\n\n".join(out))
    return kind, seconds


def write_junit(suite: SuiteResult, path: Path) -> None:
    root = ET.Element("testsuites", name="indicorderbench")
    ts = ET.SubElement(
        root,
        "testsuite",
        name=f"indicorderbench:{suite.pack.id}",
        timestamp=suite.run.started_at.isoformat(timespec="seconds"),
        hostname=suite.run.host,
    )
    props = ET.SubElement(ts, "properties")
    for name, value in [
        ("benchmark_version", suite.run.benchmark_version),
        ("pack_version", suite.pack.version),
        ("pack_hash", suite.pack.content_hash),
        ("agent", suite.agent.label),
        ("agent_spec", suite.agent.spec),
        ("trials", str(suite.run.trials)),
        ("modality", suite.run.modality),
    ]:
        ET.SubElement(props, "property", name=name, value=_clean(value))
    counts: dict[CaseKind, int] = {"pass": 0, "failure": 0, "error": 0, "skipped": 0}
    total = 0.0
    for scenario in suite.scenarios:
        kind, seconds = _testcase(ts, scenario)
        counts[kind] += 1
        total += seconds
    attrs = {
        "tests": str(len(suite.scenarios)),
        "failures": str(counts["failure"]),
        "errors": str(counts["error"]),
        "skipped": str(counts["skipped"]),
        "time": f"{total:.3f}",
    }
    for el in (root, ts):
        el.attrib.update(attrs)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
