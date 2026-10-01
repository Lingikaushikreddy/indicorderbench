import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from indicorderbench.report.json_report import ReportError, read_json, write_json
from indicorderbench.report.junit import write_junit
from indicorderbench.schemas.results import Outcome
from tests.helpers import INFRA_ERROR_TEXT, INVALID_REASON, make_suite

P, F, INV, INF = Outcome.PASS, Outcome.FAIL, Outcome.SIMULATOR_INVALID, Outcome.INFRA_ERROR

MIXED = {
    "en_quantity_01": [P, F],  # failure
    "en_modifier_01": [P, P],  # pass
    "hien_correction_01": [INV, INV],  # skipped
    "hien_cancellation_01": [INF, P],  # error
    "en_duplicate_submission_01": [F, INF],  # failure beats infra error
    "hien_quantity_01": [INV, P],  # pass: the invalid trial is excluded
}


def test_json_roundtrip(tmp_path: Path) -> None:
    suite = make_suite(MIXED)
    path = tmp_path / "nested" / "results.json"
    write_json(suite, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1 and raw["pack"]["id"] == "starter"
    assert read_json(path) == suite


def test_read_json_rejects_other_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    write_json(make_suite({"en_quantity_01": [P]}), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 99
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReportError) as e:
        read_json(path)
    assert "99" in str(e.value) and "1" in str(e.value)


@pytest.mark.parametrize(
    ("content", "needle"),
    [("{not json", "not valid JSON"), ('{"schema_version": 1}', "not a valid results file")],
)
def test_read_json_rejects_garbage(tmp_path: Path, content: str, needle: str) -> None:
    path = tmp_path / "results.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ReportError, match=needle):
        read_json(path)


def test_read_json_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="cannot read"):
        read_json(tmp_path / "nope.json")


def _cases(path: Path) -> tuple[ET.Element, dict[str, ET.Element]]:
    suite_el = ET.parse(path).getroot().find("testsuite")
    assert suite_el is not None
    return suite_el, {tc.attrib["name"]: tc for tc in suite_el.iter("testcase")}


def test_junit_counts_and_classification(tmp_path: Path) -> None:
    path = tmp_path / "out" / "junit.xml"
    write_junit(make_suite(MIXED), path)
    root = ET.parse(path).getroot()
    assert root.tag == "testsuites"
    ts, cases = _cases(path)
    assert ts.attrib["name"] == "indicorderbench:starter"
    assert (ts.attrib["tests"], ts.attrib["failures"], ts.attrib["errors"]) == ("6", "2", "1")
    assert ts.attrib["skipped"] == "1"
    assert float(ts.attrib["time"]) == pytest.approx(
        sum(float(c.attrib["time"]) for c in cases.values())
    )
    assert set(cases) == set(MIXED)
    assert cases["en_quantity_01"].attrib["classname"] == "en-IN.quantity"
    assert cases["hien_cancellation_01"].attrib["classname"] == "hi-en.cancellation"
    assert cases["en_quantity_01"].attrib["time"] == "4.900"

    def kinds(name: str) -> list[str]:
        return [child.tag for child in cases[name] if child.tag != "system-out"]

    assert kinds("en_quantity_01") == ["failure"]
    assert kinds("en_duplicate_submission_01") == ["failure"]
    assert kinds("en_modifier_01") == [] and kinds("hien_quantity_01") == []
    assert kinds("hien_correction_01") == ["skipped"]
    assert kinds("hien_cancellation_01") == ["error"]
    props = {p.attrib["name"]: p.attrib["value"] for p in ts.iter("property")}
    assert props["pack_hash"].startswith("3f2a") and props["agent"] == "agent"


def test_junit_failure_lists_failing_field_checks(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    write_junit(make_suite(MIXED), path)
    _, cases = _cases(path)
    failure = cases["en_quantity_01"].find("failure")
    assert failure is not None
    assert "1 of 2 valid trials failed" in failure.attrib["message"]
    assert "Paneer Wrap quantity" in failure.attrib["message"]
    assert failure.text is not None
    assert "trial 2: Paneer Wrap quantity: expected 1, actual 2" in failure.text
    assert "Onion" not in failure.text  # passing checks are not listed
    out = cases["en_quantity_01"].find("system-out")
    assert out is not None and out.text is not None
    assert "[caller t1] One paneer wrap, no onion, please." in out.text
    assert "[agent 160 ms] Two Paneer Wraps, no onion. Anything else?" in out.text
    assert cases["en_modifier_01"].find("system-out") is None  # passing cases stay quiet


def test_junit_error_and_skipped_messages(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    write_junit(make_suite(MIXED), path)
    _, cases = _cases(path)
    error = cases["hien_cancellation_01"].find("error")
    skipped = cases["hien_correction_01"].find("skipped")
    assert error is not None and skipped is not None
    assert INFRA_ERROR_TEXT in error.attrib["message"]
    assert INVALID_REASON in skipped.attrib["message"]


def test_junit_strips_characters_xml_cannot_hold(tmp_path: Path) -> None:
    suite = make_suite({"en_quantity_01": [F]})
    suite.scenarios[0].trials[0].transcript[1].text = "ANSI \x1b[31mred\x1b[0m and NUL \x00 bytes"
    path = tmp_path / "junit.xml"
    write_junit(suite, path)
    _, cases = _cases(path)
    out = cases["en_quantity_01"].find("system-out")
    assert out is not None and out.text is not None
    assert "ANSI [31mred[0m and NUL  bytes" in out.text


def test_source_has_no_invisible_or_unassigned_characters() -> None:
    """Characters such as U+E000 render as nothing in editors and diffs; write them as escapes."""
    import unicodedata

    src = Path(__file__).resolve().parents[1] / "src"
    bad = []
    for path in sorted(src.rglob("*")):
        if path.suffix not in (".py", ".j2"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for ch in line:
                category = unicodedata.category(ch)
                if ch == "\ufffd" or (category[0] in "CZ" and ch not in "\t "):
                    bad.append(f"{path.relative_to(src)}:{lineno}: U+{ord(ch):04X} {category}")
    assert bad == []
