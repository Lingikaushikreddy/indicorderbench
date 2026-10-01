import re
import wave
from pathlib import Path

import pytest

from indicorderbench.report.compare import compare
from indicorderbench.report.formatting import pct
from indicorderbench.report.html import write_html
from indicorderbench.runner.stats import wilson_interval
from indicorderbench.schemas.results import Outcome, SuiteResult
from tests.helpers import PACK_HASH, make_suite

P, F, INV, INF = Outcome.PASS, Outcome.FAIL, Outcome.SIMULATOR_INVALID, Outcome.INFRA_ERROR

MIXED = {
    "en_quantity_01": [P, F],
    "en_modifier_01": [P, P],
    "hien_correction_01": [INV, P],
    "hien_cancellation_01": [INF, INF],
}


def render(tmp_path: Path, suite: SuiteResult, **kw: object) -> str:
    path = write_html(suite, tmp_path / "report", **kw)
    assert path == tmp_path / "report" / "report.html"
    return path.read_text(encoding="utf-8")


def block(html: str, tag: str, element_id: str) -> str:
    m = re.search(rf'<{tag}[^>]*id="{element_id}"[^>]*>(.*?)</{tag}>', html, re.DOTALL)
    assert m, f"no <{tag} id={element_id}>"
    return m.group(1)


def card_value(html: str, card_id: str) -> str:
    m = re.search(rf'id="{card_id}".*?class="value">(.*?)</', html, re.DOTALL)
    assert m, f"no card {card_id}"
    return m.group(1).strip()


def write_silence(path: Path, seconds: float = 0.1, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def test_header_cards_and_breakdowns(tmp_path: Path) -> None:
    html = render(tmp_path, make_suite(MIXED, agent_label="buggy"))
    assert "<title>IndicOrderBench · starter · buggy</title>" in html
    for card in ("card-pass-rate", "card-pass-k", "card-invalid", "card-infra", "card-latency"):
        assert f'id="{card}"' in html
    assert card_value(html, "card-pass-rate") == "83.3%"  # mean of 0.5, 1.0, 1.0
    assert card_value(html, "card-invalid") == "1"
    assert card_value(html, "card-infra") == "2"
    # hien_correction_01 has one valid trial, so pass^2 is unavailable and the card falls to k=1
    assert "pass<sup>1</sup>" in block(html, "article", "card-pass-k")
    assert card_value(html, "card-pass-k") == "83.3%"
    by_lang = block(html, "table", "by-language")
    lo, hi = wilson_interval(3, 4)
    assert "95% CI (Wilson)" in by_lang and "en-IN" in by_lang and "hi-en" in by_lang
    assert f"{pct(lo)} – {pct(hi)}" in by_lang
    by_cat = block(html, "table", "by-category")
    assert "cancellation" in by_cat and "—" in by_cat  # no valid trials -> no rate or CI
    assert 'id="comparison"' not in html


def test_scenarios_chips_checks_transcript_and_trace(tmp_path: Path) -> None:
    html = render(tmp_path, make_suite(MIXED))
    scenarios = block(html, "section", "scenarios")
    for sid in MIXED:
        assert f'<details id="scenario-{sid}"' in scenarios
    for chip in ("chip-pass", "chip-fail", "chip-invalid", "chip-infra"):
        assert f'<span class="chip {chip}">' in scenarios
    quantity = block(html, "details", "scenario-en_quantity_01")
    assert '<table class="field-checks">' in quantity
    fail_row = re.search(r'<tr class="fail">(.*?)</tr>', quantity, re.DOTALL)
    assert fail_row and "Paneer Wrap quantity" in fail_row.group(1)
    assert ">1<" in fail_row.group(1) and ">2<" in fail_row.group(1)
    assert '<tr class="pass">' in quantity
    assert '<ol class="transcript">' in quantity and "160 ms" in quantity
    assert '<table class="trace">' in quantity and "add_item" in quantity
    assert "submit_order" in quantity
    infra = block(html, "details", "scenario-hien_cancellation_01")
    assert "agent timed out after 30.0s" in infra
    invalid = block(html, "details", "scenario-hien_correction_01")
    assert "cannot answer" in invalid
    assert "<audio" not in html


def test_self_contained_escaped_and_well_formed(tmp_path: Path) -> None:
    suite = make_suite(MIXED)
    suite.scenarios[0].trials[0].transcript[1].text = "<b>bold</b> & <script>x()</script>"
    html = render(tmp_path, suite)
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; &lt;script&gt;x()&lt;/script&gt;" in html
    assert "<b>bold</b>" not in html
    assert "<style>" in html and "prefers-color-scheme: dark" in html
    for external in ("http://", "https://", "<link", "@import", "url(", "<script src"):
        assert external not in html
    assert html.count("<script>") == 1  # only the expand/collapse toggle
    for tag in ("details", "section", "table", "tr", "ol", "li", "article", "footer"):
        opened = len(re.findall(rf"<{tag}[\s>]", html))
        assert opened == html.count(f"</{tag}>"), tag


def test_footer_has_version_hash_and_timestamps(tmp_path: Path) -> None:
    html = render(tmp_path, make_suite(MIXED))
    footer = block(html, "footer", "footer")
    assert "IndicOrderBench 0.1.0" in footer and PACK_HASH in footer
    assert "2026-10-01T09:30:00+00:00" in footer and "2026-10-01T09:32:05+00:00" in footer


def test_audio_clip_is_copied_and_referenced(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    write_silence(clip)
    html = render(tmp_path, make_suite({"en_quantity_01": [P, F]}, with_clip=clip))
    copied = tmp_path / "report" / "assets" / "en_quantity_01" / "t1.wav"
    assert copied.read_bytes() == clip.read_bytes()
    assert html.count('<audio controls src="assets/en_quantity_01/t1.wav">') == 2
    assert str(tmp_path) not in html  # only relative references


def test_audio_skipped_when_not_copying_or_missing(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    write_silence(clip)
    html = render(tmp_path, make_suite({"en_quantity_01": [P]}, with_clip=clip), copy_audio=False)
    assert "<audio" not in html and not (tmp_path / "report" / "assets").exists()
    missing = make_suite({"en_quantity_01": [P]}, with_clip=tmp_path / "gone.wav")
    html = render(tmp_path, missing)
    assert "<audio" not in html


def test_comparison_section(tmp_path: Path) -> None:
    baseline = make_suite({"en_quantity_01": [P, P], "en_modifier_01": [P, P]}, agent_label="v1")
    current = make_suite({"en_quantity_01": [P, F], "hien_correction_01": [P, P]})
    html = render(tmp_path, current, baseline=baseline, comparison=compare(baseline, current))
    section = block(html, "section", "comparison")
    assert re.search(r'<tr class="regressed">.*?en_quantity_01', section, re.DOTALL)
    assert '<tr class="new">' in section and '<tr class="removed">' in section
    assert "-25.0%" in section and "Regressed" in section and "v1" in section
    assert 'href="#scenario-en_quantity_01"' in section


def test_baseline_without_comparison_is_compared(tmp_path: Path) -> None:
    baseline = make_suite({"en_quantity_01": [P, F]})
    current = make_suite({"en_quantity_01": [P, P]})
    section = block(render(tmp_path, current, baseline=baseline), "section", "comparison")
    assert '<tr class="improved">' in section and "No regression" in section


def test_every_trial_infra_error_shows_dashes(tmp_path: Path) -> None:
    html = render(tmp_path, make_suite({"en_quantity_01": [INF], "hien_modifier_01": [INF]}))
    for card in ("card-pass-rate", "card-pass-k", "card-latency"):
        assert card_value(html, card) == "—", card
    assert card_value(html, "card-infra") == "2"
    assert '<table class="field-checks">' not in html


@pytest.mark.parametrize("trials", [1, 3])
def test_pass_k_card_uses_run_trials(tmp_path: Path, trials: int) -> None:
    html = render(tmp_path, make_suite({"en_quantity_01": [P] * trials}))
    assert f"pass<sup>{trials}</sup>" in html
    assert card_value(html, "card-pass-k") == "100.0%"


def test_pass_k_card_falls_back_to_highest_available_k(tmp_path: Path) -> None:
    # 3 trials per scenario, one simulator-invalid trial: pass^3 is None, pass^2 is not.
    suite = make_suite({"en_quantity_01": [P, P, F], "en_modifier_01": [P, INV, P]})
    assert suite.metrics.pass_k[3] is None
    assert suite.metrics.pass_k[2] == pytest.approx((1 / 3 + 1) / 2)
    html = render(tmp_path, suite)
    card = block(html, "article", "card-pass-k")
    assert "pass<sup>2</sup>" in card and "pass<sup>3</sup>" not in card
    assert card_value(html, "card-pass-k") == "66.7%"
    assert "needs 3 valid trials in every scenario" in card


def test_distinct_clips_never_overwrite_and_names_stay_inside_assets(tmp_path: Path) -> None:
    first, second, third = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"
    for i, clip in enumerate((first, second, third), start=1):
        write_silence(clip, seconds=0.1 * i)
    suite = make_suite({"en_quantity_01": [P, F]}, with_clip=first)
    trial2 = suite.scenarios[0].trials[1].transcript
    trial2[0].audio_path = str(second)  # same turn id t1, different clip
    trial2[2].audio_path, trial2[2].turn_id = str(third), "../../evil"
    html = render(tmp_path, suite)
    assets = tmp_path / "report" / "assets" / "en_quantity_01"
    assert (assets / "t1.wav").read_bytes() == first.read_bytes()
    assert (assets / "t1-2.wav").read_bytes() == second.read_bytes()
    assert (assets / "______evil.wav").read_bytes() == third.read_bytes()
    assert 'src="assets/en_quantity_01/t1-2.wav"' in html
    assert sorted(p.name for p in assets.iterdir()) == ["______evil.wav", "t1-2.wav", "t1.wav"]
