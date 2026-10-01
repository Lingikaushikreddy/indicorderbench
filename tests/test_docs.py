"""The README and docs must stay consistent with what the package actually does."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md")), REPO / "CONTRIBUTING.md"]
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")


def test_readme_first_screen():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Did it actually place the correct order?" in text
    assert "pip install indicorderbench" in text
    assert "iob demo" in text
    assert "| Field" in text and "Paneer Wrap quantity" in text
    assert "docs/demo/fixed/report.html" in text
    assert "Hinglish" in text and "unreviewed" in text


def test_local_links_resolve():
    broken: list[str] = []
    for doc in DOCS:
        for target in LINK_RE.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path = (doc.parent / target).resolve()
            if not path.exists():
                broken.append(f"{doc.relative_to(REPO)} -> {target}")
    assert broken == []


def test_demo_artifacts_are_committed():
    for sub in ("buggy", "fixed", "demo-scenario"):
        for name in ("results.json", "report.html"):
            assert (REPO / "docs" / "demo" / sub / name).exists(), f"{sub}/{name} missing"


def test_changelog_mentions_release():
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [0.1.0]" in text and "42 scenarios" in text


def test_scenarios_doc_never_recommends_an_anything_else_clarification():
    """A pack-level rule matching "anything else" pre-empts scripted turn 2 of every scenario."""
    text = (REPO / "docs" / "scenarios.md").read_text(encoding="utf-8")
    match_lists = re.findall(r"match:\s*\[([^\]]*)\]", text)
    assert match_lists, "expected at least one clarification example"
    assert [m for m in match_lists if "anything else" in m.lower()] == []
