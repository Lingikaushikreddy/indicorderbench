"""The wheel bundles the starter pack but never ships locally synthesised audio."""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")
def test_wheel_bundles_the_starter_pack_without_wav_clips(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "hatch_build.py", "README.md", "LICENSE"):
        if (REPO / name).exists():
            shutil.copy2(REPO / name, project / name)
    skip = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO / "src", project / "src", ignore=skip)
    shutil.copytree(REPO / "packs", project / "packs", ignore=skip)
    clip = project / "packs" / "starter" / "clips" / "en_quantity_01" / "t1.wav"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"RIFF" + b"\0" * 64)
    (project / "packs" / "starter" / "clips" / "manifest.json").write_text('{"clips": []}')

    dist = tmp_path / "dist"
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, wheels
    names = zipfile.ZipFile(wheels[0]).namelist()
    prefix = "indicorderbench/data/packs/starter/"
    assert f"{prefix}pack.yaml" in names
    assert f"{prefix}menu.yaml" in names
    assert f"{prefix}clips/manifest.json" in names
    assert any(n.startswith(f"{prefix}scenarios/") and n.endswith(".yaml") for n in names)
    assert [n for n in names if n.lower().endswith(".wav")] == []
