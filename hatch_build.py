"""Hatch build hook: bundle the starter pack in the wheel without any audio clips.

A static ``force-include`` ignores VCS ignores, so a maintainer who ran ``iob synth`` would
publish the WAV clips. This hook adds every pack file except ``*.wav`` instead;
``clips/manifest.json`` is kept when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PACK = Path("packs") / "starter"
TARGET = "indicorderbench/data/packs/starter"


class StarterPackBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root) / PACK
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() == ".wav":
                continue
            rel = path.relative_to(root).as_posix()
            build_data["force_include"][str(path.resolve())] = f"{TARGET}/{rel}"
