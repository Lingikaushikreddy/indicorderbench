"""Clip manifest: what text each synthesized clip contains and where it came from."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ClipEntry(BaseModel):
    path: str  # relative to the pack root, forward slashes
    sha256: str
    scenario_id: str
    turn_id: str
    text: str
    language: str
    provider: str
    voice: str
    model: str
    created_at: datetime


class ClipManifest(BaseModel):
    version: int = 1
    clips: list[ClipEntry] = Field(default_factory=list)
    derived_from: str | None = None
    perturbation: dict[str, Any] | None = None

    def by_hash(self) -> dict[str, ClipEntry]:
        return {c.sha256: c for c in self.clips}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> ClipManifest:
    return ClipManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(m: ClipManifest, path: Path) -> None:
    """Write atomically: temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(m.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)
