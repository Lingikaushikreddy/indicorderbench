"""Write and read ``results.json`` (a serialised :class:`SuiteResult`)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from indicorderbench.schemas.results import SuiteResult

SCHEMA_VERSION: int = SuiteResult.model_fields["schema_version"].default


class ReportError(Exception):
    """A results file or a comparison that cannot be used."""


def write_json(suite: SuiteResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(suite.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> SuiteResult:
    """Load a results file written by :func:`write_json` with the current schema version."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ReportError(f"cannot read {path}: {e.strerror or e}") from e
    except ValueError as e:  # JSONDecodeError and UnicodeDecodeError
        raise ReportError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ReportError(f"{path} is not a valid results file: expected a JSON object")
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ReportError(
            f"{path} has schema_version {version}; this version of indicorderbench reads "
            f"schema_version {SCHEMA_VERSION}"
        )
    try:
        return SuiteResult.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise ReportError(
            f"{path} is not a valid results file: {e.error_count()} problem(s), "
            f"first at {where}: {first['msg']}"
        ) from e
