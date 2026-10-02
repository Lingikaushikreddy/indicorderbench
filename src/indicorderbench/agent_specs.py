"""Turn an ``--agent`` spec string into an adapter factory.

Grammar::

    builtin:correct                 reference agent, no bugs
    builtin:buggy                   reference agent, every bug
    builtin:buggy:<bug>[,<bug>...]  reference agent, the named bugs
    http:<url>                      HTTP turn adapter; the runner serves the sandbox backend
    python:<module>:<factory>       in-process AgentFactory imported from your code
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from indicorderbench.adapters.inprocess import InProcessAdapter
from indicorderbench.adapters.protocol import AgentAdapter, AgentFactory, Modality
from indicorderbench.audio.transcribe import Transcriber

GRAMMAR = "builtin:correct | builtin:buggy[:<bug>,...] | http:<url> | python:<module>:<factory>"


class AgentSpecError(ValueError):
    pass


@dataclass(frozen=True)
class AgentSetup:
    factory: Callable[[], AgentAdapter]
    label: str
    needs_backend_server: bool


def _oracle_transcriber(clips_dir: Path | None) -> Transcriber | None:
    if clips_dir is None:
        return None
    manifest_path = clips_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    from indicorderbench.audio.manifest import load_manifest
    from indicorderbench.audio.transcribe import OracleTranscriber

    return OracleTranscriber(load_manifest(manifest_path))


def _builtin(spec: str, modality: Modality, clips_dir: Path | None) -> AgentSetup:
    from indicorderbench.agents.rule_based import make_factory, parse_agent_spec

    try:
        bugs = parse_agent_spec(spec)
    except ValueError as e:
        raise AgentSpecError(str(e)) from e
    if bugs is None:  # defensive: parse_agent_spec only returns None for non-builtin specs
        raise AgentSpecError(f"unknown builtin agent {spec!r}")
    transcriber = _oracle_transcriber(clips_dir) if modality is not Modality.TEXT else None
    agent_factory = make_factory(bugs, transcriber=transcriber)
    return AgentSetup(
        factory=lambda: InProcessAdapter(agent_factory), label=spec, needs_backend_server=False
    )


def _import_module(name: str) -> ModuleType:
    """Import ``name``, also from the working directory.

    Console scripts leave the working directory off ``sys.path`` (``python -m`` adds it), so
    ``python:my_agent:make_agent`` for a ``my_agent.py`` next to the shell would otherwise
    fail. The directory is added only when the import cannot be resolved without it.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        cwd = os.getcwd()
        if cwd in sys.path:
            raise
        sys.path.insert(0, cwd)
        importlib.invalidate_caches()
        return importlib.import_module(name)


def _python(spec: str) -> AgentSetup:
    _, _, rest = spec.partition(":")
    module_name, sep, attr = rest.rpartition(":")
    if not sep or not module_name or not attr:
        raise AgentSpecError(f"python spec must be python:<module>:<factory>, got {spec!r}")
    try:
        module = _import_module(module_name)
    except ImportError as e:
        raise AgentSpecError(f"cannot import module {module_name!r}: {e}") from e
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise AgentSpecError(f"{module_name}:{attr} is missing or not callable")
    typed: AgentFactory = factory
    return AgentSetup(
        factory=lambda: InProcessAdapter(typed), label=spec, needs_backend_server=False
    )


def _http(spec: str, timeout_s: float) -> AgentSetup:
    from indicorderbench.adapters.http import HttpTurnAdapter

    url = spec[len("http:") :]
    if not url.startswith(("http://", "https://")):
        raise AgentSpecError(f"http spec needs a full URL, got {url!r}")
    return AgentSetup(
        factory=lambda: HttpTurnAdapter(url, timeout_s=timeout_s),
        label=spec,
        needs_backend_server=True,
    )


def build_adapter_factory(
    spec: str,
    modality: Modality = Modality.TEXT,
    pack_root: Path | None = None,
    timeout_s: float = 30.0,
    clips_dir: Path | None = None,
) -> AgentSetup:
    """Resolve an agent spec. Raises :class:`AgentSpecError` with the grammar on bad input.

    In audio modality the built-in agents transcribe through the clip manifest, read from
    ``clips_dir`` (default ``<pack_root>/clips``).
    """
    if clips_dir is None and pack_root is not None:
        clips_dir = pack_root / "clips"
    if spec.startswith("builtin:"):
        return _builtin(spec, modality, clips_dir)
    if spec.startswith("python:"):
        return _python(spec)
    if spec.startswith("http:"):
        return _http(spec, timeout_s)
    raise AgentSpecError(f"unrecognised agent spec {spec!r}; expected {GRAMMAR}")
