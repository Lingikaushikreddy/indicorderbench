"""Load and validate scenario packs from a directory.

A pack is a directory with ``pack.yaml`` (manifest and per-language caller defaults),
a menu file, ``scenarios/*.yaml`` and optional ``clips/`` audio.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from indicorderbench.schemas.menu import Menu
from indicorderbench.schemas.scenario import (
    CallerDefaults,
    CallerTurn,
    Language,
    PackManifest,
    Scenario,
)


class PackError(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass
class Pack:
    root: Path
    manifest: PackManifest
    menu: Menu
    scenarios: list[Scenario]
    clips_dir: Path | None = None  # an alternate clip set; None means <root>/clips

    @property
    def clips(self) -> Path:
        """The directory clips are resolved from."""
        return self.clips_dir if self.clips_dir is not None else self.root / "clips"

    def with_clips(self, clips_dir: Path) -> Pack:
        """This pack resolving clips from ``clips_dir`` (``iob perturb`` output, for example)
        instead of ``<root>/clips``. The pack on disk is untouched."""
        return replace(self, clips_dir=Path(clips_dir).resolve())

    def scenario(self, scenario_id: str) -> Scenario:
        for s in self.scenarios:
            if s.id == scenario_id:
                return s
        raise KeyError(scenario_id)

    def filter(
        self,
        tags: list[str] | None = None,
        languages: list[str] | None = None,
        categories: list[str] | None = None,
        ids: list[str] | None = None,
    ) -> list[Scenario]:
        out = []
        for s in self.scenarios:
            if tags and not set(tags) & set(s.tags):
                continue
            if languages and s.language.value not in languages:
                continue
            if categories and s.category.value not in categories:
                continue
            if ids and s.id not in ids:
                continue
            out.append(s)
        return out

    def defaults_for(self, language: Language) -> CallerDefaults:
        return self.manifest.defaults.caller[language]

    def clip_path(self, scenario_id: str, turn: CallerTurn, language: Language) -> Path | None:
        """Resolve a turn's audio clip, or None when no clip exists.

        An explicit ``audio`` field is relative to the pack root. Otherwise, with ``<clips>``
        being :attr:`clips`, a turn's clip is ``<clips>/<scenario>/<turn id>.wav`` (scripted
        turns, clarification replies and a scenario's own closing/confirm/fallback/nudge
        overrides), falling back to ``<clips>/_defaults/<language>/<turn id>.wav`` (the
        pack-wide default turns and clarification replies, synthesised once per language).
        """
        if turn.audio:
            p = self.root / turn.audio
            return p if p.exists() else None
        specific = self.clips / scenario_id / f"{turn.id}.wav"
        if specific.exists():
            return specific
        shared = self.clips / "_defaults" / language.value / f"{turn.id}.wav"
        return shared if shared.exists() else None

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.root.rglob("*.yaml")):
            h.update(p.relative_to(self.root).as_posix().encode())
            h.update(p.read_bytes())
        return h.hexdigest()[:16]


def _fmt_validation(path: Path, err: ValidationError) -> list[str]:
    return [
        f"{path}: {'.'.join(str(x) for x in e['loc']) or '<root>'}: {e['msg']}"
        for e in err.errors()
    ]


def _load_yaml(path: Path, problems: list[str]) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        problems.append(f"{path}: {e}")
        return None


def _check_references(path: Path, scenario: Scenario, menu: Menu, problems: list[str]) -> None:
    if scenario.menu != menu.id:
        problems.append(
            f"{path}: menu: references {scenario.menu!r} but the pack menu is {menu.id!r}"
        )
    for es in scenario.expected:
        for oi, order in enumerate(es.orders):
            for li, line in enumerate(order.lines):
                loc = f"expected[{es.id}].orders[{oi}].lines[{li}]"
                if not menu.has_item(line.item_id):
                    problems.append(f"{path}: {loc}.item_id: unknown item {line.item_id!r}")
                    continue
                groups = {g.id: g for g in menu.groups_for(line.item_id)}
                for gid, spec in line.modifiers.items():
                    if gid not in groups:
                        problems.append(
                            f"{path}: {loc}.modifiers.{gid}: group not applicable to "
                            f"{line.item_id!r}"
                        )
                        continue
                    if spec == "*":
                        continue
                    opts = [spec] if isinstance(spec, str) else list(spec)
                    for o in opts:
                        if not any(o == x.id for x in groups[gid].options):
                            problems.append(f"{path}: {loc}.modifiers.{gid}: unknown option {o!r}")
                    if groups[gid].exclusive and len(opts) > 1:
                        problems.append(
                            f"{path}: {loc}.modifiers.{gid}: exclusive group given "
                            f"{len(opts)} options"
                        )


def _check_clips(path: Path, root: Path, scenario: Scenario, problems: list[str]) -> None:
    """Every clip a scenario names explicitly with ``audio:`` must exist under the pack root."""
    script = scenario.caller
    referenced: list[tuple[str, CallerTurn]] = [
        (f"caller.turns[{i}]", turn) for i, turn in enumerate(script.turns)
    ]
    referenced += [
        (f"caller.clarifications[{i}].reply", rule.reply)
        for i, rule in enumerate(script.clarifications)
    ]
    for name in ("closing", "confirm", "fallback", "nudge"):
        override: CallerTurn | None = getattr(script, name)
        if override is not None:
            referenced.append((f"caller.{name}", override))
    for loc, turn in referenced:
        if turn.audio and not (root / turn.audio).exists():
            problems.append(f"{path}: {loc}.audio: missing clip {turn.audio}")


def _load(root: Path, check_clips: bool = True) -> tuple[Pack | None, list[str]]:
    problems: list[str] = []
    root = Path(root).resolve()
    manifest_path = root / "pack.yaml"
    if not manifest_path.exists():
        return None, [f"{manifest_path}: missing"]
    raw = _load_yaml(manifest_path, problems)
    manifest: PackManifest | None = None
    if raw is not None:
        try:
            manifest = PackManifest.model_validate(raw)
        except ValidationError as e:
            problems.extend(_fmt_validation(manifest_path, e))
    if manifest is None:
        return None, problems

    menu: Menu | None = None
    menu_path = root / manifest.menu
    if not menu_path.exists():
        problems.append(f"{menu_path}: missing")
    else:
        raw_menu = _load_yaml(menu_path, problems)
        if raw_menu is not None:
            try:
                menu = Menu.model_validate(raw_menu)
            except ValidationError as e:
                problems.extend(_fmt_validation(menu_path, e))

    scenarios: list[Scenario] = []
    seen: dict[str, Path] = {}
    for path in sorted((root / "scenarios").glob("*.yaml")):
        raw_s = _load_yaml(path, problems)
        if raw_s is None:
            continue
        try:
            s = Scenario.model_validate(raw_s)
        except ValidationError as e:
            problems.extend(_fmt_validation(path, e))
            continue
        if s.id in seen:
            problems.append(f"{path}: duplicate scenario id {s.id!r} (also in {seen[s.id].name})")
            continue
        seen[s.id] = path
        if s.language not in manifest.defaults.caller:
            problems.append(
                f"{path}: language {s.language.value!r} has no caller defaults in pack.yaml"
            )
        if menu is not None:
            _check_references(path, s, menu, problems)
        if check_clips:
            _check_clips(path, root, s, problems)
        scenarios.append(s)
    if not scenarios and not problems:
        problems.append(f"{root / 'scenarios'}: no scenarios found")
    if problems or menu is None:
        return None, problems
    return Pack(root=root, manifest=manifest, menu=menu, scenarios=scenarios), []


def validate_pack(root: Path) -> list[str]:
    """Return every problem found in the pack; an empty list means it is valid."""
    _, problems = _load(Path(root))
    return problems


def load_pack(root: Path, check_clips: bool = True) -> Pack:
    """Load a pack or raise :class:`PackError` listing every problem.

    ``check_clips=False`` skips the check that explicitly referenced ``audio:`` clips exist,
    for ``iob synth``, which creates them.
    """
    pack, problems = _load(Path(root), check_clips)
    if pack is None:
        raise PackError(problems or ["unknown pack error"])
    return pack
