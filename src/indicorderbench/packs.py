"""Load and validate scenario packs from a directory.

A pack is a directory with ``pack.yaml`` (manifest and per-language caller defaults),
a menu file, ``scenarios/*.yaml`` and optional ``clips/`` audio.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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

DEFAULT_TURN_IDS = ("closing", "confirm", "fallback", "nudge")


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

        An explicit ``audio`` field is relative to the pack root. Otherwise scripted and
        clarification turns live at ``clips/<scenario>/<turn id>.wav`` and the language
        defaults at ``clips/_defaults/<language>/<turn id>.wav``.
        """
        if turn.audio:
            p = self.root / turn.audio
            return p if p.exists() else None
        if turn.id in DEFAULT_TURN_IDS:
            p = self.root / "clips" / "_defaults" / language.value / f"{turn.id}.wav"
        else:
            p = self.root / "clips" / scenario_id / f"{turn.id}.wav"
        return p if p.exists() else None

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


def _load(root: Path) -> tuple[Pack | None, list[str]]:
    problems: list[str] = []
    root = Path(root)
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


def load_pack(root: Path) -> Pack:
    pack, problems = _load(Path(root))
    if pack is None:
        raise PackError(problems or ["unknown pack error"])
    return pack
