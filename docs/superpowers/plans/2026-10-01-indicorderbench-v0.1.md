# IndicOrderBench v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pip-installable benchmark that drives a voice ordering agent through English and Hinglish scripted scenarios and deterministically reports whether the committed order matched, with JSON, JUnit and HTML output and a no-API-key demo.

**Architecture:** Pydantic schemas are the shared contract. A sandbox `OrderBackend` records every tool call. A `ScriptedCaller` state machine talks to the agent through an `AgentAdapter`. The `Runner` classifies each trial as PASS, FAIL, SIMULATOR_INVALID or INFRA_ERROR using a deterministic `Checker`, and reporters render the result. A rule-based reference agent with switchable bugs makes the demo and the integration tests run offline.

**Tech Stack:** Python 3.11+, uv, pydantic 2, PyYAML, Typer, Jinja2, httpx, pytest + pytest-asyncio, ruff, mypy; optional numpy + soundfile for audio.

**Spec:** `docs/superpowers/specs/2026-10-01-indicorderbench-design.md`

## Global Constraints

- Python `>=3.11`; package name `indicorderbench`; CLI entry point `iob`; src layout `src/indicorderbench/`.
- Core dependencies only: `pydantic>=2.7`, `pyyaml>=6`, `typer>=0.12`, `jinja2>=3.1`, `httpx>=0.27`. Extra `audio`: `numpy>=1.26`, `soundfile>=0.12`. Extra `dev`: `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `types-PyYAML`.
- Nothing in `schemas/` imports from any other package module.
- Ids are `^[a-z][a-z0-9_]*$`. Languages are `en-IN` and `hi-en`. Categories are `quantity`, `modifier`, `correction`, `cancellation`, `duplicate_submission`.
- Outcome priority: `INFRA_ERROR` > `SIMULATOR_INVALID` > `PASS`/`FAIL`. Metrics are computed over valid trials only; invalid and infra counts are always shown.
- Only orders with `status == "submitted"` count as committed.
- Reports must work from `file://` with no external assets.
- Exit codes: 0 ok, 1 usage/infra, 2 accuracy below threshold or regression.
- Commit messages: conventional prefix (`feat:`, `test:`, `docs:`, `chore:`), no AI attribution lines.
- Default runner limits: `max_turns=20`, `timeout_turn_s=30`, `timeout_trial_s=300`, `trials=1`.

## Review Focus

1. **Agent reply with no text (empty string or whitespace)** after the script is exhausted: the caller must treat it as "not a question", nudge, and finish valid; the checker then scores the state. Test added to Task 4.
2. **Agent submits twice with identical content**: canonical comparison must not merge orders; `Submitted orders` must read expected 1, actual 2, and the trial must FAIL. Test added to Task 3.
3. **Pack scenario referencing an unknown item id or option id in `expected`**: `iob validate` must list the file and field and exit 1 rather than crash during a run. Test added to Task 8.
4. **Adapter that hangs on one turn**: the per-turn timeout must convert that trial to `INFRA_ERROR` with a readable message, and the suite must continue with the next trial. Test added to Task 6.
5. **Baseline JSON from an older `schema_version` or a different pack id**: `iob compare` must refuse with a clear message and exit 1, never silently compare unrelated runs. Test added to Task 9.

---

## Phase A — core (sequential; these define the interfaces)

### Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/indicorderbench/__init__.py`, `src/indicorderbench/py.typed`, `tests/__init__.py`, `tests/conftest.py`, `.gitignore`, `.python-version`, `ruff.toml` (inside pyproject), `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `indicorderbench.__version__ == "0.1.0"`; `uv run pytest` works; `uv run ruff check .` and `uv run mypy src` run.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "indicorderbench"
version = "0.1.0"
description = "Catches multilingual voice agents placing the wrong order. A τ-bench style benchmark for voice ordering in Indian languages."
readme = "README.md"
requires-python = ">=3.11"
license = {text = "Apache-2.0"}
authors = [{name = "Kaushik Reddy Lingireddy"}]
keywords = ["voice-agents", "benchmark", "evaluation", "hinglish", "indic", "livekit", "tool-calling"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Developers",
  "License :: OSI Approved :: Apache Software License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Topic :: Software Development :: Testing",
]
dependencies = [
  "pydantic>=2.7",
  "pyyaml>=6.0",
  "typer>=0.12",
  "jinja2>=3.1",
  "httpx>=0.27",
]

[project.optional-dependencies]
audio = ["numpy>=1.26", "soundfile>=0.12"]

[project.scripts]
iob = "indicorderbench.cli:app"

[project.urls]
Homepage = "https://github.com/Lingikaushikreddy/indicorderbench"

[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/indicorderbench"]

[dependency-groups]
dev = [
  "pytest>=8.3",
  "pytest-asyncio>=0.24",
  "ruff>=0.6",
  "mypy>=1.11",
  "types-PyYAML",
  "numpy>=1.26",
  "soundfile>=0.12",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
filterwarnings = ["error::DeprecationWarning:indicorderbench.*"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
ignore = ["RUF001", "RUF002", "RUF003"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["indicorderbench"]
mypy_path = "src"
[[tool.mypy.overrides]]
module = ["soundfile", "soundfile.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Write package init, gitignore, python-version**

`src/indicorderbench/__init__.py`:
```python
"""IndicOrderBench: catches multilingual voice agents placing the wrong order."""

__version__ = "0.1.0"
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
dist/
build/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
results/
demo-out/
packs/*/clips/**/*.wav
!packs/*/clips/manifest.json
.DS_Store
```

`.python-version`: `3.13`

`tests/conftest.py`:
```python
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def starter_pack_dir(repo_root: Path) -> Path:
    return repo_root / "packs" / "starter"
```

- [ ] **Step 3: Write CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python }}
      - run: uv sync --all-extras --dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy
      - run: uv run pytest -q
      - run: uv run iob validate packs/starter
      - run: uv run iob run packs/starter --agent builtin:correct --tag smoke --out results/smoke --fail-under 1.0
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: smoke-report-${{ matrix.python }}
          path: results/smoke
```

- [ ] **Step 4: Sync and verify tooling**

Run: `uv sync --all-extras --dev && uv run python -c "import indicorderbench; print(indicorderbench.__version__)"`
Expected: `0.1.0`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src tests .gitignore .python-version .github
git commit -m "chore: project scaffold with uv, ruff, mypy, pytest, CI"
```

---

### Task 1: Schemas (menu, scenario, results)

**Files:**
- Create: `src/indicorderbench/schemas/__init__.py`, `schemas/menu.py`, `schemas/scenario.py`, `schemas/results.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces everything below. Later tasks import exactly these names.

- [ ] **Step 1: Write failing tests**

`tests/test_schemas.py`:
```python
import pytest
from pydantic import ValidationError

from indicorderbench.schemas.menu import Menu, MenuItem, Modifier, ModifierGroup, normalise_text
from indicorderbench.schemas.results import BackendSnapshot, CartLine, Outcome, SubmittedOrder
from indicorderbench.schemas.scenario import (
    CallerDefaults,
    CallerScript,
    CallerTurn,
    ClarificationRule,
    EndState,
    ExpectedLine,
    ExpectedOrder,
    FailureCategory,
    Language,
    Scenario,
)


def make_menu() -> Menu:
    return Menu(
        id="m",
        name="M",
        modifier_groups=[
            ModifierGroup(
                id="onion",
                name="Onion",
                exclusive=True,
                default="with_onion",
                options=[
                    Modifier(id="with_onion", name="With onion"),
                    Modifier(id="no_onion", name="No onion", aliases=["no onion", "bina pyaaz"]),
                ],
            ),
            ModifierGroup(
                id="extras",
                name="Extras",
                exclusive=False,
                default=None,
                options=[Modifier(id="extra_cheese", name="Extra cheese", aliases=["extra cheese"])],
            ),
        ],
        items=[
            MenuItem(id="paneer_wrap", name="Paneer Wrap", aliases=["paneer wrap", "paneer roll"],
                     price=180, modifier_groups=["onion", "extras"]),
            MenuItem(id="mango_lassi", name="Mango Lassi", aliases=["mango lassi"], price=90),
        ],
    )


def test_normalise_text_strips_punctuation_and_case():
    assert normalise_text("  Two Paneer Wraps... actually, ONE! ") == "two paneer wraps actually one"


def test_menu_lookups():
    menu = make_menu()
    assert menu.item("paneer_wrap").name == "Paneer Wrap"
    assert menu.option_group("no_onion").id == "onion"
    assert [g.id for g in menu.groups_for("paneer_wrap")] == ["onion", "extras"]
    assert menu.groups_for("mango_lassi") == []
    with pytest.raises(KeyError):
        menu.item("nope")


def test_menu_rejects_duplicate_ids_and_unknown_group():
    with pytest.raises(ValidationError):
        Menu(id="m", name="M", modifier_groups=[], items=[
            MenuItem(id="a", name="A", price=1), MenuItem(id="a", name="A2", price=1)])
    with pytest.raises(ValidationError):
        Menu(id="m", name="M", modifier_groups=[], items=[
            MenuItem(id="a", name="A", price=1, modifier_groups=["ghost"])])


def test_exclusive_group_requires_default_option():
    with pytest.raises(ValidationError):
        ModifierGroup(id="g", name="G", exclusive=True, default="missing",
                      options=[Modifier(id="x", name="X")])
    with pytest.raises(ValidationError):
        ModifierGroup(id="g", name="G", exclusive=False, default="x",
                      options=[Modifier(id="x", name="X")])


def test_alias_index_is_longest_first_and_normalised():
    menu = make_menu()
    index = menu.alias_index()
    assert index[0].alias == "paneer wrap" or index[0].alias == "paneer roll" or index[0].alias == "mango lassi" or index[0].alias == "bina pyaaz"
    assert all(len(index[i].alias) >= len(index[i + 1].alias) for i in range(len(index) - 1))
    kinds = {(e.kind, e.ref_id) for e in index}
    assert ("item", "paneer_wrap") in kinds and ("option", "no_onion") in kinds


def test_caller_script_assigns_turn_ids_and_resolves_defaults():
    script = CallerScript(turns=[CallerTurn(text="two wraps"), CallerTurn(text="no onion", id="custom")])
    assert [t.id for t in script.turns] == ["t1", "custom"]
    defaults = CallerDefaults(
        closing=CallerTurn(text="that's all"),
        confirm=CallerTurn(text="yes"),
        fallback=CallerTurn(text="yes that's right"),
        nudge=CallerTurn(text="please place the order"),
        confirm_patterns=["shall i place"],
        goodbye_patterns=["order placed"],
        question_patterns=["which", "how many"],
        clarifications=[ClarificationRule(id="any", match=["anything else"], reply=CallerTurn(text="no"))],
    )
    resolved = script.resolved(defaults)
    assert resolved.closing.text == "that's all"
    assert resolved.max_fallbacks == 2 and resolved.max_nudges == 2
    assert [r.id for r in resolved.clarifications] == ["any"]
    assert resolved.closing.id == "closing" and resolved.clarifications[0].reply.id == "c_any"


def test_clarification_rule_matches_case_insensitively():
    rule = ClarificationRule(id="r", match=["how many", "kitne"], reply=CallerTurn(text="two"))
    assert rule.matches("Sure, HOW MANY wraps?")
    assert not rule.matches("Added two wraps.")


def test_scenario_validates_shape():
    s = Scenario(
        id="en_quantity_01", title="t", language=Language.EN_IN, category=FailureCategory.QUANTITY,
        menu="m", caller=CallerScript(turns=[CallerTurn(text="two paneer wraps")]),
        expected=[EndState(id="one_order", orders=[ExpectedOrder(lines=[ExpectedLine(item_id="paneer_wrap", quantity=2)])])],
    )
    assert s.expected[0].orders[0].lines[0].modifiers == {}
    with pytest.raises(ValidationError):
        Scenario(id="Bad-Id", title="t", language=Language.EN_IN, category=FailureCategory.QUANTITY,
                 menu="m", caller=CallerScript(turns=[CallerTurn(text="x")]), expected=[EndState(id="e")])


def test_snapshot_active_orders_excludes_cancelled():
    snap = BackendSnapshot(cart=[], orders=[
        SubmittedOrder(order_id="o1", lines=[CartLine(line_id="l1", item_id="x", quantity=1)], submitted_at_ms=1.0, status="cancelled"),
        SubmittedOrder(order_id="o2", lines=[CartLine(line_id="l2", item_id="x", quantity=1)], submitted_at_ms=2.0),
    ], trace=[])
    assert [o.order_id for o in snap.active_orders()] == ["o2"]
    assert Outcome.PASS.value == "pass"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_schemas.py -q`
Expected: ImportError on `indicorderbench.schemas`.

- [ ] **Step 3: Implement `schemas/menu.py`**

```python
"""Menu schema: items, modifier groups, alias lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ID_PATTERN = r"^[a-z][a-z0-9_]*$"

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


@dataclass(frozen=True)
class AliasEntry:
    alias: str  # normalised
    kind: Literal["item", "option"]
    ref_id: str


class Modifier(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ID_PATTERN)
    name: str
    aliases: list[str] = Field(default_factory=list)
    price_delta: Decimal = Decimal("0")


class ModifierGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ID_PATTERN)
    name: str
    exclusive: bool = True
    default: str | None = None
    options: list[Modifier] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ModifierGroup:
        ids = [o.id for o in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate option ids in group {self.id}")
        if self.exclusive:
            if self.default is None or self.default not in ids:
                raise ValueError(f"exclusive group {self.id} needs a default that is one of its options")
        elif self.default is not None:
            raise ValueError(f"non-exclusive group {self.id} must not declare a default")
        return self

    def option(self, option_id: str) -> Modifier:
        for o in self.options:
            if o.id == option_id:
                return o
        raise KeyError(option_id)


class MenuItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ID_PATTERN)
    name: str
    aliases: list[str] = Field(default_factory=list)
    price: Decimal
    category: str = "general"
    modifier_groups: list[str] = Field(default_factory=list)


class Menu(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=ID_PATTERN)
    name: str
    currency: str = "INR"
    modifier_groups: list[ModifierGroup] = Field(default_factory=list)
    items: list[MenuItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Menu:
        group_ids = [g.id for g in self.modifier_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("duplicate modifier group ids")
        option_ids = [o.id for g in self.modifier_groups for o in g.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("option ids must be unique across groups")
        item_ids = [i.id for i in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("duplicate item ids")
        if set(item_ids) & set(option_ids):
            raise ValueError("item ids and option ids must not overlap")
        for item in self.items:
            for gid in item.modifier_groups:
                if gid not in group_ids:
                    raise ValueError(f"item {item.id} references unknown modifier group {gid}")
        return self

    def item(self, item_id: str) -> MenuItem:
        for i in self.items:
            if i.id == item_id:
                return i
        raise KeyError(item_id)

    def group(self, group_id: str) -> ModifierGroup:
        for g in self.modifier_groups:
            if g.id == group_id:
                return g
        raise KeyError(group_id)

    def option_group(self, option_id: str) -> ModifierGroup:
        for g in self.modifier_groups:
            if any(o.id == option_id for o in g.options):
                return g
        raise KeyError(option_id)

    def option(self, option_id: str) -> Modifier:
        return self.option_group(option_id).option(option_id)

    def groups_for(self, item_id: str) -> list[ModifierGroup]:
        return [self.group(gid) for gid in self.item(item_id).modifier_groups]

    def has_item(self, item_id: str) -> bool:
        return any(i.id == item_id for i in self.items)

    def has_option(self, option_id: str) -> bool:
        return any(o.id == option_id for g in self.modifier_groups for o in g.options)

    def alias_index(self) -> list[AliasEntry]:
        """All item and option aliases (plus names), normalised, longest first."""
        entries: list[AliasEntry] = []
        for item in self.items:
            for a in [item.name, *item.aliases]:
                entries.append(AliasEntry(normalise_text(a), "item", item.id))
        for g in self.modifier_groups:
            for o in g.options:
                for a in o.aliases:  # option names like "Medium" are too generic to match by name
                    entries.append(AliasEntry(normalise_text(a), "option", o.id))
        entries = [e for e in entries if e.alias]
        return sorted(set(entries), key=lambda e: (-len(e.alias), e.alias))

    def search(self, query: str) -> list[MenuItem]:
        q = normalise_text(query)
        if not q:
            return []
        out: list[MenuItem] = []
        for item in self.items:
            haystacks = [normalise_text(item.name), *(normalise_text(a) for a in item.aliases)]
            if any(q in h or h in q for h in haystacks):
                out.append(item)
        return out
```

- [ ] **Step 4: Implement `schemas/scenario.py`**

```python
"""Scenario, caller script and pack manifest schemas."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from indicorderbench.schemas.menu import ID_PATTERN


class Language(str, Enum):
    EN_IN = "en-IN"
    HI_EN = "hi-en"


class FailureCategory(str, Enum):
    QUANTITY = "quantity"
    MODIFIER = "modifier"
    CORRECTION = "correction"
    CANCELLATION = "cancellation"
    DUPLICATE_SUBMISSION = "duplicate_submission"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"


class CallerTurn(BaseModel):
    id: str | None = None
    text: str = Field(min_length=1)
    audio: str | None = None  # path relative to pack root
    is_closing: bool = False


class ClarificationRule(BaseModel):
    id: str = Field(pattern=ID_PATTERN)
    match: list[str] = Field(min_length=1)
    reply: CallerTurn
    max_uses: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _ids(self) -> ClarificationRule:
        if self.reply.id is None:
            self.reply.id = f"c_{self.id}"
        for p in self.match:
            re.compile(p)
        return self

    def matches(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in self.match)


def _any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


class CallerDefaults(BaseModel):
    """Per-language defaults shared by all scenarios in a pack."""

    closing: CallerTurn
    confirm: CallerTurn
    fallback: CallerTurn
    nudge: CallerTurn
    confirm_patterns: list[str] = Field(min_length=1)
    goodbye_patterns: list[str] = Field(min_length=1)
    question_patterns: list[str] = Field(default_factory=list)
    clarifications: list[ClarificationRule] = Field(default_factory=list)
    max_fallbacks: int = Field(default=2, ge=0)
    max_nudges: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _ids(self) -> CallerDefaults:
        for name in ("closing", "confirm", "fallback", "nudge"):
            turn: CallerTurn = getattr(self, name)
            if turn.id is None:
                turn.id = name
        return self


class ResolvedCallerScript(BaseModel):
    model_config = ConfigDict(frozen=True)

    turns: list[CallerTurn]
    clarifications: list[ClarificationRule]
    closing: CallerTurn
    confirm: CallerTurn
    fallback: CallerTurn
    nudge: CallerTurn
    confirm_patterns: list[str]
    goodbye_patterns: list[str]
    question_patterns: list[str]
    max_fallbacks: int
    max_nudges: int

    def is_confirm_request(self, text: str) -> bool:
        return _any_match(self.confirm_patterns, text)

    def is_goodbye(self, text: str) -> bool:
        return _any_match(self.goodbye_patterns, text)

    def is_question(self, text: str) -> bool:
        return "?" in text or _any_match(self.question_patterns, text)


class CallerScript(BaseModel):
    turns: list[CallerTurn] = Field(min_length=1)
    clarifications: list[ClarificationRule] = Field(default_factory=list)
    closing: CallerTurn | None = None
    confirm: CallerTurn | None = None
    fallback: CallerTurn | None = None
    nudge: CallerTurn | None = None
    confirm_patterns: list[str] | None = None
    goodbye_patterns: list[str] | None = None
    question_patterns: list[str] | None = None
    max_fallbacks: int | None = Field(default=None, ge=0)
    max_nudges: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _turn_ids(self) -> CallerScript:
        seen: set[str] = set()
        for i, t in enumerate(self.turns):
            if t.id is None:
                t.id = f"t{i + 1}"
            if t.id in seen:
                raise ValueError(f"duplicate turn id {t.id}")
            seen.add(t.id)
        return self

    def resolved(self, defaults: CallerDefaults) -> ResolvedCallerScript:
        def pick(name: str, fallback_turn: CallerTurn) -> CallerTurn:
            turn: CallerTurn | None = getattr(self, name)
            if turn is None:
                return fallback_turn
            if turn.id is None:
                turn.id = name
            return turn

        return ResolvedCallerScript(
            turns=list(self.turns),
            clarifications=[*self.clarifications, *defaults.clarifications],
            closing=pick("closing", defaults.closing),
            confirm=pick("confirm", defaults.confirm),
            fallback=pick("fallback", defaults.fallback),
            nudge=pick("nudge", defaults.nudge),
            confirm_patterns=self.confirm_patterns or defaults.confirm_patterns,
            goodbye_patterns=self.goodbye_patterns or defaults.goodbye_patterns,
            question_patterns=self.question_patterns if self.question_patterns is not None else defaults.question_patterns,
            max_fallbacks=self.max_fallbacks if self.max_fallbacks is not None else defaults.max_fallbacks,
            max_nudges=self.max_nudges if self.max_nudges is not None else defaults.max_nudges,
        )


class ExpectedLine(BaseModel):
    item_id: str = Field(pattern=ID_PATTERN)
    quantity: int = Field(ge=1)
    modifiers: dict[str, str | list[str]] = Field(default_factory=dict)  # group_id -> option | [options] | "*"


class ExpectedOrder(BaseModel):
    lines: list[ExpectedLine] = Field(min_length=1)


class EndState(BaseModel):
    id: str = Field(pattern=ID_PATTERN)
    description: str = ""
    orders: list[ExpectedOrder] = Field(default_factory=list)


class Scenario(BaseModel):
    id: str = Field(pattern=ID_PATTERN)
    title: str
    language: Language
    category: FailureCategory
    menu: str
    difficulty: Difficulty = Difficulty.MEDIUM
    tags: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer_notes: str | None = None
    version: int = 1
    caller: CallerScript
    expected: list[EndState] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_end_states(self) -> Scenario:
        ids = [e.id for e in self.expected]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate end state ids in {self.id}")
        return self


class PackDefaults(BaseModel):
    caller: dict[Language, CallerDefaults]


class PackManifest(BaseModel):
    id: str = Field(pattern=ID_PATTERN)
    name: str
    version: str
    menu: str = "menu.yaml"
    description: str = ""
    defaults: PackDefaults
```

- [ ] **Step 5: Implement `schemas/results.py`**

```python
"""Result schemas: backend state, checks, trials, suites."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SIMULATOR_INVALID = "simulator_invalid"
    INFRA_ERROR = "infra_error"


class CartLine(BaseModel):
    line_id: str
    item_id: str
    quantity: int = Field(ge=0)
    modifiers: set[str] = Field(default_factory=set)
    note: str | None = None


class SubmittedOrder(BaseModel):
    order_id: str
    lines: list[CartLine]
    submitted_at_ms: float
    status: Literal["submitted", "cancelled"] = "submitted"


class ToolCall(BaseModel):
    seq: int
    t_ms: float
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class BackendSnapshot(BaseModel):
    cart: list[CartLine] = Field(default_factory=list)
    orders: list[SubmittedOrder] = Field(default_factory=list)
    trace: list[ToolCall] = Field(default_factory=list)

    def active_orders(self) -> list[SubmittedOrder]:
        return [o for o in self.orders if o.status == "submitted"]


class FieldCheck(BaseModel):
    field: str
    expected: str
    actual: str
    passed: bool


class CheckResult(BaseModel):
    passed: bool
    best_end_state_id: str
    field_checks: list[FieldCheck]
    summary: str


CallerSource = Literal["script", "clarification", "confirm", "closing", "fallback", "nudge"]


class TurnRecord(BaseModel):
    index: int
    speaker: Literal["caller", "agent"]
    text: str | None
    audio_path: str | None = None
    t_ms: float
    latency_ms: float | None = None
    turn_id: str | None = None
    source: CallerSource | None = None


class TrialResult(BaseModel):
    scenario_id: str
    trial: int
    outcome: Outcome
    check: CheckResult | None = None
    transcript: list[TurnRecord] = Field(default_factory=list)
    snapshot: BackendSnapshot | None = None
    caller_valid: bool = True
    caller_invalid_reason: str | None = None
    error: str | None = None
    max_turns_hit: bool = False
    started_at: datetime
    duration_ms: float
    agent_meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.outcome in (Outcome.PASS, Outcome.FAIL)

    @property
    def latencies_ms(self) -> list[float]:
        return [t.latency_ms for t in self.transcript if t.speaker == "agent" and t.latency_ms is not None]


class ScenarioResult(BaseModel):
    scenario_id: str
    title: str
    language: str
    category: str
    tags: list[str] = Field(default_factory=list)
    trials: list[TrialResult]

    @property
    def n_valid(self) -> int:
        return sum(1 for t in self.trials if t.is_valid)

    @property
    def n_pass(self) -> int:
        return sum(1 for t in self.trials if t.outcome is Outcome.PASS)

    @property
    def pass_rate(self) -> float | None:
        return None if self.n_valid == 0 else self.n_pass / self.n_valid

    @property
    def worst_outcome(self) -> Outcome:
        order = [Outcome.INFRA_ERROR, Outcome.FAIL, Outcome.SIMULATOR_INVALID, Outcome.PASS]
        present = {t.outcome for t in self.trials}
        for o in order:
            if o in present:
                return o
        return Outcome.INFRA_ERROR


class GroupMetric(BaseModel):
    key: str
    n_trials: int
    n_pass: int
    pass_rate: float | None
    ci_low: float | None
    ci_high: float | None


class Metrics(BaseModel):
    n_scenarios: int
    n_trials_total: int
    n_valid: int
    n_pass: int
    n_fail: int
    n_simulator_invalid: int
    n_infra_error: int
    n_max_turns: int
    pass_rate: float | None
    pass_k: dict[int, float | None] = Field(default_factory=dict)
    by_language: list[GroupMetric] = Field(default_factory=list)
    by_category: list[GroupMetric] = Field(default_factory=list)
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None


class RunInfo(BaseModel):
    started_at: datetime
    finished_at: datetime
    trials: int
    modality: str
    seed: int | None
    host: str
    benchmark_version: str
    python: str
    max_turns: int
    timeout_turn_s: float
    timeout_trial_s: float


class PackInfo(BaseModel):
    id: str
    version: str
    content_hash: str
    path: str


class AgentInfo(BaseModel):
    label: str
    spec: str


class SuiteResult(BaseModel):
    schema_version: int = 1
    run: RunInfo
    pack: PackInfo
    agent: AgentInfo
    scenarios: list[ScenarioResult]
    metrics: Metrics
```

`schemas/__init__.py` re-exports nothing (keep imports explicit).

- [ ] **Step 6: Run tests, lint, types**

Run: `uv run pytest tests/test_schemas.py -q && uv run ruff check src tests && uv run mypy`
Expected: all pass. Fix any mypy strictness issues inline (for example `Any` for `ToolCall.result` is deliberate).

- [ ] **Step 7: Commit**

```bash
git add src/indicorderbench/schemas tests/test_schemas.py
git commit -m "feat(schemas): menu, scenario and result models"
```

---

### Task 2: Sandbox order backend

**Files:**
- Create: `src/indicorderbench/backend/__init__.py`, `backend/state.py`
- Test: `tests/test_backend.py`

**Interfaces:**
- Consumes: `Menu`, `CartLine`, `SubmittedOrder`, `ToolCall`, `BackendSnapshot`.
- Produces:
  ```python
  class BackendError(Exception): code: str; message: str
  class OrderBackend:
      def __init__(self, menu: Menu, clock: Callable[[], float] | None = None) -> None
      menu: Menu
      def lookup_menu(self, query: str) -> list[dict[str, Any]]
      def add_item(self, item_id: str, quantity: int = 1, modifiers: list[str] | None = None) -> CartLine
      def update_line(self, line_id: str, quantity: int | None = None, modifiers: list[str] | None = None) -> CartLine | None
      def remove_line(self, line_id: str) -> None
      def clear_cart(self) -> None
      def get_cart(self) -> list[CartLine]
      def submit_order(self) -> SubmittedOrder
      def cancel_order(self, order_id: str) -> SubmittedOrder
      def list_orders(self) -> list[SubmittedOrder]
      def active_orders(self) -> list[SubmittedOrder]
      def snapshot(self) -> BackendSnapshot
      def call(self, name: str, args: dict[str, Any]) -> Any            # dispatch by tool name; raises BackendError
      @staticmethod
      def tool_specs() -> list[dict[str, Any]]                          # JSON-schema function definitions
      TOOL_NAMES: tuple[str, ...]
  ```
  `clock()` returns milliseconds since session start (default uses `time.perf_counter`).

- [ ] **Step 1: Write failing tests**

`tests/test_backend.py`:
```python
import pytest

from indicorderbench.backend.state import BackendError, OrderBackend
from tests.test_schemas import make_menu


def make_backend() -> OrderBackend:
    ticks = iter(range(0, 100000, 10))
    return OrderBackend(make_menu(), clock=lambda: float(next(ticks)))


def test_add_item_validates_item_and_modifiers():
    b = make_backend()
    line = b.add_item("paneer_wrap", 2, ["no_onion"])
    assert line.quantity == 2 and line.modifiers == {"no_onion"}
    with pytest.raises(BackendError) as e:
        b.add_item("ghost")
    assert e.value.code == "unknown_item"
    with pytest.raises(BackendError) as e:
        b.add_item("mango_lassi", 1, ["no_onion"])  # group not applicable
    assert e.value.code == "invalid_modifier"
    with pytest.raises(BackendError) as e:
        b.add_item("paneer_wrap", 1, ["no_onion", "with_onion"])  # two options of exclusive group
    assert e.value.code == "invalid_modifier"
    with pytest.raises(BackendError) as e:
        b.add_item("paneer_wrap", 0)
    assert e.value.code == "invalid_quantity"


def test_update_remove_clear():
    b = make_backend()
    line = b.add_item("paneer_wrap", 2)
    updated = b.update_line(line.line_id, quantity=1, modifiers=["no_onion"])
    assert updated is not None and updated.quantity == 1 and updated.modifiers == {"no_onion"}
    assert b.update_line(line.line_id, quantity=0) is None
    assert b.get_cart() == []
    b.add_item("mango_lassi")
    b.clear_cart()
    assert b.get_cart() == []
    with pytest.raises(BackendError) as e:
        b.remove_line("nope")
    assert e.value.code == "unknown_line"


def test_submit_and_cancel():
    b = make_backend()
    with pytest.raises(BackendError) as e:
        b.submit_order()
    assert e.value.code == "empty_cart"
    b.add_item("mango_lassi")
    order = b.submit_order()
    assert order.status == "submitted" and b.get_cart() == []
    assert [o.order_id for o in b.active_orders()] == [order.order_id]
    cancelled = b.cancel_order(order.order_id)
    assert cancelled.status == "cancelled" and b.active_orders() == []
    with pytest.raises(BackendError) as e:
        b.cancel_order(order.order_id)
    assert e.value.code == "already_cancelled"


def test_trace_records_every_call_including_errors_with_clock():
    b = make_backend()
    b.add_item("mango_lassi")
    with pytest.raises(BackendError):
        b.add_item("ghost")
    snap = b.snapshot()
    assert [c.name for c in snap.trace] == ["add_item", "add_item"]
    assert snap.trace[0].error is None and snap.trace[0].t_ms == 0.0
    assert snap.trace[1].error is not None and snap.trace[1].t_ms == 10.0
    assert snap.trace[0].result["item_id"] == "mango_lassi"


def test_snapshot_is_a_copy():
    b = make_backend()
    b.add_item("mango_lassi")
    snap = b.snapshot()
    b.clear_cart()
    assert len(snap.cart) == 1


def test_call_dispatch_and_tool_specs():
    b = make_backend()
    result = b.call("add_item", {"item_id": "mango_lassi", "quantity": 3})
    assert result["quantity"] == 3
    with pytest.raises(BackendError) as e:
        b.call("fly", {})
    assert e.value.code == "unknown_tool"
    names = {s["name"] for s in OrderBackend.tool_specs()}
    assert names == set(OrderBackend.TOOL_NAMES)
    assert all("parameters" in s and "description" in s for s in OrderBackend.tool_specs())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_backend.py -q` → ImportError.

- [ ] **Step 3: Implement `backend/state.py`**

```python
"""In-memory sandbox order backend with a tool-call trace."""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from indicorderbench.schemas.menu import Menu
from indicorderbench.schemas.results import BackendSnapshot, CartLine, SubmittedOrder, ToolCall


class BackendError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _default_clock() -> Callable[[], float]:
    t0 = time.perf_counter()
    return lambda: (time.perf_counter() - t0) * 1000.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


class OrderBackend:
    TOOL_NAMES: tuple[str, ...] = (
        "lookup_menu", "add_item", "update_line", "remove_line", "clear_cart",
        "get_cart", "submit_order", "cancel_order", "list_orders",
    )

    def __init__(self, menu: Menu, clock: Callable[[], float] | None = None) -> None:
        self.menu = menu
        self._clock = clock or _default_clock()
        self._cart: dict[str, CartLine] = {}
        self._orders: list[SubmittedOrder] = []
        self._trace: list[ToolCall] = []
        self._line_ids = itertools.count(1)
        self._order_ids = itertools.count(1)

    # -- tracing -------------------------------------------------------------
    def _traced(self, name: str, args: dict[str, Any], fn: Callable[[], Any]) -> Any:
        call = ToolCall(seq=len(self._trace) + 1, t_ms=self._clock(), name=name, args=args)
        try:
            result = fn()
        except BackendError as e:
            call.error = f"{e.code}: {e.message}"
            self._trace.append(call)
            raise
        call.result = _jsonable(result)
        self._trace.append(call)
        return result

    # -- validation -----------------------------------------------------------
    def _validate_modifiers(self, item_id: str, modifiers: list[str]) -> set[str]:
        groups = {g.id: g for g in self.menu.groups_for(item_id)}
        chosen: dict[str, list[str]] = {}
        for opt in modifiers:
            if not self.menu.has_option(opt):
                raise BackendError("invalid_modifier", f"unknown modifier {opt!r}")
            g = self.menu.option_group(opt)
            if g.id not in groups:
                raise BackendError("invalid_modifier", f"{opt!r} does not apply to {item_id!r}")
            chosen.setdefault(g.id, []).append(opt)
        for gid, opts in chosen.items():
            if groups[gid].exclusive and len(set(opts)) > 1:
                raise BackendError("invalid_modifier", f"group {gid!r} allows one option, got {sorted(set(opts))}")
        return set(modifiers)

    # -- tools ----------------------------------------------------------------
    def lookup_menu(self, query: str) -> list[dict[str, Any]]:
        def run() -> list[dict[str, Any]]:
            return [
                {"item_id": i.id, "name": i.name, "price": str(i.price),
                 "modifier_groups": [
                     {"group_id": g.id, "name": g.name, "exclusive": g.exclusive, "default": g.default,
                      "options": [{"option_id": o.id, "name": o.name} for o in g.options]}
                     for g in self.menu.groups_for(i.id)]}
                for i in self.menu.search(query)
            ]
        return self._traced("lookup_menu", {"query": query}, run)

    def add_item(self, item_id: str, quantity: int = 1, modifiers: list[str] | None = None) -> CartLine:
        mods = list(modifiers or [])

        def run() -> CartLine:
            if not self.menu.has_item(item_id):
                raise BackendError("unknown_item", f"no item {item_id!r}")
            if quantity < 1:
                raise BackendError("invalid_quantity", "quantity must be at least 1")
            line = CartLine(line_id=f"l{next(self._line_ids)}", item_id=item_id, quantity=quantity,
                            modifiers=self._validate_modifiers(item_id, mods))
            self._cart[line.line_id] = line
            return line
        return self._traced("add_item", {"item_id": item_id, "quantity": quantity, "modifiers": mods}, run)

    def update_line(self, line_id: str, quantity: int | None = None,
                    modifiers: list[str] | None = None) -> CartLine | None:
        def run() -> CartLine | None:
            line = self._cart.get(line_id)
            if line is None:
                raise BackendError("unknown_line", f"no cart line {line_id!r}")
            if quantity is not None and quantity < 0:
                raise BackendError("invalid_quantity", "quantity must be 0 or more")
            if quantity == 0:
                del self._cart[line_id]
                return None
            new_q = line.quantity if quantity is None else quantity
            new_m = line.modifiers if modifiers is None else self._validate_modifiers(line.item_id, modifiers)
            updated = line.model_copy(update={"quantity": new_q, "modifiers": new_m})
            self._cart[line_id] = updated
            return updated
        return self._traced("update_line", {"line_id": line_id, "quantity": quantity, "modifiers": modifiers}, run)

    def remove_line(self, line_id: str) -> None:
        def run() -> None:
            if line_id not in self._cart:
                raise BackendError("unknown_line", f"no cart line {line_id!r}")
            del self._cart[line_id]
        self._traced("remove_line", {"line_id": line_id}, run)

    def clear_cart(self) -> None:
        self._traced("clear_cart", {}, self._cart.clear)

    def get_cart(self) -> list[CartLine]:
        return self._traced("get_cart", {}, lambda: list(self._cart.values()))

    def submit_order(self) -> SubmittedOrder:
        def run() -> SubmittedOrder:
            if not self._cart:
                raise BackendError("empty_cart", "cannot submit an empty cart")
            order = SubmittedOrder(order_id=f"o{next(self._order_ids)}", lines=list(self._cart.values()),
                                   submitted_at_ms=self._clock())
            self._orders.append(order)
            self._cart.clear()
            return order
        return self._traced("submit_order", {}, run)

    def cancel_order(self, order_id: str) -> SubmittedOrder:
        def run() -> SubmittedOrder:
            for i, o in enumerate(self._orders):
                if o.order_id == order_id:
                    if o.status == "cancelled":
                        raise BackendError("already_cancelled", f"order {order_id!r} is already cancelled")
                    cancelled = o.model_copy(update={"status": "cancelled"})
                    self._orders[i] = cancelled
                    return cancelled
            raise BackendError("unknown_order", f"no order {order_id!r}")
        return self._traced("cancel_order", {"order_id": order_id}, run)

    def list_orders(self) -> list[SubmittedOrder]:
        return self._traced("list_orders", {}, lambda: list(self._orders))

    # -- non-traced helpers ---------------------------------------------------
    def active_orders(self) -> list[SubmittedOrder]:
        return [o for o in self._orders if o.status == "submitted"]

    def snapshot(self) -> BackendSnapshot:
        return copy.deepcopy(BackendSnapshot(cart=list(self._cart.values()), orders=list(self._orders),
                                             trace=list(self._trace)))

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if name not in self.TOOL_NAMES:
            raise BackendError("unknown_tool", f"no tool {name!r}")
        fn = getattr(self, name)
        try:
            result = fn(**args)
        except TypeError as e:
            raise BackendError("invalid_args", str(e)) from e
        return _jsonable(result)

    @staticmethod
    def tool_specs() -> list[dict[str, Any]]:
        def spec(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
            return {"name": name, "description": description,
                    "parameters": {"type": "object", "properties": props, "required": required}}
        s = {"type": "string"}
        i = {"type": "integer"}
        mods = {"type": "array", "items": {"type": "string"}, "description": "modifier option ids"}
        return [
            spec("lookup_menu", "Search menu items by name or alias.", {"query": s}, ["query"]),
            spec("add_item", "Add an item to the cart.", {"item_id": s, "quantity": i, "modifiers": mods}, ["item_id"]),
            spec("update_line", "Change quantity (0 removes) or replace modifiers of a cart line.",
                 {"line_id": s, "quantity": i, "modifiers": mods}, ["line_id"]),
            spec("remove_line", "Remove a cart line.", {"line_id": s}, ["line_id"]),
            spec("clear_cart", "Empty the cart.", {}, []),
            spec("get_cart", "Return the current cart lines.", {}, []),
            spec("submit_order", "Submit the cart as an order. The cart must not be empty.", {}, []),
            spec("cancel_order", "Cancel a submitted order.", {"order_id": s}, ["order_id"]),
            spec("list_orders", "List all orders and their status.", {}, []),
        ]
```

- [ ] **Step 4: Run tests, lint, types** → all green.
- [ ] **Step 5: Commit** `feat(backend): sandbox order backend with tool trace`

---

### Task 3: Checker

**Files:**
- Create: `src/indicorderbench/checker/__init__.py`, `checker/canonical.py`, `checker/checker.py`
- Test: `tests/test_checker.py`

**Interfaces:**
- Consumes: `Menu`, `CartLine`, `SubmittedOrder`, `BackendSnapshot`, `EndState`, `ExpectedOrder`, `ExpectedLine`, `FieldCheck`, `CheckResult`.
- Produces:
  ```python
  # canonical.py
  @dataclass(frozen=True, order=True) class CanonicalLine: item_id: str; modifiers: tuple[tuple[str, tuple[str, ...]], ...]; quantity: int
  def resolve_modifiers(menu: Menu, item_id: str, selected: Iterable[str]) -> dict[str, tuple[str, ...]]
  def canonical_lines(menu: Menu, lines: Iterable[CartLine]) -> tuple[CanonicalLine, ...]
  @dataclass(frozen=True) class ExpectedCanonicalLine: item_id: str; quantity: int; modifiers: dict[str, tuple[str, ...] | None]   # None = wildcard
  def canonical_expected(menu: Menu, order: ExpectedOrder) -> list[ExpectedCanonicalLine]
  def line_matches(expected: ExpectedCanonicalLine, actual: CanonicalLine) -> bool
  def order_matches(menu: Menu, expected: ExpectedOrder, lines: Iterable[CartLine]) -> bool
  # checker.py
  def end_state_matches(menu: Menu, end_state: EndState, active: list[SubmittedOrder]) -> bool
  def field_checks(menu: Menu, end_state: EndState, active: list[SubmittedOrder]) -> list[FieldCheck]
  def check(expected: list[EndState], snapshot: BackendSnapshot, menu: Menu) -> CheckResult
  ```

- [ ] **Step 1: Write failing tests**

`tests/test_checker.py`:
```python
from indicorderbench.checker.canonical import canonical_lines, order_matches, resolve_modifiers
from indicorderbench.checker.checker import check, field_checks
from indicorderbench.schemas.results import BackendSnapshot, CartLine, SubmittedOrder
from indicorderbench.schemas.scenario import EndState, ExpectedLine, ExpectedOrder
from tests.test_schemas import make_menu

MENU = make_menu()


def order(*lines: CartLine, status: str = "submitted", oid: str = "o1") -> SubmittedOrder:
    return SubmittedOrder(order_id=oid, lines=list(lines), submitted_at_ms=0.0, status=status)  # type: ignore[arg-type]


def line(item: str, qty: int, *mods: str, lid: str = "l1") -> CartLine:
    return CartLine(line_id=lid, item_id=item, quantity=qty, modifiers=set(mods))


def snap(*orders: SubmittedOrder) -> BackendSnapshot:
    return BackendSnapshot(cart=[], orders=list(orders), trace=[])


def expect(*lines: ExpectedLine, es_id: str = "e1") -> EndState:
    return EndState(id=es_id, orders=[ExpectedOrder(lines=list(lines))])


def test_resolve_modifiers_applies_defaults():
    assert resolve_modifiers(MENU, "paneer_wrap", []) == {"onion": ("with_onion",), "extras": ()}
    assert resolve_modifiers(MENU, "paneer_wrap", ["no_onion", "extra_cheese"]) == {
        "onion": ("no_onion",), "extras": ("extra_cheese",)}
    assert resolve_modifiers(MENU, "mango_lassi", []) == {}


def test_canonical_lines_merge_and_sort():
    lines = canonical_lines(MENU, [line("paneer_wrap", 1, lid="a"), line("mango_lassi", 1, lid="b"),
                                   line("paneer_wrap", 1, lid="c"), line("paneer_wrap", 0, lid="d")])
    assert [(c.item_id, c.quantity) for c in lines] == [("mango_lassi", 1), ("paneer_wrap", 2)]


def test_order_matches_with_wildcard_and_defaults():
    actual = [line("paneer_wrap", 1, "no_onion"), line("mango_lassi", 1, lid="l2")]
    assert order_matches(MENU, ExpectedOrder(lines=[
        ExpectedLine(item_id="paneer_wrap", quantity=1, modifiers={"onion": "no_onion"}),
        ExpectedLine(item_id="mango_lassi", quantity=1)]), actual)
    assert not order_matches(MENU, ExpectedOrder(lines=[
        ExpectedLine(item_id="paneer_wrap", quantity=1), ExpectedLine(item_id="mango_lassi", quantity=1)]), actual)
    assert order_matches(MENU, ExpectedOrder(lines=[
        ExpectedLine(item_id="paneer_wrap", quantity=1, modifiers={"onion": "*"}),
        ExpectedLine(item_id="mango_lassi", quantity=1)]), actual)
    assert order_matches(MENU, ExpectedOrder(lines=[
        ExpectedLine(item_id="paneer_wrap", quantity=1, modifiers={"onion": "no_onion", "extras": ["extra_cheese"]})]),
        [line("paneer_wrap", 1, "no_onion", "extra_cheese")])


def test_unasked_modifier_fails():
    assert not order_matches(MENU, ExpectedOrder(lines=[ExpectedLine(item_id="paneer_wrap", quantity=1)]),
                             [line("paneer_wrap", 1, "extra_cheese")])


def test_check_demo_table():
    expected = [expect(ExpectedLine(item_id="paneer_wrap", quantity=1, modifiers={"onion": "no_onion"}),
                       ExpectedLine(item_id="mango_lassi", quantity=1))]
    actual = snap(order(line("paneer_wrap", 2, "no_onion"), line("mango_lassi", 1, lid="l2")))
    result = check(expected, actual, MENU)
    assert not result.passed and result.best_end_state_id == "e1"
    rows = [(f.field, f.expected, f.actual, f.passed) for f in result.field_checks]
    assert rows == [
        ("Submitted orders", "1", "1", True),
        ("Paneer Wrap quantity", "1", "2", False),
        ("Paneer Wrap · Onion", "No onion", "No onion", True),
        ("Mango Lassi quantity", "1", "1", True),
    ]


def test_check_passes_and_ignores_cancelled():
    expected = [expect(ExpectedLine(item_id="mango_lassi", quantity=1))]
    actual = snap(order(line("paneer_wrap", 1), status="cancelled", oid="o0"), order(line("mango_lassi", 1), oid="o1"))
    assert check(expected, actual, MENU).passed


def test_duplicate_submission_is_not_merged():
    expected = [expect(ExpectedLine(item_id="mango_lassi", quantity=1))]
    actual = snap(order(line("mango_lassi", 1), oid="o1"), order(line("mango_lassi", 1), oid="o2"))
    result = check(expected, actual, MENU)
    assert not result.passed
    assert result.field_checks[0].model_dump() == {"field": "Submitted orders", "expected": "1", "actual": "2", "passed": False}


def test_cancellation_end_state():
    expected = [EndState(id="none", orders=[])]
    assert check(expected, snap(), MENU).passed
    result = check(expected, snap(order(line("mango_lassi", 1))), MENU)
    assert not result.passed
    assert [(f.field, f.expected, f.actual) for f in result.field_checks] == [
        ("Submitted orders", "0", "1"), ("Mango Lassi quantity", "0", "1")]


def test_best_end_state_is_the_closest():
    expected = [expect(ExpectedLine(item_id="paneer_wrap", quantity=3), es_id="far"),
                expect(ExpectedLine(item_id="paneer_wrap", quantity=1), es_id="near")]
    result = check(expected, snap(order(line("paneer_wrap", 1, "extra_cheese"))), MENU)
    assert result.best_end_state_id == "near"


def test_missing_and_extra_lines():
    expected = [expect(ExpectedLine(item_id="paneer_wrap", quantity=1))]
    result = check(expected, snap(order(line("mango_lassi", 2))), MENU)
    assert [(f.field, f.expected, f.actual, f.passed) for f in result.field_checks] == [
        ("Submitted orders", "1", "1", True),
        ("Paneer Wrap quantity", "1", "0", False),
        ("Mango Lassi quantity", "0", "2", False),
    ]
```

- [ ] **Step 2: Run to verify failure** → ImportError.

- [ ] **Step 3: Implement `checker/canonical.py`**

```python
"""Canonical forms of actual and expected orders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from indicorderbench.schemas.menu import Menu
from indicorderbench.schemas.results import CartLine
from indicorderbench.schemas.scenario import ExpectedOrder

ModMap = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, order=True)
class CanonicalLine:
    item_id: str
    modifiers: ModMap
    quantity: int


@dataclass(frozen=True)
class ExpectedCanonicalLine:
    item_id: str
    quantity: int
    modifiers: dict[str, tuple[str, ...] | None]  # None means wildcard


def resolve_modifiers(menu: Menu, item_id: str, selected: Iterable[str]) -> dict[str, tuple[str, ...]]:
    chosen = set(selected)
    out: dict[str, tuple[str, ...]] = {}
    for g in menu.groups_for(item_id):
        picked = tuple(sorted(o.id for o in g.options if o.id in chosen))
        if g.exclusive and not picked and g.default is not None:
            picked = (g.default,)
        out[g.id] = picked
    return out


def _modmap(resolved: dict[str, tuple[str, ...]]) -> ModMap:
    return tuple(sorted(resolved.items()))


def canonical_lines(menu: Menu, lines: Iterable[CartLine]) -> tuple[CanonicalLine, ...]:
    merged: dict[tuple[str, ModMap], int] = {}
    for line in lines:
        if line.quantity <= 0:
            continue
        key = (line.item_id, _modmap(resolve_modifiers(menu, line.item_id, line.modifiers)))
        merged[key] = merged.get(key, 0) + line.quantity
    return tuple(sorted(CanonicalLine(item, mods, q) for (item, mods), q in merged.items()))


def canonical_expected(menu: Menu, order: ExpectedOrder) -> list[ExpectedCanonicalLine]:
    merged: dict[tuple[str, tuple[tuple[str, tuple[str, ...] | None], ...]], int] = {}
    for el in order.lines:
        resolved: dict[str, tuple[str, ...] | None] = dict(resolve_modifiers(menu, el.item_id, []))
        for gid, spec in el.modifiers.items():
            if spec == "*":
                resolved[gid] = None
            elif isinstance(spec, str):
                resolved[gid] = (spec,)
            else:
                resolved[gid] = tuple(sorted(spec))
        key = (el.item_id, tuple(sorted(resolved.items())))
        merged[key] = merged.get(key, 0) + el.quantity
    return [ExpectedCanonicalLine(item, q, dict(mods)) for (item, mods), q in merged.items()]


def line_matches(expected: ExpectedCanonicalLine, actual: CanonicalLine) -> bool:
    if expected.item_id != actual.item_id or expected.quantity != actual.quantity:
        return False
    actual_mods = dict(actual.modifiers)
    return all(want is None or actual_mods.get(gid) == want for gid, want in expected.modifiers.items())


def _match_all(expected: list[ExpectedCanonicalLine], actual: list[CanonicalLine]) -> bool:
    if not expected:
        return not actual
    head, rest = expected[0], expected[1:]
    for i, a in enumerate(actual):
        if line_matches(head, a) and _match_all(rest, actual[:i] + actual[i + 1:]):
            return True
    return False


def order_matches(menu: Menu, expected: ExpectedOrder, lines: Iterable[CartLine]) -> bool:
    return _match_all(canonical_expected(menu, expected), list(canonical_lines(menu, lines)))
```

- [ ] **Step 4: Implement `checker/checker.py`**

```python
"""Deterministic end-state checker producing field-level results."""

from __future__ import annotations

from indicorderbench.checker.canonical import (
    CanonicalLine,
    ExpectedCanonicalLine,
    canonical_expected,
    canonical_lines,
    line_matches,
    order_matches,
)
from indicorderbench.schemas.menu import Menu
from indicorderbench.schemas.results import BackendSnapshot, CheckResult, FieldCheck, SubmittedOrder
from indicorderbench.schemas.scenario import EndState, ExpectedOrder


def end_state_matches(menu: Menu, end_state: EndState, active: list[SubmittedOrder]) -> bool:
    if len(end_state.orders) != len(active):
        return False
    remaining = list(active)
    for exp in end_state.orders:
        for i, act in enumerate(remaining):
            if order_matches(menu, exp, act.lines):
                remaining.pop(i)
                break
        else:
            return False
    return True


def _option_names(menu: Menu, gid: str, opts: tuple[str, ...] | None) -> str:
    if opts is None:
        return "any"
    if not opts:
        return "none"
    return ", ".join(menu.option(o).name for o in opts)


def _pick_actual(exp: ExpectedCanonicalLine, pool: list[CanonicalLine]) -> CanonicalLine | None:
    same_item = [a for a in pool if a.item_id == exp.item_id]
    if not same_item:
        return None
    for a in same_item:  # prefer full modifier agreement
        if all(w is None or dict(a.modifiers).get(g) == w for g, w in exp.modifiers.items()):
            return a
    return same_item[0]


def _order_field_checks(menu: Menu, expected: ExpectedOrder | None, actual: SubmittedOrder | None) -> list[FieldCheck]:
    checks: list[FieldCheck] = []
    pool = list(canonical_lines(menu, actual.lines)) if actual else []
    exp_lines = canonical_expected(menu, expected) if expected else []
    for exp in exp_lines:
        item_name = menu.item(exp.item_id).name
        act = _pick_actual(exp, pool)
        if act is not None:
            pool.remove(act)
        checks.append(FieldCheck(field=f"{item_name} quantity", expected=str(exp.quantity),
                                 actual=str(act.quantity if act else 0),
                                 passed=act is not None and act.quantity == exp.quantity))
        explicit = [gid for gid, want in exp.modifiers.items()
                    if want is None or want != tuple(sorted(o for o in [menu.group(gid).default] if o)) or _explicitly_set(expected, exp.item_id, gid)]
        for gid in explicit:
            want = exp.modifiers[gid]
            got = dict(act.modifiers).get(gid) if act else None
            checks.append(FieldCheck(field=f"{item_name} · {menu.group(gid).name}",
                                     expected=_option_names(menu, gid, want),
                                     actual=_option_names(menu, gid, got) if act else "—",
                                     passed=act is not None and (want is None or got == want)))
    for extra in pool:
        checks.append(FieldCheck(field=f"{menu.item(extra.item_id).name} quantity", expected="0",
                                 actual=str(extra.quantity), passed=False))
    return checks


def _explicitly_set(expected: ExpectedOrder | None, item_id: str, gid: str) -> bool:
    return expected is not None and any(gid in el.modifiers for el in expected.lines if el.item_id == item_id)


def field_checks(menu: Menu, end_state: EndState, active: list[SubmittedOrder]) -> list[FieldCheck]:
    checks = [FieldCheck(field="Submitted orders", expected=str(len(end_state.orders)),
                         actual=str(len(active)), passed=len(end_state.orders) == len(active))]
    n = max(len(end_state.orders), len(active))
    for i in range(n):
        exp = end_state.orders[i] if i < len(end_state.orders) else None
        act = active[i] if i < len(active) else None
        checks.extend(_order_field_checks(menu, exp, act))
    return checks


def check(expected: list[EndState], snapshot: BackendSnapshot, menu: Menu) -> CheckResult:
    active = snapshot.active_orders()
    scored: list[tuple[EndState, bool, list[FieldCheck]]] = []
    for es in expected:
        scored.append((es, end_state_matches(menu, es, active), field_checks(menu, es, active)))
    best = min(scored, key=lambda s: (not s[1], sum(1 for f in s[2] if not f.passed)))
    es, matched, checks = best
    passed = any(s[1] for s in scored)
    failing = [f for f in checks if not f.passed]
    summary = "matched end state " + es.id if passed else (
        f"{len(failing)} field(s) wrong vs {es.id}: " + "; ".join(f"{f.field} expected {f.expected} got {f.actual}" for f in failing))
    return CheckResult(passed=passed, best_end_state_id=es.id, field_checks=checks, summary=summary)
```

Note on `explicit`: a modifier row is shown when the scenario author set that group explicitly (the `_explicitly_set` test), or when the expected value differs from the default, or it is a wildcard. Simplify the expression to `if want is None or _explicitly_set(expected, exp.item_id, gid)` once tests pass; the default-comparison branch is redundant because `canonical_expected` fills unspecified groups with defaults, and those groups should not be shown. **Use the simplified form in the final code.**

- [ ] **Step 5: Run tests, lint, types** → green.
- [ ] **Step 6: Commit** `feat(checker): canonical order comparison and field checks`

---

### Task 4: Scripted caller

**Files:**
- Create: `src/indicorderbench/caller/__init__.py`, `caller/scripted.py`
- Test: `tests/test_caller.py`

**Interfaces:**
- Consumes: `ResolvedCallerScript`, `CallerTurn`, `ClarificationRule`.
- Produces:
  ```python
  @dataclass(frozen=True) class CallerMove: turn: CallerTurn; source: CallerSource   # CallerSource from schemas.results
  class ScriptedCaller:
      def __init__(self, script: ResolvedCallerScript) -> None
      def first_move(self) -> CallerMove
      def next_move(self, agent_text: str) -> CallerMove | None   # None: caller has stopped
      script_exhausted: bool; closing_spoken: bool; invalid: bool; invalid_reason: str | None
      fallbacks_used: int; nudges_used: int
      def should_end(self, agent_text: str, has_active_order: bool) -> bool
  ```

- [ ] **Step 1: Write failing tests**

`tests/test_caller.py`:
```python
from indicorderbench.caller.scripted import ScriptedCaller
from indicorderbench.schemas.scenario import CallerDefaults, CallerScript, CallerTurn, ClarificationRule


def script(turns: list[str], clar: list[ClarificationRule] | None = None, closing_last: bool = False):
    ts = [CallerTurn(text=t) for t in turns]
    if closing_last:
        ts[-1].is_closing = True
    defaults = CallerDefaults(
        closing=CallerTurn(text="that's all"), confirm=CallerTurn(text="yes, place it"),
        fallback=CallerTurn(text="yes, that's right"), nudge=CallerTurn(text="please place the order"),
        confirm_patterns=["shall i place", "confirm"], goodbye_patterns=["order placed"],
        question_patterns=["which", "how many"],
        clarifications=[ClarificationRule(id="size", match=["what size"], reply=CallerTurn(text="regular"), max_uses=1)],
    )
    return CallerScript(turns=ts, clarifications=clar or []).resolved(defaults)


def test_happy_path_then_closing_then_end():
    c = ScriptedCaller(script(["two wraps", "and a lassi"]))
    assert c.first_move().turn.text == "two wraps"
    m = c.next_move("Added two wraps. Anything else?")
    assert m and m.source == "script" and m.turn.text == "and a lassi"
    m = c.next_move("Added a lassi. Anything else?")
    assert m and m.source == "closing" and c.script_exhausted and c.closing_spoken
    assert c.should_end("Order placed, thank you!", has_active_order=True)
    assert c.should_end("Thanks, order placed!", has_active_order=False)
    assert not c.should_end("Okay.", has_active_order=False)


def test_clarification_mid_script_then_resumes():
    c = ScriptedCaller(script(["two wraps", "and a lassi"]))
    c.first_move()
    m = c.next_move("What size would you like?")
    assert m and m.source == "clarification" and m.turn.text == "regular" and m.turn.id == "c_size"
    m = c.next_move("What size would you like?")  # max_uses exhausted -> continue script
    assert m and m.source == "script" and m.turn.text == "and a lassi"


def test_unmatched_question_mid_script_continues_script():
    c = ScriptedCaller(script(["two wraps", "no onion"]))
    c.first_move()
    m = c.next_move("Do you want them spicy?")
    assert m and m.source == "script" and m.turn.text == "no onion"


def test_confirm_request_after_closing_is_answered_once():
    c = ScriptedCaller(script(["a lassi"]))
    c.first_move()
    assert c.next_move("Anything else?").source == "closing"
    m = c.next_move("Shall I place the order?")
    assert m and m.source == "confirm"
    m = c.next_move("Shall I place the order?")
    assert m and m.source == "fallback"


def test_unanswerable_questions_make_caller_invalid():
    c = ScriptedCaller(script(["a lassi"]))
    c.first_move()
    c.next_move("Anything else?")  # closing
    assert c.next_move("Which branch are you ordering from?").source == "fallback"
    assert c.next_move("Which branch?").source == "fallback"
    assert c.next_move("Which branch?") is None
    assert c.invalid and "question" in (c.invalid_reason or "")


def test_non_question_without_submission_nudges_then_stops_valid():
    c = ScriptedCaller(script(["a lassi"]))
    c.first_move()
    c.next_move("Anything else?")
    assert c.next_move("Okay.").source == "nudge"
    assert c.next_move("Sure.").source == "nudge"
    assert c.next_move("Sure.") is None
    assert not c.invalid


def test_empty_agent_reply_is_not_a_question():
    c = ScriptedCaller(script(["a lassi"]))
    c.first_move()
    c.next_move("   ")
    assert c.closing_spoken
    assert c.next_move("").source == "nudge"
    assert not c.invalid


def test_is_closing_turn_skips_default_closing():
    c = ScriptedCaller(script(["a lassi", "that's it, thanks"], closing_last=True))
    c.first_move()
    c.next_move("Added. Anything else?")
    assert c.closing_spoken
    assert c.should_end("Order placed.", has_active_order=True)
```

- [ ] **Step 2: Run to verify failure** → ImportError.

- [ ] **Step 3: Implement `caller/scripted.py`**

```python
"""Scripted caller: a state machine over a resolved caller script."""

from __future__ import annotations

from dataclasses import dataclass

from indicorderbench.schemas.results import CallerSource
from indicorderbench.schemas.scenario import CallerTurn, ResolvedCallerScript


@dataclass(frozen=True)
class CallerMove:
    turn: CallerTurn
    source: CallerSource


class ScriptedCaller:
    def __init__(self, script: ResolvedCallerScript) -> None:
        self.script = script
        self._next_index = 0
        self._rule_uses: dict[str, int] = {}
        self.closing_spoken = False
        self.confirm_spoken = False
        self.fallbacks_used = 0
        self.nudges_used = 0
        self.invalid = False
        self.invalid_reason: str | None = None

    @property
    def script_exhausted(self) -> bool:
        return self._next_index >= len(self.script.turns)

    def first_move(self) -> CallerMove:
        return self._speak_script()

    def _speak_script(self) -> CallerMove:
        turn = self.script.turns[self._next_index]
        self._next_index += 1
        if turn.is_closing:
            self.closing_spoken = True
        return CallerMove(turn, "script")

    def _clarification(self, agent_text: str) -> CallerMove | None:
        for rule in self.script.clarifications:
            if self._rule_uses.get(rule.id, 0) < rule.max_uses and rule.matches(agent_text):
                self._rule_uses[rule.id] = self._rule_uses.get(rule.id, 0) + 1
                return CallerMove(rule.reply, "clarification")
        return None

    def next_move(self, agent_text: str) -> CallerMove | None:
        text = agent_text or ""
        if self.invalid:
            return None
        clar = self._clarification(text)
        if clar is not None:
            return clar
        if not self.script_exhausted:
            return self._speak_script()
        if not self.confirm_spoken and self.script.is_confirm_request(text):
            self.confirm_spoken = True
            return CallerMove(self.script.confirm, "confirm")
        if not self.closing_spoken:
            self.closing_spoken = True
            return CallerMove(self.script.closing, "closing")
        if text.strip() and self.script.is_question(text):
            if self.fallbacks_used < self.script.max_fallbacks:
                self.fallbacks_used += 1
                return CallerMove(self.script.fallback, "fallback")
            self.invalid = True
            self.invalid_reason = f"agent asked a question the script could not answer: {text.strip()[:120]!r}"
            return None
        if self.nudges_used < self.script.max_nudges:
            self.nudges_used += 1
            return CallerMove(self.script.nudge, "nudge")
        return None

    def should_end(self, agent_text: str, has_active_order: bool) -> bool:
        if not (self.script_exhausted and self.closing_spoken):
            return False
        return has_active_order or self.script.is_goodbye(agent_text or "")
```

- [ ] **Step 4: Run tests, lint, types** → green.
- [ ] **Step 5: Commit** `feat(caller): scripted caller state machine`

---

### Task 5: Adapter protocol and in-process adapter

**Files:**
- Create: `src/indicorderbench/adapters/__init__.py`, `adapters/protocol.py`, `adapters/inprocess.py`
- Test: `tests/test_adapters_inprocess.py`

**Interfaces:**
- Produces:
  ```python
  class Modality(str, Enum): TEXT = "text"; AUDIO = "audio"; BOTH = "both"
  @dataclass class CallerUtterance: turn_id: str; text: str | None; audio_path: Path | None; audio_bytes: bytes | None; language: str
  @dataclass class AgentReply: text: str; audio_bytes: bytes | None = None; meta: dict[str, Any] = field(default_factory=dict)
  @dataclass class SessionInfo: scenario_id: str; trial: int; language: str; modality: Modality; backend: OrderBackend; session_id: str; backend_url: str | None = None
  class AgentAdapter(Protocol):
      async def start(self, session: SessionInfo) -> None: ...
      async def respond(self, utterance: CallerUtterance) -> AgentReply: ...
      async def stop(self) -> None: ...
  class Agent(Protocol):
      def handle(self, utterance: CallerUtterance) -> AgentReply | Awaitable[AgentReply]: ...
  AgentFactory = Callable[[OrderBackend, SessionInfo], Agent]
  class InProcessAdapter:
      def __init__(self, agent_factory: AgentFactory) -> None
  ```

- [ ] **Step 1: Write failing tests**

`tests/test_adapters_inprocess.py`:
```python
import asyncio

from indicorderbench.adapters.inprocess import InProcessAdapter
from indicorderbench.adapters.protocol import AgentReply, CallerUtterance, Modality, SessionInfo
from indicorderbench.backend.state import OrderBackend
from tests.test_schemas import make_menu


class SyncAgent:
    def __init__(self, backend: OrderBackend) -> None:
        self.backend = backend
        self.closed = False

    def handle(self, u: CallerUtterance) -> AgentReply:
        self.backend.add_item("mango_lassi")
        return AgentReply(text=f"got {u.text}")

    def close(self) -> None:
        self.closed = True


class AsyncAgent:
    def __init__(self, backend: OrderBackend) -> None:
        self.backend = backend

    async def handle(self, u: CallerUtterance) -> AgentReply:
        await asyncio.sleep(0)
        return AgentReply(text="async", meta={"cost_usd": 0.01})


def session(backend: OrderBackend) -> SessionInfo:
    return SessionInfo(scenario_id="s", trial=1, language="en-IN", modality=Modality.TEXT,
                       backend=backend, session_id="sess")


async def test_sync_agent_roundtrip_and_close():
    backend = OrderBackend(make_menu())
    agents: list[SyncAgent] = []

    def factory(b: OrderBackend, s: SessionInfo) -> SyncAgent:
        a = SyncAgent(b)
        agents.append(a)
        return a

    adapter = InProcessAdapter(factory)
    await adapter.start(session(backend))
    reply = await adapter.respond(CallerUtterance(turn_id="t1", text="hi", audio_path=None, audio_bytes=None, language="en-IN"))
    assert reply.text == "got hi" and len(backend.get_cart()) == 1
    await adapter.stop()
    assert agents[0].closed


async def test_async_agent():
    adapter = InProcessAdapter(lambda b, s: AsyncAgent(b))
    await adapter.start(session(OrderBackend(make_menu())))
    reply = await adapter.respond(CallerUtterance(turn_id="t1", text="hi", audio_path=None, audio_bytes=None, language="en-IN"))
    assert reply.text == "async" and reply.meta["cost_usd"] == 0.01
```

- [ ] **Step 2: Run to verify failure** → ImportError.

- [ ] **Step 3: Implement `adapters/protocol.py`**

```python
"""Adapter contracts between the runner and an agent under test."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from indicorderbench.backend.state import OrderBackend


class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    BOTH = "both"


@dataclass
class CallerUtterance:
    turn_id: str
    text: str | None
    audio_path: Path | None
    audio_bytes: bytes | None
    language: str


@dataclass
class AgentReply:
    text: str
    audio_bytes: bytes | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionInfo:
    scenario_id: str
    trial: int
    language: str
    modality: Modality
    backend: OrderBackend
    session_id: str
    backend_url: str | None = None


@runtime_checkable
class AgentAdapter(Protocol):
    async def start(self, session: SessionInfo) -> None: ...
    async def respond(self, utterance: CallerUtterance) -> AgentReply: ...
    async def stop(self) -> None: ...


class Agent(Protocol):
    def handle(self, utterance: CallerUtterance) -> AgentReply | Awaitable[AgentReply]: ...


AgentFactory = Callable[[OrderBackend, SessionInfo], Agent]
```

- [ ] **Step 4: Implement `adapters/inprocess.py`**

```python
"""Adapter for agents that run inside the benchmark process."""

from __future__ import annotations

import inspect

from indicorderbench.adapters.protocol import Agent, AgentFactory, AgentReply, CallerUtterance, SessionInfo


class InProcessAdapter:
    def __init__(self, agent_factory: AgentFactory) -> None:
        self._factory = agent_factory
        self._agent: Agent | None = None

    async def start(self, session: SessionInfo) -> None:
        self._agent = self._factory(session.backend, session)

    async def respond(self, utterance: CallerUtterance) -> AgentReply:
        if self._agent is None:
            raise RuntimeError("adapter not started")
        result = self._agent.handle(utterance)
        if inspect.isawaitable(result):
            return await result
        return result

    async def stop(self) -> None:
        agent, self._agent = self._agent, None
        close = getattr(agent, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
```

- [ ] **Step 5: Run tests, lint, types** → green.
- [ ] **Step 6: Commit** `feat(adapters): adapter protocol and in-process adapter`

---

### Task 6: Pack loader, runner and stats

**Files:**
- Create: `src/indicorderbench/packs.py`, `src/indicorderbench/runner/__init__.py`, `runner/runner.py`, `runner/stats.py`
- Test: `tests/test_packs.py`, `tests/test_runner.py`, `tests/test_stats.py`, fixture pack at `tests/fixtures/minipack/` (pack.yaml, menu.yaml, scenarios/en_quantity_01.yaml, scenarios/en_cancellation_01.yaml)

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  ```python
  # packs.py
  class PackError(Exception): problems: list[str]
  @dataclass class Pack: root: Path; manifest: PackManifest; menu: Menu; scenarios: list[Scenario]
      def scenario(self, scenario_id: str) -> Scenario
      def filter(self, tags: list[str] | None = None, languages: list[str] | None = None, categories: list[str] | None = None, ids: list[str] | None = None) -> list[Scenario]
      def defaults_for(self, language: Language) -> CallerDefaults
      def clip_path(self, scenario_id: str, turn: CallerTurn) -> Path | None
      def content_hash(self) -> str          # sha256 over sorted yaml file bytes, 16 hex chars
  def load_pack(root: Path) -> Pack           # raises PackError listing every problem
  def validate_pack(root: Path) -> list[str]  # [] when valid
  # runner/runner.py
  @dataclass class RunConfig: trials: int = 1; modality: Modality = Modality.TEXT; seed: int | None = None; timeout_turn_s: float = 30.0; timeout_trial_s: float = 300.0; max_turns: int = 20; agent_label: str = "agent"; agent_spec: str = ""
  AdapterFactory = Callable[[], AgentAdapter]
  class SessionRegistry: def register(self, session_id: str, backend: OrderBackend) -> None; def get(self, session_id: str) -> OrderBackend; def unregister(self, session_id: str) -> None   # thread-safe
  async def run_trial(pack: Pack, scenario: Scenario, adapter: AgentAdapter, config: RunConfig, trial: int, backend_url: str | None = None, registry: SessionRegistry | None = None) -> TrialResult
  async def run_suite(pack: Pack, scenarios: list[Scenario], adapter_factory: AdapterFactory, config: RunConfig, backend_url: str | None = None, registry: SessionRegistry | None = None, on_scenario: Callable[[ScenarioResult], None] | None = None) -> SuiteResult
  def run_suite_sync(...same...) -> SuiteResult
  # runner/stats.py
  def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]
  def percentile(values: list[float], p: float) -> float | None
  def pass_k(scenarios: list[ScenarioResult], k: int) -> float | None
  def group_metrics(scenarios: list[ScenarioResult], key: Callable[[ScenarioResult], str]) -> list[GroupMetric]
  def compute_metrics(scenarios: list[ScenarioResult], trials: int) -> Metrics
  ```

- [ ] **Step 1: Write the fixture mini-pack**

`tests/fixtures/minipack/pack.yaml`:
```yaml
id: minipack
name: Mini pack for tests
version: "0.0.1"
menu: menu.yaml
defaults:
  caller:
    en-IN:
      closing: {text: "That's all."}
      confirm: {text: "Yes, place it."}
      fallback: {text: "Yes, that's right."}
      nudge: {text: "Please place the order."}
      confirm_patterns: ["shall i place", "place the order\\?", "confirm"]
      goodbye_patterns: ["order placed", "order is placed", "already placed"]
      question_patterns: ["which", "how many", "what "]
      clarifications: []
    hi-en:
      closing: {text: "Bas itna hi."}
      confirm: {text: "Haan, order kar do."}
      fallback: {text: "Haan, theek hai."}
      nudge: {text: "Order place kar do."}
      confirm_patterns: ["order kar d(oo|u)n", "confirm"]
      goodbye_patterns: ["order place ho gaya", "pehle se place"]
      question_patterns: ["kitne", "konsa", "kaunsa", "kya"]
      clarifications: []
```

`tests/fixtures/minipack/menu.yaml`:
```yaml
id: mini
name: Mini menu
modifier_groups:
  - id: onion
    name: Onion
    exclusive: true
    default: with_onion
    options:
      - {id: with_onion, name: With onion, aliases: [with onion]}
      - {id: no_onion, name: No onion, aliases: [no onion, without onion, bina pyaaz]}
items:
  - {id: paneer_wrap, name: Paneer Wrap, aliases: [paneer wrap, paneer wraps], price: 180, modifier_groups: [onion]}
  - {id: mango_lassi, name: Mango Lassi, aliases: [mango lassi], price: 90}
```

`tests/fixtures/minipack/scenarios/en_quantity_01.yaml`:
```yaml
id: en_quantity_01
title: Two wraps
language: en-IN
category: quantity
menu: mini
tags: [smoke]
caller:
  turns:
    - text: "Two paneer wraps please."
expected:
  - id: two_wraps
    orders:
      - lines:
          - {item_id: paneer_wrap, quantity: 2}
```

`tests/fixtures/minipack/scenarios/en_cancellation_01.yaml`:
```yaml
id: en_cancellation_01
title: Cancel after ordering
language: en-IN
category: cancellation
menu: mini
tags: [smoke]
caller:
  turns:
    - text: "One mango lassi."
    - text: "Actually cancel the order."
expected:
  - id: nothing
    orders: []
```

- [ ] **Step 2: Write failing tests**

`tests/test_packs.py`:
```python
from pathlib import Path

import pytest
import yaml

from indicorderbench.packs import PackError, load_pack, validate_pack
from indicorderbench.schemas.scenario import Language

FIX = Path(__file__).parent / "fixtures" / "minipack"


def test_load_minipack():
    pack = load_pack(FIX)
    assert pack.manifest.id == "minipack" and len(pack.scenarios) == 2
    assert pack.scenario("en_quantity_01").category.value == "quantity"
    assert pack.defaults_for(Language.EN_IN).closing.text == "That's all."
    assert len(pack.content_hash()) == 16
    assert [s.id for s in pack.filter(categories=["cancellation"])] == ["en_cancellation_01"]
    assert len(pack.filter(tags=["smoke"])) == 2 and pack.filter(tags=["nope"]) == []


def test_clip_path_convention(tmp_path: Path):
    pack = load_pack(FIX)
    turn = pack.scenario("en_quantity_01").caller.turns[0]
    assert pack.clip_path("en_quantity_01", turn) is None
    (FIX / "clips" / "en_quantity_01").mkdir(parents=True, exist_ok=True)
    clip = FIX / "clips" / "en_quantity_01" / "t1.wav"
    clip.write_bytes(b"RIFF")
    try:
        assert pack.clip_path("en_quantity_01", turn) == clip
    finally:
        clip.unlink()


def test_validate_reports_unknown_item_and_option(tmp_path: Path):
    import shutil
    shutil.copytree(FIX, tmp_path / "p")
    bad = tmp_path / "p" / "scenarios" / "bad.yaml"
    doc = yaml.safe_load((FIX / "scenarios" / "en_quantity_01.yaml").read_text())
    doc["id"] = "en_quantity_99"
    doc["expected"][0]["orders"][0]["lines"][0]["item_id"] = "ghost"
    doc["expected"][0]["orders"][0]["lines"].append({"item_id": "paneer_wrap", "quantity": 1, "modifiers": {"onion": "burnt"}})
    bad.write_text(yaml.safe_dump(doc))
    problems = validate_pack(tmp_path / "p")
    assert any("bad.yaml" in p and "ghost" in p for p in problems)
    assert any("bad.yaml" in p and "burnt" in p for p in problems)
    with pytest.raises(PackError):
        load_pack(tmp_path / "p")


def test_validate_rejects_duplicate_ids_and_missing_language_defaults(tmp_path: Path):
    import shutil
    shutil.copytree(FIX, tmp_path / "p")
    src = tmp_path / "p" / "scenarios" / "en_quantity_01.yaml"
    (tmp_path / "p" / "scenarios" / "dup.yaml").write_text(src.read_text())
    problems = validate_pack(tmp_path / "p")
    assert any("duplicate scenario id" in p for p in problems)
```

`tests/test_stats.py`:
```python
from datetime import datetime

import pytest

from indicorderbench.runner.stats import compute_metrics, pass_k, percentile, wilson_interval
from indicorderbench.schemas.results import Outcome, ScenarioResult, TrialResult


def trial(sid: str, n: int, outcome: Outcome, latency: float = 100.0) -> TrialResult:
    from indicorderbench.schemas.results import TurnRecord
    return TrialResult(scenario_id=sid, trial=n, outcome=outcome, started_at=datetime(2026, 1, 1), duration_ms=1.0,
                       transcript=[TurnRecord(index=0, speaker="agent", text="x", t_ms=0.0, latency_ms=latency)])


def scen(sid: str, lang: str, cat: str, outcomes: list[Outcome]) -> ScenarioResult:
    return ScenarioResult(scenario_id=sid, title=sid, language=lang, category=cat,
                          trials=[trial(sid, i + 1, o) for i, o in enumerate(outcomes)])


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = wilson_interval(9, 10)
    assert 0.59 < lo < 0.60 and 0.98 < hi < 0.99
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)


def test_pass_k_tau_bench_estimator():
    s = [scen("a", "en-IN", "quantity", [Outcome.PASS, Outcome.PASS, Outcome.FAIL]),
         scen("b", "en-IN", "quantity", [Outcome.PASS, Outcome.PASS, Outcome.PASS])]
    assert pass_k(s, 1) == pytest.approx((2 / 3 + 1) / 2)
    assert pass_k(s, 2) == pytest.approx((1 / 3 + 1) / 2)
    assert pass_k(s, 3) == pytest.approx((0 + 1) / 2)
    assert pass_k(s, 4) is None


def test_compute_metrics_excludes_invalid_and_groups():
    s = [scen("a", "en-IN", "quantity", [Outcome.PASS, Outcome.SIMULATOR_INVALID]),
         scen("b", "hi-en", "modifier", [Outcome.FAIL, Outcome.INFRA_ERROR]),
         scen("c", "hi-en", "modifier", [Outcome.SIMULATOR_INVALID, Outcome.SIMULATOR_INVALID])]
    m = compute_metrics(s, trials=2)
    assert (m.n_valid, m.n_pass, m.n_fail, m.n_simulator_invalid, m.n_infra_error) == (2, 1, 1, 3, 1)
    assert m.pass_rate == pytest.approx(0.5)  # scenario-weighted over a and b; c has no valid trials
    by_lang = {g.key: g for g in m.by_language}
    assert by_lang["en-IN"].n_trials == 1 and by_lang["en-IN"].pass_rate == 1.0
    assert by_lang["hi-en"].n_trials == 1 and by_lang["hi-en"].pass_rate == 0.0
    assert m.latency_p50_ms == 100.0
    assert m.pass_k[1] == pytest.approx(0.5) and m.pass_k[2] is None
```

`tests/test_runner.py`:
```python
import asyncio
from pathlib import Path

from indicorderbench.adapters.protocol import AgentReply, CallerUtterance, Modality, SessionInfo
from indicorderbench.packs import load_pack
from indicorderbench.runner.runner import RunConfig, run_suite, run_trial
from indicorderbench.schemas.results import Outcome

FIX = Path(__file__).parent / "fixtures" / "minipack"


class ScriptedAgent:
    """Adapter + agent in one: replies from a list; optional backend actions per turn."""

    def __init__(self, replies: list[str], actions: dict[int, str] | None = None, hang_turn: int | None = None,
                 raise_turn: int | None = None) -> None:
        self.replies, self.actions, self.hang_turn, self.raise_turn = replies, actions or {}, hang_turn, raise_turn
        self.n = 0

    async def start(self, session: SessionInfo) -> None:
        self.backend = session.backend

    async def respond(self, u: CallerUtterance) -> AgentReply:
        self.n += 1
        if self.n == self.hang_turn:
            await asyncio.sleep(10)
        if self.n == self.raise_turn:
            raise RuntimeError("agent exploded")
        action = self.actions.get(self.n)
        if action == "add2":
            self.backend.add_item("paneer_wrap", 2)
        elif action == "add1":
            self.backend.add_item("paneer_wrap", 1)
        elif action == "submit":
            self.backend.submit_order()
        elif action == "lassi":
            self.backend.add_item("mango_lassi", 1)
        elif action == "cancel":
            for o in self.backend.active_orders():
                self.backend.cancel_order(o.order_id)
        return AgentReply(text=self.replies[min(self.n - 1, len(self.replies) - 1)])

    async def stop(self) -> None:
        pass


def cfg(**kw) -> RunConfig:
    return RunConfig(**{"timeout_turn_s": 0.2, "timeout_trial_s": 2.0, **kw})


async def test_pass_trial():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["Added two paneer wraps. Anything else?", "Order placed, thank you!"], {1: "add2", 2: "submit"})
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.PASS and r.check and r.check.passed
    assert [t.source for t in r.transcript if t.speaker == "caller"] == ["script", "closing"]
    assert r.transcript[1].latency_ms is not None and r.snapshot is not None
    assert len(r.snapshot.trace) == 2


async def test_fail_trial_with_field_checks():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["Added. Anything else?", "Order placed!"], {1: "add1", 2: "submit"})
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.FAIL
    assert any(f.field == "Paneer Wrap quantity" and not f.passed for f in r.check.field_checks)


async def test_simulator_invalid_when_agent_keeps_asking():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["Which branch?"] * 10)
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.SIMULATOR_INVALID and not r.caller_valid and r.check is None


async def test_no_submission_after_nudges_is_fail():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["Okay."] * 10, {1: "add2"})
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(), trial=1)
    assert r.outcome is Outcome.FAIL and r.caller_valid
    assert r.check.field_checks[0].field == "Submitted orders" and r.check.field_checks[0].actual == "0"


async def test_timeout_and_exception_are_infra_errors():
    pack = load_pack(FIX)
    hang = ScriptedAgent(["x"], hang_turn=1)
    r = await run_trial(pack, pack.scenario("en_quantity_01"), hang, cfg(), trial=1)
    assert r.outcome is Outcome.INFRA_ERROR and "timed out" in (r.error or "")
    boom = ScriptedAgent(["x"], raise_turn=1)
    r = await run_trial(pack, pack.scenario("en_quantity_01"), boom, cfg(), trial=1)
    assert r.outcome is Outcome.INFRA_ERROR and "agent exploded" in (r.error or "")


async def test_max_turns_hit_is_fail_not_invalid():
    pack = load_pack(FIX)
    # a clarification rule that always matches keeps the caller talking forever
    scen = pack.scenario("en_quantity_01").model_copy(deep=True)
    from indicorderbench.schemas.scenario import CallerTurn, ClarificationRule
    scen.caller.clarifications = [ClarificationRule(id="loop", match=["."], reply=CallerTurn(text="two"), max_uses=99)]
    agent = ScriptedAgent(["hmm?"] * 50)
    r = await run_trial(pack, scen, agent, cfg(max_turns=6), trial=1)
    assert r.outcome is Outcome.FAIL and r.max_turns_hit and r.caller_valid


async def test_audio_modality_missing_clip_is_infra_error():
    pack = load_pack(FIX)
    agent = ScriptedAgent(["x"])
    r = await run_trial(pack, pack.scenario("en_quantity_01"), agent, cfg(modality=Modality.AUDIO), trial=1)
    assert r.outcome is Outcome.INFRA_ERROR and "clip" in (r.error or "")


async def test_run_suite_continues_after_infra_error_and_computes_metrics():
    pack = load_pack(FIX)
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            return ScriptedAgent(["x"], raise_turn=1)
        return ScriptedAgent(["Cancelled. Anything else?", "Order placed"], {1: "lassi"})

    suite = await run_suite(pack, pack.scenarios, factory, cfg(trials=1, agent_label="t", agent_spec="test"))
    assert suite.metrics.n_infra_error == 1 and suite.metrics.n_scenarios == 2
    assert suite.pack.id == "minipack" and suite.run.trials == 1
    assert suite.scenarios[0].trials[0].outcome is Outcome.INFRA_ERROR
```

- [ ] **Step 3: Run to verify failure** → ImportError.

- [ ] **Step 4: Implement `packs.py`**

```python
"""Load and validate scenario packs from a directory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from indicorderbench.schemas.menu import Menu
from indicorderbench.schemas.scenario import CallerDefaults, CallerTurn, Language, PackManifest, Scenario


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

    def filter(self, tags: list[str] | None = None, languages: list[str] | None = None,
               categories: list[str] | None = None, ids: list[str] | None = None) -> list[Scenario]:
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

    def clip_path(self, scenario_id: str, turn: CallerTurn) -> Path | None:
        if turn.audio:
            p = self.root / turn.audio
            return p if p.exists() else None
        p = self.root / "clips" / scenario_id / f"{turn.id}.wav"
        return p if p.exists() else None

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.root.rglob("*.yaml")):
            h.update(p.relative_to(self.root).as_posix().encode())
            h.update(p.read_bytes())
        return h.hexdigest()[:16]


def _fmt_validation(path: Path, err: ValidationError) -> list[str]:
    return [f"{path}: {'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in err.errors()]


def _load_yaml(path: Path, problems: list[str]) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        problems.append(f"{path}: {e}")
        return None


def _check_references(path: Path, scenario: Scenario, menu: Menu, problems: list[str]) -> None:
    if scenario.menu != menu.id:
        problems.append(f"{path}: menu: references {scenario.menu!r} but pack menu is {menu.id!r}")
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
                        problems.append(f"{path}: {loc}.modifiers.{gid}: group not applicable to {line.item_id!r}")
                        continue
                    opts = [] if spec == "*" else ([spec] if isinstance(spec, str) else list(spec))
                    for o in opts:
                        if not any(o == x.id for x in groups[gid].options):
                            problems.append(f"{path}: {loc}.modifiers.{gid}: unknown option {o!r}")
                    if groups[gid].exclusive and len(opts) > 1:
                        problems.append(f"{path}: {loc}.modifiers.{gid}: exclusive group given {len(opts)} options")


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
    menu_path = root / manifest.menu
    menu: Menu | None = None
    raw_menu = _load_yaml(menu_path, problems) if menu_path.exists() else problems.append(f"{menu_path}: missing")
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
            problems.append(f"{path}: language {s.language.value!r} has no caller defaults in pack.yaml")
        if menu is not None:
            _check_references(path, s, menu, problems)
        scenarios.append(s)
    if not scenarios and not problems:
        problems.append(f"{root / 'scenarios'}: no scenarios found")
    if problems or menu is None:
        return None, problems
    return Pack(root=root, manifest=manifest, menu=menu, scenarios=scenarios), []


def validate_pack(root: Path) -> list[str]:
    _, problems = _load(root)
    return problems


def load_pack(root: Path) -> Pack:
    pack, problems = _load(root)
    if pack is None:
        raise PackError(problems or ["unknown pack error"])
    return pack
```

Note the `raw_menu = ... if ... else problems.append(...)` line is too clever; write it as a plain `if/else` in the final code.

- [ ] **Step 5: Implement `runner/stats.py`**

```python
"""Suite statistics: pass rate, pass^k, Wilson intervals, latency percentiles."""

from __future__ import annotations

import math
from collections.abc import Callable

from indicorderbench.schemas.results import GroupMetric, Metrics, Outcome, ScenarioResult


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def pass_k(scenarios: list[ScenarioResult], k: int) -> float | None:
    vals: list[float] = []
    for s in scenarios:
        n, c = s.n_valid, s.n_pass
        if n == 0:
            continue
        if k > n:
            return None
        vals.append(math.comb(c, k) / math.comb(n, k))
    return sum(vals) / len(vals) if vals else None


def group_metrics(scenarios: list[ScenarioResult], key: Callable[[ScenarioResult], str]) -> list[GroupMetric]:
    buckets: dict[str, tuple[int, int]] = {}
    for s in scenarios:
        n, c = buckets.get(key(s), (0, 0))
        buckets[key(s)] = (n + s.n_valid, c + s.n_pass)
    out = []
    for k in sorted(buckets):
        n, c = buckets[k]
        lo, hi = wilson_interval(c, n)
        out.append(GroupMetric(key=k, n_trials=n, n_pass=c, pass_rate=(c / n if n else None),
                               ci_low=(lo if n else None), ci_high=(hi if n else None)))
    return out


def compute_metrics(scenarios: list[ScenarioResult], trials: int) -> Metrics:
    all_trials = [t for s in scenarios for t in s.trials]
    rates = [s.pass_rate for s in scenarios if s.pass_rate is not None]
    latencies = [x for t in all_trials for x in t.latencies_ms]
    return Metrics(
        n_scenarios=len(scenarios),
        n_trials_total=len(all_trials),
        n_valid=sum(1 for t in all_trials if t.is_valid),
        n_pass=sum(1 for t in all_trials if t.outcome is Outcome.PASS),
        n_fail=sum(1 for t in all_trials if t.outcome is Outcome.FAIL),
        n_simulator_invalid=sum(1 for t in all_trials if t.outcome is Outcome.SIMULATOR_INVALID),
        n_infra_error=sum(1 for t in all_trials if t.outcome is Outcome.INFRA_ERROR),
        n_max_turns=sum(1 for t in all_trials if t.max_turns_hit),
        pass_rate=(sum(rates) / len(rates) if rates else None),
        pass_k={k: pass_k(scenarios, k) for k in range(1, max(1, trials) + 1)},
        by_language=group_metrics(scenarios, lambda s: s.language),
        by_category=group_metrics(scenarios, lambda s: s.category),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
    )
```

- [ ] **Step 6: Implement `runner/runner.py`**

```python
"""Trial and suite execution."""

from __future__ import annotations

import asyncio
import platform
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from indicorderbench import __version__
from indicorderbench.adapters.protocol import AgentAdapter, AgentReply, CallerUtterance, Modality, SessionInfo
from indicorderbench.backend.state import OrderBackend
from indicorderbench.caller.scripted import ScriptedCaller
from indicorderbench.checker.checker import check
from indicorderbench.packs import Pack
from indicorderbench.runner.stats import compute_metrics
from indicorderbench.schemas.results import (
    AgentInfo, Outcome, PackInfo, RunInfo, ScenarioResult, SuiteResult, TrialResult, TurnRecord,
)
from indicorderbench.schemas.scenario import Scenario


@dataclass
class RunConfig:
    trials: int = 1
    modality: Modality = Modality.TEXT
    seed: int | None = None
    timeout_turn_s: float = 30.0
    timeout_trial_s: float = 300.0
    max_turns: int = 20
    agent_label: str = "agent"
    agent_spec: str = ""


AdapterFactory = Callable[[], AgentAdapter]


class SessionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, OrderBackend] = {}

    def register(self, session_id: str, backend: OrderBackend) -> None:
        with self._lock:
            self._sessions[session_id] = backend

    def get(self, session_id: str) -> OrderBackend:
        with self._lock:
            return self._sessions[session_id]

    def unregister(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


class _MissingClip(Exception):
    pass


def _utterance(pack: Pack, scenario: Scenario, turn_id: str, text: str, audio_path: object, modality: Modality) -> CallerUtterance:
    from pathlib import Path
    path = audio_path if isinstance(audio_path, Path) else None
    if modality in (Modality.AUDIO, Modality.BOTH) and path is None:
        raise _MissingClip(f"no audio clip for {scenario.id}/{turn_id}; run `iob synth` or use --modality text")
    return CallerUtterance(
        turn_id=turn_id,
        text=None if modality is Modality.AUDIO else text,
        audio_path=path if modality is not Modality.TEXT else None,
        audio_bytes=path.read_bytes() if (path and modality is not Modality.TEXT) else None,
        language=scenario.language.value,
    )


async def run_trial(pack: Pack, scenario: Scenario, adapter: AgentAdapter, config: RunConfig, trial: int,
                    backend_url: str | None = None, registry: SessionRegistry | None = None) -> TrialResult:
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    clock = lambda: (time.perf_counter() - t0) * 1000.0  # noqa: E731
    backend = OrderBackend(pack.menu, clock=clock)
    session_id = f"{scenario.id}-{trial}-{uuid.uuid4().hex[:8]}"
    if registry is not None:
        registry.register(session_id, backend)
    session = SessionInfo(scenario_id=scenario.id, trial=trial, language=scenario.language.value,
                          modality=config.modality, backend=backend, session_id=session_id, backend_url=backend_url)
    caller = ScriptedCaller(scenario.caller.resolved(pack.defaults_for(scenario.language)))
    transcript: list[TurnRecord] = []
    error: str | None = None
    max_turns_hit = False
    agent_meta: dict[str, object] = {}

    async def converse() -> None:
        nonlocal max_turns_hit
        await adapter.start(session)
        move = caller.first_move()
        turns = 0
        while True:
            turn = move.turn
            assert turn.id is not None
            clip = pack.clip_path(scenario.id, turn)
            utt = _utterance(pack, scenario, turn.id, turn.text, clip, config.modality)
            transcript.append(TurnRecord(index=len(transcript), speaker="caller", text=turn.text,
                                         audio_path=(str(clip) if clip else None), t_ms=clock(),
                                         turn_id=turn.id, source=move.source))
            sent = time.perf_counter()
            try:
                reply: AgentReply = await asyncio.wait_for(adapter.respond(utt), timeout=config.timeout_turn_s)
            except TimeoutError as e:
                raise RuntimeError(f"agent timed out after {config.timeout_turn_s}s on turn {turn.id}") from e
            latency = (time.perf_counter() - sent) * 1000.0
            transcript.append(TurnRecord(index=len(transcript), speaker="agent", text=reply.text, t_ms=clock(),
                                         latency_ms=latency))
            if reply.meta:
                agent_meta.update(reply.meta)
            turns += 1
            if caller.should_end(reply.text, bool(backend.active_orders())):
                return
            if turns >= config.max_turns:
                max_turns_hit = True
                return
            nxt = caller.next_move(reply.text)
            if nxt is None:
                return
            move = nxt

    try:
        try:
            await asyncio.wait_for(converse(), timeout=config.timeout_trial_s)
        except TimeoutError:
            error = f"trial timed out after {config.timeout_trial_s}s"
        except _MissingClip as e:
            error = str(e)
        except Exception as e:  # noqa: BLE001 - any adapter failure is an infra error
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    finally:
        try:
            await asyncio.wait_for(adapter.stop(), timeout=config.timeout_turn_s)
        except Exception as e:  # noqa: BLE001
            error = error or f"adapter.stop failed: {type(e).__name__}: {e}"
        if registry is not None:
            registry.unregister(session_id)

    snapshot = backend.snapshot()
    duration = (time.perf_counter() - t0) * 1000.0
    if error is not None:
        outcome, result = Outcome.INFRA_ERROR, None
    elif caller.invalid:
        outcome, result = Outcome.SIMULATOR_INVALID, None
    else:
        result = check(scenario.expected, snapshot, pack.menu)
        outcome = Outcome.PASS if result.passed else Outcome.FAIL
    return TrialResult(scenario_id=scenario.id, trial=trial, outcome=outcome, check=result, transcript=transcript,
                       snapshot=snapshot, caller_valid=not caller.invalid, caller_invalid_reason=caller.invalid_reason,
                       error=error, max_turns_hit=max_turns_hit, started_at=started, duration_ms=duration,
                       agent_meta=dict(agent_meta))


async def run_suite(pack: Pack, scenarios: list[Scenario], adapter_factory: AdapterFactory, config: RunConfig,
                    backend_url: str | None = None, registry: SessionRegistry | None = None,
                    on_scenario: Callable[[ScenarioResult], None] | None = None) -> SuiteResult:
    started = datetime.now(UTC)
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        trials = []
        for n in range(1, config.trials + 1):
            adapter = adapter_factory()
            trials.append(await run_trial(pack, scenario, adapter, config, n, backend_url, registry))
        sr = ScenarioResult(scenario_id=scenario.id, title=scenario.title, language=scenario.language.value,
                            category=scenario.category.value, tags=list(scenario.tags), trials=trials)
        results.append(sr)
        if on_scenario:
            on_scenario(sr)
    finished = datetime.now(UTC)
    return SuiteResult(
        run=RunInfo(started_at=started, finished_at=finished, trials=config.trials, modality=config.modality.value,
                    seed=config.seed, host=socket.gethostname(), benchmark_version=__version__,
                    python=platform.python_version(), max_turns=config.max_turns,
                    timeout_turn_s=config.timeout_turn_s, timeout_trial_s=config.timeout_trial_s),
        pack=PackInfo(id=pack.manifest.id, version=pack.manifest.version, content_hash=pack.content_hash(),
                      path=str(pack.root)),
        agent=AgentInfo(label=config.agent_label, spec=config.agent_spec),
        scenarios=results,
        metrics=compute_metrics(results, config.trials),
    )


def run_suite_sync(pack: Pack, scenarios: list[Scenario], adapter_factory: AdapterFactory, config: RunConfig,
                   backend_url: str | None = None, registry: SessionRegistry | None = None,
                   on_scenario: Callable[[ScenarioResult], None] | None = None) -> SuiteResult:
    return asyncio.run(run_suite(pack, scenarios, adapter_factory, config, backend_url, registry, on_scenario))
```

Python 3.11 note: `asyncio.TimeoutError` is an alias of the builtin `TimeoutError` from 3.11, so `except TimeoutError` is correct. Use `datetime.UTC` (3.11+).

- [ ] **Step 7: Run tests, lint, types** → green.
- [ ] **Step 8: Commit** `feat(runner): pack loader, trial runner, suite statistics`

---

## Phase B — parallel tracks (each depends only on Phase A)

### Task 7: Lexicon, parser and rule-based reference agent

**Files:**
- Create: `src/indicorderbench/agents/__init__.py`, `agents/lexicon.py`, `agents/parsing.py`, `agents/rule_based.py`
- Test: `tests/test_parsing.py`, `tests/test_rule_based_agent.py`

**Interfaces:**
- Consumes: `Menu`, `normalise_text`, `OrderBackend`, `BackendError`, `CallerUtterance`, `AgentReply`, `SessionInfo`.
- Produces:
  ```python
  # lexicon.py  (all phrases stored normalised; language "en-IN" | "hi-en")
  NUMBER_WORDS: dict[str, dict[str, int]]      # per language; "do"→2 only in hi-en, "a"/"an"→1 only in en-IN; digits 1-10 in both
  CONNECTORS: list[str]                        # ["and also", "and", "aur", "also", "plus", "then", "phir", "ke saath"? no] -> exactly: ["and also", "and", "aur", "also", "plus", "then", "phir"]
  CORRECTION_MARKERS, REMOVE_MARKERS, CANCEL_ORDER_MARKERS, CLOSING_MARKERS, CONFIRM_MARKERS, QUESTION_MARKERS: list[str]
  # parsing.py
  class Intent(str, Enum): ADD, CORRECT, REMOVE, CANCEL_ORDER, CLOSING, CONFIRM, UNKNOWN
  @dataclass class Clause: intent: Intent; item_id: str | None; quantity: int | None; modifiers: list[str]; raw: str
  def split_clauses(text: str) -> list[str]
  def parse_utterance(text: str, menu: Menu, language: str) -> list[Clause]
  # rule_based.py
  BUGS: tuple[str, ...] = ("ignore_corrections", "drop_modifiers", "double_submit", "ignore_cancellation", "quantity_default_one")
  class RuleBasedAgent:
      def __init__(self, backend: OrderBackend, language: str, bugs: frozenset[str] = frozenset(), transcriber: Transcriber | None = None) -> None
      def handle(self, utterance: CallerUtterance) -> AgentReply
  def make_factory(bugs: frozenset[str] = frozenset()) -> AgentFactory      # for InProcessAdapter
  def parse_agent_spec(spec: str) -> frozenset[str] | None                  # "builtin:correct" -> frozenset(); "builtin:buggy" -> all; "builtin:buggy:a,b" -> {a,b}; else None
  ```
  `Transcriber` is a Protocol defined here as `def transcribe(self, audio: bytes, language: str) -> str` (Task 12 re-exports it; keep the Protocol in `agents/rule_based.py` to avoid a dependency on the audio extra — Task 12 imports it from here).

Lexicon content (normalised, authoritative; the pack author in Task 8 must only use these):

```python
NUMBER_WORDS = {
  "en-IN": {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "ek": 1, "do": 2, "teen": 3},
  "hi-en": {"ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "che": 6, "chhe": 6,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6},
}
CONNECTORS = ["and also", "and", "aur", "also", "plus", "then", "phir"]
CORRECTION_MARKERS = ["actually", "wait", "sorry", "no wait", "instead", "make it", "make that", "make them",
                      "change it to", "change that to", "change to", "nahi nahi", "nahi", "badal do", "badal ke",
                      "ki jagah", "rehne do", "correction", "i meant", "matlab", "scratch that"]
REMOVE_MARKERS = ["remove", "take off", "hata do", "hata", "nikal do", "nikaal do", "mat do", "nahi chahiye",
                  "don t want", "do not want", "dont want", "drop", "skip", "cancel the", "cancel that", "cancel"]
CANCEL_ORDER_MARKERS = ["cancel the order", "cancel my order", "cancel the whole order", "cancel everything",
                        "cancel it all", "cancel order", "order cancel", "poora order cancel", "pura order cancel",
                        "sab cancel", "sab kuch cancel", "order cancel kar do", "cancel kar do poora"]
CLOSING_MARKERS = ["that s all", "thats all", "that is all", "that s it", "thats it", "that will be all",
                   "that ll be all", "nothing else", "that s everything", "bas", "bas itna hi", "itna hi",
                   "aur kuch nahi", "bas yahi", "ho gaya", "done", "i m done"]
CONFIRM_MARKERS = ["yes", "yeah", "yep", "yup", "haan", "ha", "ji", "ji haan", "ok", "okay", "theek hai",
                   "sahi hai", "confirm", "confirmed", "go ahead", "place it", "place the order",
                   "order kar do", "order place kar do", "correct", "right", "sure", "please place the order"]
QUESTION_MARKERS = ["which", "what", "how many", "kitne", "kitna", "konsa", "kaunsa", "kya", "would you like", "do you want", "chahiye"]
```

Parser rules (implement exactly):
1. `norm = normalise_text(text)`.
2. `split_clauses`: split `norm` on connectors as whole words (`\b(and also|and|aur|also|plus|then|phir)\b`) and on the original punctuation boundaries (`[,.;!?]`, `...`) applied before normalisation. Return non-empty stripped fragments.
3. For each fragment, in order:
   a. If it contains a `CANCEL_ORDER_MARKERS` phrase → `CANCEL_ORDER`.
   b. Else detect `is_correction` (any `CORRECTION_MARKERS` phrase as whole words), `is_remove` (`REMOVE_MARKERS`).
   c. Find item aliases: scan `menu.alias_index()` longest-first; a match is a whole-word substring; mark its span consumed; first item match is the clause item.
   d. Find option aliases the same way on the remaining text; collect option ids.
   e. Quantity: the nearest `NUMBER_WORDS[language]` token before the item span; else the first number token in the fragment; `ek hi` / `one only` / `only one` → 1.
   f. If fragment equals or starts with a `CLOSING_MARKERS` phrase and has no item → `CLOSING`.
   g. Else if every token is in `CONFIRM_MARKERS` (after removing "please") → `CONFIRM`.
   h. Else intent: `CANCEL_ORDER` handled; `REMOVE` if `is_remove`; `CORRECT` if `is_correction`; `ADD` if item found; `CORRECT` if no item but (quantity or modifiers) found and a previous clause in this utterance exists or `is_correction`; `UNKNOWN` otherwise.
4. Post-process: a `CORRECT` clause with only modifiers that immediately follows an `ADD` clause in the same utterance is merged into that `ADD` (modifiers appended) **unless** `is_correction` was detected on it. Example: "two paneer wraps, no onion, and one mango lassi" → `[ADD paneer_wrap 2 [no_onion], ADD mango_lassi 1]`.

Agent behaviour (implement exactly; replies are plain strings per language):

| Clause | Correct action | Reply en-IN / hi-en |
|---|---|---|
| ADD | `add_item(item, qty or 1, mods)`; set `last_line_id` | `Added {qty} {name}{, mods}. Anything else?` / `{qty} {name}{, mods} add kar diya. Aur kuch?` |
| CORRECT with item (swap) | `update_line(last, quantity=qty or current, modifiers=compatible)` **then** if item differs: `remove_line(last)` + `add_item(new, qty, mods)` | `Changed that to {qty} {name}. Anything else?` / `Usko {qty} {name} kar diya. Aur kuch?` |
| CORRECT qty only | `update_line(last, quantity=qty)` | `Changed {name} to {qty}. Anything else?` / `{name} {qty} kar diya. Aur kuch?` |
| CORRECT mods only | `update_line(last, modifiers=merge(current, mods))` where merge replaces options of the same exclusive group | `Updated {name}: {mods}. Anything else?` / `{name} {mods} kar diya. Aur kuch?` |
| REMOVE with item | remove the most recent line with that item | `Removed {name}. Anything else?` / `{name} hata diya. Aur kuch?` |
| REMOVE without item | remove `last_line_id` | same |
| CANCEL_ORDER | cancel every active order, clear cart | `Your order has been cancelled. Anything else?` / `Aapka order cancel kar diya. Aur kuch?` |
| CLOSING or CONFIRM | if cart non-empty: readback + `submit_order()` → `Your order: {lines}. Order placed, thank you!` / `Aapka order: {lines}. Order place ho gaya, shukriya!`; if cart empty and an active order exists: `Your order is already placed. Thank you!` / `Aapka order pehle se place ho gaya hai. Shukriya!`; if nothing at all: `Nothing in your cart yet. What would you like?` / `Cart khaali hai. Kya chahiye?` |
| UNKNOWN | nothing | `Sorry, I didn't catch that. Which item would you like?` / `Maaf kijiye, samajh nahi aaya. Konsa item chahiye?` |

`BackendError` from any call → reply `Sorry, I couldn't do that: {message}` / `Maaf kijiye, yeh nahi ho paya: {message}` (never crash).

Bug effects (apply inside the action layer; replies stay identical to the correct agent so the bug is only visible in state):

- `ignore_corrections`: CORRECT clauses skip all backend calls.
- `drop_modifiers`: every `add_item`/`update_line` is called with `modifiers=[]`.
- `double_submit`: after a successful `submit_order()`, immediately re-add the same lines and submit again.
- `ignore_cancellation`: CANCEL_ORDER and REMOVE clauses skip backend calls.
- `quantity_default_one`: ADD uses quantity 1 regardless of the parsed quantity (corrections still apply).

- [ ] **Step 1: Write failing parser tests** (`tests/test_parsing.py`) covering: English add with quantity; Hinglish "do paneer wrap" → 2; "actually ek hi karo" → CORRECT qty 1; "no onion" merge into preceding ADD; standalone "no onion, actually" → CORRECT mods; "cancel the order" → CANCEL_ORDER; "that's all" → CLOSING; "haan" → CONFIRM; "remove the lassi" → REMOVE mango_lassi; "do" in en-IN is not a number (`"do you have paneer wrap"` → ADD paneer_wrap qty None); unknown item → UNKNOWN; item swap "actually make it a chicken wrap" (use a second item in the test menu).
- [ ] **Step 2: Run → fail.** 
- [ ] **Step 3: Implement lexicon.py and parsing.py** per the rules above.
- [ ] **Step 4: Run → pass. Commit** `feat(agents): lexicon and utterance parser`
- [ ] **Step 5: Write failing agent tests** (`tests/test_rule_based_agent.py`): drive `RuleBasedAgent` directly with `CallerUtterance`s and assert on `backend.snapshot()` for the demo flow ("Two paneer wraps... actually make it one. No onion. And one mango lassi." then "That's all") → one active order, wrap qty 1 with no_onion, lassi 1; each bug produces its characteristic wrong state in the same flow (write one test per bug, asserting the precise state); `parse_agent_spec` grammar; `make_factory` returns an agent usable through `InProcessAdapter`; audio-only utterance uses the transcriber.
- [ ] **Step 6: Run → fail. Implement rule_based.py. Run → pass.**
- [ ] **Step 7: Lint, types. Commit** `feat(agents): rule-based reference agent with switchable bugs`

---

### Task 8: Starter pack and bug-isolation integration test

**Files:**
- Create: `packs/starter/pack.yaml`, `packs/starter/menu.yaml`, `packs/starter/scenarios/*.yaml` (40 files), `packs/starter/REVIEW.md`, `packs/starter/clips/.gitkeep`
- Test: `tests/test_starter_pack.py`

**Interfaces:**
- Consumes: Task 6 `load_pack`, Task 7 agent and lexicon, Task 5 `InProcessAdapter`, Task 6 `run_suite`.

Menu (12 items; add Hinglish aliases; prices in INR):

| id | name | groups |
|---|---|---|
| paneer_wrap | Paneer Wrap | onion, spice, extras |
| chicken_wrap | Chicken Wrap | onion, spice, extras |
| veg_biryani | Veg Biryani | spice |
| chicken_biryani | Chicken Biryani | spice |
| butter_chicken | Butter Chicken | spice |
| dal_makhani | Dal Makhani | — |
| butter_naan | Butter Naan | — |
| samosa | Samosa | — |
| masala_chai | Masala Chai | sugar |
| mango_lassi | Mango Lassi | sugar |
| cold_coffee | Cold Coffee | sugar |
| gulab_jamun | Gulab Jamun | — |

Groups: `onion` (exclusive, default `with_onion`, options `with_onion`, `no_onion`), `spice` (exclusive, default `medium`, options `mild`, `medium`, `spicy`), `extras` (non-exclusive, options `extra_cheese`, `extra_paneer`), `sugar` (exclusive, default `normal_sugar`, options `normal_sugar`, `less_sugar`, `no_sugar`). Every option alias must be a phrase the parser finds as a whole-word substring (for example `no_onion`: "no onion", "no onions", "without onion", "bina pyaaz", "pyaaz nahi", "onion nahi", "pyaaz mat dalna", "onion mat dalna"; `spicy`: "spicy", "extra spicy", "teekha", "zyada teekha"; `mild`: "mild", "less spicy", "kam teekha"; `less_sugar`: "less sugar", "kam cheeni", "kam meetha"; `no_sugar`: "no sugar", "without sugar", "bina cheeni", "cheeni nahi"). Do not give `with_onion`, `medium` or `normal_sugar` aliases that could appear in a correction phrase accidentally ("with onion" is fine).

Scenario naming: `{en|hien}_{category}_{01..04}`; tags: `smoke` on `_01` of each language and category (10 total), plus free-form tags. Every `_01` smoke scenario must exercise only its own category's behaviour. Each scenario uses only phrases the parser supports (Task 7 lexicon and menu aliases). Each has at least one `clarifications` rule for the obvious clarifying question (quantity or modifier) so other agents get a fair run. Hinglish scenarios are romanised, natural, code-mixed ("Do paneer wrap dena, bina pyaaz. Aur ek mango lassi.").

`pack.yaml` defaults (use these exact patterns):

```yaml
id: starter
name: IndicOrderBench starter pack
version: "0.1.0"
menu: menu.yaml
description: English and Hinglish ordering scenarios across five failure categories.
defaults:
  caller:
    en-IN:
      closing: {text: "That's all, thanks."}
      confirm: {text: "Yes, please place the order."}
      fallback: {text: "Yes, that's right."}
      nudge: {text: "Please place the order now."}
      confirm_patterns: ["shall i (place|confirm)", "place (the|your) order\\?", "confirm (the|your) order", "should i place"]
      goodbye_patterns: ["order (is )?placed", "already placed", "thank you for (your )?order", "goodbye"]
      question_patterns: ["which", "what ", "how many", "would you like", "do you want"]
      clarifications:
        - {id: anything_else_en, match: ["anything else"], reply: {text: "No, that's all."}, max_uses: 1}
    hi-en:
      closing: {text: "Bas itna hi, thanks."}
      confirm: {text: "Haan, order place kar do."}
      fallback: {text: "Haan, theek hai."}
      nudge: {text: "Ab order place kar do."}
      confirm_patterns: ["order (place )?kar d(oo|u)n", "confirm kar", "place kar d(oo|u)n"]
      goodbye_patterns: ["order place ho gaya", "pehle se place", "shukriya", "dhanyavaad"]
      question_patterns: ["kitne", "kitna", "konsa", "kaunsa", "kya ", "chahiye\\?"]
      clarifications:
        - {id: aur_kuch_hi, match: ["aur kuch"], reply: {text: "Nahi, bas itna hi."}, max_uses: 1}
```

Note: the `anything_else` clarification has `max_uses: 1` so the caller answers "Anything else?" once with a closing-like reply; because that reply is a clarification, `closing_spoken` stays false and the default closing follows on the next agent turn. That is acceptable: the reference agent submits on either phrase. Mark the reply with `is_closing: true` so the caller does not say "that's all" twice: `reply: {text: "No, that's all.", is_closing: true}`. The `ScriptedCaller._clarification` path must honour `is_closing` on replies — **add that to Task 4's implementation** (`if rule.reply.is_closing: self.closing_spoken = True`) and a test.

`tests/test_starter_pack.py`:
```python
from indicorderbench.adapters.inprocess import InProcessAdapter
from indicorderbench.agents.rule_based import BUGS, make_factory
from indicorderbench.packs import load_pack, validate_pack
from indicorderbench.runner.runner import RunConfig, run_suite_sync
from indicorderbench.schemas.results import Outcome

CATEGORY_FOR_BUG = {"quantity_default_one": "quantity", "drop_modifiers": "modifier", "ignore_corrections": "correction",
                    "ignore_cancellation": "cancellation", "double_submit": "duplicate_submission"}


def test_pack_is_valid_and_complete(starter_pack_dir):
    assert validate_pack(starter_pack_dir) == []
    pack = load_pack(starter_pack_dir)
    assert len(pack.scenarios) == 40
    for lang in ("en-IN", "hi-en"):
        for cat in ("quantity", "modifier", "correction", "cancellation", "duplicate_submission"):
            assert len(pack.filter(languages=[lang], categories=[cat])) == 4
    assert len(pack.filter(tags=["smoke"])) == 10


def test_correct_agent_passes_everything(starter_pack_dir):
    pack = load_pack(starter_pack_dir)
    suite = run_suite_sync(pack, pack.scenarios, lambda: InProcessAdapter(make_factory()), RunConfig(trials=1))
    failures = [(s.scenario_id, t.outcome, t.check.summary if t.check else t.error) for s in suite.scenarios for t in s.trials if t.outcome is not Outcome.PASS]
    assert failures == []


def test_each_bug_breaks_only_its_category(starter_pack_dir):
    pack = load_pack(starter_pack_dir)
    for bug in BUGS:
        suite = run_suite_sync(pack, pack.scenarios, lambda b=bug: InProcessAdapter(make_factory(frozenset({b}))), RunConfig(trials=1))
        by_id = {s.scenario_id: s for s in suite.scenarios}
        cat = CATEGORY_FOR_BUG[bug]
        for s in pack.filter(categories=[cat]):
            assert by_id[s.id].trials[0].outcome is Outcome.FAIL, (bug, s.id, by_id[s.id].trials[0].check)
        for s in pack.filter(tags=["smoke"]):
            if s.category.value != cat:
                assert by_id[s.id].trials[0].outcome is Outcome.PASS, (bug, s.id, by_id[s.id].trials[0].check)
```

- [ ] **Step 1: Write menu.yaml and pack.yaml.** Run `uv run python -c "from indicorderbench.packs import validate_pack; print(validate_pack('packs/starter'))"` → expect only "no scenarios found".
- [ ] **Step 2: Write the 10 smoke scenarios first** (one per language and category, `_01`). Run the three tests; fix parser/agent gaps found (coordinate: parser fixes belong in Task 7 files and are allowed here).
- [ ] **Step 3: Write the remaining 30 scenarios.** Run tests → green.
- [ ] **Step 4: Write `packs/starter/REVIEW.md`**: a checklist for native-speaker review (naturalness, code-mixing, number words, regional variants, whether a human would phrase the correction that way, suggested recording notes), with the instruction to flip `review_status` to `reviewed` and fill `reviewer_notes` per scenario.
- [ ] **Step 5: Commit** `feat(pack): starter pack with 40 English and Hinglish scenarios`

---

### Task 9: JSON, JUnit and compare

**Files:**
- Create: `src/indicorderbench/report/__init__.py`, `report/json_report.py`, `report/junit.py`, `report/compare.py`
- Test: `tests/test_report_json_junit.py`, `tests/test_compare.py`

**Interfaces:**
- Consumes: `SuiteResult`, `ScenarioResult`, `Outcome`.
- Produces:
  ```python
  # json_report.py
  def write_json(suite: SuiteResult, path: Path) -> None
  def read_json(path: Path) -> SuiteResult                       # raises ReportError on bad schema_version
  class ReportError(Exception)
  # junit.py
  def write_junit(suite: SuiteResult, path: Path) -> None        # testsuite name "indicorderbench:<pack id>"; one testcase per scenario; classname = language.category
  # compare.py
  @dataclass class ScenarioDelta: scenario_id: str; title: str; baseline_rate: float | None; current_rate: float | None; delta: float | None; status: Literal["regressed", "improved", "same", "new", "removed", "unknown"]
  @dataclass class Comparison: baseline_rate: float | None; current_rate: float | None; delta: float | None; max_regression: float; regressed: bool; scenarios: list[ScenarioDelta]; problems: list[str]
  def compare(baseline: SuiteResult, current: SuiteResult, max_regression: float = 0.0) -> Comparison   # raises ReportError if pack ids differ or schema_version differs
  def render_comparison_text(c: Comparison) -> str
  ```
  `regressed` is true when suite `delta < -max_regression` **or** any scenario went from rate 1.0 to below 1.0 by more than `max_regression`.

- [ ] **Step 1: Write failing tests** using a helper `make_suite(outcomes_by_scenario: dict[str, list[Outcome]], pack_id="starter")` built from the result schemas (put the helper in `tests/helpers.py` so Task 10 reuses it). Tests: JSON roundtrip equality; `read_json` rejects `schema_version: 99` with `ReportError`; JUnit has `tests="N"`, `failures`, `errors`, `skipped` counts and a `<failure>` whose message contains a failing field check; compare detects a regression, marks new/removed scenarios, refuses different pack ids, respects `max_regression`.
- [ ] **Step 2: Run → fail. Implement.** JUnit via `xml.etree.ElementTree`; include `<system-out>` with the transcript for failed cases.
- [ ] **Step 3: Run → pass. Lint, types. Commit** `feat(report): JSON, JUnit and baseline comparison`

---

### Task 10: HTML report

**Files:**
- Create: `src/indicorderbench/report/html.py`, `src/indicorderbench/report/templates/report.html.j2`
- Modify: `pyproject.toml` (hatch include for templates: `[tool.hatch.build.targets.wheel] packages = ["src/indicorderbench"]` already includes package data; verify the template ships by building a wheel and listing it)
- Test: `tests/test_report_html.py`

**Interfaces:**
- Consumes: `SuiteResult`, `Comparison` (Task 9), `Pack` optional for menu names.
- Produces: `def write_html(suite: SuiteResult, out_dir: Path, baseline: SuiteResult | None = None, comparison: Comparison | None = None, copy_audio: bool = True) -> Path` returning `out_dir / "report.html"`.

Required content (tests assert on these strings/structures):
- `<title>IndicOrderBench · {pack id} · {agent label}</title>`
- summary cards with ids `card-pass-rate`, `card-pass-k`, `card-invalid`, `card-infra`, `card-latency`
- tables `#by-language`, `#by-category` with Wilson interval columns
- `#comparison` section only when a comparison is given; rows with class `regressed`
- `#scenarios` list; each scenario a `<details id="scenario-{id}">` with outcome chips `<span class="chip chip-pass|chip-fail|chip-invalid|chip-infra">`
- per trial: `<table class="field-checks">` rows with class `pass`/`fail`; transcript `<ol class="transcript">` with latency; `<audio controls src="assets/{scenario}/{turn}.wav">` when a clip path exists and was copied; `<table class="trace">`
- inline CSS only; `prefers-color-scheme: dark` supported; no external URLs
- footer with benchmark version, pack hash, run timestamps

Template guidance: keep the Jinja template under 300 lines; Python side pre-computes view models (dicts) so the template holds no logic beyond loops and ifs. Escape everything (autoescape on). Format floats as percentages with one decimal.

- [ ] **Step 1: Write failing tests**: renders for a suite with a PASS, a FAIL and an INFRA_ERROR trial (from `tests/helpers.make_suite`); asserts the strings above; a fixture WAV (write 0.1 s of silence with the `wave` stdlib module in the test) is copied to `assets/` and referenced; renders with a comparison and shows a `regressed` row; renders when every trial is infra error (cards show "—").
- [ ] **Step 2: Run → fail. Implement html.py + template.**
- [ ] **Step 3: Run → pass. Lint, types. Build wheel (`uv build`) and confirm `unzip -l dist/*.whl | grep report.html.j2`. Commit** `feat(report): single-file HTML report`

---

### Task 11: Audio perturbations

**Files:**
- Create: `src/indicorderbench/audio/__init__.py`, `audio/perturb.py`, `audio/wav.py`
- Test: `tests/test_audio_perturb.py`

**Interfaces:**
- Produces:
  ```python
  # wav.py (no numpy dependency; stdlib wave)
  def read_wav(path: Path) -> tuple[list[float], int]        # mono float samples in [-1, 1], sample rate; stereo is averaged
  def write_wav(path: Path, samples: Sequence[float], sr: int) -> None  # 16-bit PCM mono
  # perturb.py (numpy; raise AudioExtraMissing with install hint if numpy is absent)
  class AudioExtraMissing(RuntimeError)
  def add_noise(samples: np.ndarray, sr: int, snr_db: float, kind: Literal["white", "pink"] = "white", seed: int | None = None) -> np.ndarray
  def gain(samples: np.ndarray, db: float) -> np.ndarray
  def telephone(samples: np.ndarray, sr: int) -> np.ndarray   # band-pass 300-3400 Hz by FFT mask, then resample to 8 kHz and back with linear interpolation
  def normalise_peak(samples: np.ndarray, peak: float = 0.95) -> np.ndarray
  @dataclass class PerturbSpec: snr_db: float | None = None; noise: Literal["white", "pink"] = "white"; gain_db: float = 0.0; telephone: bool = False; seed: int | None = None
  def apply(samples: np.ndarray, sr: int, spec: PerturbSpec) -> np.ndarray
  def perturb_dir(src: Path, dst: Path, spec: PerturbSpec) -> list[Path]   # mirrors the clips tree, copies manifest.json with "derived_from" and the spec
  ```

- [ ] **Step 1: Write failing tests**: wav roundtrip; `add_noise` achieves the requested SNR within 0.5 dB on a 1 kHz sine; `gain(-6)` halves RMS within 1%; `telephone` reduces energy at 100 Hz and 6 kHz by at least 20 dB relative to 1 kHz; `normalise_peak`; `perturb_dir` writes the same relative paths and a manifest with `derived_from`.
- [ ] **Step 2: Run → fail. Implement. Run → pass. Lint, types. Commit** `feat(audio): WAV helpers and perturbations`

---

### Task 12: TTS synth, transcribers and clip manifest

**Files:**
- Create: `src/indicorderbench/audio/tts.py`, `audio/transcribe.py`, `audio/manifest.py`
- Test: `tests/test_audio_tts.py`, `tests/test_audio_transcribe.py`

**Interfaces:**
- Consumes: `Pack`, `Transcriber` Protocol from `agents/rule_based.py`.
- Produces:
  ```python
  # manifest.py
  class ClipEntry(BaseModel): path: str; sha256: str; scenario_id: str; turn_id: str; text: str; language: str; provider: str; voice: str; model: str; created_at: datetime
  class ClipManifest(BaseModel): version: int = 1; clips: list[ClipEntry]; derived_from: str | None = None; perturbation: dict[str, Any] | None = None
      def by_hash(self) -> dict[str, ClipEntry]
  def load_manifest(path: Path) -> ClipManifest; def save_manifest(m: ClipManifest, path: Path) -> None
  def sha256_file(path: Path) -> str
  # tts.py
  class AudioProviderError(RuntimeError): status: int | None; detail: str
  class TTSProvider(Protocol): name: str; model: str; def synthesize(self, text: str, language: str, voice: str) -> bytes
  class SarvamTTS: def __init__(self, api_key: str, client: httpx.Client | None = None, model: str = "bulbul:v3", timeout: float = 25.0)
      # POST https://api.sarvam.ai/text-to-speech, header API-Subscription-Key, json {"inputs":[text], "target_language_code": "hi-IN" for hi-en else "en-IN", "speaker": voice, "model": model}; returns base64-decoded audios[0]
  def language_code(language: str) -> str     # "hi-en" -> "hi-IN", "en-IN" -> "en-IN"
  @dataclass class SynthReport: written: list[Path]; skipped: list[Path]; failed: list[tuple[str, str]]
  def synth_pack(pack: Pack, provider: TTSProvider, voice: str, only_missing: bool = True, scenario_ids: list[str] | None = None) -> SynthReport
      # writes clips/<scenario>/<turn>.wav for every scripted turn, clarification reply, and the language defaults (closing/confirm/fallback/nudge → clips/_defaults/<language>/<id>.wav); updates manifest.json
  # transcribe.py
  class OracleTranscriber: def __init__(self, manifest: ClipManifest); def transcribe(self, audio: bytes, language: str) -> str   # sha256 of bytes → text; KeyError → raises AudioProviderError("clip not in manifest")
  class SarvamTranscriber: def __init__(self, api_key: str, client: httpx.Client | None = None, model: str = "saaras:v3", timeout: float = 30.0); def transcribe(self, audio: bytes, language: str) -> str
      # POST https://api.sarvam.ai/speech-to-text multipart: file=("audio.wav", audio, "audio/wav"), model, language_code; returns json["transcript"]
  ```
  The Sarvam request shapes come from verified production code; do not change field names.

  **Pack.clip_path must also resolve default turns**: extend `Pack.clip_path` (Task 6 file) so a turn whose id is one of `closing|confirm|fallback|nudge` resolves to `clips/_defaults/<language>/<id>.wav`; it needs the language, so the signature becomes `clip_path(scenario_id: str, turn: CallerTurn, language: Language) -> Path | None`. Update the runner call site and Task 6 tests accordingly.

- [ ] **Step 1: Write failing tests** with `httpx.MockTransport` for both Sarvam classes (assert URL, header, JSON/multipart fields, base64 decode, error → `AudioProviderError` with status); manifest roundtrip; `synth_pack` on the minipack fixture with a fake provider that returns a valid WAV, `only_missing` skips, failures are collected not raised; `OracleTranscriber` lookup and miss.
- [ ] **Step 2: Run → fail. Implement. Run → pass. Lint, types. Commit** `feat(audio): Sarvam TTS/STT providers, oracle transcriber, clip manifest`

---

### Task 13: HTTP backend server, HTTP turn adapter, example shim

**Files:**
- Create: `src/indicorderbench/backend/http.py`, `src/indicorderbench/adapters/http.py`, `examples/http_agent_shim.py`
- Test: `tests/test_http.py`

**Interfaces:**
- Consumes: `SessionRegistry`, `OrderBackend`, `BackendError`, adapter protocol.
- Produces:
  ```python
  # backend/http.py
  class BackendServer:
      def __init__(self, registry: SessionRegistry, host: str = "127.0.0.1", port: int = 0) -> None
      def start(self) -> None; def stop(self) -> None; url: str   # "http://127.0.0.1:<port>"
      # routes: GET /healthz → {"ok": true}; GET /sessions/{id}/menu → menu JSON; GET /sessions/{id}/tools → tool_specs;
      #         POST /sessions/{id}/tools/{name} body JSON args → {"ok": true, "result": ...} | 400 {"ok": false, "error": {"code", "message"}}; unknown session → 404
  # adapters/http.py
  class HttpTurnAdapter:
      def __init__(self, url: str, timeout_s: float = 30.0, client: httpx.AsyncClient | None = None) -> None
      # start: POST url {"event":"start","session_id","scenario_id","backend_url","language","modality"}
      # respond: POST url {"event":"turn","session_id","turn_id","text","audio_b64","audio_format":"wav","language"} → {"text","audio_b64"?, "meta"?}
      # stop: POST url {"event":"stop","session_id"}; non-2xx or malformed reply → raise AdapterError (subclass RuntimeError)
  ```
  `examples/http_agent_shim.py`: a stdlib `http.server` script that implements the turn protocol by wrapping `RuleBasedAgent` and calling the sandbox backend **over HTTP** through a tiny `RemoteBackend` class (same method names as `OrderBackend`, each doing a POST). It demonstrates exactly what a user's agent must do. Run with `python examples/http_agent_shim.py --port 8900`.

- [ ] **Step 1: Write failing tests**: start `BackendServer` on port 0, register a backend, call tools through `httpx`, verify trace on the in-process object, error shape, 404; `HttpTurnAdapter` against a local stdlib handler in a thread that echoes and performs one backend call via the server; a full `run_trial` through `HttpTurnAdapter` + the shim's handler class for the minipack quantity scenario → PASS.
- [ ] **Step 2: Run → fail. Implement. Run → pass. Lint, types. Commit** `feat(http): sandbox backend server, HTTP turn adapter and example shim`

---

## Phase C — integration

### Task 14: CLI

**Files:**
- Create: `src/indicorderbench/cli.py`, `src/indicorderbench/agent_specs.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything.
- Produces: Typer app `app` with commands `validate`, `ls`, `run`, `report`, `compare`, `demo`, `synth`, `perturb`, `serve-backend`, and `agent_specs.build_adapter_factory(spec: str, registry: SessionRegistry | None) -> tuple[AdapterFactory, str label, bool needs_backend_server]`.

Agent spec grammar: `builtin:correct` | `builtin:buggy` | `builtin:buggy:<bug,...>` | `http:<url>` | `python:<module>:<factory>` (factory is an `AgentFactory`). Unknown bug name → usage error listing `BUGS`.

`run` behaviour: load pack (problems → print and exit 1); filter scenarios (none → exit 1 with message); if adapter needs a backend server, start `BackendServer` and pass `backend_url`; print a one-line progress per scenario (`✔ en_quantity_01 pass`, `✘ ... fail`, `! ... infra`, `? ... invalid`); write `results.json`, `junit.xml`, `report.html` to `--out` (default `results/<timestamp>`); if `--baseline`, read it, compare, include in HTML; print the summary table (pass rate, pass^k, invalid, infra, p50/p95) and the comparison text; exit 2 if `pass_rate < --fail-under` (when given) or comparison regressed; exit 1 if every trial was infra error.

`demo`: run `builtin:buggy` on tag `smoke` into `<out>/buggy/`, then `builtin:correct` with the buggy run as baseline into `<out>/fixed/`, and print the demo table from the correction smoke scenario (`field_checks` of `hien_correction_01` buggy trial) as text. Also write `<out>/README.md` explaining the two reports.

- [ ] **Step 1: Write failing CLI tests** with `typer.testing.CliRunner`: `validate` ok and failing pack (exit 1 and problems printed); `ls --tag smoke` lists 10; `run --agent builtin:correct --tag smoke --out tmp` exit 0 and writes the three files; `run --agent builtin:buggy --tag smoke --fail-under 0.9` exit 2; `run` with `--baseline` from a correct run then buggy current → exit 2 and comparison in output; `compare` with mismatched pack id → exit 1; `demo --out tmp` exit 0 and prints "Paneer Wrap quantity"; `--agent builtin:buggy:nope` → exit 2 (Typer usage) with `BUGS` listed; `python:` spec loading a factory from `tests/helpers.py`.
- [ ] **Step 2: Run → fail. Implement. Run → pass. Lint, types. Commit** `feat(cli): iob command line`

---

### Task 15: Docs, examples, demo artefacts, release polish

**Files:**
- Create: `README.md`, `LICENSE` (Apache-2.0 full text from https://www.apache.org/licenses/LICENSE-2.0.txt), `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/methodology.md`, `docs/adapters.md`, `docs/scenarios.md`, `examples/ci/github-actions.yml`, `docs/demo/` (generated by `iob demo --out docs/demo`, committed: `buggy/report.html`, `fixed/report.html`, both `results.json`, no clips)
- Test: `tests/test_docs.py` (README contains the install command, the demo table header and a link to `docs/demo/fixed/report.html`; every `docs/*.md` link to a local file resolves)

README first screen, in this order: one-line promise ("Your voice agent sounds convincing. Did it actually place the correct order?"), the demo table (reproduced from the actual buggy run output), `pip install indicorderbench` / `uvx indicorderbench demo`, a quickstart (`iob demo`, `iob run packs/starter --agent builtin:correct --tag smoke`), "Connect your agent" (three options with 10-line examples: Python factory, HTTP shim, link to adapters doc), what is measured and what is not (turn-based, unreviewed Hinglish, no LLM pieces yet), prior art (τ-bench, EVA-Bench, VoiceAgentBench, VoiceTest with links), roadmap (Telugu, LiveKit adapter, generative caller, native-speaker review), contributing, license.

`docs/methodology.md`: outcomes and priority, validity rules, pass rate and pass^k with the formula, Wilson intervals, latency definition, what counts as committed, reproducibility fields in results.json, limitations, how to report results responsibly (always publish invalid and infra counts, trials, config, pack hash).

`docs/adapters.md`: protocol tables from the spec §4.8, the shim walkthrough, LiveKit guidance (how a LiveKit agent would expose a turn endpoint; mark adapter as planned).

`docs/scenarios.md`: YAML schema with every field, the lexicon rules a scenario must stay inside when it targets the reference agent, the bug-isolation contract, the review checklist link.

`CONTRIBUTING.md`: small tasks (language review, scenarios, adapters, report improvements), dev setup (`uv sync --all-extras --dev`, `uv run pytest`), PR checklist, consent note for contributed recordings ("removing names does not grant permission; include written consent").

- [ ] **Step 1: Write docs and tests; generate `docs/demo` with `uv run iob demo --out docs/demo`; verify `tests/test_docs.py` passes.**
- [ ] **Step 2: Full verification**: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q && uv run iob validate packs/starter && uv run iob run packs/starter --agent builtin:correct --out /tmp/iob-full --fail-under 1.0 && uv build`.
- [ ] **Step 3: Commit** `docs: README, methodology, adapters, scenarios, contributing, demo reports`

---

## Self-review notes

- Spec coverage: §4.1–4.2 Tasks 1–2; §4.3–4.4 Tasks 1, 4, 6; §4.5–4.7 Tasks 3, 6; §4.8 Tasks 5, 13; §4.9 Task 7; §4.10 Task 8; §4.11 Tasks 9–10; §4.12 Task 14; §4.13 Tasks 11–12; §5 error handling spread across Tasks 6, 12, 13, 14; §6 testing per task; §7 layout Tasks 0 and 15; §8 limitations Task 15.
- Type consistency checked: `CallerSource` lives in `schemas/results.py` and is imported by `caller/scripted.py`; `Transcriber` Protocol lives in `agents/rule_based.py`; `Pack.clip_path` signature changes in Task 12 and that task owns updating Task 6 code and tests; `ScriptedCaller._clarification` honours `is_closing` (added by Task 8 note; apply it during Task 4 if executing in order).
- Review Focus items are pinned to Tasks 4, 3, 8 (`validate_pack` test in Task 6 covers unknown ids), 6, 9.
