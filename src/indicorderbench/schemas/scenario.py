"""Scenario, caller script and pack manifest schemas."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from indicorderbench.schemas.menu import ID_PATTERN


class Language(StrEnum):
    EN_IN = "en-IN"
    HI_EN = "hi-en"


class FailureCategory(StrEnum):
    QUANTITY = "quantity"
    MODIFIER = "modifier"
    CORRECTION = "correction"
    CANCELLATION = "cancellation"
    DUPLICATE_SUBMISSION = "duplicate_submission"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"


class CallerTurn(BaseModel):
    id: str | None = None
    text: str = Field(min_length=1)
    audio: str | None = None  # path relative to the pack root
    is_closing: bool = False


def _compile_all(patterns: list[str]) -> None:
    for p in patterns:
        try:
            re.compile(p)
        except re.error as e:
            raise ValueError(f"invalid regex {p!r}: {e}") from e


def _any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


class ClarificationRule(BaseModel):
    id: str = Field(pattern=ID_PATTERN)
    match: list[str] = Field(min_length=1)
    reply: CallerTurn
    max_uses: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _check(self) -> ClarificationRule:
        if self.reply.id is None:
            self.reply.id = f"c_{self.id}"
        _compile_all(self.match)
        return self

    def matches(self, text: str) -> bool:
        return _any_match(self.match, text)


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
    def _check(self) -> CallerDefaults:
        for name in ("closing", "confirm", "fallback", "nudge"):
            turn: CallerTurn = getattr(self, name)
            if turn.id is None:
                turn.id = name
        _compile_all(self.confirm_patterns)
        _compile_all(self.goodbye_patterns)
        _compile_all(self.question_patterns)
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
    def _check(self) -> CallerScript:
        seen: set[str] = set()
        for i, t in enumerate(self.turns):
            if t.id is None:
                t.id = f"t{i + 1}"
            if t.id in seen:
                raise ValueError(f"duplicate turn id {t.id}")
            seen.add(t.id)
        for patterns in (self.confirm_patterns, self.goodbye_patterns, self.question_patterns):
            if patterns is not None:
                _compile_all(patterns)
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
            question_patterns=(
                self.question_patterns
                if self.question_patterns is not None
                else defaults.question_patterns
            ),
            max_fallbacks=(
                self.max_fallbacks if self.max_fallbacks is not None else defaults.max_fallbacks
            ),
            max_nudges=self.max_nudges if self.max_nudges is not None else defaults.max_nudges,
        )


class ExpectedLine(BaseModel):
    item_id: str = Field(pattern=ID_PATTERN)
    quantity: int = Field(ge=1)
    # group_id -> option id | [option ids] | "*"
    modifiers: dict[str, str | list[str]] = Field(default_factory=dict)


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
    def _check(self) -> Scenario:
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
