"""Scripted caller: a state machine over a resolved caller script.

The caller speaks its scripted turns in order, answers clarifying questions it has rules
for, then closes the order. After closing it either confirms, falls back on an unanswerable
question (and eventually declares itself invalid), or nudges a silent agent to submit (and
eventually stops while remaining valid).
"""

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
            used = self._rule_uses.get(rule.id, 0)
            if used < rule.max_uses and rule.matches(agent_text):
                self._rule_uses[rule.id] = used + 1
                if rule.reply.is_closing:
                    self.closing_spoken = True
                return CallerMove(rule.reply, "clarification")
        return None

    def next_move(self, agent_text: str) -> CallerMove | None:
        """Return the caller's next utterance, or None when the caller has stopped."""
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
            snippet = text.strip()[:120]
            self.invalid_reason = f"agent asked a question the script could not answer: {snippet!r}"
            return None
        if self.nudges_used < self.script.max_nudges:
            self.nudges_used += 1
            return CallerMove(self.script.nudge, "nudge")
        return None

    def should_end(self, agent_text: str, has_active_order: bool) -> bool:
        """True once the script is done, closing was said, and the order is placed or
        the agent said goodbye."""
        if not (self.script_exhausted and self.closing_spoken):
            return False
        return has_active_order or self.script.is_goodbye(agent_text or "")
