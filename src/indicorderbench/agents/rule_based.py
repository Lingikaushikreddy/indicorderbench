"""Rule-based reference agent with switchable, realistic bugs.

The agent parses each caller utterance with :mod:`indicorderbench.agents.parsing`, acts on
the sandbox backend and replies with a short readback in the caller's language. ``bugs``
toggles failures that leave the replies unchanged so that only the committed state reveals
them; each bug breaks exactly one scenario category.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from indicorderbench.adapters.protocol import (
    AgentFactory,
    AgentReply,
    CallerUtterance,
    SessionInfo,
)
from indicorderbench.agents.parsing import TERMINAL, Clause, Intent, parse_utterance
from indicorderbench.backend.state import BackendError, OrderBackend
from indicorderbench.schemas.results import CartLine

BUGS: tuple[str, ...] = (
    "ignore_corrections",
    "drop_modifiers",
    "double_submit",
    "ignore_cancellation",
    "quantity_default_one",
)


@runtime_checkable
class Transcriber(Protocol):
    """Turns caller audio into text; the audio extra provides a real implementation."""

    def transcribe(self, audio: bytes, language: str) -> str: ...


_MESSAGES: dict[str, dict[str, str]] = {
    "en-IN": {
        "added": "Added {qty} {name}{mods}.",
        "swapped": "Changed that to {qty} {name}{mods}.",
        "qty": "Changed {name} to {qty}.",
        "mods": "Updated {name}: {mods}.",
        "removed": "Removed {name}.",
        "cancelled": "Your order has been cancelled.",
        "tail": "Anything else?",
        "placed": "Your order: {lines}. Order placed, thank you!",
        "already": "Your order is already placed. Thank you!",
        "cancelled_bye": "Your order has been cancelled. Thank you, goodbye!",
        "nothing": "Nothing in your cart yet. What would you like?",
        "so_far": "So far: {lines}.",
        "status": "Your order: {lines}. It has been placed.",
        "unknown": "Sorry, I didn't catch that. Which item would you like?",
        "error": "Sorry, I couldn't do that: {message}",
        "not_in_cart": "{name} is not in your cart.",
        "nothing_to_remove": "there is nothing to remove.",
        "removed_mods": "Removed {mods} from {name}.",
        "not_applicable": "{mods} does not apply to {name}.",
        "already_without": "{name} has no {mods}.",
        "not_in_order": "There is no {name} in your order.",
    },
    "hi-en": {
        "added": "{qty} {name}{mods} add kar diya.",
        "swapped": "Usko {qty} {name}{mods} kar diya.",
        "qty": "{name} {qty} kar diya.",
        "mods": "{name} {mods} kar diya.",
        "removed": "{name} hata diya.",
        "cancelled": "Aapka order cancel kar diya.",
        "tail": "Aur kuch?",
        "placed": "Aapka order: {lines}. Order place ho gaya, shukriya!",
        "already": "Aapka order pehle se place ho gaya hai. Shukriya!",
        "cancelled_bye": "Aapka order cancel ho gaya hai. Shukriya, namaste!",
        "nothing": "Cart khaali hai. Kya chahiye?",
        "so_far": "Abhi tak: {lines}.",
        "status": "Aapka order: {lines}. Place ho chuka hai.",
        "unknown": "Maaf kijiye, samajh nahi aaya. Konsa item chahiye?",
        "error": "Maaf kijiye, yeh nahi ho paya: {message}",
        "not_in_cart": "{name} cart mein nahi hai.",
        "nothing_to_remove": "hataane ke liye kuch nahi hai.",
        "removed_mods": "{name} se {mods} hata diya.",
        "not_applicable": "{mods} {name} pe lagu nahi hota.",
        "already_without": "{name} mein {mods} pehle se nahi hai.",
        "not_in_order": "Aapke order mein {name} nahi hai.",
    },
}


def _check_bugs(bugs: frozenset[str]) -> frozenset[str]:
    unknown = sorted(set(bugs) - set(BUGS))
    if unknown:
        raise ValueError(f"unknown bug(s): {', '.join(unknown)}; known bugs: {', '.join(BUGS)}")
    return frozenset(bugs)


@dataclass
class _Line:
    """A cart line as the agent believes it to be. Replies are built from these so a bug
    changes only the backend state, never the wording."""

    backend_id: str | None
    item_id: str
    quantity: int
    modifiers: list[str]


# How a clause reply combines with the others in the same utterance.
ACTION = "action"  # followed by "Anything else?" unless the utterance also closes
TERMINAL_REPLY = "terminal"  # closes the exchange; no tail
PLAIN = "plain"  # a question or apology that stands on its own


class RuleBasedAgent:
    def __init__(
        self,
        backend: OrderBackend,
        language: str,
        bugs: frozenset[str] = frozenset(),
        transcriber: Transcriber | None = None,
    ) -> None:
        self.backend = backend
        self.language = language if language in _MESSAGES else "en-IN"
        self.bugs = _check_bugs(bugs)
        self.transcriber = transcriber
        self._cart: list[_Line] = []
        self._placed: list[_Line] | None = None
        self._cancelled = False

    @property
    def last_line_id(self) -> str | None:
        """Backend id of the most recent cart line, which corrections apply to."""
        return self._cart[-1].backend_id if self._cart else None

    # -- entry point ---------------------------------------------------------------
    def handle(self, utterance: CallerUtterance) -> AgentReply:
        text = self._resolve_text(utterance)
        if text is None:
            return AgentReply(text=self._m("unknown"))
        clauses = parse_utterance(text, self.backend.menu, self.language)
        if not clauses:
            if not self._cart and self._placed:
                return AgentReply(text=self._m("already"))
            return AgentReply(text=self._m("unknown"))
        parts: list[str] = []
        acted = closed = False
        for clause in clauses:
            if clause.intent in TERMINAL and closed:
                continue
            try:
                part, kind = self._act(clause)
            except BackendError as e:
                part, kind = self._m("error", message=e.message), PLAIN
            parts.append(part)
            acted = acted or kind == ACTION
            closed = closed or kind == TERMINAL_REPLY
        if acted and not closed:
            parts.append(self._m("tail"))
        return AgentReply(text=" ".join(parts))

    def _resolve_text(self, utterance: CallerUtterance) -> str | None:
        if utterance.text is not None:
            return utterance.text
        audio = utterance.audio_bytes
        if audio is None and utterance.audio_path is not None:
            try:
                audio = Path(utterance.audio_path).read_bytes()
            except OSError:
                return None
        if audio is None or self.transcriber is None:
            return None
        return self.transcriber.transcribe(audio, self.language)

    # -- clause actions ----------------------------------------------------------------
    def _act(self, c: Clause) -> tuple[str, str]:
        if c.intent is Intent.ADD and c.item_id is not None:
            return self._add(c.item_id, c.quantity, c.modifiers)
        if c.intent is Intent.CORRECT:
            return self._correct(c)
        if c.intent is Intent.REMOVE:
            return self._remove(c)
        if c.intent is Intent.CANCEL_ORDER:
            return self._cancel()
        if c.intent in TERMINAL:
            return self._close()
        if c.intent is Intent.READBACK:
            return self._readback()
        return self._m("unknown"), PLAIN

    def _add(
        self, item_id: str, quantity: int | None, mods: list[str], do_backend: bool = True
    ) -> tuple[str, str]:
        qty_said = quantity or 1
        qty_used = 1 if "quantity_default_one" in self.bugs else qty_said
        backend_id = self._add_item(item_id, qty_used, mods).line_id if do_backend else None
        self._cart.append(_Line(backend_id, item_id, qty_said, list(mods)))
        msg = self._m("added", qty=qty_said, name=self._name(item_id), mods=self._fmt_mods(mods))
        return msg, ACTION

    def _correct(self, c: Clause) -> tuple[str, str]:
        skip = "ignore_corrections" in self.bugs
        if (
            c.targets_named_item
            and c.item_id is not None
            and not any(ln.item_id == c.item_id for ln in self._cart)
        ):
            # "no onion in the paneer wrap" with no wrap ordered: say so, change nothing
            return self._m("not_in_order", name=self._name(c.item_id)), ACTION
        if not self._cart:
            if c.item_id is not None:
                return self._add(c.item_id, c.quantity, c.modifiers, do_backend=not skip)
            return self._m("unknown"), PLAIN
        line = self._cart[-1]
        named = [ln for ln in self._cart if ln.item_id == c.item_id] if c.item_id else []
        if named:
            line = named[-1]  # "change the chai to three" targets the chai line
        elif c.item_id is None:
            groups = self._groups_of(c.modifiers + c.negated)
            if groups:
                # "teekha nahi chahiye" edits the most recent line that has a spice level
                with_group = [ln for ln in self._cart if groups & self._line_groups(ln)]
                if with_group:
                    line = with_group[-1]
                elif c.quantity is None:
                    names = self._mod_names(c.modifiers + c.negated)
                    msg = self._m("not_applicable", mods=names, name=self._name(line.item_id))
                    return msg, ACTION
                # else: the quantity still applies to the last line; the modifier is reported
                # as not applicable below
        else:
            qty = c.quantity or line.quantity
            kept, dropped = self._applicable(c.item_id, c.modifiers)
            mods = self._merge(c.item_id, line.modifiers, kept, c.negated)
            if not skip and line.backend_id is not None:
                # add first, then remove, so a refused add leaves the old line in place
                new_id = self._add_item(c.item_id, qty, mods).line_id
                self.backend.remove_line(line.backend_id)
                line.backend_id = new_id
            line.item_id, line.quantity, line.modifiers = c.item_id, qty, mods
            name = self._name(c.item_id)
            msg = self._m("swapped", qty=qty, name=name, mods=self._fmt_mods(kept))
            return self._with_dropped(msg, name, dropped), ACTION
        name = self._name(line.item_id)
        kept, dropped = self._applicable(line.item_id, c.modifiers)
        neg_ok, neg_dropped = self._applicable(line.item_id, c.negated)
        neg_present = [m for m in neg_ok if m in line.modifiers]
        dropped += neg_dropped
        if c.quantity is None and not kept and not neg_present:
            # nothing changes on this line: say why and make no backend call
            if dropped:
                return self._m("not_applicable", mods=self._mod_names(dropped), name=name), ACTION
            if neg_ok:
                return self._m("already_without", mods=self._mod_names(neg_ok), name=name), ACTION
            return self._m("swapped", qty=line.quantity, name=name, mods=""), ACTION
        merged = (
            self._merge(line.item_id, line.modifiers, kept, neg_present)
            if kept or neg_present
            else None
        )
        if not skip and line.backend_id is not None:
            self._update_line(line.backend_id, c.quantity, merged)
        if c.quantity is not None:
            line.quantity = c.quantity
        if merged is not None:
            line.modifiers = merged
        if c.quantity is not None and kept:
            msg = self._m("swapped", qty=c.quantity, name=name, mods=self._fmt_mods(kept))
        elif c.quantity is not None:
            msg = self._m("qty", name=name, qty=c.quantity)
        elif kept:
            msg = self._m("mods", name=name, mods=self._mod_names(kept))
        else:
            msg = self._m("removed_mods", name=name, mods=self._mod_names(neg_present))
        return self._with_dropped(msg, name, dropped), ACTION

    def _remove(self, c: Clause) -> tuple[str, str]:
        if c.item_id is not None:
            matches = [ln for ln in self._cart if ln.item_id == c.item_id]
            if not matches:
                detail = self._m("not_in_cart", name=self._name(c.item_id))
                return self._m("error", message=detail), PLAIN
            line = matches[-1]
        elif self._cart:
            line = self._cart[-1]
        else:
            return self._m("error", message=self._m("nothing_to_remove")), PLAIN
        backend_id = None if "ignore_cancellation" in self.bugs else line.backend_id
        if c.quantity is not None and 0 < c.quantity < line.quantity:
            line.quantity -= c.quantity
            if backend_id is not None:
                self._update_line(backend_id, line.quantity, None)
        else:
            self._cart.remove(line)
            if backend_id is not None:
                self.backend.remove_line(backend_id)
        return self._m("removed", name=self._name(line.item_id)), ACTION

    def _cancel(self) -> tuple[str, str]:
        if "ignore_cancellation" not in self.bugs:
            for order in self.backend.active_orders():
                self.backend.cancel_order(order.order_id)
            if self.backend.snapshot().cart:
                self.backend.clear_cart()
        self._cart, self._placed, self._cancelled = [], None, True
        return self._m("cancelled"), ACTION

    def _close(self) -> tuple[str, str]:
        if self._cart:
            self.backend.submit_order()
            self._placed, self._cart = self._cart, []
            return self._m("placed", lines=self._fmt_lines(self._placed)), TERMINAL_REPLY
        if "ignore_cancellation" in self.bugs and self.backend.snapshot().cart:
            self.backend.submit_order()  # the cart the agent "cancelled" is still there
        if self._placed:
            if "double_submit" in self.bugs:
                for order in self.backend.active_orders()[-1:]:
                    for ln in order.lines:
                        self._add_item(ln.item_id, ln.quantity, sorted(ln.modifiers))
                    self.backend.submit_order()
            return self._m("already"), TERMINAL_REPLY
        if self._cancelled:
            return self._m("cancelled_bye"), TERMINAL_REPLY
        return self._m("nothing"), TERMINAL_REPLY

    def _readback(self) -> tuple[str, str]:
        if self._cart:
            return self._m("so_far", lines=self._fmt_lines(self._cart)), ACTION
        if self._placed:
            return self._m("status", lines=self._fmt_lines(self._placed)), PLAIN
        if self._cancelled:
            return self._m("cancelled_bye"), PLAIN
        return self._m("nothing"), PLAIN

    # -- backend calls with bug effects ----------------------------------------------------
    def _add_item(self, item_id: str, quantity: int, mods: list[str]) -> CartLine:
        return self.backend.add_item(item_id, quantity, self._mods_arg(mods) or [])

    def _update_line(self, line_id: str, quantity: int | None, mods: list[str] | None) -> None:
        self.backend.update_line(line_id, quantity=quantity, modifiers=self._mods_arg(mods))

    def _mods_arg(self, mods: list[str] | None) -> list[str] | None:
        return [] if "drop_modifiers" in self.bugs else mods

    # -- helpers ---------------------------------------------------------------------------
    def _m(self, key: str, **kw: object) -> str:
        return _MESSAGES[self.language][key].format(**kw)

    def _name(self, item_id: str) -> str:
        return self.backend.menu.item(item_id).name

    def _mod_names(self, mods: Iterable[str]) -> str:
        menu = self.backend.menu
        return ", ".join(menu.option(m).name if menu.has_option(m) else m for m in mods)

    def _fmt_mods(self, mods: Iterable[str]) -> str:
        names = self._mod_names(mods)
        return f", {names}" if names else ""

    def _fmt_lines(self, lines: Iterable[_Line]) -> str:
        return "; ".join(
            f"{ln.quantity} {self._name(ln.item_id)}{self._fmt_mods(sorted(ln.modifiers))}"
            for ln in lines
        )

    def _applicable(self, item_id: str, mods: list[str]) -> tuple[list[str], list[str]]:
        """Split options into those that apply to ``item_id`` and those that do not."""
        menu = self.backend.menu
        groups = {g.id for g in menu.groups_for(item_id)}
        kept = [m for m in mods if menu.has_option(m) and menu.option_group(m).id in groups]
        return kept, [m for m in mods if m not in kept]

    def _merge(
        self, item_id: str, current: Iterable[str], new: list[str], negated: Iterable[str] = ()
    ) -> list[str]:
        """Current modifiers that still apply to ``item_id``, minus ``negated``, plus ``new``;
        a new option replaces the current option of the same exclusive group."""
        menu = self.backend.menu
        out, _ = self._applicable(item_id, [m for m in current if m not in set(negated)])
        for m in new:
            group = menu.option_group(m)
            if group.exclusive:
                out = [x for x in out if menu.option_group(x).id != group.id]
            if m not in out:
                out.append(m)
        return out

    def _groups_of(self, options: Iterable[str]) -> set[str]:
        menu = self.backend.menu
        return {menu.option_group(o).id for o in options if menu.has_option(o)}

    def _line_groups(self, line: _Line) -> set[str]:
        return {g.id for g in self.backend.menu.groups_for(line.item_id)}

    def _with_dropped(self, msg: str, name: str, dropped: list[str]) -> str:
        if not dropped:
            return msg
        return f"{msg} {self._m('not_applicable', mods=self._mod_names(dropped), name=name)}"


def make_factory(
    bugs: frozenset[str] = frozenset(), transcriber: Transcriber | None = None
) -> AgentFactory:
    """An :class:`AgentFactory` for ``InProcessAdapter`` building a fresh agent per session."""
    bugs = _check_bugs(bugs)

    def factory(backend: OrderBackend, session: SessionInfo) -> RuleBasedAgent:
        return RuleBasedAgent(backend, session.language, bugs, transcriber)

    return factory


def parse_agent_spec(spec: str) -> frozenset[str] | None:
    """``builtin:correct`` -> no bugs; ``builtin:buggy`` -> all; ``builtin:buggy:a,b`` -> those.

    A spec that is not ``builtin:`` at all returns None (it belongs to another adapter). A
    malformed builtin spec or an unknown bug name raises ValueError.
    """
    s = spec.strip()
    if not s.startswith("builtin:"):
        return None
    if s == "builtin:correct":
        return frozenset()
    if s == "builtin:buggy":
        return frozenset(BUGS)
    prefix = "builtin:buggy:"
    if s.startswith(prefix):
        names = [n.strip() for n in s[len(prefix) :].split(",") if n.strip()]
        if not names:
            raise ValueError(f"{spec!r} names no bugs; known bugs: {', '.join(BUGS)}")
        return _check_bugs(frozenset(names))
    raise ValueError(
        f"unknown builtin agent {spec!r}; use builtin:correct, builtin:buggy or builtin:buggy:a,b"
    )
