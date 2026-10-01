"""Deterministic utterance parser for the rule-based reference agent.

An utterance is split into clauses on punctuation and connector words; each clause is
classified by the lexicon phrases and the menu aliases it contains. Everything here is a
plain function of the text, the menu and the language, so a scenario parses the same way
on every run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from indicorderbench.agents import lexicon
from indicorderbench.schemas.menu import Menu, normalise_text


class Intent(StrEnum):
    ADD = "add"
    CORRECT = "correct"
    REMOVE = "remove"
    CANCEL_ORDER = "cancel_order"
    CLOSING = "closing"
    CONFIRM = "confirm"
    READBACK = "readback"
    UNKNOWN = "unknown"


@dataclass
class Clause:
    intent: Intent
    item_id: str | None = None
    quantity: int | None = None
    modifiers: list[str] = field(default_factory=list)
    raw: str = ""
    is_correction: bool = False
    negated: list[str] = field(default_factory=list)  # options the caller does not want


TERMINAL = frozenset({Intent.CLOSING, Intent.CONFIRM})
ACTIONABLE = frozenset({Intent.ADD, Intent.CORRECT, Intent.REMOVE, Intent.CANCEL_ORDER})

_PUNCT_SPLIT_RE = re.compile(r"[,.;!?]+")


def _phrase_re(phrases: list[str]) -> re.Pattern[str]:
    alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})\b")


def _build_connector_re() -> re.Pattern[str]:
    """Split on connectors, except where a connector starts a lexicon phrase ("aur kuch
    nahi"). The "ek aur X" construction (one more X) is re-joined by :func:`split_clauses`."""
    phrases = (
        lexicon.CLOSING_MARKERS
        + lexicon.CANCEL_ORDER_MARKERS
        + lexicon.CONFIRM_MARKERS
        + lexicon.READBACK_MARKERS
    )
    protected = sorted(
        {p for p in phrases if any(p.startswith(c + " ") for c in lexicon.CONNECTORS)}
    )
    alts = []
    for c in sorted(lexicon.CONNECTORS, key=len, reverse=True):
        tails = [p[len(c) + 1 :] for p in protected if p.startswith(c + " ")]
        alt = re.escape(c)
        if tails:
            alt += rf"(?! (?:{'|'.join(re.escape(t) for t in tails)})\b)"
        alts.append(alt)
    consuming = rf"\b(?:{'|'.join(alts)})\b"
    if not protected:
        return re.compile(consuming)
    lookahead = "|".join(re.escape(p) for p in protected)
    return re.compile(rf"(?=\b(?:{lookahead})\b)|{consuming}")


_CONNECTOR_RE = _build_connector_re()
_CORRECTION_RE = _phrase_re(lexicon.CORRECTION_MARKERS)
_REMOVE_RE = _phrase_re(lexicon.REMOVE_MARKERS)
_CANCEL_ORDER_RE = _phrase_re(lexicon.CANCEL_ORDER_MARKERS)
_CLOSING_RE = _phrase_re(lexicon.CLOSING_MARKERS)
_READBACK_RE = _phrase_re(lexicon.READBACK_MARKERS)
_CONFIRM_RE = _phrase_re(lexicon.CONFIRM_MARKERS)
_CONFIRM_STRIP_RE = _phrase_re(lexicon.CONFIRM_MARKERS + lexicon.CONFIRM_FILLERS)
_BARE_STRIP_RE = _phrase_re(
    lexicon.CORRECTION_MARKERS + lexicon.IGNORED_CLAUSES + lexicon.CONFIRM_FILLERS
)
_CORRECTION_FILLER_RE = _phrase_re(
    lexicon.CORRECTION_FILLERS + lexicon.CORRECTION_MARKERS + lexicon.CONFIRM_FILLERS
)
_QUANTITY_PHRASE_RES = [
    (re.compile(rf"\b{re.escape(p)}\b"), n) for p, n in lexicon.QUANTITY_PHRASES.items()
]
_DO_VERB_STEMS = frozenset(lexicon.DO_VERB_STEMS)
_REHNE_RE = re.compile(r"\brehne do\b")
_IN_RE = _phrase_re(lexicon.IN_MARKERS)
_ARTICLES = frozenset({"a", "an"})


def split_clauses(text: str) -> list[str]:
    """Split raw text into normalised clause fragments."""
    out: list[str] = []
    for piece in _PUNCT_SPLIT_RE.split(text):
        norm = normalise_text(piece)
        if not norm:
            continue
        frags = [f for f in (x.strip() for x in _CONNECTOR_RE.split(norm)) if f]
        i = 0
        while i < len(frags):
            if frags[i] == "ek" and i + 1 < len(frags):
                out.append(f"ek {frags[i + 1]}")  # "ek aur lassi" = one more lassi
                i += 2
            else:
                out.append(frags[i])
                i += 1
    return out


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _scan_aliases(fragment: str, menu: Menu) -> tuple[list[tuple[int, int, str]], list[str], str]:
    """Find item and option aliases (longest first, whole words, non-overlapping).

    Returns the item spans, the option ids and the fragment with every alias blanked out.
    """
    masked = fragment
    items: list[tuple[int, int, str]] = []
    options: list[str] = []
    for entry in menu.alias_index():
        for m in re.finditer(rf"\b{re.escape(entry.alias)}\b", masked):
            if entry.kind == "item":
                items.append((m.start(), m.end(), entry.ref_id))
            elif entry.ref_id not in options:
                options.append(entry.ref_id)
            masked = _blank(masked, m.start(), m.end())
    items.sort()
    return items, options, masked


def _pick_item(fragment: str, items: list[tuple[int, int, str]]) -> str | None:
    """The clause item: leftmost, except "X ki jagah Y" and "Y instead of X" pick the
    replacement."""
    if not items:
        return None
    if len(items) > 1:
        m = re.search(r"\bki jagah\b", fragment)
        if m:
            after = [i for i in items if i[0] >= m.end()]
            if after:
                return after[0][2]
        m = re.search(r"\binstead of\b", fragment)
        if m:
            replaced = next((i for i in items if i[0] >= m.end()), None)
            others = [i for i in items if i is not replaced]
            if others:
                return others[0][2]
    return items[0][2]


def _quantity(masked: str, language: str, item_start: int | None) -> int | None:
    for pattern, n in _QUANTITY_PHRASE_RES:
        if pattern.search(masked):
            return n
    table = lexicon.NUMBER_WORDS[language]
    tokens = list(re.finditer(r"\S+", masked))
    numbers: list[tuple[int, int]] = []  # (start, value)
    for i, tok in enumerate(tokens):
        word = tok.group()
        if word not in table:
            continue
        if language == "hi-en" and word == "do":
            if i > 0 and tokens[i - 1].group() in _DO_VERB_STEMS:
                continue
            if item_start is not None and tok.start() > item_start:
                continue
        if word in _ARTICLES and not _directly_before(masked, tok.end(), item_start):
            continue
        numbers.append((tok.start(), table[word]))
    if not numbers:
        return None
    if item_start is not None:
        before = [n for n in numbers if n[0] < item_start]
        if before:
            return before[-1][1]
    return numbers[0][1]


def _directly_before(masked: str, end: int, item_start: int | None) -> bool:
    """True when only blanks (consumed aliases or spaces) separate ``end`` from the item."""
    return item_start is not None and end < item_start and not masked[end:item_start].strip()


def _is_confirm(fragment: str) -> bool:
    return bool(_CONFIRM_RE.search(fragment)) and not _CONFIRM_STRIP_RE.sub(" ", fragment).strip()


def _is_bare(masked: str) -> bool:
    return not _BARE_STRIP_RE.sub(" ", masked).strip()


def _only_fillers(masked: str, language: str) -> bool:
    rest = _CORRECTION_FILLER_RE.sub(" ", masked)
    table = lexicon.NUMBER_WORDS[language]
    return all(tok in table for tok in rest.split())


def _refused_options(masked: str, menu: Menu) -> tuple[list[str], list[str]]:
    """Options implied by a refusal that names a group's subject ("onion", "pyaaz", "sugar")
    or an extra ("cheese"): (options to set, options to drop)."""
    tokens = set(masked.split())
    to_set: list[str] = []
    to_drop: list[str] = []
    for gid, words in lexicon.GROUP_SUBJECTS.items():
        if not tokens & set(words):
            continue
        try:
            group = menu.group(gid)
        except KeyError:
            continue
        negative = next(
            (o.id for o in group.options if o.id.startswith(lexicon.NEGATIVE_OPTION_PREFIX)), None
        )
        chosen = negative or group.default
        if chosen is not None and chosen not in to_set:
            to_set.append(chosen)
    for word, option in lexicon.EXTRA_SUBJECTS.items():
        if word in tokens and menu.has_option(option) and option not in to_drop:
            to_drop.append(option)
    return to_set, to_drop


def _only_rehne_do(masked: str) -> bool:
    """True when "rehne do" is the only removal marker in the clause."""
    return not _REMOVE_RE.search(_REHNE_RE.sub(" ", masked))


def _parse_fragment(fragment: str, menu: Menu, language: str, carried: bool) -> Clause | None:
    """Classify one fragment. Returns None for a dropped (bare) fragment."""
    if _CANCEL_ORDER_RE.search(fragment):
        return Clause(Intent.CANCEL_ORDER, raw=fragment, is_correction=carried)
    items, options, masked = _scan_aliases(fragment, menu)
    item_id = _pick_item(fragment, items)
    item_start = items[0][0] if items else None
    quantity = _quantity(masked, language, item_start)
    if item_id is None and _READBACK_RE.search(fragment):
        return Clause(Intent.READBACK, raw=fragment)
    if item_id is None and not options and quantity is None:
        if _CLOSING_RE.search(fragment):
            return Clause(Intent.CLOSING, raw=fragment)
        if _is_confirm(fragment):
            return Clause(Intent.CONFIRM, raw=fragment)
    if item_id is None and not options and _is_bare(masked):
        return None
    is_correction = carried or bool(_CORRECTION_RE.search(masked))
    is_remove = bool(_REMOVE_RE.search(masked))
    negated: list[str] = []
    if is_remove:
        to_set, to_drop = _refused_options(masked, menu)
        if options or to_set or to_drop:
            # "teekha nahi chahiye", "pyaaz wala nahi chahiye", "remove the extra cheese":
            # a refusal of modifiers, applied to a line, never a removal of the line
            negated, options = options + to_drop, to_set
            is_remove, is_correction = False, True
        elif quantity is not None and _only_rehne_do(masked):
            is_remove, is_correction = False, True  # "ek hi rehne do" = keep just one
    elif item_id is not None and options and quantity is None and _IN_RE.search(masked):
        is_correction = True  # "paneer wrap mein pyaaz nahi chahiye" edits the existing wrap
    if is_remove:
        intent = Intent.REMOVE
    elif is_correction:
        intent = Intent.CORRECT
    elif item_id is not None:
        intent = Intent.ADD
    elif (quantity is not None or options) and _only_fillers(masked, language):
        intent = Intent.CORRECT
    else:
        intent = Intent.UNKNOWN
    if (
        intent is Intent.CORRECT
        and item_id is None
        and quantity is None
        and not options
        and not negated
    ):
        intent = Intent.UNKNOWN
    return Clause(intent, item_id, quantity, options, fragment, is_correction, negated)


def _carries_correction(fragment: str) -> bool:
    return bool(_CORRECTION_RE.search(fragment))


def _post_process(clauses: list[Clause], text: str) -> list[Clause]:
    merged: list[Clause] = []
    for c in clauses:
        mods_only = c.item_id is None and c.quantity is None and c.modifiers
        if (
            c.intent is Intent.CORRECT
            and mods_only
            and not c.is_correction
            and merged
            and merged[-1].intent is Intent.ADD
        ):
            for m in c.modifiers:
                if m not in merged[-1].modifiers:
                    merged[-1].modifiers.append(m)
            continue
        merged.append(c)
    # Terminal clauses before an actionable clause are affirmative fillers ("Haan, do ...").
    out: list[Clause] = []
    for i, c in enumerate(merged):
        if c.intent in TERMINAL and any(x.intent in ACTIONABLE for x in merged[i + 1 :]):
            continue
        out.append(c)
    # Keep only the first of consecutive terminal clauses ("Okay, that's all.").
    deduped: list[Clause] = []
    for c in out:
        if c.intent in TERMINAL and deduped and deduped[-1].intent in TERMINAL:
            continue
        deduped.append(c)
    # Unknown clauses matter only when nothing else was understood.
    known = [c for c in deduped if c.intent is not Intent.UNKNOWN]
    if known:
        return known
    if deduped:
        return [Clause(Intent.UNKNOWN, raw=normalise_text(text))]
    return []


def parse_utterance(text: str, menu: Menu, language: str) -> list[Clause]:
    """Parse a caller utterance into ordered clauses."""
    if language not in lexicon.NUMBER_WORDS:
        language = "en-IN"
    clauses: list[Clause] = []
    carried = False
    for fragment in split_clauses(text):
        clause = _parse_fragment(fragment, menu, language, carried)
        if clause is None:
            carried = carried or _carries_correction(fragment)
            continue
        carried = False
        clauses.append(clause)
    return _post_process(clauses, text)
