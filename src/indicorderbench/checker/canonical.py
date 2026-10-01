"""Canonical forms of actual and expected orders.

An actual order is reduced to merged, sorted lines with every applicable modifier group
resolved (explicit option or group default). An expected order is reduced the same way,
with ``"*"`` kept as a wildcard.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from indicorderbench.schemas.menu import Menu
from indicorderbench.schemas.results import CartLine
from indicorderbench.schemas.scenario import ExpectedOrder

ModMap = tuple[tuple[str, tuple[str, ...]], ...]
ExpKey = tuple[str, tuple[tuple[str, tuple[str, ...] | None], ...]]


@dataclass(frozen=True, order=True)
class CanonicalLine:
    item_id: str
    modifiers: ModMap
    quantity: int

    def resolved(self) -> dict[str, tuple[str, ...]]:
        return dict(self.modifiers)


@dataclass(frozen=True)
class ExpectedCanonicalLine:
    item_id: str
    quantity: int
    modifiers: dict[str, tuple[str, ...] | None]  # None means wildcard
    explicit: frozenset[str]  # groups the scenario author set


def resolve_modifiers(
    menu: Menu, item_id: str, selected: Iterable[str]
) -> dict[str, tuple[str, ...]]:
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
    merged: dict[ExpKey, tuple[int, set[str]]] = {}
    for el in order.lines:
        resolved: dict[str, tuple[str, ...] | None] = dict(resolve_modifiers(menu, el.item_id, []))
        for gid, spec in el.modifiers.items():
            if spec == "*":
                resolved[gid] = None
            elif isinstance(spec, str):
                resolved[gid] = (spec,)
            else:
                resolved[gid] = tuple(sorted(spec))
        key: ExpKey = (el.item_id, tuple(sorted(resolved.items())))
        qty, explicit = merged.get(key, (0, set()))
        merged[key] = (qty + el.quantity, explicit | set(el.modifiers))
    return [
        ExpectedCanonicalLine(item, qty, dict(mods), frozenset(explicit))
        for (item, mods), (qty, explicit) in merged.items()
    ]


def modifiers_agree(expected: ExpectedCanonicalLine, actual: CanonicalLine) -> bool:
    got = actual.resolved()
    return all(want is None or got.get(gid) == want for gid, want in expected.modifiers.items())


def line_matches(expected: ExpectedCanonicalLine, actual: CanonicalLine) -> bool:
    return (
        expected.item_id == actual.item_id
        and expected.quantity == actual.quantity
        and modifiers_agree(expected, actual)
    )


def _match_all(expected: list[ExpectedCanonicalLine], actual: list[CanonicalLine]) -> bool:
    if not expected:
        return not actual
    head, rest = expected[0], expected[1:]
    for i, a in enumerate(actual):
        if line_matches(head, a) and _match_all(rest, actual[:i] + actual[i + 1 :]):
            return True
    return False


def order_matches(menu: Menu, expected: ExpectedOrder, lines: Iterable[CartLine]) -> bool:
    return _match_all(canonical_expected(menu, expected), list(canonical_lines(menu, lines)))
