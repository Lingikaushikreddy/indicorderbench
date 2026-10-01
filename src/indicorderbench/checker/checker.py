"""Deterministic end-state checker producing field-level results."""

from __future__ import annotations

from indicorderbench.checker.canonical import (
    CanonicalLine,
    ExpectedCanonicalLine,
    canonical_expected,
    canonical_lines,
    modifiers_agree,
    order_matches,
)
from indicorderbench.schemas.menu import Menu
from indicorderbench.schemas.results import BackendSnapshot, CheckResult, FieldCheck, SubmittedOrder
from indicorderbench.schemas.scenario import EndState, ExpectedOrder

NO_VALUE = "—"


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


def _option_names(menu: Menu, opts: tuple[str, ...] | None) -> str:
    if opts is None:
        return "any"
    if not opts:
        return "none"
    return ", ".join(menu.option(o).name for o in opts)


def _pick_actual(exp: ExpectedCanonicalLine, pool: list[CanonicalLine]) -> CanonicalLine | None:
    same_item = [a for a in pool if a.item_id == exp.item_id]
    if not same_item:
        return None
    for a in same_item:
        if modifiers_agree(exp, a):
            return a
    return same_item[0]


def _order_field_checks(
    menu: Menu, expected: ExpectedOrder | None, actual: SubmittedOrder | None
) -> list[FieldCheck]:
    checks: list[FieldCheck] = []
    pool = list(canonical_lines(menu, actual.lines)) if actual else []
    exp_lines = canonical_expected(menu, expected) if expected else []
    for exp in exp_lines:
        item_name = menu.item(exp.item_id).name
        act = _pick_actual(exp, pool)
        if act is not None:
            pool.remove(act)
        checks.append(
            FieldCheck(
                field=f"{item_name} quantity",
                expected=str(exp.quantity),
                actual=str(act.quantity if act else 0),
                passed=act is not None and act.quantity == exp.quantity,
            )
        )
        got_all = act.resolved() if act else {}
        for gid, want in exp.modifiers.items():
            got = got_all.get(gid) if act else None
            agrees = act is not None and (want is None or got == want)
            # Show a modifier row when the author set it, or when the agent got it wrong.
            if gid not in exp.explicit and (act is None or agrees):
                continue
            checks.append(
                FieldCheck(
                    field=f"{item_name} · {menu.group(gid).name}",
                    expected=_option_names(menu, want),
                    actual=_option_names(menu, got) if act else NO_VALUE,
                    passed=agrees,
                )
            )
    for extra in pool:
        checks.append(
            FieldCheck(
                field=f"{menu.item(extra.item_id).name} quantity",
                expected="0",
                actual=str(extra.quantity),
                passed=False,
            )
        )
    return checks


def field_checks(menu: Menu, end_state: EndState, active: list[SubmittedOrder]) -> list[FieldCheck]:
    checks = [
        FieldCheck(
            field="Submitted orders",
            expected=str(len(end_state.orders)),
            actual=str(len(active)),
            passed=len(end_state.orders) == len(active),
        )
    ]
    for i in range(max(len(end_state.orders), len(active))):
        exp = end_state.orders[i] if i < len(end_state.orders) else None
        act = active[i] if i < len(active) else None
        checks.extend(_order_field_checks(menu, exp, act))
    return checks


def check(expected: list[EndState], snapshot: BackendSnapshot, menu: Menu) -> CheckResult:
    """Compare the committed state with every acceptable end state.

    Passes when any end state matches. The reported field checks belong to the best end
    state: a matching one, else the one with the fewest failing (then most passing) rows.
    """
    active = snapshot.active_orders()
    scored = [
        (es, end_state_matches(menu, es, active), field_checks(menu, es, active)) for es in expected
    ]

    def rank(s: tuple[EndState, bool, list[FieldCheck]]) -> tuple[bool, int, int]:
        failed = sum(1 for f in s[2] if not f.passed)
        return (not s[1], failed, -(len(s[2]) - failed))

    es, _, checks = min(scored, key=rank)
    passed = any(s[1] for s in scored)
    failing = [f for f in checks if not f.passed]
    if passed:
        summary = f"matched end state {es.id}"
    else:
        detail = "; ".join(f"{f.field} expected {f.expected} got {f.actual}" for f in failing)
        summary = f"{len(failing)} field(s) wrong vs {es.id}: {detail}"
    return CheckResult(passed=passed, best_end_state_id=es.id, field_checks=checks, summary=summary)
