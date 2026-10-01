from typing import Literal

from indicorderbench.checker.canonical import canonical_lines, order_matches, resolve_modifiers
from indicorderbench.checker.checker import check
from indicorderbench.schemas.results import BackendSnapshot, CartLine, SubmittedOrder
from indicorderbench.schemas.scenario import EndState, ExpectedLine, ExpectedOrder
from tests.test_schemas import make_menu

MENU = make_menu()


def order(
    *lines: CartLine, status: Literal["submitted", "cancelled"] = "submitted", oid: str = "o1"
) -> SubmittedOrder:
    return SubmittedOrder(order_id=oid, lines=list(lines), submitted_at_ms=0.0, status=status)


def line(item: str, qty: int, *mods: str, lid: str = "l1") -> CartLine:
    return CartLine(line_id=lid, item_id=item, quantity=qty, modifiers=set(mods))


def snap(*orders: SubmittedOrder) -> BackendSnapshot:
    return BackendSnapshot(cart=[], orders=list(orders), trace=[])


def expect(*lines: ExpectedLine, es_id: str = "e1") -> EndState:
    return EndState(id=es_id, orders=[ExpectedOrder(lines=list(lines))])


def test_resolve_modifiers_applies_defaults():
    assert resolve_modifiers(MENU, "paneer_wrap", []) == {"onion": ("with_onion",), "extras": ()}
    assert resolve_modifiers(MENU, "paneer_wrap", ["no_onion", "extra_cheese"]) == {
        "onion": ("no_onion",),
        "extras": ("extra_cheese",),
    }
    assert resolve_modifiers(MENU, "mango_lassi", []) == {}


def test_canonical_lines_merge_and_sort():
    lines = canonical_lines(
        MENU,
        [
            line("paneer_wrap", 1, lid="a"),
            line("mango_lassi", 1, lid="b"),
            line("paneer_wrap", 1, lid="c"),
            line("paneer_wrap", 0, lid="d"),
        ],
    )
    assert [(c.item_id, c.quantity) for c in lines] == [("mango_lassi", 1), ("paneer_wrap", 2)]


def test_canonical_lines_keep_different_modifiers_apart():
    lines = canonical_lines(
        MENU, [line("paneer_wrap", 1, lid="a"), line("paneer_wrap", 1, "no_onion", lid="b")]
    )
    assert [c.quantity for c in lines] == [1, 1]


def test_order_matches_with_wildcard_and_defaults():
    actual = [line("paneer_wrap", 1, "no_onion"), line("mango_lassi", 1, lid="l2")]
    assert order_matches(
        MENU,
        ExpectedOrder(
            lines=[
                ExpectedLine(item_id="paneer_wrap", quantity=1, modifiers={"onion": "no_onion"}),
                ExpectedLine(item_id="mango_lassi", quantity=1),
            ]
        ),
        actual,
    )
    assert not order_matches(
        MENU,
        ExpectedOrder(
            lines=[
                ExpectedLine(item_id="paneer_wrap", quantity=1),
                ExpectedLine(item_id="mango_lassi", quantity=1),
            ]
        ),
        actual,
    )
    assert order_matches(
        MENU,
        ExpectedOrder(
            lines=[
                ExpectedLine(item_id="paneer_wrap", quantity=1, modifiers={"onion": "*"}),
                ExpectedLine(item_id="mango_lassi", quantity=1),
            ]
        ),
        actual,
    )
    assert order_matches(
        MENU,
        ExpectedOrder(
            lines=[
                ExpectedLine(
                    item_id="paneer_wrap",
                    quantity=1,
                    modifiers={"onion": "no_onion", "extras": ["extra_cheese"]},
                )
            ]
        ),
        [line("paneer_wrap", 1, "no_onion", "extra_cheese")],
    )


def test_expected_lines_merge_like_actual_lines():
    expected = ExpectedOrder(
        lines=[
            ExpectedLine(item_id="mango_lassi", quantity=1),
            ExpectedLine(item_id="mango_lassi", quantity=1),
        ]
    )
    assert order_matches(MENU, expected, [line("mango_lassi", 2)])


def test_unasked_modifier_fails():
    assert not order_matches(
        MENU,
        ExpectedOrder(lines=[ExpectedLine(item_id="paneer_wrap", quantity=1)]),
        [line("paneer_wrap", 1, "extra_cheese")],
    )


def test_check_demo_table():
    expected = [
        expect(
            ExpectedLine(item_id="paneer_wrap", quantity=1, modifiers={"onion": "no_onion"}),
            ExpectedLine(item_id="mango_lassi", quantity=1),
        )
    ]
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
    assert "Paneer Wrap quantity expected 1 got 2" in result.summary


def test_check_passes_and_ignores_cancelled():
    expected = [expect(ExpectedLine(item_id="mango_lassi", quantity=1))]
    actual = snap(
        order(line("paneer_wrap", 1), status="cancelled", oid="o0"),
        order(line("mango_lassi", 1), oid="o1"),
    )
    result = check(expected, actual, MENU)
    assert result.passed and all(f.passed for f in result.field_checks)


def test_duplicate_submission_is_not_merged():
    expected = [expect(ExpectedLine(item_id="mango_lassi", quantity=1))]
    actual = snap(order(line("mango_lassi", 1), oid="o1"), order(line("mango_lassi", 1), oid="o2"))
    result = check(expected, actual, MENU)
    assert not result.passed
    assert result.field_checks[0].model_dump() == {
        "field": "Submitted orders",
        "expected": "1",
        "actual": "2",
        "passed": False,
    }
    # the second, unexpected order is reported line by line
    assert ("Mango Lassi quantity", "0", "1", False) in [
        (f.field, f.expected, f.actual, f.passed) for f in result.field_checks
    ]


def test_cancellation_end_state():
    expected = [EndState(id="none", orders=[])]
    assert check(expected, snap(), MENU).passed
    result = check(expected, snap(order(line("mango_lassi", 1))), MENU)
    assert not result.passed
    assert [(f.field, f.expected, f.actual) for f in result.field_checks] == [
        ("Submitted orders", "0", "1"),
        ("Mango Lassi quantity", "0", "1"),
    ]


def test_best_end_state_is_the_closest():
    expected = [
        expect(ExpectedLine(item_id="paneer_wrap", quantity=3), es_id="far"),
        expect(ExpectedLine(item_id="paneer_wrap", quantity=1), es_id="near"),
    ]
    result = check(expected, snap(order(line("paneer_wrap", 1, "extra_cheese"))), MENU)
    assert result.best_end_state_id == "near" and not result.passed


def test_any_matching_end_state_passes():
    expected = [
        expect(ExpectedLine(item_id="paneer_wrap", quantity=3), es_id="far"),
        expect(ExpectedLine(item_id="paneer_wrap", quantity=1), es_id="near"),
    ]
    result = check(expected, snap(order(line("paneer_wrap", 1))), MENU)
    assert result.passed and result.best_end_state_id == "near"


def test_missing_and_extra_lines():
    expected = [expect(ExpectedLine(item_id="paneer_wrap", quantity=1))]
    result = check(expected, snap(order(line("mango_lassi", 2))), MENU)
    assert [(f.field, f.expected, f.actual, f.passed) for f in result.field_checks] == [
        ("Submitted orders", "1", "1", True),
        ("Paneer Wrap quantity", "1", "0", False),
        ("Mango Lassi quantity", "0", "2", False),
    ]


def test_wildcard_and_wrong_modifier_rows():
    expected = [
        expect(
            ExpectedLine(
                item_id="paneer_wrap",
                quantity=1,
                modifiers={"onion": "*", "extras": ["extra_cheese"]},
            )
        )
    ]
    result = check(expected, snap(order(line("paneer_wrap", 1, "no_onion"))), MENU)
    rows = [(f.field, f.expected, f.actual, f.passed) for f in result.field_checks]
    assert ("Paneer Wrap · Onion", "any", "No onion", True) in rows
    assert ("Paneer Wrap · Extras", "Extra cheese", "none", False) in rows
    assert not result.passed


def test_no_order_at_all_reports_every_expected_line():
    expected = [
        expect(
            ExpectedLine(item_id="paneer_wrap", quantity=1, modifiers={"onion": "no_onion"}),
        )
    ]
    result = check(expected, snap(), MENU)
    assert [(f.field, f.expected, f.actual, f.passed) for f in result.field_checks] == [
        ("Submitted orders", "1", "0", False),
        ("Paneer Wrap quantity", "1", "0", False),
        ("Paneer Wrap · Onion", "No onion", "—", False),
    ]
