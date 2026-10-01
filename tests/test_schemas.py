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
                options=[
                    Modifier(id="extra_cheese", name="Extra cheese", aliases=["extra cheese"])
                ],
            ),
        ],
        items=[
            MenuItem(
                id="paneer_wrap",
                name="Paneer Wrap",
                aliases=["paneer wrap", "paneer roll"],
                price=180,
                modifier_groups=["onion", "extras"],
            ),
            MenuItem(id="mango_lassi", name="Mango Lassi", aliases=["mango lassi"], price=90),
        ],
    )


def test_normalise_text_strips_punctuation_and_case():
    assert (
        normalise_text("  Two Paneer Wraps... actually, ONE! ") == "two paneer wraps actually one"
    )


def test_menu_lookups():
    menu = make_menu()
    assert menu.item("paneer_wrap").name == "Paneer Wrap"
    assert menu.option_group("no_onion").id == "onion"
    assert [g.id for g in menu.groups_for("paneer_wrap")] == ["onion", "extras"]
    assert menu.groups_for("mango_lassi") == []
    assert menu.has_item("mango_lassi") and not menu.has_item("ghost")
    assert menu.has_option("extra_cheese") and not menu.has_option("ghost")
    with pytest.raises(KeyError):
        menu.item("nope")


def test_menu_rejects_duplicate_ids_and_unknown_group():
    with pytest.raises(ValidationError):
        Menu(
            id="m",
            name="M",
            modifier_groups=[],
            items=[MenuItem(id="a", name="A", price=1), MenuItem(id="a", name="A2", price=1)],
        )
    with pytest.raises(ValidationError):
        Menu(
            id="m",
            name="M",
            modifier_groups=[],
            items=[MenuItem(id="a", name="A", price=1, modifier_groups=["ghost"])],
        )


def test_exclusive_group_requires_default_option():
    with pytest.raises(ValidationError):
        ModifierGroup(
            id="g",
            name="G",
            exclusive=True,
            default="missing",
            options=[Modifier(id="x", name="X")],
        )
    with pytest.raises(ValidationError):
        ModifierGroup(
            id="g", name="G", exclusive=False, default="x", options=[Modifier(id="x", name="X")]
        )


def test_alias_index_is_longest_first_and_normalised():
    menu = make_menu()
    index = menu.alias_index()
    assert all(len(index[i].alias) >= len(index[i + 1].alias) for i in range(len(index) - 1))
    kinds = {(e.kind, e.ref_id) for e in index}
    assert ("item", "paneer_wrap") in kinds and ("option", "no_onion") in kinds
    assert all(e.alias == normalise_text(e.alias) for e in index)


def test_menu_search_matches_names_and_aliases():
    menu = make_menu()
    assert [i.id for i in menu.search("Paneer")] == ["paneer_wrap"]
    assert [i.id for i in menu.search("lassi")] == ["mango_lassi"]
    assert menu.search("") == [] and menu.search("pizza") == []


def test_caller_script_assigns_turn_ids_and_resolves_defaults():
    script = CallerScript(
        turns=[CallerTurn(text="two wraps"), CallerTurn(text="no onion", id="custom")]
    )
    assert [t.id for t in script.turns] == ["t1", "custom"]
    defaults = CallerDefaults(
        closing=CallerTurn(text="that's all"),
        confirm=CallerTurn(text="yes"),
        fallback=CallerTurn(text="yes that's right"),
        nudge=CallerTurn(text="please place the order"),
        confirm_patterns=["shall i place"],
        goodbye_patterns=["order placed"],
        question_patterns=["which", "how many"],
        clarifications=[
            ClarificationRule(id="any", match=["anything else"], reply=CallerTurn(text="no"))
        ],
    )
    resolved = script.resolved(defaults)
    assert resolved.closing.text == "that's all"
    assert resolved.max_fallbacks == 2 and resolved.max_nudges == 2
    assert [r.id for r in resolved.clarifications] == ["any"]
    assert resolved.closing.id == "closing" and resolved.clarifications[0].reply.id == "c_any"
    assert resolved.is_confirm_request("Great. Shall I place the order?")
    assert resolved.is_goodbye("Order placed, thanks!")
    assert resolved.is_question("Which one?") and resolved.is_question("How many wraps")
    assert not resolved.is_question("Added two wraps.")


def test_caller_script_scenario_overrides_win():
    script = CallerScript(
        turns=[CallerTurn(text="x")],
        closing=CallerTurn(text="bas"),
        max_fallbacks=0,
        clarifications=[ClarificationRule(id="mine", match=["a"], reply=CallerTurn(text="b"))],
    )
    defaults = CallerDefaults(
        closing=CallerTurn(text="that's all"),
        confirm=CallerTurn(text="yes"),
        fallback=CallerTurn(text="right"),
        nudge=CallerTurn(text="place it"),
        confirm_patterns=["confirm"],
        goodbye_patterns=["bye"],
        clarifications=[ClarificationRule(id="theirs", match=["c"], reply=CallerTurn(text="d"))],
    )
    resolved = script.resolved(defaults)
    assert resolved.closing.text == "bas" and resolved.closing.id == "closing"
    assert resolved.max_fallbacks == 0
    assert [r.id for r in resolved.clarifications] == ["mine", "theirs"]


def test_caller_script_rejects_duplicate_turn_ids():
    with pytest.raises(ValidationError):
        CallerScript(turns=[CallerTurn(text="a", id="t1"), CallerTurn(text="b", id="t1")])


def test_clarification_rule_matches_case_insensitively():
    rule = ClarificationRule(id="r", match=["how many", "kitne"], reply=CallerTurn(text="two"))
    assert rule.matches("Sure, HOW MANY wraps?")
    assert not rule.matches("Added two wraps.")
    with pytest.raises(ValidationError):
        ClarificationRule(id="bad", match=["("], reply=CallerTurn(text="x"))


def test_scenario_validates_shape():
    s = Scenario(
        id="en_quantity_01",
        title="t",
        language=Language.EN_IN,
        category=FailureCategory.QUANTITY,
        menu="m",
        caller=CallerScript(turns=[CallerTurn(text="two paneer wraps")]),
        expected=[
            EndState(
                id="one_order",
                orders=[ExpectedOrder(lines=[ExpectedLine(item_id="paneer_wrap", quantity=2)])],
            )
        ],
    )
    assert s.expected[0].orders[0].lines[0].modifiers == {}
    with pytest.raises(ValidationError):
        Scenario(
            id="Bad-Id",
            title="t",
            language=Language.EN_IN,
            category=FailureCategory.QUANTITY,
            menu="m",
            caller=CallerScript(turns=[CallerTurn(text="x")]),
            expected=[EndState(id="e")],
        )
    with pytest.raises(ValidationError):
        Scenario(
            id="dup_end_states",
            title="t",
            language=Language.EN_IN,
            category=FailureCategory.QUANTITY,
            menu="m",
            caller=CallerScript(turns=[CallerTurn(text="x")]),
            expected=[EndState(id="e"), EndState(id="e")],
        )


def test_snapshot_active_orders_excludes_cancelled():
    snap = BackendSnapshot(
        cart=[],
        orders=[
            SubmittedOrder(
                order_id="o1",
                lines=[CartLine(line_id="l1", item_id="x", quantity=1)],
                submitted_at_ms=1.0,
                status="cancelled",
            ),
            SubmittedOrder(
                order_id="o2",
                lines=[CartLine(line_id="l2", item_id="x", quantity=1)],
                submitted_at_ms=2.0,
            ),
        ],
        trace=[],
    )
    assert [o.order_id for o in snap.active_orders()] == ["o2"]
    assert Outcome.PASS.value == "pass"


def test_cart_line_modifiers_serialise_sorted():
    """Sets have hash-seed-dependent iteration order; JSON output must be deterministic."""
    line = CartLine(line_id="l1", item_id="x", quantity=1, modifiers={"zeta", "alpha", "mid"})
    assert line.model_dump(mode="json")["modifiers"] == ["alpha", "mid", "zeta"]
    assert '"modifiers":["alpha","mid","zeta"]' in line.model_dump_json()
    # round-trips back into a set
    assert CartLine.model_validate_json(line.model_dump_json()).modifiers == {
        "zeta",
        "alpha",
        "mid",
    }
