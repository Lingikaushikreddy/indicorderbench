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
        b.add_item("paneer_wrap", 1, ["no_onion", "with_onion"])  # two options, exclusive
    assert e.value.code == "invalid_modifier"
    with pytest.raises(BackendError) as e:
        b.add_item("paneer_wrap", 1, ["ghost"])
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
    with pytest.raises(BackendError) as e:
        b.update_line("nope", quantity=1)
    assert e.value.code == "unknown_line"


def test_update_keeps_unspecified_fields():
    b = make_backend()
    line = b.add_item("paneer_wrap", 2, ["no_onion"])
    updated = b.update_line(line.line_id, quantity=3)
    assert updated is not None and updated.modifiers == {"no_onion"} and updated.quantity == 3
    updated = b.update_line(line.line_id, modifiers=[])
    assert updated is not None and updated.modifiers == set() and updated.quantity == 3


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
    assert [o.status for o in b.list_orders()] == ["cancelled"]
    with pytest.raises(BackendError) as e:
        b.cancel_order(order.order_id)
    assert e.value.code == "already_cancelled"
    with pytest.raises(BackendError) as e:
        b.cancel_order("o99")
    assert e.value.code == "unknown_order"


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
    assert snap.trace[0].args == {"item_id": "mango_lassi", "quantity": 1, "modifiers": []}


def test_snapshot_is_a_copy():
    b = make_backend()
    b.add_item("mango_lassi")
    snap = b.snapshot()
    b.clear_cart()
    assert len(snap.cart) == 1 and len(snap.trace) == 1


def test_call_dispatch_and_tool_specs():
    b = make_backend()
    result = b.call("add_item", {"item_id": "mango_lassi", "quantity": 3})
    assert result["quantity"] == 3
    assert b.call("get_cart", {})[0]["item_id"] == "mango_lassi"
    with pytest.raises(BackendError) as e:
        b.call("fly", {})
    assert e.value.code == "unknown_tool"
    with pytest.raises(BackendError) as e:
        b.call("add_item", {"bogus": 1})
    assert e.value.code == "invalid_args"
    names = {s["name"] for s in OrderBackend.tool_specs()}
    assert names == set(OrderBackend.TOOL_NAMES)
    assert all("parameters" in s and "description" in s for s in OrderBackend.tool_specs())


def test_lookup_menu_returns_items_with_groups():
    b = make_backend()
    hits = b.lookup_menu("paneer")
    assert [h["item_id"] for h in hits] == ["paneer_wrap"]
    assert [g["group_id"] for g in hits[0]["modifier_groups"]] == ["onion", "extras"]
    assert b.lookup_menu("pizza") == []


@pytest.mark.parametrize(
    ("name", "args", "code"),
    [
        ("fly", {}, "unknown_tool"),
        ("add_item", {"item": "mango_lassi"}, "invalid_args"),
        ("add_item", {"item_id": "mango_lassi", "quantity": 2.5}, "invalid_args"),
    ],
)
def test_refused_calls_are_traced(name: str, args: dict[str, object], code: str):
    b = make_backend()
    with pytest.raises(BackendError) as e:
        b.call(name, args)
    assert e.value.code == code
    trace = b.snapshot().trace
    assert len(trace) == 1, trace
    assert trace[0].name == name and args.items() <= trace[0].args.items()
    assert trace[0].error is not None and trace[0].error.startswith(f"{code}: ")
    assert trace[0].result is None and b.get_cart() == []


def test_lookup_menu_rejects_a_non_string_query_and_traces_it():
    b = make_backend()
    for query in (None, 5):
        with pytest.raises(BackendError) as e:
            b.call("lookup_menu", {"query": query})
        assert e.value.code == "invalid_args"
    trace = b.snapshot().trace
    assert [c.name for c in trace] == ["lookup_menu", "lookup_menu"]
    assert all(c.error and c.error.startswith("invalid_args") for c in trace)


@pytest.mark.parametrize("quantity", [2.5, True, "2"])
def test_update_line_rejects_a_non_integer_quantity(quantity: object):
    b = make_backend()
    line = b.add_item("paneer_wrap", 1)
    with pytest.raises(BackendError) as e:
        b.update_line(line.line_id, quantity=quantity)  # type: ignore[arg-type]
    assert e.value.code == "invalid_args" and "quantity" in e.value.message
    assert b.get_cart()[0].quantity == 1


def test_add_item_rejects_a_bool_quantity():
    b = make_backend()
    with pytest.raises(BackendError) as e:
        b.add_item("paneer_wrap", True)  # JSON true is not a count
    assert e.value.code == "invalid_args" and "quantity" in e.value.message
    assert b.get_cart() == []


@pytest.mark.parametrize("modifiers", ["no_onion", 5, [1]])
def test_modifiers_must_be_a_list_of_strings(modifiers: object):
    b = make_backend()
    line = b.add_item("paneer_wrap", 1)
    with pytest.raises(BackendError) as e:
        b.add_item("paneer_wrap", 1, modifiers)  # type: ignore[arg-type]
    assert e.value.code == "invalid_args" and "modifiers" in e.value.message
    with pytest.raises(BackendError) as e:
        b.update_line(line.line_id, modifiers=modifiers)  # type: ignore[arg-type]
    assert e.value.code == "invalid_args" and "modifiers" in e.value.message
    assert b.get_cart()[0].modifiers == set()
    # every refusal is in the trace, with the raw argument
    trace = [c for c in b.snapshot().trace if c.error]
    assert [c.name for c in trace] == ["add_item", "update_line"]
    assert all(c.args["modifiers"] == modifiers for c in trace)


def test_line_quantities_in_a_snapshot_always_round_trip_through_json():
    b = make_backend()
    line = b.add_item("paneer_wrap", 1)
    b.update_line(line.line_id, quantity=2.0)  # type: ignore[arg-type]
    b.submit_order()
    from indicorderbench.schemas.results import BackendSnapshot

    snapshot = b.snapshot()
    assert snapshot.orders[0].lines[0].quantity == 2
    assert isinstance(snapshot.orders[0].lines[0].quantity, int)
    assert BackendSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
