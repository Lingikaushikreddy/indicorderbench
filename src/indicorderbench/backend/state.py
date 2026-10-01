"""In-memory sandbox order backend with a tool-call trace."""

from __future__ import annotations

import copy
import inspect
import itertools
import time
from collections.abc import Callable
from typing import Any, NoReturn, TypeVar

from pydantic import BaseModel, ValidationError

from indicorderbench.schemas.menu import Menu
from indicorderbench.schemas.results import BackendSnapshot, CartLine, SubmittedOrder, ToolCall

T = TypeVar("T")


class BackendError(Exception):
    """A tool call the sandbox refused. The state is unchanged when this is raised."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _default_clock() -> Callable[[], float]:
    t0 = time.perf_counter()
    return lambda: (time.perf_counter() - t0) * 1000.0


def _describe(error: Exception) -> str:
    """A one-line message for a bad-argument error raised inside a tool."""
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(str(x) for x in e['loc']) or 'value'}: {e['msg']}" for e in error.errors()
        )
    return str(error)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


class OrderBackend:
    """Menu lookup, cart operations and order submission for one call session.

    Every public tool method appends a :class:`ToolCall` to the trace, including failed
    calls, so a report can show exactly what the agent did and when. Calls through
    :meth:`call` are traced even when the tool name or the argument names are wrong.
    """

    TOOL_NAMES: tuple[str, ...] = (
        "lookup_menu",
        "add_item",
        "update_line",
        "remove_line",
        "clear_cart",
        "get_cart",
        "submit_order",
        "cancel_order",
        "list_orders",
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
    def _traced(self, name: str, args: dict[str, Any], fn: Callable[[], T]) -> T:
        call = ToolCall(seq=len(self._trace) + 1, t_ms=self._clock(), name=name, args=args)
        try:
            result = fn()
        except BackendError as e:
            call.error = f"{e.code}: {e.message}"
            self._trace.append(call)
            raise
        except (ValueError, TypeError) as e:
            # e.g. add_item(quantity=2.5): the CartLine model rejects it
            refused = BackendError("invalid_args", _describe(e))
            call.error = f"{refused.code}: {refused.message}"
            self._trace.append(call)
            raise refused from e
        call.result = _jsonable(result)
        self._trace.append(call)
        return result

    # -- validation -----------------------------------------------------------
    def _validate_modifiers(self, item_id: str, modifiers: list[str]) -> set[str]:
        groups = {g.id: g for g in self.menu.groups_for(item_id)}
        chosen: dict[str, set[str]] = {}
        for opt in modifiers:
            if not self.menu.has_option(opt):
                raise BackendError("invalid_modifier", f"unknown modifier {opt!r}")
            g = self.menu.option_group(opt)
            if g.id not in groups:
                raise BackendError("invalid_modifier", f"{opt!r} does not apply to {item_id!r}")
            chosen.setdefault(g.id, set()).add(opt)
        for gid, opts in chosen.items():
            if groups[gid].exclusive and len(opts) > 1:
                raise BackendError(
                    "invalid_modifier", f"group {gid!r} allows one option, got {sorted(opts)}"
                )
        return set(modifiers)

    # -- tools ----------------------------------------------------------------
    def lookup_menu(self, query: str) -> list[dict[str, Any]]:
        def run() -> list[dict[str, Any]]:
            return [
                {
                    "item_id": i.id,
                    "name": i.name,
                    "price": str(i.price),
                    "modifier_groups": [
                        {
                            "group_id": g.id,
                            "name": g.name,
                            "exclusive": g.exclusive,
                            "default": g.default,
                            "options": [{"option_id": o.id, "name": o.name} for o in g.options],
                        }
                        for g in self.menu.groups_for(i.id)
                    ],
                }
                for i in self.menu.search(query)
            ]

        return self._traced("lookup_menu", {"query": query}, run)

    def add_item(
        self, item_id: str, quantity: int = 1, modifiers: list[str] | None = None
    ) -> CartLine:
        mods = list(modifiers or [])

        def run() -> CartLine:
            if not self.menu.has_item(item_id):
                raise BackendError("unknown_item", f"no item {item_id!r}")
            if quantity < 1:
                raise BackendError("invalid_quantity", "quantity must be at least 1")
            line = CartLine(
                line_id=f"l{next(self._line_ids)}",
                item_id=item_id,
                quantity=quantity,
                modifiers=self._validate_modifiers(item_id, mods),
            )
            self._cart[line.line_id] = line
            return line

        args = {"item_id": item_id, "quantity": quantity, "modifiers": mods}
        return self._traced("add_item", args, run)

    def update_line(
        self, line_id: str, quantity: int | None = None, modifiers: list[str] | None = None
    ) -> CartLine | None:
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
            new_m = (
                line.modifiers
                if modifiers is None
                else self._validate_modifiers(line.item_id, modifiers)
            )
            updated = line.model_copy(update={"quantity": new_q, "modifiers": new_m})
            self._cart[line_id] = updated
            return updated

        args = {"line_id": line_id, "quantity": quantity, "modifiers": modifiers}
        return self._traced("update_line", args, run)

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
            order = SubmittedOrder(
                order_id=f"o{next(self._order_ids)}",
                lines=list(self._cart.values()),
                submitted_at_ms=self._clock(),
            )
            self._orders.append(order)
            self._cart.clear()
            return order

        return self._traced("submit_order", {}, run)

    def cancel_order(self, order_id: str) -> SubmittedOrder:
        def run() -> SubmittedOrder:
            for i, o in enumerate(self._orders):
                if o.order_id == order_id:
                    if o.status == "cancelled":
                        raise BackendError(
                            "already_cancelled", f"order {order_id!r} is already cancelled"
                        )
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
        return copy.deepcopy(
            BackendSnapshot(
                cart=list(self._cart.values()), orders=list(self._orders), trace=list(self._trace)
            )
        )

    def _refuse(self, name: str, args: dict[str, Any], error: BackendError) -> NoReturn:
        """Trace a call refused before the tool ran, then raise ``error``."""
        self._trace.append(
            ToolCall(
                seq=len(self._trace) + 1,
                t_ms=self._clock(),
                name=name,
                args=dict(args),
                error=f"{error.code}: {error.message}",
            )
        )
        raise error

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch a tool by name. Used by the HTTP server and LLM tool-use loops.

        Raises :class:`BackendError` for an unknown tool or bad arguments; every refusal is
        recorded in the trace.
        """
        if name not in self.TOOL_NAMES:
            self._refuse(name, args, BackendError("unknown_tool", f"no tool {name!r}"))
        fn = getattr(self, name)
        try:
            inspect.signature(fn).bind(**args)
        except TypeError as e:
            self._refuse(name, args, BackendError("invalid_args", str(e)))
        try:
            result = fn(**args)
        except (ValueError, TypeError) as e:
            # raised before the tool reached its traced body, e.g. modifiers=5
            self._refuse(name, args, BackendError("invalid_args", _describe(e)))
        return _jsonable(result)

    @staticmethod
    def tool_specs() -> list[dict[str, Any]]:
        """The tools as JSON-schema function definitions for LLM function calling."""

        def spec(
            name: str, description: str, props: dict[str, Any], required: list[str]
        ) -> dict[str, Any]:
            return {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": props, "required": required},
            }

        s = {"type": "string"}
        i = {"type": "integer"}
        mods = {
            "type": "array",
            "items": {"type": "string"},
            "description": "modifier option ids",
        }
        return [
            spec("lookup_menu", "Search menu items by name or alias.", {"query": s}, ["query"]),
            spec(
                "add_item",
                "Add an item to the cart.",
                {"item_id": s, "quantity": i, "modifiers": mods},
                ["item_id"],
            ),
            spec(
                "update_line",
                "Change quantity (0 removes) or replace modifiers of a cart line.",
                {"line_id": s, "quantity": i, "modifiers": mods},
                ["line_id"],
            ),
            spec("remove_line", "Remove a cart line.", {"line_id": s}, ["line_id"]),
            spec("clear_cart", "Empty the cart.", {}, []),
            spec("get_cart", "Return the current cart lines.", {}, []),
            spec(
                "submit_order",
                "Submit the cart as an order. The cart must not be empty.",
                {},
                [],
            ),
            spec("cancel_order", "Cancel a submitted order.", {"order_id": s}, ["order_id"]),
            spec("list_orders", "List all orders and their status.", {}, []),
        ]
