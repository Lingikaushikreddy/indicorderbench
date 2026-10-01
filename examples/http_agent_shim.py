#!/usr/bin/env python3
"""Example: expose an agent to IndicOrderBench over the HTTP turn protocol.

This wraps the built-in reference agent, but the shape is what your own agent needs:

1. On ``{"event": "start"}`` remember the session and the ``backend_url``.
2. On ``{"event": "turn"}`` feed the text (or the base64 WAV) to your agent, let it call the
   sandbox backend over HTTP (``POST {backend_url}/sessions/{session_id}/tools/{tool}``),
   and reply ``{"text": "<what the agent said>"}``.
3. On ``{"event": "stop"}`` forget the session.

Run it, then point the benchmark at it:

    python examples/http_agent_shim.py --port 8900
    iob run starter --agent http:http://127.0.0.1:8900/iob --tag smoke

Add ``--bugs ignore_corrections,drop_modifiers`` to see failures in the report.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

from indicorderbench.adapters.protocol import AgentReply, CallerUtterance
from indicorderbench.agents.rule_based import RuleBasedAgent
from indicorderbench.backend.state import BackendError
from indicorderbench.schemas.menu import Menu
from indicorderbench.schemas.results import BackendSnapshot, CartLine, SubmittedOrder


class RemoteBackend:
    """The sandbox backend reached over HTTP, with the same method names as OrderBackend.

    Your agent's function tools would call these. ``BackendError`` is raised for refused
    calls exactly as the in-process backend does, so agent code works unchanged.
    """

    def __init__(self, backend_url: str, session_id: str, timeout: float = 10.0) -> None:
        self._base = f"{backend_url.rstrip('/')}/sessions/{session_id}"
        self._client = httpx.Client(timeout=timeout)
        self._menu: Menu | None = None

    @property
    def menu(self) -> Menu:
        if self._menu is None:
            self._menu = Menu.model_validate(self._client.get(f"{self._base}/menu").json())
        return self._menu

    def _call(self, name: str, **args: Any) -> Any:
        body = self._client.post(f"{self._base}/tools/{name}", json=args).json()
        if not body.get("ok"):
            err = body.get("error") or {}
            raise BackendError(err.get("code", "error"), err.get("message", "backend refused"))
        return body["result"]

    def lookup_menu(self, query: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._call("lookup_menu", query=query)
        return result

    def add_item(
        self, item_id: str, quantity: int = 1, modifiers: list[str] | None = None
    ) -> CartLine:
        return CartLine.model_validate(
            self._call("add_item", item_id=item_id, quantity=quantity, modifiers=modifiers or [])
        )

    def update_line(
        self, line_id: str, quantity: int | None = None, modifiers: list[str] | None = None
    ) -> CartLine | None:
        result = self._call("update_line", line_id=line_id, quantity=quantity, modifiers=modifiers)
        return CartLine.model_validate(result) if result is not None else None

    def remove_line(self, line_id: str) -> None:
        self._call("remove_line", line_id=line_id)

    def clear_cart(self) -> None:
        self._call("clear_cart")

    def get_cart(self) -> list[CartLine]:
        return [CartLine.model_validate(c) for c in self._call("get_cart")]

    def submit_order(self) -> SubmittedOrder:
        return SubmittedOrder.model_validate(self._call("submit_order"))

    def cancel_order(self, order_id: str) -> SubmittedOrder:
        return SubmittedOrder.model_validate(self._call("cancel_order", order_id=order_id))

    def list_orders(self) -> list[SubmittedOrder]:
        return [SubmittedOrder.model_validate(o) for o in self._call("list_orders")]

    def active_orders(self) -> list[SubmittedOrder]:
        return [o for o in self.list_orders() if o.status == "submitted"]

    def snapshot(self) -> BackendSnapshot:
        return BackendSnapshot(cart=self.get_cart(), orders=self.list_orders(), trace=[])

    def close(self) -> None:
        self._client.close()


class ShimState:
    def __init__(self, bugs: frozenset[str]) -> None:
        self.bugs = bugs
        self.lock = threading.Lock()
        self.sessions: dict[str, tuple[RemoteBackend, RuleBasedAgent]] = {}


def make_handler(state: ShimState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
                event = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._reply(400, {"error": "malformed JSON"})
                return
            kind = event.get("event")
            sid = event.get("session_id", "")
            if kind == "start":
                backend = RemoteBackend(event["backend_url"], sid)
                # OrderBackend and RemoteBackend share the same method surface; the agent
                # only uses that surface, so the remote one can stand in for it.
                agent = RuleBasedAgent(backend, event.get("language", "en-IN"), state.bugs)  # type: ignore[arg-type]
                with state.lock:
                    state.sessions[sid] = (backend, agent)
                self._reply(200, {"ok": True})
            elif kind == "turn":
                with state.lock:
                    pair = state.sessions.get(sid)
                if pair is None:
                    self._reply(404, {"error": f"unknown session {sid}"})
                    return
                audio = event.get("audio_b64")
                utterance = CallerUtterance(
                    turn_id=event.get("turn_id", "turn"),
                    text=event.get("text"),
                    audio_path=None,
                    audio_bytes=base64.b64decode(audio) if audio else None,
                    language=event.get("language", "en-IN"),
                )
                reply: AgentReply = pair[1].handle(utterance)
                self._reply(200, {"text": reply.text, "meta": {"agent": "reference-shim"}})
            elif kind == "stop":
                with state.lock:
                    pair = state.sessions.pop(sid, None)
                if pair is not None:
                    pair[0].close()
                self._reply(200, {"ok": True})
            else:
                self._reply(400, {"error": f"unknown event {kind!r}"})

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def serve(
    host: str = "127.0.0.1", port: int = 0, bugs: frozenset[str] = frozenset()
) -> ThreadingHTTPServer:
    """Start the shim in a daemon thread and return the server (``server_address`` has the port)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(ShimState(bugs)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--bugs", default="", help="comma-separated reference-agent bugs to inject")
    args = parser.parse_args()
    bugs = frozenset(b.strip() for b in args.bugs.split(",") if b.strip())
    httpd = serve(args.host, args.port, bugs)
    url = f"http://{args.host}:{httpd.server_address[1]}/iob"
    print(f"agent shim listening at {url}")
    print(f"  iob run starter --agent http:{url} --tag smoke")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
