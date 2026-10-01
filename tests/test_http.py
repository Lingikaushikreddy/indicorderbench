import asyncio
import base64
import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from indicorderbench.adapters.http import AdapterError, HttpTurnAdapter
from indicorderbench.adapters.protocol import CallerUtterance, Modality, SessionInfo
from indicorderbench.backend.http import BackendServer
from indicorderbench.backend.state import OrderBackend
from indicorderbench.packs import load_pack
from indicorderbench.runner.runner import RunConfig, SessionRegistry, run_trial
from indicorderbench.schemas.results import Outcome

FIX = Path(__file__).parent / "fixtures" / "minipack"


@pytest.fixture
def served() -> Iterator[tuple[BackendServer, SessionRegistry, OrderBackend]]:
    pack = load_pack(FIX)
    registry = SessionRegistry()
    backend = OrderBackend(pack.menu)
    registry.register("s1", backend)
    server = BackendServer(registry)
    server.start()
    try:
        yield server, registry, backend
    finally:
        server.stop()


# -- backend server ---------------------------------------------------------------


def test_healthz(served: Any) -> None:
    server, _, _ = served
    assert server.url.startswith("http://127.0.0.1:")
    r = httpx.get(f"{server.url}/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_menu_and_tools(served: Any) -> None:
    server, _, backend = served
    r = httpx.get(f"{server.url}/sessions/s1/menu")
    assert r.status_code == 200 and r.json() == backend.menu.model_dump(mode="json")
    r = httpx.get(f"{server.url}/sessions/s1/tools")
    assert r.status_code == 200 and r.json() == OrderBackend.tool_specs()


def test_tool_call_success_is_traced(served: Any) -> None:
    server, _, backend = served
    r = httpx.post(
        f"{server.url}/sessions/s1/tools/add_item", json={"item_id": "paneer_wrap", "quantity": 2}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["result"]["item_id"] == "paneer_wrap"
    assert body["result"]["quantity"] == 2
    snap = backend.snapshot()
    assert [c.name for c in snap.trace] == ["add_item"] and len(snap.cart) == 1


def test_tool_call_without_body_means_empty_args(served: Any) -> None:
    server, _, _ = served
    r = httpx.post(f"{server.url}/sessions/s1/tools/get_cart")
    assert r.status_code == 200 and r.json() == {"ok": True, "result": []}


def test_backend_error_is_400(served: Any) -> None:
    server, _, backend = served
    r = httpx.post(f"{server.url}/sessions/s1/tools/submit_order", json={})
    assert r.status_code == 400
    assert r.json() == {
        "ok": False,
        "error": {"code": "empty_cart", "message": "cannot submit an empty cart"},
    }
    assert backend.snapshot().trace[-1].error is not None
    r = httpx.post(f"{server.url}/sessions/s1/tools/nope", json={})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_tool"


def test_float_quantity_is_400_and_the_server_keeps_serving(served: Any) -> None:
    server, _, backend = served
    with httpx.Client(base_url=server.url) as client:
        r = client.post(
            "/sessions/s1/tools/add_item", json={"item_id": "mango_lassi", "quantity": 2.5}
        )
        assert r.status_code == 400
        body = r.json()
        assert body["ok"] is False and body["error"]["code"] == "invalid_args"
        assert isinstance(body["error"]["message"], str) and "quantity" in body["error"]["message"]
        r = client.post("/sessions/s1/tools/add_item", json={"item_id": "mango_lassi"})
        assert r.status_code == 200 and r.json()["result"]["quantity"] == 1
    trace = backend.snapshot().trace
    assert [c.error is not None for c in trace] == [True, False]


def test_unexpected_tool_exception_is_500(served: Any) -> None:
    server, registry, _ = served

    class Exploding(OrderBackend):
        def get_cart(self) -> Any:
            raise RuntimeError("disk on fire")

    registry.register("boom", Exploding(load_pack(FIX).menu))
    with httpx.Client(base_url=server.url) as client:
        r = client.post("/sessions/boom/tools/get_cart")
        assert r.status_code == 500
        assert r.json() == {
            "ok": False,
            "error": {"code": "internal", "message": "RuntimeError: disk on fire"},
        }
        assert client.get("/healthz").status_code == 200


def test_unknown_session_and_route_are_404(served: Any) -> None:
    server, _, _ = served
    assert httpx.get(f"{server.url}/sessions/zzz/menu").status_code == 404
    assert httpx.get(f"{server.url}/sessions/zzz/tools").status_code == 404
    assert httpx.post(f"{server.url}/sessions/zzz/tools/get_cart").status_code == 404
    r = httpx.get(f"{server.url}/nowhere")
    assert r.status_code == 404 and r.json()["ok"] is False
    assert httpx.post(f"{server.url}/nowhere", json={}).status_code == 404


def test_malformed_json_is_400(served: Any) -> None:
    server, _, _ = served
    r = httpx.post(
        f"{server.url}/sessions/s1/tools/get_cart",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400 and r.json()["ok"] is False
    r = httpx.post(f"{server.url}/sessions/s1/tools/get_cart", json=[1, 2])
    assert r.status_code == 400


def test_stop_releases_port() -> None:
    server = BackendServer(SessionRegistry())
    server.start()
    url = server.url
    server.stop()
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{url}/healthz", timeout=1.0)


# -- test-local turn handler ------------------------------------------------------


class RemoteBackend:
    """Minimal remote client: every tool is one POST to the sandbox backend."""

    def __init__(self, base_url: str, session_id: str) -> None:
        self._base = f"{base_url}/sessions/{session_id}"

    def call(self, name: str, **args: Any) -> Any:
        r = httpx.post(f"{self._base}/tools/{name}", json=args)
        body = r.json()
        if not body["ok"]:
            raise RuntimeError(body["error"])
        return body["result"]


class TurnServer:
    """Stdlib HTTP server implementing the turn protocol; records received payloads."""

    def __init__(self, respond: Callable[[dict[str, Any], "TurnServer"], tuple[int, bytes]]):
        self.events: list[dict[str, Any]] = []
        self.remotes: dict[str, RemoteBackend] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n))
                outer.events.append(payload)
                status, body = respond(payload, outer)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/turn"

    def __enter__(self) -> "TurnServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join()


def reply_json(obj: dict[str, Any]) -> tuple[int, bytes]:
    return 200, json.dumps(obj).encode()


def order_handler(payload: dict[str, Any], srv: TurnServer) -> tuple[int, bytes]:
    event, sid = payload["event"], payload["session_id"]
    if event == "start":
        srv.remotes[sid] = RemoteBackend(payload["backend_url"], sid)
        return reply_json({})
    if event == "turn":
        remote = srv.remotes[sid]
        if "all" in payload["text"]:
            remote.call("submit_order")
            return reply_json({"text": "Order placed!", "meta": {"agent": "shim"}})
        remote.call("add_item", item_id="paneer_wrap", quantity=1)
        remote.call("add_item", item_id="paneer_wrap", quantity=1)
        return reply_json({"text": "Added two paneer wraps. Anything else?"})
    return reply_json({})


def make_session(backend_url: str | None = "http://x", modality: Modality = Modality.TEXT) -> Any:
    pack = load_pack(FIX)
    return SessionInfo(
        scenario_id="en_quantity_01",
        trial=1,
        language="en-IN",
        modality=modality,
        backend=OrderBackend(pack.menu),
        session_id="sess-1",
        backend_url=backend_url,
    )


async def test_run_trial_through_http_adapter(served: Any) -> None:
    server, registry, _ = served
    pack = load_pack(FIX)
    with TurnServer(order_handler) as turns:
        adapter = HttpTurnAdapter(turns.url)
        cfg = RunConfig(timeout_turn_s=5.0, timeout_trial_s=20.0)
        r = await run_trial(
            pack,
            pack.scenario("en_quantity_01"),
            adapter,
            cfg,
            trial=1,
            backend_url=server.url,
            registry=registry,
        )
    assert r.error is None, r.error
    assert r.outcome is Outcome.PASS
    assert r.agent_meta == {"agent": "shim"}
    assert [e["event"] for e in turns.events] == ["start", "turn", "turn", "stop"]
    assert r.snapshot is not None and [c.name for c in r.snapshot.trace] == [
        "add_item",
        "add_item",
        "submit_order",
    ]


async def test_adapter_payloads() -> None:
    def handler(payload: dict[str, Any], srv: TurnServer) -> tuple[int, bytes]:
        if payload["event"] == "turn":
            return reply_json(
                {
                    "text": "hello",
                    "audio_b64": base64.b64encode(b"RIFFreply").decode(),
                    "meta": {"k": 1},
                }
            )
        return reply_json({})

    with TurnServer(handler) as turns:
        adapter = HttpTurnAdapter(turns.url)
        await adapter.start(make_session("http://backend:1", Modality.BOTH))
        reply = await adapter.respond(
            CallerUtterance("t1", "two wraps", None, b"RIFFclip", "en-IN")
        )
        await adapter.respond(CallerUtterance("t2", None, None, None, "hi-en"))
        await adapter.stop()
    start, turn, turn2, stop = turns.events
    assert start == {
        "event": "start",
        "session_id": "sess-1",
        "scenario_id": "en_quantity_01",
        "backend_url": "http://backend:1",
        "language": "en-IN",
        "modality": "both",
    }
    assert turn == {
        "event": "turn",
        "session_id": "sess-1",
        "turn_id": "t1",
        "text": "two wraps",
        "audio_b64": base64.b64encode(b"RIFFclip").decode(),
        "audio_format": "wav",
        "language": "en-IN",
    }
    assert turn2["text"] is None and turn2["audio_b64"] is None
    assert stop == {"event": "stop", "session_id": "sess-1"}
    assert reply.text == "hello" and reply.audio_bytes == b"RIFFreply" and reply.meta == {"k": 1}


async def test_reply_without_optional_fields() -> None:
    with TurnServer(lambda p, s: reply_json({"text": "ok"})) as turns:
        adapter = HttpTurnAdapter(turns.url)
        await adapter.start(make_session())
        reply = await adapter.respond(CallerUtterance("t1", "hi", None, None, "en-IN"))
        await adapter.stop()
    assert reply.text == "ok" and reply.audio_bytes is None and reply.meta == {}


async def test_non_2xx_raises_adapter_error() -> None:
    def handler(payload: dict[str, Any], srv: TurnServer) -> tuple[int, bytes]:
        if payload["event"] == "turn":
            return 500, b'{"error": "boom"}'
        return reply_json({})

    with TurnServer(handler) as turns:
        adapter = HttpTurnAdapter(turns.url)
        await adapter.start(make_session())
        with pytest.raises(AdapterError, match="500"):
            await adapter.respond(CallerUtterance("t1", "hi", None, None, "en-IN"))
        await adapter.stop()
    assert issubclass(AdapterError, RuntimeError)


async def test_start_failure_raises_adapter_error() -> None:
    with TurnServer(lambda p, s: (503, b"{}")) as turns:
        adapter = HttpTurnAdapter(turns.url)
        with pytest.raises(AdapterError, match="503"):
            await adapter.start(make_session())
        await adapter.stop()


@pytest.mark.parametrize(
    "body",
    [b"not json", b'{"nope": 1}', b'{"text": 5}', b"[1]", b'{"text": "x", "meta": 3}'],
)
async def test_malformed_reply_raises_adapter_error(body: bytes) -> None:
    def handler(payload: dict[str, Any], srv: TurnServer) -> tuple[int, bytes]:
        return (200, body) if payload["event"] == "turn" else reply_json({})

    with TurnServer(handler) as turns:
        adapter = HttpTurnAdapter(turns.url)
        await adapter.start(make_session())
        with pytest.raises(AdapterError):
            await adapter.respond(CallerUtterance("t1", "hi", None, None, "en-IN"))
        await adapter.stop()


async def test_stop_failure_raises_adapter_error() -> None:
    def handler(payload: dict[str, Any], srv: TurnServer) -> tuple[int, bytes]:
        return (500, b"{}") if payload["event"] == "stop" else reply_json({})

    with TurnServer(handler) as turns:
        adapter = HttpTurnAdapter(turns.url)
        await adapter.start(make_session())
        with pytest.raises(AdapterError, match="stop"):
            await adapter.stop()


async def test_timeout_raises_adapter_error() -> None:
    release = threading.Event()

    def handler(payload: dict[str, Any], srv: TurnServer) -> tuple[int, bytes]:
        if payload["event"] == "turn":
            release.wait(2)
        return reply_json({"text": "late"})

    with TurnServer(handler) as turns:
        adapter = HttpTurnAdapter(turns.url, timeout_s=0.1)
        await adapter.start(make_session())
        with pytest.raises(AdapterError):
            await adapter.respond(CallerUtterance("t1", "hi", None, None, "en-IN"))
        release.set()
        await asyncio.sleep(0)
        await adapter.stop()


async def test_respond_before_start_raises() -> None:
    with pytest.raises(AdapterError):
        await HttpTurnAdapter("http://127.0.0.1:1/turn").respond(
            CallerUtterance("t1", "hi", None, None, "en-IN")
        )


async def test_injected_client_is_not_closed() -> None:
    async with httpx.AsyncClient() as client:
        with TurnServer(lambda p, s: reply_json({"text": "ok"})) as turns:
            adapter = HttpTurnAdapter(turns.url, client=client)
            await adapter.start(make_session())
            await adapter.stop()
        assert not client.is_closed
