"""HTTP face of the sandbox backend, so agents in any language can use the order tools."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from indicorderbench.backend.state import BackendError, OrderBackend
from indicorderbench.runner.runner import SessionRegistry


class BackendServer:
    """Serves ``GET /healthz``, ``GET /sessions/{id}/menu|tools`` and tool POSTs.

    Sessions are created by the runner (via the shared registry), never by agents.
    """

    def __init__(self, registry: SessionRegistry, host: str = "127.0.0.1", port: int = 0) -> None:
        self._registry = registry
        self._host = host
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    def start(self) -> None:
        if self._httpd is not None:
            return
        registry = self._registry

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send(self, status: int, payload: Any) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _error(self, status: int, code: str, message: str) -> None:
                self._send(status, {"ok": False, "error": {"code": code, "message": message}})

            def _backend(self, session_id: str) -> OrderBackend | None:
                try:
                    return registry.get(session_id)
                except KeyError:
                    self._error(404, "unknown_session", f"no session {session_id!r}")
                    return None

            def do_GET(self) -> None:
                parts = self.path.split("?", 1)[0].strip("/").split("/")
                if parts == ["healthz"]:
                    self._send(200, {"ok": True})
                elif len(parts) == 3 and parts[0] == "sessions" and parts[2] in ("menu", "tools"):
                    backend = self._backend(parts[1])
                    if backend is None:
                        return
                    if parts[2] == "menu":
                        self._send(200, backend.menu.model_dump(mode="json"))
                    else:
                        self._send(200, OrderBackend.tool_specs())
                else:
                    self._error(404, "not_found", f"no route {self.path!r}")

            def do_POST(self) -> None:
                parts = self.path.split("?", 1)[0].strip("/").split("/")
                if not (len(parts) == 4 and parts[0] == "sessions" and parts[2] == "tools"):
                    self._error(404, "not_found", f"no route {self.path!r}")
                    return
                backend = self._backend(parts[1])
                if backend is None:
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    args = json.loads(raw) if raw.strip() else {}
                except ValueError:
                    self._error(400, "invalid_json", "request body is not valid JSON")
                    return
                if not isinstance(args, dict):
                    self._error(400, "invalid_args", "request body must be a JSON object")
                    return
                try:
                    result = backend.call(parts[3], args)
                except BackendError as e:
                    self._error(400, e.code, e.message)
                    return
                except Exception as e:  # never drop the connection on an unexpected failure
                    self._error(500, "internal", f"{type(e).__name__}: {e}")
                    return
                self._send(200, {"ok": True, "result": result})

        httpd = ThreadingHTTPServer((self._host, self._port), Handler)
        httpd.daemon_threads = True
        self._httpd = httpd
        self.url = f"http://{self._host}:{httpd.server_address[1]}"
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd = self._thread = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join()
