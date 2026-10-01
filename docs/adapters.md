# Connecting your agent

The benchmark talks to your agent through an **adapter**. Two ship in v0.1:

| Adapter | Spec string | Use when |
|---|---|---|
| In-process | `python:<module>:<factory>` | Your agent's logic is importable Python and can call the sandbox backend object directly. |
| HTTP turn | `http:<url>` | Your agent runs anywhere else. It receives each caller turn by HTTP and calls the sandbox backend over HTTP. |

Built-in reference agents use `builtin:correct`, `builtin:buggy` and `builtin:buggy:<bug,...>`.

## The session model

For every trial the runner creates a fresh sandbox `OrderBackend` with the pack's menu, a
session id, and (for HTTP agents) registers the session with the backend server. The adapter
is started with a `SessionInfo`, receives one `CallerUtterance` per caller turn and returns an
`AgentReply`, and is stopped at the end. Your agent must place the order through the sandbox
backend; the checker reads only that state.

```python
@dataclass
class CallerUtterance:
    turn_id: str
    text: str | None          # None in audio modality
    audio_path: Path | None   # None in text modality
    audio_bytes: bytes | None # WAV bytes, same condition
    language: str             # "en-IN" | "hi-en"

@dataclass
class AgentReply:
    text: str                 # what the agent said (its own transcript)
    audio_bytes: bytes | None = None
    meta: dict[str, Any] = {} # e.g. {"cost_usd": 0.004, "model": "..."}
```

Modality: `--modality text` sends text only, `audio` sends audio only (every turn needs a
clip; run `iob synth` first), `both` sends both.

## In-process adapter

Write a factory that takes the backend and session and returns an object with a `handle`
method (sync or async). Point the CLI at it with `python:my_pkg.my_module:make_agent`.

```python
# my_pkg/my_module.py
from indicorderbench.adapters.protocol import AgentReply, CallerUtterance, SessionInfo
from indicorderbench.backend.state import OrderBackend


class MyAgent:
    def __init__(self, backend: OrderBackend, session: SessionInfo) -> None:
        self.backend = backend
        self.tools = OrderBackend.tool_specs()  # JSON-schema tool definitions for your LLM

    async def handle(self, u: CallerUtterance) -> AgentReply:
        text = u.text or self.transcribe(u.audio_bytes)
        # ... your LLM loop: when it calls a tool, dispatch to the sandbox:
        #     result = self.backend.call(tool_name, tool_args)
        return AgentReply(text="Added two paneer wraps. Anything else?")

    async def close(self) -> None:  # optional
        pass


def make_agent(backend: OrderBackend, session: SessionInfo) -> MyAgent:
    return MyAgent(backend, session)
```

`OrderBackend.call(name, args)` raises `BackendError(code, message)` for invalid calls; the
call is still recorded in the trace, so the report shows what your agent tried.

## HTTP turn adapter

Your agent exposes one endpoint. The runner POSTs three kinds of JSON events to it and runs a
sandbox backend server your agent calls back into.

### Events sent to your endpoint

```json
{"event": "start", "session_id": "...", "scenario_id": "hien_correction_01",
 "backend_url": "http://127.0.0.1:54321", "language": "hi-en", "modality": "text"}

{"event": "turn", "session_id": "...", "turn_id": "t1", "text": "Do paneer wrap...",
 "audio_b64": null, "audio_format": "wav", "language": "hi-en"}

{"event": "stop", "session_id": "..."}
```

Reply to `turn` with `{"text": "...", "audio_b64": null, "meta": {}}`. Reply to `start` and
`stop` with any 2xx. A non-2xx status or a reply without `text` is an infra error for that
trial.

### The sandbox backend over HTTP

Within a session, call tools at the `backend_url` you received:

```
GET  {backend_url}/healthz
GET  {backend_url}/sessions/{session_id}/menu
GET  {backend_url}/sessions/{session_id}/tools        -> JSON-schema tool definitions
POST {backend_url}/sessions/{session_id}/tools/{name}  body: JSON args object
     200 {"ok": true, "result": ...}
     400 {"ok": false, "error": {"code": "unknown_item", "message": "..."}}
     404 unknown session
```

Tool names and arguments are exactly those of `OrderBackend`: `lookup_menu(query)`,
`add_item(item_id, quantity=1, modifiers=[])`, `update_line(line_id, quantity=None,
modifiers=None)`, `remove_line(line_id)`, `clear_cart()`, `get_cart()`, `submit_order()`,
`cancel_order(order_id)`, `list_orders()`.

`examples/http_agent_shim.py` is a complete working example that wraps the reference agent
behind this protocol. Run it, then point the benchmark at it:

```bash
python examples/http_agent_shim.py --port 8900
iob run packs/starter --agent http:http://127.0.0.1:8900/iob --tag smoke
```

For manual development without the runner, `iob serve-backend --port 8765` serves a single
long-lived session so you can poke the tools with curl.

## LiveKit, Pipecat and other frameworks

v0.1 ships no framework-specific adapter. The HTTP turn protocol is the integration point:
expose a small endpoint inside your agent process that feeds each turn's text or audio into
your pipeline, waits for the pipeline's reply for that turn, and returns it. For LiveKit
Agents that means driving the `AgentSession` with the turn's input and collecting the
generated reply before responding; the function tools you give the LLM should call the
sandbox backend URL from the `start` event instead of your production ordering API.

A native LiveKit adapter that joins a room and streams audio is the next adapter planned.
Contributions welcome; see `CONTRIBUTING.md`.

## Writing your own adapter

Implement the protocol and pass a factory to `run_suite`:

```python
class AgentAdapter(Protocol):
    async def start(self, session: SessionInfo) -> None: ...
    async def respond(self, utterance: CallerUtterance) -> AgentReply: ...
    async def stop(self) -> None: ...
```

Raise on transport failure; the runner converts exceptions and timeouts into `infra_error`
for that trial and continues.
