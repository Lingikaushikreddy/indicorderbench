"""The example HTTP shim must survive the real runner and the real CLI."""

import importlib.util
import socket
from pathlib import Path

from typer.testing import CliRunner

from indicorderbench.adapters.http import HttpTurnAdapter
from indicorderbench.backend.http import BackendServer
from indicorderbench.cli import app
from indicorderbench.packs import load_pack
from indicorderbench.report.json_report import read_json
from indicorderbench.runner.runner import RunConfig, SessionRegistry, run_trial
from indicorderbench.schemas.results import Outcome

REPO = Path(__file__).resolve().parents[1]


def load_shim():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "http_agent_shim", REPO / "examples" / "http_agent_shim.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_shim_passes_a_smoke_scenario_through_the_runner():
    shim = load_shim()
    httpd = shim.serve(port=0)
    registry = SessionRegistry()
    backend_server = BackendServer(registry)
    backend_server.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/iob"
        pack = load_pack(REPO / "packs" / "starter")
        result = await run_trial(
            pack,
            pack.scenario("hien_modifier_01"),
            HttpTurnAdapter(url, timeout_s=10),
            RunConfig(timeout_turn_s=10, timeout_trial_s=60),
            trial=1,
            backend_url=backend_server.url,
            registry=registry,
        )
        assert result.outcome is Outcome.PASS, result.check or result.error
        assert result.agent_meta == {"agent": "reference-shim"}
        assert result.snapshot is not None and result.snapshot.trace  # calls arrived over HTTP
    finally:
        backend_server.stop()
        httpd.shutdown()


async def test_shim_with_injected_bug_fails_its_category():
    shim = load_shim()
    httpd = shim.serve(port=0, bugs=frozenset({"ignore_corrections"}))
    registry = SessionRegistry()
    backend_server = BackendServer(registry)
    backend_server.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/iob"
        pack = load_pack(REPO / "packs" / "starter")
        result = await run_trial(
            pack,
            pack.scenario("hien_correction_05"),
            HttpTurnAdapter(url, timeout_s=10),
            RunConfig(timeout_turn_s=10, timeout_trial_s=60),
            trial=1,
            backend_url=backend_server.url,
            registry=registry,
        )
        assert result.outcome is Outcome.FAIL and result.check is not None
        rows = {f.field: (f.expected, f.actual) for f in result.check.field_checks}
        assert rows["Paneer Wrap quantity"] == ("1", "2")
    finally:
        backend_server.stop()
        httpd.shutdown()


def test_cli_runs_against_the_shim(tmp_path: Path):
    shim = load_shim()
    httpd = shim.serve(port=0)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/iob"
        result = CliRunner().invoke(
            app,
            [
                "run",
                "starter",
                "--agent",
                f"http:{url}",
                "--id",
                "en_quantity_01",
                "--id",
                "hien_cancellation_01",
                "--out",
                str(tmp_path / "o"),
                "--fail-under",
                "1.0",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        suite = read_json(tmp_path / "o" / "results.json")
        assert suite.metrics.pass_rate == 1.0 and "sandbox backend at" in result.output
    finally:
        httpd.shutdown()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_cli_backend_port_and_advertised_url_reach_the_agent(tmp_path: Path):
    from tests.test_http import TurnServer, order_handler

    port = _free_port()
    advertised = f"http://127.0.0.1:{port}"
    with TurnServer(order_handler) as turns:
        result = CliRunner().invoke(
            app,
            [
                "run",
                str(REPO / "tests" / "fixtures" / "minipack"),
                "--agent",
                f"http:{turns.url}",
                "--id",
                "en_quantity_01",
                "--backend-port",
                str(port),
                "--backend-url",
                advertised,
                "--out",
                str(tmp_path / "o"),
                "--fail-under",
                "1.0",
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    starts = [e for e in turns.events if e["event"] == "start"]
    assert [e["backend_url"] for e in starts] == [advertised]
    assert read_json(tmp_path / "o" / "results.json").metrics.pass_rate == 1.0


def test_cli_advertises_backend_url_different_from_bind_address(tmp_path: Path):
    """Bind by name (localhost), advertise by address: the agent gets the advertised URL."""
    from tests.test_http import TurnServer, order_handler

    port = _free_port()
    advertised = f"http://127.0.0.1:{port}"
    with TurnServer(order_handler) as turns:
        result = CliRunner().invoke(
            app,
            [
                "run",
                str(REPO / "tests" / "fixtures" / "minipack"),
                "--agent",
                f"http:{turns.url}",
                "--id",
                "en_quantity_01",
                "--backend-host",
                "localhost",
                "--backend-port",
                str(port),
                "--backend-url",
                advertised,
                "--out",
                str(tmp_path / "o"),
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert [e["backend_url"] for e in turns.events if e["event"] == "start"] == [advertised]
    assert f"http://localhost:{port}" in result.output and advertised in result.output


def test_cli_rejects_a_backend_url_without_scheme(tmp_path: Path):
    result = CliRunner().invoke(
        app,
        [
            "run",
            "starter",
            "--agent",
            "http:http://127.0.0.1:9/iob",
            "--backend-url",
            "127.0.0.1:8765",
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code == 2 and "--backend-url" in result.output
    assert "must start with http:// or https://" in result.output
