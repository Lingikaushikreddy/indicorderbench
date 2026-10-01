"""The ``iob`` command line."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Annotated

import typer

from indicorderbench import __version__
from indicorderbench.adapters.protocol import Modality
from indicorderbench.agent_specs import AgentSpecError, build_adapter_factory
from indicorderbench.packs import Pack, PackError, load_pack, validate_pack
from indicorderbench.report.compare import Comparison, compare, render_comparison_text
from indicorderbench.report.formatting import ms, pct
from indicorderbench.report.html import write_html
from indicorderbench.report.json_report import ReportError, read_json, write_json
from indicorderbench.report.junit import write_junit
from indicorderbench.runner.runner import RunConfig, SessionRegistry, run_suite_sync
from indicorderbench.schemas.results import Outcome, ScenarioResult, SuiteResult

app = typer.Typer(
    help="IndicOrderBench: catches multilingual voice agents placing the wrong order.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)

EXIT_OK, EXIT_ERROR, EXIT_THRESHOLD = 0, 1, 2
OUTCOME_MARK = {
    Outcome.PASS: "ok  ",
    Outcome.FAIL: "FAIL",
    Outcome.SIMULATOR_INVALID: "inv ",
    Outcome.INFRA_ERROR: "err ",
}


def _fail(message: str, code: int = EXIT_ERROR) -> typer.Exit:
    typer.echo(message, err=False)
    return typer.Exit(code)


def _bundled_pack(name: str) -> Path | None:
    """A pack shipped inside the installed package (``packs/starter`` in the wheel)."""
    try:
        root = resources.files("indicorderbench").joinpath("data", "packs", name)
    except ModuleNotFoundError:
        return None
    path = Path(str(root))
    return path if path.is_dir() else None


def _resolve_pack_dir(pack: str) -> Path:
    candidate = Path(pack)
    if candidate.is_dir():
        return candidate
    repo_pack = Path(__file__).resolve().parents[2] / "packs" / pack
    if repo_pack.is_dir():
        return repo_pack
    bundled = _bundled_pack(pack)
    if bundled is not None:
        return bundled
    raise _fail(f"pack {pack!r} is neither a directory nor a bundled pack name (try 'starter')")


def _load(pack: str) -> Pack:
    try:
        return load_pack(_resolve_pack_dir(pack))
    except PackError as e:
        typer.echo(f"pack is not valid ({len(e.problems)} problem(s)):")
        for problem in e.problems:
            typer.echo(f"  - {problem}")
        raise typer.Exit(EXIT_ERROR) from e


def _select(
    pack: Pack,
    tags: list[str] | None,
    languages: list[str] | None,
    categories: list[str] | None,
    ids: list[str] | None,
) -> list:  # type: ignore[type-arg]
    scenarios = pack.filter(tags=tags, languages=languages, categories=categories, ids=ids)
    if not scenarios:
        raise _fail("no scenarios match the given filters")
    return scenarios


def _print_summary(suite: SuiteResult) -> None:
    m = suite.metrics
    available = [k for k, v in m.pass_k.items() if v is not None]
    k = max(available) if available else suite.run.trials
    typer.echo("")
    typer.echo(f"pack {suite.pack.id} v{suite.pack.version} ({suite.pack.content_hash})")
    typer.echo(
        f"agent {suite.agent.label}   trials {suite.run.trials}   modality {suite.run.modality}"
    )
    typer.echo(f"pass rate      {pct(m.pass_rate):>7}   (scenario-weighted, valid trials only)")
    typer.echo(f"pass^{k:<9} {pct(m.pass_k.get(k)):>7}")
    typer.echo(f"valid / pass / fail   {m.n_valid} / {m.n_pass} / {m.n_fail}")
    typer.echo(f"simulator invalid     {m.n_simulator_invalid}")
    typer.echo(f"infra errors          {m.n_infra_error}")
    typer.echo(f"latency p50 / p95     {ms(m.latency_p50_ms)} / {ms(m.latency_p95_ms)}")
    for title, groups in (("by language", m.by_language), ("by category", m.by_category)):
        typer.echo(title)
        for g in groups:
            ci = f"[{pct(g.ci_low)} – {pct(g.ci_high)}]" if g.ci_low is not None else ""
            typer.echo(f"  {g.key:<22} {pct(g.pass_rate):>7}  n={g.n_trials:<3} {ci}")


def _progress(sr: ScenarioResult) -> None:
    marks = " ".join(OUTCOME_MARK[t.outcome].strip() for t in sr.trials)
    typer.echo(f"{OUTCOME_MARK[sr.worst_outcome]} {sr.scenario_id:<32} {marks}")


def _run_suite(
    pack: Pack,
    scenarios: list,  # type: ignore[type-arg]
    agent: str,
    config: RunConfig,
    backend_host: str = "127.0.0.1",
    backend_port: int = 0,
    advertised_backend_url: str | None = None,
) -> SuiteResult:
    """Run the suite; for HTTP agents also serve the sandbox backend.

    The backend binds ``backend_host:backend_port`` (port 0 picks a free port) and the agent
    is told ``advertised_backend_url`` in its ``start`` event, or the server's own URL when
    that is None.
    """
    try:
        setup = build_adapter_factory(
            agent, modality=config.modality, pack_root=pack.root, timeout_s=config.timeout_turn_s
        )
    except AgentSpecError as e:
        raise typer.BadParameter(str(e), param_hint="--agent") from e
    config.agent_label = setup.label
    config.agent_spec = agent
    registry: SessionRegistry | None = None
    backend_url: str | None = None
    server = None
    if setup.needs_backend_server:
        from indicorderbench.backend.http import BackendServer

        registry = SessionRegistry()
        server = BackendServer(registry, host=backend_host, port=backend_port)
        try:
            server.start()
        except OSError as e:
            raise _fail(
                f"cannot serve the sandbox backend on {backend_host}:{backend_port}: {e}"
            ) from e
        backend_url = advertised_backend_url or server.url
        if backend_url == server.url:
            typer.echo(f"sandbox backend at {backend_url}")
        else:
            typer.echo(f"sandbox backend at {server.url}, advertised to the agent as {backend_url}")
    try:
        return run_suite_sync(
            pack, scenarios, setup.factory, config, backend_url, registry, on_scenario=_progress
        )
    finally:
        if server is not None:
            server.stop()


def _write_outputs(
    suite: SuiteResult, out_dir: Path, baseline_path: Path | None, max_regression: float
) -> Comparison | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(suite, out_dir / "results.json")
    write_junit(suite, out_dir / "junit.xml")
    comparison: Comparison | None = None
    baseline: SuiteResult | None = None
    if baseline_path is not None:
        try:
            baseline = read_json(baseline_path)
            comparison = compare(baseline, suite, max_regression)
        except ReportError as e:
            raise _fail(f"cannot compare with baseline: {e}") from e
    write_html(suite, out_dir, baseline=baseline, comparison=comparison)
    typer.echo(f"wrote {out_dir / 'results.json'}, junit.xml and report.html")
    return comparison


def _gate(
    suite: SuiteResult,
    comparison: Comparison | None,
    fail_under: float | None,
    allow_infra: int = 0,
) -> int:
    """Exit code for a finished run: infra errors (1) take precedence over accuracy (2).

    The accuracy checks still print when the run also exits 1 for infra errors. A run where
    every trial was an infra error always exits 1, whatever ``allow_infra`` says.
    """
    m = suite.metrics
    if m.n_trials_total and m.n_infra_error == m.n_trials_total:
        typer.echo("every trial was an infra error; check the agent connection")
        return EXIT_ERROR
    code = EXIT_OK
    if fail_under is not None and (m.pass_rate is None or m.pass_rate < fail_under):
        typer.echo(f"pass rate {pct(m.pass_rate)} is below --fail-under {pct(fail_under)}")
        code = EXIT_THRESHOLD
    if comparison is not None:
        typer.echo("")
        typer.echo(render_comparison_text(comparison))
        if comparison.regressed:
            typer.echo("ordering accuracy regressed against the baseline")
            code = EXIT_THRESHOLD
    if m.n_infra_error > allow_infra:
        typer.echo(
            f"{m.n_infra_error} trial(s) hit infra errors (allowed: {allow_infra}); "
            "fix the agent connection"
        )
        return EXIT_ERROR
    return code


# --------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------


@app.callback()
def _main() -> None:
    """IndicOrderBench command line."""


@app.command()
def version() -> None:
    """Print the benchmark version."""
    typer.echo(__version__)


@app.command()
def validate(
    pack: Annotated[str, typer.Argument(help="pack directory or bundled pack name")],
) -> None:
    """Check a pack's schema and references; exit 1 and list every problem if invalid."""
    problems = validate_pack(_resolve_pack_dir(pack))
    if problems:
        typer.echo(f"{len(problems)} problem(s):")
        for p in problems:
            typer.echo(f"  - {p}")
        raise typer.Exit(EXIT_ERROR)
    loaded = load_pack(_resolve_pack_dir(pack))
    typer.echo(f"pack {loaded.manifest.id} is valid: {len(loaded.scenarios)} scenarios")


@app.command()
def ls(
    pack: Annotated[str, typer.Argument(help="pack directory or bundled pack name")],
    tag: Annotated[list[str] | None, typer.Option("--tag")] = None,
    language: Annotated[list[str] | None, typer.Option("--language")] = None,
    category: Annotated[list[str] | None, typer.Option("--category")] = None,
) -> None:
    """List scenarios, optionally filtered."""
    loaded = _load(pack)
    for s in _select(loaded, tag, language, category, None):
        typer.echo(
            f"{s.id:<34} {s.language.value:<6} {s.category.value:<21} "
            f"{s.difficulty.value:<7} {','.join(s.tags)}"
        )


@app.command()
def run(
    pack: Annotated[str, typer.Argument(help="pack directory or bundled pack name")],
    agent: Annotated[
        str,
        typer.Option(
            "--agent",
            help="builtin:correct | builtin:buggy[:bug,..] | http:<url> | python:<mod>:<factory>",
        ),
    ],
    trials: Annotated[int, typer.Option(min=1)] = 1,
    modality: Annotated[Modality, typer.Option()] = Modality.TEXT,
    tag: Annotated[list[str] | None, typer.Option("--tag")] = None,
    language: Annotated[list[str] | None, typer.Option("--language")] = None,
    category: Annotated[list[str] | None, typer.Option("--category")] = None,
    id: Annotated[list[str] | None, typer.Option("--id")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="output directory (default results/<timestamp>)")
    ] = None,
    baseline: Annotated[
        Path | None, typer.Option("--baseline", help="results.json of a previous run")
    ] = None,
    fail_under: Annotated[float | None, typer.Option("--fail-under", min=0.0, max=1.0)] = None,
    max_regression: Annotated[float, typer.Option("--max-regression", min=0.0, max=1.0)] = 0.0,
    seed: Annotated[int | None, typer.Option()] = None,
    timeout_turn: Annotated[
        float, typer.Option("--timeout-turn", help="seconds per agent reply")
    ] = 30.0,
    timeout_trial: Annotated[
        float, typer.Option("--timeout-trial", help="seconds per trial")
    ] = 300.0,
    max_turns: Annotated[int, typer.Option("--max-turns", min=1)] = 20,
    allow_infra: Annotated[
        int,
        typer.Option(
            "--allow-infra",
            min=0,
            help="infra-error trials tolerated before exiting 1",
        ),
    ] = 0,
    backend_host: Annotated[
        str,
        typer.Option(
            "--backend-host",
            help="address the sandbox backend binds for http: agents (0.0.0.0 for remote agents)",
        ),
    ] = "127.0.0.1",
    backend_port: Annotated[
        int,
        typer.Option(
            "--backend-port", min=0, max=65535, help="sandbox backend port (0 picks a free one)"
        ),
    ] = 0,
    backend_url: Annotated[
        str | None,
        typer.Option(
            "--backend-url",
            help="backend URL sent to http: agents in the start event "
            "(default: the bound server's own URL)",
        ),
    ] = None,
) -> None:
    """Run scenarios against an agent and write results.json, junit.xml and report.html."""
    if backend_url is not None and not backend_url.startswith(("http://", "https://")):
        raise typer.BadParameter(
            f"must start with http:// or https://, got {backend_url!r}",
            param_hint="--backend-url",
        )
    loaded = _load(pack)
    scenarios = _select(loaded, tag, language, category, id)
    config = RunConfig(
        trials=trials,
        modality=modality,
        seed=seed,
        timeout_turn_s=timeout_turn,
        timeout_trial_s=timeout_trial,
        max_turns=max_turns,
    )
    typer.echo(f"running {len(scenarios)} scenario(s) x {trials} trial(s) against {agent}")
    suite = _run_suite(
        loaded,
        scenarios,
        agent,
        config,
        backend_host=backend_host,
        backend_port=backend_port,
        advertised_backend_url=backend_url,
    )
    out_dir = out or Path("results") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    comparison = _write_outputs(suite, out_dir, baseline, max_regression)
    _print_summary(suite)
    raise typer.Exit(_gate(suite, comparison, fail_under, allow_infra))


@app.command()
def report(
    results: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out")],
    baseline: Annotated[Path | None, typer.Option("--baseline")] = None,
    max_regression: Annotated[float, typer.Option("--max-regression", min=0.0, max=1.0)] = 0.0,
) -> None:
    """Regenerate the HTML report from a results.json."""
    try:
        suite = read_json(results)
        base = read_json(baseline) if baseline else None
        comparison = compare(base, suite, max_regression) if base else None
    except ReportError as e:
        raise _fail(str(e)) from e
    path = write_html(suite, out, baseline=base, comparison=comparison)
    typer.echo(f"wrote {path}")


@app.command()
def compare_cmd(
    baseline: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    current: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    max_regression: Annotated[float, typer.Option("--max-regression", min=0.0, max=1.0)] = 0.0,
) -> None:
    """Compare two results.json files; exit 2 when ordering accuracy regressed."""
    try:
        c = compare(read_json(baseline), read_json(current), max_regression)
    except ReportError as e:
        raise _fail(str(e)) from e
    typer.echo(render_comparison_text(c))
    raise typer.Exit(EXIT_THRESHOLD if c.regressed else EXIT_OK)


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out")] = Path("demo-out"),
    pack: Annotated[str, typer.Option("--pack")] = "starter",
    trials: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Run the buggy and the fixed reference agents on the smoke suite and print the demo table.

    Needs no API keys. Writes <out>/buggy/ and <out>/fixed/ (the fixed run is compared
    against the buggy one) plus <out>/README.md.
    """
    loaded = _load(pack)
    scenarios = _select(loaded, ["smoke"], None, None, None)
    results: dict[str, SuiteResult] = {}
    for label in ("buggy", "fixed"):
        spec = "builtin:buggy" if label == "buggy" else "builtin:correct"
        typer.echo(f"\n== {label}: {spec} ==")
        config = RunConfig(trials=trials)
        suite = _run_suite(loaded, scenarios, spec, config)
        base = (out / "buggy" / "results.json") if label == "fixed" else None
        _write_outputs(suite, out / label, base, 0.0)
        results[label] = suite
    demo = next(
        (s for s in loaded.filter(tags=["demo"]) if s.language.value == "hi-en"),
        next(iter(loaded.filter(tags=["demo"])), scenarios[0]),
    )
    # The headline table comes from a single correction-only bug so the failure is the one
    # the pitch describes: the agent confirms the correction but commits the first quantity.
    typer.echo(f"\n== demo table: {demo.id} with builtin:buggy:ignore_corrections ==")
    headline = _run_suite(loaded, [demo], "builtin:buggy:ignore_corrections", RunConfig(trials=1))
    _write_outputs(headline, out / "demo-scenario", None, 0.0)
    trial = headline.scenarios[0].trials[0]
    for turn in trial.transcript:
        who = "caller" if turn.speaker == "caller" else "agent "
        typer.echo(f"  {who}: {turn.text}")
    if trial.check is not None:
        typer.echo("")
        typer.echo(f"  {'Field':<28} {'Expected':<12} {'Actual':<12} Result")
        for f in trial.check.field_checks:
            verdict = "Pass" if f.passed else "Fail"
            typer.echo(f"  {f.field:<28} {f.expected:<12} {f.actual:<12} {verdict}")
    for label in ("buggy", "fixed"):
        typer.echo(f"\n{label}: pass rate {pct(results[label].metrics.pass_rate)}")
    (out / "README.md").write_text(
        "# IndicOrderBench demo\n\n"
        "Two runs of the smoke suite with the built-in reference agent, no API keys:\n\n"
        "- `buggy/report.html`: every injected bug enabled (ignored corrections, dropped "
        "modifiers, double submission, ignored cancellation, quantity defaulting to one).\n"
        "- `fixed/report.html`: the correct agent, compared against the buggy run.\n"
        "- `demo-scenario/report.html`: the demo correction scenario with only the "
        "ignore_corrections bug, the run behind the headline table.\n\n"
        f"Regenerate with `iob demo --out {out}`.\n",
        encoding="utf-8",
    )
    typer.echo(f"\nreports: {out / 'buggy' / 'report.html'} and {out / 'fixed' / 'report.html'}")


@app.command()
def synth(
    pack: Annotated[str, typer.Argument(help="pack directory or bundled pack name")],
    provider: Annotated[str, typer.Option("--provider", help="sarvam | silence")] = "sarvam",
    voice: Annotated[str, typer.Option("--voice")] = "ishita",
    only_missing: Annotated[bool, typer.Option("--only-missing/--all")] = True,
    scenario: Annotated[list[str] | None, typer.Option("--scenario")] = None,
) -> None:
    """Generate caller audio clips and clips/manifest.json for a pack."""
    from indicorderbench.audio.tts import PlaceholderTTS, SarvamTTS, TTSProvider, synth_pack

    loaded = _load(pack)
    tts: TTSProvider
    if provider == "sarvam":
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise _fail("set SARVAM_API_KEY to use the sarvam provider (or use --provider silence)")
        tts = SarvamTTS(key)
    elif provider == "silence":
        tts = PlaceholderTTS()
    else:
        raise typer.BadParameter("expected 'sarvam' or 'silence'", param_hint="--provider")
    result = synth_pack(loaded, tts, voice, only_missing=only_missing, scenario_ids=scenario)
    typer.echo(
        f"written {len(result.written)}, skipped {len(result.skipped)}, failed {len(result.failed)}"
    )
    for rel, err in result.failed:
        typer.echo(f"  failed {rel}: {err}")
    if result.failed:
        raise typer.Exit(EXIT_ERROR)


@app.command()
def perturb(
    pack: Annotated[str, typer.Argument(help="pack directory or bundled pack name")],
    out: Annotated[Path, typer.Option("--out", help="directory for the derived clips")],
    snr: Annotated[float | None, typer.Option("--snr", help="noise SNR in dB")] = None,
    noise: Annotated[str, typer.Option("--noise", help="white | pink")] = "white",
    gain: Annotated[float, typer.Option("--gain", help="gain in dB")] = 0.0,
    telephone: Annotated[bool, typer.Option("--telephone")] = False,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
) -> None:
    """Derive perturbed copies of a pack's clips (needs the audio extra)."""
    from indicorderbench.audio.perturb import AudioExtraMissing, PerturbSpec, perturb_dir

    if noise not in ("white", "pink"):
        raise typer.BadParameter("expected 'white' or 'pink'", param_hint="--noise")
    loaded = _load(pack)
    src = loaded.root / "clips"
    if not src.is_dir():
        raise _fail(f"{src} does not exist; run `iob synth` first")
    spec = PerturbSpec(snr_db=snr, noise=noise, gain_db=gain, telephone=telephone, seed=seed)  # type: ignore[arg-type]
    try:
        written = perturb_dir(src, out, spec)
    except AudioExtraMissing as e:
        raise _fail(str(e)) from e
    typer.echo(f"wrote {len(written)} clip(s) to {out}")


@app.command(name="serve-backend")
def serve_backend(
    pack: Annotated[str, typer.Argument(help="pack directory or bundled pack name")] = "starter",
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8765,
    session: Annotated[str, typer.Option("--session", help="session id to serve")] = "dev",
) -> None:
    """Serve one long-lived sandbox session for manual agent development (Ctrl-C to stop)."""
    from indicorderbench.backend.http import BackendServer
    from indicorderbench.backend.state import OrderBackend

    loaded = _load(pack)
    registry = SessionRegistry()
    registry.register(session, OrderBackend(loaded.menu))
    server = BackendServer(registry, host=host, port=port)
    server.start()
    typer.echo(f"sandbox backend at {server.url}  session {session}")
    typer.echo(f"  curl {server.url}/sessions/{session}/menu")
    typer.echo(
        f"  curl -X POST {server.url}/sessions/{session}/tools/add_item "
        '-d \'{"item_id": "paneer_wrap", "quantity": 2}\''
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        typer.echo("stopped")


app.command(name="compare")(compare_cmd)


def main() -> None:  # pragma: no cover - console entry point
    sys.exit(app())
