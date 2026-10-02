# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.1] - 2026-10-01

### Added
- `iob run --clips <dir>` runs an audio suite on an alternate clip set such as `iob perturb`
  output. `results.json` records it (`pack.clips`), the HTML report shows it and
  `iob compare` flags runs on different clip sets. `iob perturb` prints the follow-up command.

### Fixed
- `iob --help` listed the compare command twice (`compare` and `compare-cmd`).
- `python:<module>:<factory>` agent specs now find a module in the working directory, as the
  README example implies; console scripts do not put it on `sys.path`.
- Sandbox backend: `lookup_menu` with a non-string `query` raised an untraced
  `AttributeError` (HTTP 500) instead of `invalid_args`; `update_line` accepted non-integer
  quantities (`2.5`, `true`), which then made `results.json` unreadable; `modifiers` must be
  a list of option-id strings, refused as `invalid_args` otherwise.
- `iob synth` crashed with an `AssertionError` on a scenario that overrides `closing`,
  `confirm`, `fallback` or `nudge` without an `id`. Such overrides are now synthesised to
  `clips/<scenario>/<id>.wav` and `Pack.clip_path` prefers that clip, so audio runs no
  longer fail with a missing clip or play the language default instead. The turn ids
  `closing`, `confirm`, `fallback` and `nudge` are reserved for those overrides.
- A text-modality run no longer records caller clips the agent was never sent, so the HTML
  report of a text run shows no audio players and copies no clips.

### Changed
- README: the reference-agent bugs are described accurately; a bug can also fail scenarios
  of another category that depend on the same behaviour.
- README rewritten for the public repository: comparison table, report screenshot.
- 0.1.0 published to PyPI on 2026-10-01 through the trusted-publishing release workflow.

## [0.1.0] - 2026-10-01

First release.

### Added
- Pydantic schemas for menus, scenarios with acceptable end states, and results.
- Sandbox order backend with a timestamped tool-call trace and JSON-schema tool definitions.
- Deterministic checker producing field-level expected/actual rows.
- Scripted caller with clarification rules, confirm/closing/fallback/nudge behaviour and
  simulator-validity tracking.
- Runner with per-turn and per-trial timeouts, outcome classification
  (pass / fail / simulator_invalid / infra_error), pass^k, Wilson intervals and latency
  percentiles.
- In-process and HTTP turn adapters; HTTP sandbox backend server.
- Rule-based reference agent with switchable bugs (`ignore_corrections`, `drop_modifiers`,
  `double_submit`, `ignore_cancellation`, `quantity_default_one`).
- Starter pack: 42 scenarios (21 English, 21 Hinglish) across five failure categories,
  including two `demo`-tagged correction scenarios; bundled in the wheel.
- JSON, JUnit and single-file HTML reports; baseline comparison with regression exit code.
- Audio: WAV helpers, noise/gain/telephone perturbations, Sarvam bulbul TTS synthesis,
  Sarvam saaras STT, oracle transcriber, clip manifest, and an offline placeholder TTS
  (`--provider silence`) so audio-modality runs work without API keys.
- `examples/http_agent_shim.py`: the reference agent behind the HTTP turn protocol.
- `iob` CLI: validate, ls, run, report, compare, demo, synth, perturb, serve-backend;
  `iob run` takes `--allow-infra`, `--backend-host`, `--backend-port` and `--backend-url`.
