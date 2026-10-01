# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
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
