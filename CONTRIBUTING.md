# Contributing

Thanks for helping make voice ordering agents measurably better. Small contributions are
the most useful ones right now.

## Good first contributions

| Area | What helps | Where |
|---|---|---|
| Language review | A native Hindi/Hinglish speaker reviewing the 20 Hinglish scenarios for naturalness and regional phrasing | `packs/starter/REVIEW.md`, set `review_status: reviewed` |
| New scenarios | Real failures you have seen an agent make, as a scenario with acceptable end states | `docs/scenarios.md` |
| New language packs | Telugu and Tamil are next; copy `packs/starter`, translate, add aliases, get a native review | `packs/<lang>/` |
| Adapters | LiveKit, Pipecat, Vapi, Retell: a turn endpoint or a native adapter | `docs/adapters.md`, `src/indicorderbench/adapters/` |
| Reports | Better HTML, a Markdown summary for PR comments, a JSON schema file | `src/indicorderbench/report/` |
| Reference agents | An LLM-backed reference agent using `OrderBackend.tool_specs()` | `src/indicorderbench/agents/` |

## Development setup

```bash
git clone https://github.com/Lingikaushikreddy/indicorderbench
cd indicorderbench
uv sync --all-extras --dev
uv run pytest -q
uv run iob demo --out demo-out
```

Before opening a pull request:

```bash
uv run ruff format src tests
uv run ruff check src tests
uv run mypy
uv run pytest -q
uv run iob validate packs/starter
```

CI runs the same commands on Python 3.11, 3.12 and 3.13.

## Pull request checklist

- One change per PR, with a test that failed before the change.
- Scenario changes keep `tests/test_starter_pack.py` green (the bug-isolation contract).
- No new core dependencies. Audio-only dependencies go under the `audio` extra.
- Public functions have type hints; `mypy --strict` passes.
- `CHANGELOG.md` gets a line under "Unreleased".

## Recordings and consent

If you contribute human recordings, include written consent from each speaker for
publication under this repository's license. Removing a name from a file does not grant
permission to publish it. Synthesised clips must name the provider, voice and model in the
manifest.

## Reporting results

If you publish numbers produced with this benchmark, follow `docs/methodology.md` section 9:
always include trial counts, invalid and infra counts, pack hash and agent configuration.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0.
