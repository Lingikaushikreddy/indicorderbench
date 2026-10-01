# IndicOrderBench v0.1 — Design Spec

Date: 2026-10-01
Status: approved for implementation (built autonomously on the owner's instruction; review after the fact)

## 1. Purpose

Voice ordering agents sound convincing and still submit the wrong order. IndicOrderBench
drives an agent through realistic spoken ordering scenarios in English and Hinglish and
answers one question deterministically: **did the committed order match what the caller
asked for?**

The deliverable for v0.1 is a `pip install`-able Python package plus a starter scenario pack.
A developer connects their agent through an adapter, runs the suite, and gets:

- expected versus actual cart, field by field;
- transcript, per-turn latency and a timestamped tool-call trace;
- failures grouped by quantity, modifier, correction, cancellation and duplicate submission;
- a comparison against a previous run;
- JSON, JUnit and a single-file HTML report;
- a non-zero exit code in CI when ordering accuracy regresses.

Framing used in docs and README: **"τ-bench for voice ordering, in Indian languages."**
The end-state comparison and the pass^k metric follow τ-bench. Simulator validation as a
separate outcome follows EVA-Bench. Both are cited in `docs/methodology.md`.

## 2. Scope of v0.1

| Area | In scope | Out of scope (documented as next) |
|---|---|---|
| Languages | English (en-IN) control, Hinglish (hi-en, romanised) | Telugu, Tamil, Devanagari script packs |
| Scenarios | 40 scenarios, 20 per language, 8 per failure category, `review_status: unreviewed` | Native-speaker reviewed status |
| Caller | Scripted caller with clarification rules; audio clips optional | Generative LLM caller |
| Agents under test | In-process Python agent; any agent reachable by an HTTP turn endpoint | LiveKit / Pipecat adapters |
| Reference agent | Rule-based, no API keys, switchable bugs | LLM reference agent |
| Audio | Sarvam bulbul TTS synth, Sarvam saaras STT, oracle transcriber, perturbations (noise, gain, telephone band) | Barge-in, full-duplex timing |
| Scoring | Deterministic end-state checker, pass rate, pass^k, Wilson CI | LLM-written failure explanations |
| Reports | JSON, JUnit XML, HTML, compare with baseline | Hosted dashboard |
| CI | GitHub Actions for the repo; template workflow for users | — |
| Demo | `iob demo` regenerates the buggy-vs-fixed report with no keys | Video |

The runner is **turn-based**: the caller speaks, the agent replies, repeat. That is an honest
limitation stated in the methodology doc. It is enough to measure order correctness; it does
not measure barge-in handling.

## 3. Architecture

```
packs/starter/                    scenario pack (YAML): menu, scenarios, clips, manifest
        │
        ▼
Pack loader ──► Scenario ──► Runner ──► ScriptedCaller ◄──► AgentAdapter ◄──► agent under test
                                │                                │
                                │                                ▼
                                │                         OrderBackend (sandbox)
                                │                          cart, orders, tool trace
                                ▼
                            Checker ──► TrialResult ──► SuiteResult ──► JSON / JUnit / HTML / compare
```

Package layout (`src/indicorderbench/`):

| Module | Responsibility | Depends on |
|---|---|---|
| `schemas/menu.py` | `Menu`, `MenuItem`, `ModifierGroup`, `Modifier` | pydantic |
| `schemas/scenario.py` | `Scenario`, `CallerScript`, `CallerTurn`, `ClarificationRule`, `EndState`, `ExpectedOrder`, `ExpectedLine`, `Pack`, enums | menu |
| `schemas/results.py` | `ToolCall`, `CartLine`, `SubmittedOrder`, `BackendSnapshot`, `FieldCheck`, `CheckResult`, `TrialResult`, `ScenarioResult`, `SuiteResult`, `Outcome`, `TurnRecord` | menu, scenario |
| `backend/state.py` | `OrderBackend`: menu lookup, cart ops, submit, cancel, trace, `tool_specs()` | schemas |
| `backend/http.py` | stdlib HTTP server exposing `OrderBackend` per session for remote agents | backend |
| `checker/canonical.py` | canonicalise expected and actual orders (resolve modifier defaults, merge lines) | schemas |
| `checker/checker.py` | `check(expected_end_states, snapshot) -> CheckResult` with field checks | canonical |
| `caller/scripted.py` | `ScriptedCaller`: state machine over `CallerScript` | schemas |
| `adapters/protocol.py` | `AgentAdapter`, `CallerUtterance`, `AgentReply`, `SessionInfo`, `Modality` | schemas |
| `adapters/inprocess.py` | `InProcessAdapter` wrapping a Python `Agent` factory | protocol, backend |
| `adapters/http.py` | `HttpTurnAdapter`: POST each turn to an agent URL | protocol, backend/http |
| `agents/lexicon.py` | shared phrase inventory (numbers, intents, connectors) for en-IN and hi-en | — |
| `agents/parsing.py` | utterance → list of `Clause` (intent, item, quantity, modifiers) | lexicon, menu |
| `agents/rule_based.py` | `RuleBasedAgent(backend, bugs)` reference agent | parsing, backend |
| `runner/runner.py` | `run_suite(...)`, `run_trial(...)`, outcome classification, timeouts | caller, adapters, checker |
| `runner/stats.py` | pass rate, pass^k, Wilson CI, latency percentiles, grouping | results |
| `report/json_report.py` | write/read `SuiteResult` JSON with `schema_version` | results |
| `report/junit.py` | JUnit XML | results |
| `report/html.py` + `templates/report.html.j2` | single-file HTML report, copies clips into `assets/` | results, jinja2 |
| `report/compare.py` | `compare(baseline, current, max_regression) -> Comparison` | results |
| `audio/perturb.py` | numpy perturbations on WAV | numpy, soundfile (extra `audio`) |
| `audio/tts.py` | `TTSProvider`, `SarvamTTS`, manifest writer | httpx (extra `audio`) |
| `audio/transcribe.py` | `Transcriber`, `OracleTranscriber`, `SarvamTranscriber` | httpx |
| `packs.py` | load/validate a pack directory | schemas, pyyaml |
| `cli.py` | Typer app `iob` | everything |

Each module has one job and is testable alone. Nothing in `schemas/` imports from elsewhere
in the package.

## 4. Data model

### 4.1 Menu

```yaml
id: starter-qsr
name: Starter QSR menu
currency: INR
modifier_groups:
  - id: onion
    name: Onion
    exclusive: true
    default: with_onion
    options:
      - {id: with_onion, name: With onion, aliases: [with onion, pyaaz ke saath]}
      - {id: no_onion,   name: No onion,   aliases: [no onion, without onion, bina pyaaz, pyaaz nahi, onion mat dalna, pyaaz mat]}
  - id: spice
    name: Spice level
    exclusive: true
    default: medium
    options: [{id: mild, ...}, {id: medium, ...}, {id: spicy, aliases: [spicy, extra spicy, teekha, zyada teekha]}]
  - id: extras
    name: Extras
    exclusive: false
    default: null
    options: [{id: extra_cheese, aliases: [extra cheese, cheese zyada]}, ...]
items:
  - id: paneer_wrap
    name: Paneer Wrap
    aliases: [paneer wrap, paneer wraps, paneer ka wrap, paneer roll, paneer kathi roll]
    price: 180
    category: wraps
    modifier_groups: [onion, spice, extras]
  - id: mango_lassi
    name: Mango Lassi
    aliases: [mango lassi, aam lassi, mango lassie]
    price: 90
    category: drinks
    modifier_groups: []
```

Rules: ids are `snake_case`, unique across items and across options. Aliases are matched
case-insensitively after whitespace and punctuation normalisation. Alias lookup returns the
longest matching alias. Items declare which groups apply. An exclusive group has exactly one
selected option; when none is selected the default applies. A non-exclusive group holds a set
of options and its default is the empty set.

### 4.2 Order state and tool trace

```python
class CartLine:      line_id: str; item_id: str; quantity: int (>=1); modifiers: set[str]; note: str | None
class SubmittedOrder: order_id: str; lines: list[CartLine]; submitted_at_ms: float; status: Literal["submitted", "cancelled"]
class ToolCall:      seq: int; t_ms: float; name: str; args: dict; result: Any | None; error: str | None
class BackendSnapshot: cart: list[CartLine]; orders: list[SubmittedOrder]; trace: list[ToolCall]
```

`OrderBackend(menu)` methods, every one appended to the trace with wall-clock ms relative to
session start:

| Tool | Args | Behaviour |
|---|---|---|
| `lookup_menu` | `query: str` | items whose name or alias contains the query; empty list if none |
| `add_item` | `item_id, quantity=1, modifiers=[]` | validates item and modifiers (option must belong to a group applicable to the item; exclusive group may carry one option); returns the new `CartLine` |
| `update_line` | `line_id, quantity=None, modifiers=None` | quantity 0 removes the line; `modifiers` replaces the set |
| `remove_line` | `line_id` | |
| `clear_cart` | | |
| `get_cart` | | current lines |
| `submit_order` | | requires a non-empty cart; creates a `SubmittedOrder`, empties the cart, returns it |
| `cancel_order` | `order_id` | marks the order cancelled; error if unknown or already cancelled |
| `list_orders` | | all orders with status |

Invalid calls raise `BackendError(code, message)`; the trace records the error and the state is
unchanged. `OrderBackend.tool_specs()` returns the tools as JSON-schema function definitions
so an LLM agent can bind them directly. `OrderBackend.snapshot()` returns a deep copy.

### 4.3 Scenario

```yaml
id: hien_correction_03
title: Two wraps corrected to one, no onion, add a lassi
language: hi-en            # en-IN | hi-en
category: correction       # quantity | modifier | correction | cancellation | duplicate_submission
menu: starter-qsr
difficulty: medium         # easy | medium | hard
tags: [smoke, mid-order-correction]
review_status: unreviewed  # unreviewed | reviewed
reviewer_notes: null
caller:
  turns:
    - id: t1
      text: "Do paneer wrap... actually ek hi karo. Bina pyaaz. Aur ek mango lassi."
      audio: null          # optional, relative to pack root; resolved by convention clips/<scenario>/<turn>.wav
  clarifications:
    - id: how_many
      match: ["how many", "kitne", "ek ya do", "one or two"]
      reply: {text: "Ek. Sirf ek paneer wrap."}
      max_uses: 2
    - id: onion
      match: ["onion", "pyaaz"]
      reply: {text: "Bina pyaaz."}
      max_uses: 1
  # closing / confirm / fallback inherit from pack defaults unless overridden
expected:
  - id: single_order
    description: One active order with one wrap without onion and one lassi
    orders:
      - lines:
          - {item_id: paneer_wrap, quantity: 1, modifiers: {onion: no_onion}}
          - {item_id: mango_lassi, quantity: 1}
```

Pack-level `pack.yaml` carries `id`, `name`, `version`, `menu: menu.yaml`, and
`defaults.caller.<language>` with, per language: `closing`, `confirm`, `fallback` and `nudge`
turns, `confirm_patterns` (agent asks whether to place the order), `goodbye_patterns` (agent
signals the call is over), `question_patterns` (agent reply is a question), `max_fallbacks`,
`max_nudges`, and `clarifications` shared by all scenarios of that language. Scenario-level
rules are checked before pack defaults. A scenario may override any of these fields.

Semantics of `expected`:

- `expected` lists **acceptable end states**. The trial passes if the committed state matches
  any one of them.
- `orders: []` means no active order may exist (cancellation scenarios).
- `ExpectedLine.modifiers` maps `group_id` to an option id, a list of option ids (non-exclusive
  groups) or `"*"` (any value accepted). Groups applicable to the item but not mentioned must
  equal the group default. An agent that adds "extra spicy" unasked fails.
- Only orders with `status == "submitted"` count. A cancelled order is not committed.

### 4.4 Caller script semantics

`ScriptedCaller` is a state machine. Given the agent's latest utterance it returns the next
`CallerUtterance` or signals the end of the call.

1. The first move is `turns[0]`. A scripted turn may carry `is_closing: true`, which tells the
   caller that this turn already said "that's all" so the default closing turn is not added.
2. After each agent reply, in this order:
   a. If the script still has turns: if a scenario or pack clarification rule matches the
      agent reply (regex, case-insensitive, searched anywhere in the text) and has uses left,
      reply with it. Otherwise speak the next scripted turn. Scripts are written so that the
      next turn is a sensible continuation even after an unmatched question.
   b. If the script is exhausted: if a clarification rule matches, use it; else if the
      pack `confirm_patterns` match ("confirm", "place the order", "shall I", "order kar",
      "theek hai?"), speak the confirm turn (once; later matches fall through); else if the
      closing turn has not been spoken, speak it; else decide by whether the agent reply is a
      **question** (`?` present, or a `question_patterns` match such as "which", "what",
      "how many", "kitne", "konsa", "kya"):
      - question: speak the `fallback` turn (a generic affirmative), up to `max_fallbacks`
        (default 2). When fallbacks are exhausted the caller stops and marks itself
        **invalid**: the agent asked something the script could not answer, so the agent's
        correctness is unknown.
      - not a question: speak the `nudge` turn ("Please place the order" / "Order place kar
        do"), up to `max_nudges` (default 2). When nudges are exhausted the caller stops and
        remains **valid**: the agent had every chance to submit and did not, which the
        checker scores as a failure.
3. The runner ends the call when the script is exhausted, the closing has been spoken, and
   either an active order exists or the agent's reply matches the pack `goodbye_patterns`
   ("order placed", "order place ho gaya", "thank you for ordering", "bye"). It also ends on
   `max_turns` (default 20), which counts as agent failure, not simulator invalid, because the
   caller was still able to respond.
4. The caller's own validity is independent of the agent's result. A caller that answered
   every question from the script or rules is valid even if the agent never submitted.

### 4.5 Outcomes

```python
class Outcome(str, Enum): PASS, FAIL, SIMULATOR_INVALID, INFRA_ERROR
```

Priority when several apply: `INFRA_ERROR` > `SIMULATOR_INVALID` > `PASS`/`FAIL`.

- `INFRA_ERROR`: adapter raised, per-turn or per-trial timeout, referenced audio clip missing
  in audio modality, backend server failure. The error message and traceback summary go in
  `TrialResult.error`.
- `SIMULATOR_INVALID`: caller marked itself invalid (4.4 step 2c).
- `PASS` / `FAIL`: from the checker.

Suite-level metrics are computed over valid trials only (`PASS` + `FAIL`); invalid and infra
counts are always reported alongside, never hidden.

### 4.6 Checker

`check(expected: list[EndState], snapshot: BackendSnapshot, menu: Menu) -> CheckResult`

- Canonical actual: active orders -> each order's lines merged by `(item_id, resolved
  modifier map)` with quantities summed; lines with quantity 0 dropped; sorted by `item_id`
  then modifier map. The resolved modifier map covers every group applicable to the item with
  the explicit option or the group default.
- Canonical expected: same shape; `"*"` is a wildcard matched against any resolved value.
- An end state matches when the number of active orders is equal and the canonical orders
  match as a multiset.
- `CheckResult.passed` is true if any end state matches. `best_end_state_id` is the end state
  with the fewest failing field checks (ties: first declared).
- `field_checks` for the best end state, in this order, each `FieldCheck(field, expected,
  actual, passed)`:
  1. `Submitted orders` (active order count).
  2. For each expected line: `<Item name> quantity`.
  3. For each explicitly expected modifier group of that line: `<Item name> · <Group name>`.
  4. For each actual line with no expected counterpart: `<Item name> quantity` expected `0`.
- For an end state with `orders: []` only rule 1 and rule 4 apply.
- Field checks are what the HTML report renders as the demo table.

### 4.7 Statistics (`runner/stats.py`)

- Per scenario: `n_valid`, `n_pass`, `pass_rate = n_pass / n_valid` (None when `n_valid == 0`).
- Suite `pass_rate`: scenario-weighted mean of per-scenario pass rates over scenarios with
  `n_valid > 0`.
- `pass^k` for `k <= min n_valid`: mean over scenarios of `C(n_pass, k) / C(n_valid, k)`
  (τ-bench estimator). Reported for k = 1 .. trials.
- Grouped (by language, by category): pooled trial-level pass rate with Wilson 95% interval.
- Latency: per-turn agent reply latency in ms, p50 and p95 per suite and per scenario.
- Counts: `simulator_invalid`, `infra_error`, `max_turns_hit`.

### 4.8 Adapters

```python
class Modality(str, Enum): TEXT, AUDIO, BOTH
@dataclass class CallerUtterance: turn_id, text: str | None, audio_path: Path | None, audio_bytes: bytes | None, language: str
@dataclass class AgentReply: text: str; audio_bytes: bytes | None = None; meta: dict = {}
@dataclass class SessionInfo: scenario_id, trial, language, modality, backend: OrderBackend, backend_url: str | None, session_id: str

class AgentAdapter(Protocol):
    async def start(self, session: SessionInfo) -> None
    async def respond(self, utterance: CallerUtterance) -> AgentReply
    async def stop(self) -> None
```

- Modality `TEXT` sends text only; `AUDIO` sends audio only and requires a clip for every turn
  spoken (missing clip -> `INFRA_ERROR`); `BOTH` sends both.
- `InProcessAdapter(agent_factory)`: `agent_factory(backend, session) -> Agent` where
  `Agent.handle(utterance) -> AgentReply` (sync or async).
- `HttpTurnAdapter(url)`: on `start`, the runner has already started `backend/http.py` and the
  adapter POSTs `{"event": "start", "session_id", "scenario_id", "backend_url", "language",
  "modality"}`. Each turn POSTs `{"event": "turn", "session_id", "turn_id", "text",
  "audio_b64", "audio_format": "wav", "language"}` and expects `{"text": str,
  "audio_b64": str | null, "meta": {}}`. `stop` POSTs `{"event": "stop", "session_id"}`.
  Timeouts are per request.
- `backend/http.py` serves `POST /sessions/{id}/tools/{tool_name}` with a JSON args body and
  returns `{"ok": true, "result": ...}` or `{"ok": false, "error": {"code", "message"}}`, plus
  `GET /sessions/{id}/menu`. Sessions are created by the runner, not by agents.

### 4.9 Reference agent (`agents/rule_based.py`)

Deterministic, no network. Handles the starter pack's full phrase inventory (`agents/lexicon.py`
is the single source of truth and the scenario author must stay inside it). Behaviour:

- Parses each caller utterance into clauses: `add(item, qty, modifiers)`,
  `correct_quantity(qty)`, `correct_modifier(modifiers)`, `remove(item | last)`,
  `cancel_order`, `closing`, `confirm`, `unknown`.
- Keeps `last_line_id` so "actually make it one" applies to the most recent line.
- Replies in the caller's language style with a readback of what it did, then "Anything
  else?" / "Aur kuch?".
- On closing: reads back the full cart and submits in the same turn, replying "Order placed."
  / "Order place ho gaya." On cancel: cancels the active order (or clears the cart if nothing
  is submitted).
- Unknown clause: asks "Sorry, which item?" / "Konsa item?" (a clarification the scripts can
  answer).

`bugs: frozenset[str]` toggles realistic failures. Each bug must break only its category:

| Bug | Effect | Category it should fail |
|---|---|---|
| `ignore_corrections` | readback confirms the correction verbally but the cart keeps the first value | correction |
| `drop_modifiers` | readback mentions the modifier, cart line omits it | modifier |
| `double_submit` | submits twice on closing | duplicate_submission |
| `ignore_cancellation` | says "cancelled" but leaves the order active | cancellation |
| `quantity_default_one` | ignores spoken quantities above one and adds one | quantity |

Agent configs exposed to the CLI: `builtin:correct`, `builtin:buggy` (all bugs),
`builtin:buggy:<bug1,bug2>`.

### 4.10 Starter pack

`packs/starter/`: `pack.yaml`, `menu.yaml`, `scenarios/<id>.yaml` × 40, `clips/` (generated,
git-ignored except `clips/manifest.json`), `REVIEW.md` (native-speaker review checklist).

Distribution: 5 categories × 2 languages × 4 scenarios = 40. Each category covers at least:
quantity (number words, "a"/"ek", quantities above two, two items with different quantities);
modifier (exclusion, addition, non-exclusive extras, modifier on one of two lines); correction
(quantity correction mid-turn, item swap, modifier correction after readback, correction after
agent asks); cancellation (remove one line, cancel whole order before submit, cancel after
submit, cancel then re-order); duplicate submission (closing phrased twice, "haan confirm"
after order placed, agent asked to repeat the order back, long pause phrasing). Scenario
tags include `smoke` for a 10-scenario quick suite (one per category per language).

Menu: 12 items across wraps, mains, sides, drinks, desserts with Hinglish aliases.

Bug isolation contract, enforced by an integration test: every scenario passes with
`builtin:correct`; with exactly one bug enabled, every scenario in that bug's category fails,
and every `smoke` scenario outside that category still passes. Scenario authors keep smoke
scenarios free of behaviour from other categories (a smoke quantity scenario contains no
correction, for example) so the contract is satisfiable.

### 4.11 Reports

- JSON: `SuiteResult` with `schema_version: 1`, `benchmark_version`, `pack` (id, version,
  content hash), `agent` (label, config string), `run` (started_at, trials, modality, seed,
  host), scenarios, metrics, and per-trial transcripts and traces. Readable back for compare.
- JUnit: one `testcase` per scenario; `FAIL` -> `<failure>` with the failing field checks;
  `SIMULATOR_INVALID` -> `<skipped>`; `INFRA_ERROR` -> `<error>`. A scenario with mixed trial
  outcomes is a failure if any valid trial failed.
- HTML: one file plus `assets/` for copied clips. Sections: header (pack, agent, run info);
  summary cards (pass rate, pass^k for k=trials, invalid, infra, p50/p95 latency); tables by
  language and by category with Wilson intervals; baseline comparison table when a baseline is
  given (per-scenario delta, regressions highlighted, new/removed scenarios listed); scenario
  list with outcome chips per trial; expandable scenario detail with the field-check table,
  transcript with latency per turn and `<audio>` players where clips exist, and the tool-call
  trace. No external assets, no JavaScript dependencies beyond inline toggles. Works from
  `file://`.
- Compare: `compare(baseline, current, max_regression)` returns per-scenario deltas, suite
  delta and a `regressed: bool` that the CLI turns into exit code 2.

### 4.12 CLI (`iob`)

| Command | Purpose |
|---|---|
| `iob validate PACK` | schema and referential integrity; clips present when referenced |
| `iob ls PACK [--tag] [--language] [--category]` | list scenarios |
| `iob run PACK --agent SPEC [--trials N] [--modality text|audio|both] [--tag T] [--language L] [--category C] [--out DIR] [--baseline results.json] [--fail-under 0.9] [--max-regression 0.05] [--seed S] [--timeout-turn 30] [--timeout-trial 300]` | run the suite; writes `results.json`, `junit.xml`, `report.html` into `--out` |
| `iob report results.json --out DIR` | regenerate HTML from JSON |
| `iob compare BASE CURRENT [--max-regression]` | print regression table; exit 2 on regression |
| `iob demo --out DIR` | run `builtin:buggy` then `builtin:correct` on the smoke tag and write a comparison report |
| `iob synth PACK --provider sarvam [--voice ishita] [--only-missing]` | generate clips and the manifest |
| `iob perturb PACK --snr 10 --telephone --out clips-noisy` | derive perturbed clips |
| `iob serve-backend --port 8765` | stand-alone sandbox backend for manual agent development |

Agent `SPEC` grammar: `builtin:correct`, `builtin:buggy[:bug,bug]`, `http:<url>`,
`python:<module>:<factory>`.

Exit codes: 0 ok, 1 usage or infra error, 2 accuracy below `--fail-under` or regression.

### 4.13 Audio

- `perturb.py`: pure functions on `(samples: np.ndarray, sr: int)`: `add_noise(snr_db,
  kind="white"|"pink", seed)`, `gain(db)`, `telephone()` (band-pass 300–3400 Hz via FFT mask,
  resample to 8 kHz and back), `normalise_peak()`. CLI composes them.
- `tts.py`: `TTSProvider.synthesize(text, language, voice) -> bytes (wav)`. `SarvamTTS` posts
  `{"inputs": [text], "target_language_code": "hi-IN" | "en-IN", "speaker": voice, "model":
  "bulbul:v3"}` to `https://api.sarvam.ai/text-to-speech` with header `API-Subscription-Key`
  and decodes `audios[0]` from base64. Hinglish text is synthesised with `hi-IN`. Manifest
  `clips/manifest.json` records per clip: path, sha256, scenario, turn, text, language,
  provider, voice, model, created_at.
- `transcribe.py`: `Transcriber.transcribe(audio_bytes, language) -> str`.
  `OracleTranscriber(manifest)` returns the manifest text for the clip hash (lets the audio
  pipeline be tested without STT). `SarvamTranscriber` posts multipart `file`, `model:
  saaras:v3`, optional `language_code` to `/speech-to-text` and returns `transcript`.
- The reference agent accepts a `Transcriber`; in audio modality it transcribes before parsing.

## 5. Error handling

- Pack loading errors list every problem with file and field, then exit 1.
- Backend errors never crash the runner; they are trace entries the agent must handle.
- Adapter exceptions and timeouts are caught per trial and become `INFRA_ERROR`; the suite
  continues.
- The HTML report renders even when every trial is an infra error.
- Audio providers raise `AudioProviderError(status, body_excerpt)`; `iob synth` reports
  per-clip failures and continues, exiting 1 at the end if any failed.

## 6. Testing strategy

- Unit tests per module with no network and no audio providers.
- Checker: table-driven tests for every field-check rule, wildcard, defaults, multiset order
  matching, cancelled orders ignored.
- Caller: tests for each branch of 4.4, including the invalid path.
- Runner: tests with fake adapters for `PASS`, `FAIL`, `SIMULATOR_INVALID`, `INFRA_ERROR`
  (exception and timeout), `max_turns`.
- Integration: the full starter pack passes with `builtin:correct`; each single bug fails its
  category and only its category; the smoke tag has 10 scenarios.
- HTTP: `HttpTurnAdapter` against a local stdlib shim that proxies to the backend server.
- Reports: JUnit validates against expected structure; HTML contains the field-check table
  and audio tags for a clip fixture; compare flags a regression.
- CLI: Typer `CliRunner` for `validate`, `ls`, `run`, `compare`, `demo` exit codes.
- Audio: perturbation tests on synthetic signals (SNR within tolerance, band energy outside
  300–3400 Hz reduced); TTS and STT tests with a mocked HTTP transport.
- CI: ruff, mypy (strict on `schemas`, `checker`, `caller`, `runner`), pytest on 3.11–3.13.

## 7. Repository layout

```
indicorderbench/
  pyproject.toml           uv-managed; extras: audio, dev
  README.md                promise, demo table, install, quickstart, sample report
  LICENSE                  Apache-2.0
  CONTRIBUTING.md          contribution tasks: scenarios, language review, adapters, reports
  CHANGELOG.md
  .github/workflows/ci.yml
  src/indicorderbench/...
  packs/starter/...
  examples/http_agent_shim.py       minimal agent reachable by HttpTurnAdapter
  examples/ci/github-actions.yml    template for users
  docs/methodology.md      outcomes, metrics, validity rules, limitations, prior art
  docs/adapters.md         protocol and shim walkthrough
  docs/scenarios.md        authoring guide and review checklist
  docs/superpowers/specs/, docs/superpowers/plans/
  tests/
```

## 8. Non-goals and known limitations (also in README)

- Turn-based: no barge-in or overlapping speech measurement.
- Hinglish scenarios are unreviewed by native speakers until `review_status: reviewed`.
- No LLM agent, caller or explanation in v0.1; interfaces exist for all three.
- No cost accounting; adapters may attach cost in `AgentReply.meta` and the report shows it
  if present.
