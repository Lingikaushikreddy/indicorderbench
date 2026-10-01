# Methodology

IndicOrderBench answers one question about a voice ordering agent: **did the order it
committed match what the caller asked for?** This document defines exactly what is
measured, how trials are classified, which numbers are reported, and what the benchmark
does not measure. Read it before publishing results.

## 1. What a trial is

A trial is one conversation between the scripted caller and the agent under test on one
scenario. The caller speaks its scripted turns (text, audio, or both), answers clarifying
questions it has rules for, says it is done, and then either confirms, nudges, or gives up.
The agent calls the sandbox order backend through tools: `lookup_menu`, `add_item`,
`update_line`, `remove_line`, `clear_cart`, `get_cart`, `submit_order`, `cancel_order`,
`list_orders`. Every call is recorded with a timestamp.

The runner is **turn-based**. The caller speaks, the agent replies, repeat. There is no
overlapping speech, so barge-in handling is not measured. A caller turn that contains a
mid-sentence correction ("two wraps... actually one") tests whether the agent applies the
correction, not whether it can be interrupted.

## 2. What counts as committed

Only orders with status `submitted` count. A cart that was never submitted is nothing. A
cancelled order is nothing. Two active orders with identical content are two orders, not one.
The checker never reads the agent's words, only the backend state.

## 3. Outcomes

Every trial gets exactly one outcome, assigned in this priority order:

| Outcome | Meaning | Counted in pass rate? |
|---|---|---|
| `infra_error` | The adapter raised, a turn or trial timed out, a required audio clip was missing, or the backend server failed. The agent was not given a fair run. | No |
| `simulator_invalid` | After the caller had finished its script and said it was done, the agent asked a question the script could not answer, and the generic fallbacks ran out. The agent's correctness is unknown. | No |
| `pass` | The committed state matches one of the scenario's acceptable end states. | Yes |
| `fail` | The caller was valid, the infrastructure worked, and the committed state matched no acceptable end state. This includes an agent that never submitted after being told the order was complete and nudged twice. | Yes |

Invalid and infra counts are always reported next to the pass rate. A result that hides them
is not a benchmark result.

### Why an agent that never submits is a failure, not "invalid"

The caller distinguishes between an agent that asks something unanswerable (the simulator
cannot continue, so the agent is not blamed) and an agent that simply does not place the
order after "that's all" and two nudges (the agent had every chance). The first is
`simulator_invalid`; the second is `fail`.

## 4. Acceptable end states

A scenario lists one or more acceptable end states. Each is a list of expected orders, each
order a list of lines with item id, quantity and modifier constraints. The trial passes if
any end state matches. This lets a scenario accept, for example, both "the agent applied the
correction" and "the agent asked and the caller confirmed" when both are legitimate.

Modifier semantics: a modifier group the scenario does not mention must be at its menu
default. An agent that adds "extra spicy" when nobody asked fails. `"*"` accepts any value.

## 5. Metrics

Let a scenario have `n` valid trials (pass or fail) and `c` passes.

- **Scenario pass rate** = `c / n`.
- **Suite pass rate** = mean of scenario pass rates over scenarios with `n > 0`
  (scenario-weighted, so a scenario with more valid trials does not dominate).
- **pass^k** (from τ-bench) = mean over scenarios of `C(c, k) / C(n, k)`: the probability
  that `k` randomly chosen trials of a scenario all pass. `pass^1` is the pass rate;
  `pass^3` punishes agents that are right two times out of three. Reported for `k = 1..trials`.
- **Grouped rates** (by language, by category) pool trials and report a **Wilson 95%
  interval**. With 40 scenarios and one trial each, intervals are wide; treat differences
  inside overlapping intervals as noise.
- **Latency** = wall-clock time from sending the caller's utterance to receiving the agent's
  reply, per turn, reported as p50 and p95. It includes the agent's STT, reasoning, tool calls
  and TTS, and the adapter's transport. It is not a model latency figure.
- **Cost** is not measured in v0.1. Adapters may attach cost in `AgentReply.meta` and the
  report shows it when present.

## 6. Reproducibility

`results.json` records the benchmark version, pack id, pack version and a content hash of
every YAML file in the pack, the agent label and spec string, trial count, modality, seed,
timeouts, max turns, host and Python version, and per trial the full transcript, tool trace
and backend snapshot. Two runs with the same pack hash and agent configuration are comparable;
`iob compare` refuses to compare runs from different packs.

Repeating audio does not guarantee an identical agent response. Run several trials and report
pass^k with the trial count rather than a single number.

## 7. Audio

Clips are synthesised by a TTS provider (Sarvam bulbul in v0.1) and recorded in
`clips/manifest.json` with text, language, provider, voice, model and SHA-256. Perturbed
variants (noise at a stated SNR, gain, telephone band) are derived from the clean clips and
carry the perturbation spec in their manifest. Report which clip set was used.

An agent run in audio modality receives only audio. The `OracleTranscriber` returns the
manifest text for a clip and exists to test the audio pipeline without an STT provider; it is
not an evaluation configuration and results produced with it must say so.

## 8. Limitations

- Turn-based, no barge-in or overlapping speech.
- Hinglish scenarios are romanised and marked `unreviewed` until a native speaker reviews
  them. Treat Hinglish results as provisional until the pack's `review_status` says otherwise.
- The scripted caller is deterministic and small; it cannot answer unexpected questions. A
  high `simulator_invalid` rate against a particular agent means the scripts need rules for
  that agent's clarification style, not that the agent is wrong.
- The reference agent is rule-based and handles only the pack's phrase inventory; it exists
  to validate the pack and to demonstrate detection of injected bugs, not as a baseline for
  real agents.
- One sandbox menu. Real menus are larger and noisier.

## 9. Reporting results responsibly

Publish, together: pass rate and pass^k with the trial count, by language and by category
with intervals, the `simulator_invalid` and `infra_error` counts, the pack id, version and
content hash, the agent configuration, the modality and clip set, and the benchmark version.
Add a percentage to a headline only after the above exists for it.

## Prior art

- τ-bench (Sierra): end-state comparison against a database and the pass^k metric.
- EVA-Bench (ServiceNow, 2026): validated bot-to-bot audio conversations with simulator
  error detection as a separate outcome.
- VoiceAgentBench (Krutrim, 2025): spoken tool-calling evaluation across Indian languages
  on static queries.
- VoiceTest: open-source multi-turn simulation and LLM-judge scoring for several voice
  platforms.

IndicOrderBench combines the first two ideas for one narrow task, ordering, in Indian
languages, and scores committed state instead of transcripts.
