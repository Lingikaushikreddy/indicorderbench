# IndicOrderBench

**Your voice agent sounds convincing. Did it actually place the correct order?**

IndicOrderBench drives a voice ordering agent through scripted English and Hinglish
ordering calls, lets it talk to a sandbox order backend, and then checks one thing
deterministically: whether the **committed order** matches what the caller asked for. It
scores state, not transcripts. Think τ-bench for voice ordering, in Indian languages.

```
caller:  Do paneer wrap... actually ek hi karo. Bina pyaaz. Aur ek mango lassi.
agent:   2 Paneer Wrap add kar diya. Paneer Wrap 1 kar diya. Paneer Wrap No onion kar diya.
         1 Mango Lassi add kar diya. Aur kuch?
caller:  Bas itna hi, thanks.
agent:   Aapka order: 1 Paneer Wrap, No onion; 1 Mango Lassi. Order place ho gaya, shukriya!
```

The agent *said* one wrap without onion. Here is what it *committed*:

| Field                | Expected | Actual     | Result |
|----------------------|----------|------------|--------|
| Submitted orders     | 1        | 1          | Pass   |
| Paneer Wrap quantity | 1        | 2          | Fail   |
| Paneer Wrap · Onion  | No onion | With onion | Fail   |
| Mango Lassi quantity | 1        | 1          | Pass   |

That table is real output from `iob demo`: the built-in reference agent with its
"ignore corrections" bug switched on. See the full reports in
[docs/demo/buggy/report.html](docs/demo/buggy/report.html) and
[docs/demo/fixed/report.html](docs/demo/fixed/report.html).

## Install and run the demo (no API keys)

```bash
pip install indicorderbench        # or: uv tool install indicorderbench
iob demo --out demo-out            # buggy vs fixed reference agent, HTML reports, no network
iob run starter --agent builtin:correct --tag smoke
```

`iob run` writes `results.json`, `junit.xml` and a single-file `report.html` with the
expected-versus-actual table, the transcript with per-turn latency, the tool-call trace and
audio players when clips exist.

## Connect your agent

Three ways. Full details and payloads in [docs/adapters.md](docs/adapters.md).

**1. In-process Python.** A factory that returns an object with `handle(utterance)`:

```python
# my_agent.py
from indicorderbench.adapters.protocol import AgentReply, CallerUtterance, SessionInfo
from indicorderbench.backend.state import OrderBackend

class MyAgent:
    def __init__(self, backend: OrderBackend, session: SessionInfo) -> None:
        self.backend = backend                  # add_item, update_line, submit_order, ...
        self.tools = OrderBackend.tool_specs()  # JSON-schema tools for your LLM

    async def handle(self, u: CallerUtterance) -> AgentReply:
        text = u.text or transcribe(u.audio_bytes)
        ...  # your LLM loop; dispatch tool calls with self.backend.call(name, args)
        return AgentReply(text="Added two paneer wraps. Anything else?")

def make_agent(backend, session):
    return MyAgent(backend, session)
```

```bash
iob run starter --agent python:my_agent:make_agent
```

**2. HTTP.** Your agent runs anywhere and receives each caller turn by HTTP; it calls the
sandbox backend over HTTP too. `examples/http_agent_shim.py` is a complete working example:

```bash
python examples/http_agent_shim.py --port 8900
iob run starter --agent http:http://127.0.0.1:8900/iob --tag smoke
```

**3. CI gate.** Fail the build when ordering accuracy drops or regresses against a baseline:

```bash
iob run starter --agent http:http://127.0.0.1:8900/iob --tag smoke \
    --baseline baselines/smoke.json --fail-under 0.9 --max-regression 0.0
```

A ready-made workflow is in [examples/ci/github-actions.yml](examples/ci/github-actions.yml).

## What it measures

- **Outcome per trial:** `pass`, `fail`, `simulator_invalid` (the agent asked something the
  script could not answer, so the agent is not blamed) or `infra_error`. Invalid and infra
  counts are always reported next to the pass rate, never hidden.
- **Pass rate and pass^k** (τ-bench's repeated-run metric), by language and by category,
  with Wilson 95% intervals. Run `--trials 3` to see whether your agent is right three times
  in a row.
- **Five failure categories:** quantity, modifier, mid-order correction, cancellation,
  duplicate submission.
- **Latency** per turn (p50 / p95) and a timestamped tool-call trace.

Methodology, validity rules and how to publish results responsibly:
[docs/methodology.md](docs/methodology.md).

## The starter pack

42 scenarios: 21 English (en-IN) and 21 Hinglish (romanised, hi-en), four per category per
language plus two `demo`-tagged correction scenarios. Ten are tagged `smoke` for a quick
gate. Every scenario lists its acceptable end states and carries clarification rules so an
agent that asks "how many?" gets a fair run. Authoring guide: [docs/scenarios.md](docs/scenarios.md).

Audio: `iob synth starter --provider sarvam` generates caller clips with Sarvam bulbul
(set `SARVAM_API_KEY`); `iob perturb` derives noisy, quiet or telephone-band variants;
`iob run --modality audio` sends only audio. `--provider silence` makes placeholder clips so
the audio pipeline runs offline with the oracle transcriber (results from it are pipeline
tests, not evaluations).

## What it does not do yet

- The runner is **turn-based**: no barge-in or overlapping speech is measured.
- The Hinglish scenarios are marked **unreviewed** until native speakers review them. Treat
  Hinglish numbers as provisional until the pack says `reviewed`.
- No LLM-based reference agent, generative caller or failure explanations in v0.1; the
  interfaces exist (`OrderBackend.tool_specs()`, `AgentAdapter`, `Transcriber`).
- No LiveKit or Pipecat adapter; the HTTP turn protocol is the integration point.

## Prior art

- [τ-bench](https://arxiv.org/abs/2406.12045): end-state comparison and pass^k.
- [EVA-Bench](https://huggingface.co/blog/ServiceNow-AI/eva): bot-to-bot audio conversations
  with simulator error detection.
- [VoiceAgentBench](https://arxiv.org/abs/2510.07978): spoken tool-calling across Indian
  languages on static queries.
- [VoiceTest](https://github.com/voicetestdev/voicetest): multi-turn simulation with
  LLM-judge scoring for several voice platforms.

IndicOrderBench narrows the problem to ordering, adds Indian languages, and scores the order
that was actually committed.

## Roadmap

Telugu and Tamil packs, native-speaker review of the Hinglish pack, a LiveKit adapter, a
generative caller for unscripted clarifications, and an LLM reference agent to publish
measured ordering accuracy for real configurations.

## Contributing

Language review, scenarios, adapters and report improvements are all welcome and small.
See [CONTRIBUTING.md](CONTRIBUTING.md). Development:

```bash
uv sync --all-extras --dev
uv run pytest -q
uv run iob validate starter
```

## License

Apache-2.0. See [LICENSE](LICENSE).
