# IndicOrderBench demo

Two runs of the smoke suite with the built-in reference agent, no API keys:

- `buggy/report.html`: every injected bug enabled (ignored corrections, dropped modifiers, double submission, ignored cancellation, quantity defaulting to one).
- `fixed/report.html`: the correct agent, compared against the buggy run.
- `demo-scenario/report.html`: the demo correction scenario with only the ignore_corrections bug, the run behind the headline table.

Regenerate with `iob demo --out docs/demo`.
