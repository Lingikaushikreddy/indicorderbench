# Native-speaker review checklist for the starter pack

Every scenario in `scenarios/` was written by a non-native author and ships with
`review_status: unreviewed`. A reviewer who speaks the scenario's language natively should
go through the checklist below for each file, then flip `review_status` to `reviewed` and
write their observations into `reviewer_notes` (free text, keep it short; `null` means no
notes). Commit the YAML change; nothing else needs to be edited.

Scenarios are grouped by id: `en_<category>_NN` are Indian English, `hien_<category>_NN`
are romanised Hinglish. The five categories are quantity, modifier, correction,
cancellation and duplicate_submission. Scenarios tagged `smoke` are the quick suite and are
deliberately minimal.

## What to check in each scenario

1. **Naturalness.** Would a real customer phone a restaurant and say this, in this order,
   with this amount of detail? Flag anything that reads like a test case rather than speech
   (over-complete sentences, unusual word order, missing fillers).
2. **Code-mixing (Hinglish only).** Is the mix of Hindi and English what a city caller
   would actually use? Watch for English words where Hindi is more natural
   ("cancel" is fine, "order" is fine; "remove" may not be) and for Hindi words a
   Hinglish speaker would say in English instead.
3. **Number words.** Are `ek`, `do`, `teen`, `chaar`, `paanch`, `che` and the English
   numbers used the way the scenario intends? In particular confirm that `do` in each
   Hinglish turn reads as "two" and not as the verb "give/do" ("de do", "kar do").
4. **Regional variants.** Note spellings or words that are specific to one region
   (`samose` vs `samosas`, `cheeni` vs `shakkar`, `pyaaz` vs `kanda`). The menu aliases
   list the variants the reference agent understands; suggest additions in the notes.
5. **Corrections.** For correction scenarios, would a human phrase the correction that way
   ("nahi nahi, ek hi karo", "make that a chicken biryani instead")? Is the moment of the
   correction (mid-sentence, after the readback, after the agent asks) realistic?
6. **Cancellations and closings.** Does "cancel the order", "poora order cancel kar do",
   "bas itna hi", "that's all" sound like what a caller says, and does the turn that closes
   the call (`is_closing: true`) actually read as a closing?
7. **Clarification replies.** Each scenario carries `clarifications` for questions another
   agent might ask (how many, which spice level, with or without onion). Check that the
   replies sound like answers a caller would give, not like menu ids.
8. **Expected end state.** Confirm that the `expected` block is what the caller meant. If
   the words could reasonably be read another way, say so; ambiguity is a finding.

## Recording notes

When the clips are recorded or synthesised for the audio modality, note for each scenario:

- the pace and any hesitation implied by `...` in the text (a short pause, not a long one);
- which words should carry the emphasis in corrections ("ek hi", "actually, make it one");
- whether a turn should sound hurried or interrupted (the "long pause" scenarios, the
  "hello? are you there?" turns);
- the accent and register you recommend (the default assumption is an urban North-Indian
  caller in a casual register).

Put these in `reviewer_notes` too, prefixed with `recording:` so they are easy to find.

## How to record your review

```yaml
review_status: reviewed
reviewer_notes: |
  Natural. "Chai hata do" is fine in Delhi; a Mumbai caller might say "chai cancel kar do".
  recording: say "nahi nahi" quickly, emphasis on "ek hi".
```

Run `uv run pytest tests/test_starter_pack.py`
after editing to make sure the pack still loads and the reference agent still passes every
scenario. If a wording change makes the reference agent fail, open an issue with the
scenario id and the transcript rather than reverting the wording: the agent's phrase list
(`src/indicorderbench/agents/lexicon.py`) and the menu aliases are what need to grow.
