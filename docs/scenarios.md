# Writing scenarios

A scenario pack is a directory:

```
packs/<name>/
  pack.yaml          manifest + per-language caller defaults
  menu.yaml          items, modifier groups, aliases
  scenarios/*.yaml   one scenario per file
  clips/             generated audio (git-ignored) + manifest.json
  REVIEW.md          native-speaker review checklist
```

Validate with `iob validate packs/<name>`. It lists every problem with file and field,
including any clip named with an explicit `audio:` path that does not exist (`iob synth`
creates those).

## Scenario file

```yaml
id: hien_correction_03            # ^[a-z][a-z0-9_]*$, unique in the pack
title: Two wraps corrected to one, no onion, add a lassi
language: hi-en                   # en-IN | hi-en
category: correction              # quantity | modifier | correction | cancellation | duplicate_submission
menu: starter_qsr                 # must equal the pack menu id
difficulty: medium                # easy | medium | hard
tags: [smoke, mid-order-correction]
review_status: unreviewed         # unreviewed | reviewed
reviewer_notes: null
version: 1
caller:
  turns:
    - text: "Do paneer wrap... actually ek hi karo. Bina pyaaz. Aur ek mango lassi."
      # id defaults to t1, t2, ...; audio defaults to clips/<scenario id>/<turn id>.wav
      # is_closing: true  marks a turn that already says "that's all"
  clarifications:
    - id: how_many
      match: ["how many", "kitne", "ek ya do"]   # regexes, case-insensitive
      reply: {text: "Ek. Sirf ek paneer wrap."}
      max_uses: 2
  # closing / confirm / fallback / nudge and the pattern lists inherit from pack.yaml
expected:
  - id: single_order
    description: One active order with one wrap without onion and one lassi
    orders:
      - lines:
          - {item_id: paneer_wrap, quantity: 1, modifiers: {onion: no_onion}}
          - {item_id: mango_lassi, quantity: 1}
```

### `expected` semantics

- A list of **acceptable end states**; the trial passes if any one matches.
- `orders: []` means no active order may exist.
- `modifiers` maps a group id to one option id, a list of option ids (non-exclusive groups),
  or `"*"` for "any value". Groups you do not mention must be at the menu default.
- Only submitted, non-cancelled orders count.

### How the caller behaves

1. Speaks `turns` in order. After each agent reply it first checks clarification rules
   (scenario rules, then pack defaults); a matching rule with uses left answers the question.
   Otherwise it speaks the next turn, even if the agent asked something unmatched, so write
   turns that still make sense after an unexpected question.
2. When the turns are exhausted: a confirm request from the agent gets the confirm turn once;
   otherwise the closing turn is spoken once; after that, questions get the generic fallback
   (twice, then the trial is `simulator_invalid`) and non-questions get a nudge to place the
   order (twice, then the trial ends and is scored).
3. The call ends when the script is done, closing was said, and an order is active or the
   agent's reply matches a goodbye pattern.

Give every scenario at least one clarification rule for the obvious question (how many? which
modifier?) so agents that ask before acting get a fair run.

## Pack defaults (`pack.yaml`)

```yaml
defaults:
  caller:
    en-IN:
      closing: {text: "That's all, thanks."}
      confirm: {text: "Yes, please place the order."}
      fallback: {text: "Yes, that's right."}
      nudge: {text: "Please place the order now."}
      confirm_patterns: ["shall i (place|confirm)", "place (the|your) order\\?"]
      goodbye_patterns: ["order (is )?placed", "already placed"]
      question_patterns: ["which", "what ", "how many"]
      max_fallbacks: 2
      max_nudges: 2
      clarifications:
        - {id: how_many_en, match: ["how many"],
           reply: {text: "Just the quantities I said, please."}, max_uses: 1}
```

**Pack-level clarification rules must match only genuine questions.** They apply to every
scenario in that language, and after each agent reply the caller checks clarification rules
*before* it speaks the next scripted turn. A pack rule that matches something the agent says
routinely, such as its readback or "Anything else?", fires after the agent's first reply and
replaces scripted turn 2 in every multi-turn scenario. A rule matching `anything else` with
`is_closing: true` would end each of those scenarios after its first turn. Leave "Anything
else?" to the closing turn, which the caller speaks once the script is exhausted, and put
scenario-specific answers (which wrap? how many lassis?) in the scenario's own
`clarifications`.

## Menu

Items and modifier options carry **aliases**: every phrase a caller might say. Aliases are
matched case-insensitively after punctuation is stripped, longest first. Exclusive groups
must declare a default option; non-exclusive groups (extras) have none. Ids are unique across
items, groups and options.

## Staying inside the reference agent's grammar

The starter pack is validated by the rule-based reference agent, which handles a fixed phrase
inventory (`src/indicorderbench/agents/lexicon.py`): number words in both languages,
connectors (`and`, `aur`, `also`, `plus`, `then`, `phir`), correction markers (`actually`,
`make it`, `instead`, `nahi`, `badal do`), removal markers, whole-order cancellation phrases,
closing phrases and confirmations. Scenarios for the starter pack must stay inside that
inventory and the menu's aliases, or the integration test fails. Scenarios written for real
agents can use any phrasing; the reference agent is a validation tool, not a ceiling.

## Bug-isolation contract

The starter pack must satisfy, as enforced by `tests/test_starter_pack.py`:

- every scenario passes with `builtin:correct`;
- with exactly one bug enabled, every scenario in that bug's category fails;
- every `smoke` scenario outside that category still passes.

Keep smoke scenarios free of behaviour from other categories (a smoke quantity scenario has
no corrections, no cancellations, no modifiers).

## Review checklist for native speakers

See `packs/starter/REVIEW.md`. When a scenario has been reviewed, set `review_status:
reviewed` and record what changed and why in `reviewer_notes`. Contributed recordings need
written consent from the speaker; removing a name is not consent.
