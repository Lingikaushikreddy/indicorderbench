from indicorderbench.caller.scripted import ScriptedCaller
from indicorderbench.schemas.scenario import (
    CallerDefaults,
    CallerScript,
    CallerTurn,
    ClarificationRule,
    ResolvedCallerScript,
)


def script(
    turns: list[str],
    clar: list[ClarificationRule] | None = None,
    closing_last: bool = False,
    default_clar: list[ClarificationRule] | None = None,
) -> ResolvedCallerScript:
    ts = [CallerTurn(text=t) for t in turns]
    if closing_last:
        ts[-1].is_closing = True
    size_rule = ClarificationRule(
        id="size", match=["what size"], reply=CallerTurn(text="regular"), max_uses=1
    )
    defaults = CallerDefaults(
        closing=CallerTurn(text="that's all"),
        confirm=CallerTurn(text="yes, place it"),
        fallback=CallerTurn(text="yes, that's right"),
        nudge=CallerTurn(text="please place the order"),
        confirm_patterns=["shall i place", "confirm"],
        goodbye_patterns=["order placed"],
        question_patterns=["which", "how many"],
        clarifications=[size_rule, *(default_clar or [])],
    )
    return CallerScript(turns=ts, clarifications=clar or []).resolved(defaults)


def test_happy_path_then_closing_then_end():
    c = ScriptedCaller(script(["two wraps", "and a lassi"]))
    assert c.first_move().turn.text == "two wraps"
    m = c.next_move("Added two wraps. Anything else?")
    assert m and m.source == "script" and m.turn.text == "and a lassi"
    m = c.next_move("Added a lassi. Anything else?")
    assert m and m.source == "closing" and c.script_exhausted and c.closing_spoken
    assert c.should_end("Order placed, thank you!", has_active_order=True)
    assert c.should_end("Thanks, order placed!", has_active_order=False)
    assert not c.should_end("Okay.", has_active_order=False)


def test_should_end_requires_exhausted_script_and_closing():
    c = ScriptedCaller(script(["two wraps", "and a lassi"]))
    c.first_move()
    assert not c.should_end("Order placed!", has_active_order=True)


def test_clarification_mid_script_then_resumes():
    c = ScriptedCaller(script(["two wraps", "and a lassi"]))
    c.first_move()
    m = c.next_move("What size would you like?")
    assert m and m.source == "clarification"
    assert m.turn.text == "regular" and m.turn.id == "c_size"
    m = c.next_move("What size would you like?")  # max_uses exhausted -> continue script
    assert m and m.source == "script" and m.turn.text == "and a lassi"


def test_scenario_rule_wins_over_pack_rule():
    mine = ClarificationRule(id="mine", match=["size"], reply=CallerTurn(text="large"))
    c = ScriptedCaller(script(["two wraps"], clar=[mine]))
    c.first_move()
    m = c.next_move("What size?")
    assert m and m.turn.text == "large"


def test_unmatched_question_mid_script_continues_script():
    c = ScriptedCaller(script(["two wraps", "no onion"]))
    c.first_move()
    m = c.next_move("Do you want them spicy?")
    assert m and m.source == "script" and m.turn.text == "no onion"


def test_confirm_request_after_closing_is_answered_once():
    c = ScriptedCaller(script(["a lassi"]))
    c.first_move()
    m = c.next_move("Anything else?")
    assert m and m.source == "closing"
    m = c.next_move("Shall I place the order?")
    assert m and m.source == "confirm"
    m = c.next_move("Shall I place the order?")
    assert m and m.source == "fallback"


def test_confirm_request_before_closing_is_answered_then_closing_follows():
    c = ScriptedCaller(script(["a lassi"]))
    c.first_move()
    m = c.next_move("Shall I place the order?")
    assert m and m.source == "confirm"
    m = c.next_move("Anything else?")
    assert m and m.source == "closing"


def test_unanswerable_questions_make_caller_invalid():
    c = ScriptedCaller(script(["a lassi"]))
    c.first_move()
    c.next_move("Anything else?")  # closing
    m = c.next_move("Which branch are you ordering from?")
    assert m and m.source == "fallback"
    m = c.next_move("Which branch?")
    assert m and m.source == "fallback"
    assert c.next_move("Which branch?") is None
    assert c.invalid and "question" in (c.invalid_reason or "")
    assert c.next_move("Hello?") is None


def test_non_question_without_submission_nudges_then_stops_valid():
    c = ScriptedCaller(script(["a lassi"]))
    c.first_move()
    c.next_move("Anything else?")
    m = c.next_move("Okay.")
    assert m and m.source == "nudge"
    m = c.next_move("Sure.")
    assert m and m.source == "nudge"
    assert c.next_move("Sure.") is None
    assert not c.invalid and c.nudges_used == 2


def test_empty_agent_reply_is_not_a_question():
    c = ScriptedCaller(script(["a lassi"]))
    c.first_move()
    c.next_move("   ")
    assert c.closing_spoken
    m = c.next_move("")
    assert m and m.source == "nudge"
    assert not c.invalid


def test_is_closing_turn_skips_default_closing():
    c = ScriptedCaller(script(["a lassi", "that's it, thanks"], closing_last=True))
    c.first_move()
    m = c.next_move("Added. Anything else?")
    assert m and m.source == "script" and c.closing_spoken
    assert c.should_end("Order placed.", has_active_order=True)


def test_is_closing_on_clarification_reply_counts_as_closing():
    rule = ClarificationRule(
        id="anything_else",
        match=["anything else"],
        reply=CallerTurn(text="No, that's all.", is_closing=True),
        max_uses=1,
    )
    c = ScriptedCaller(script(["a lassi"], default_clar=[rule]))
    c.first_move()
    m = c.next_move("Added a lassi. Anything else?")
    assert m and m.source == "clarification" and c.closing_spoken
    assert c.should_end("Order placed.", has_active_order=True)
    # with closing already spoken, a non-question reply goes straight to nudging
    m = c.next_move("Okay.")
    assert m and m.source == "nudge"
