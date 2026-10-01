from pathlib import Path

import pytest

from indicorderbench.adapters.inprocess import InProcessAdapter
from indicorderbench.adapters.protocol import AgentReply, CallerUtterance, Modality, SessionInfo
from indicorderbench.agents.rule_based import (
    BUGS,
    RuleBasedAgent,
    Transcriber,
    make_factory,
    parse_agent_spec,
)
from indicorderbench.backend.state import BackendError, OrderBackend
from indicorderbench.schemas.results import BackendSnapshot
from tests.test_parsing import make_test_menu

DEMO = [
    "Two paneer wraps... actually make it one. No onion. And one mango lassi.",
    "That's all",
]


def utt(text: str | None, language: str = "en-IN", audio: bytes | None = None) -> CallerUtterance:
    return CallerUtterance(
        turn_id="t", text=text, audio_path=None, audio_bytes=audio, language=language
    )


def agent(bugs: set[str] | None = None, language: str = "en-IN") -> RuleBasedAgent:
    return RuleBasedAgent(OrderBackend(make_test_menu()), language, frozenset(bugs or set()))


def drive(a: RuleBasedAgent, texts: list[str], language: str = "en-IN") -> list[str]:
    return [a.handle(utt(t, language)).text for t in texts]


def lines(snap: BackendSnapshot, order_index: int = 0) -> list[tuple[str, int, set[str]]]:
    active = snap.active_orders()
    return sorted((ln.item_id, ln.quantity, set(ln.modifiers)) for ln in active[order_index].lines)


# -- correct agent -----------------------------------------------------------------


def test_demo_flow_commits_corrected_order():
    a = agent()
    replies = drive(a, DEMO)
    snap = a.backend.snapshot()
    assert len(snap.active_orders()) == 1 and snap.cart == []
    assert lines(snap) == [("mango_lassi", 1, set()), ("paneer_wrap", 1, {"no_onion"})]
    assert replies[0] == (
        "Added 2 Paneer Wrap. Changed Paneer Wrap to 1. Updated Paneer Wrap: No onion. "
        "Added 1 Mango Lassi. Anything else?"
    )
    assert (
        replies[1] == "Your order: 1 Paneer Wrap, No onion; 1 Mango Lassi. Order placed, thank you!"
    )
    names = [c.name for c in snap.trace]
    assert names == ["add_item", "update_line", "update_line", "add_item", "submit_order"]


def test_hinglish_flow_and_replies():
    a = agent(language="hi-en")
    replies = drive(
        a, ["Do paneer wrap dena, bina pyaaz. Aur ek mango lassi.", "Bas itna hi."], "hi-en"
    )
    assert (
        replies[0] == "2 Paneer Wrap, No onion add kar diya. 1 Mango Lassi add kar diya. Aur kuch?"
    )
    assert replies[1] == (
        "Aapka order: 2 Paneer Wrap, No onion; 1 Mango Lassi. Order place ho gaya, shukriya!"
    )
    assert lines(a.backend.snapshot()) == [
        ("mango_lassi", 1, set()),
        ("paneer_wrap", 2, {"no_onion"}),
    ]


def test_second_closing_after_placement_does_not_resubmit():
    a = agent()
    replies = drive(a, [*DEMO, "Yes, that's it. Please place the order."])
    assert replies[2] == "Your order is already placed. Thank you!"
    assert len(a.backend.snapshot().active_orders()) == 1


def test_closing_with_nothing_asks_and_after_cancel_says_goodbye():
    a = agent()
    assert drive(a, ["That's all."]) == ["Nothing in your cart yet. What would you like?"]
    drive(a, ["One mango lassi, that's all.", "Cancel the order."])
    snap = a.backend.snapshot()
    assert snap.active_orders() == [] and len(snap.orders) == 1
    assert drive(a, ["No, that's all."]) == ["Your order has been cancelled. Thank you, goodbye!"]
    b = agent(language="hi-en")
    drive(b, ["Ek chai.", "Poora order cancel kar do.", "Bas."], "hi-en")
    assert b.backend.snapshot().cart == [] and b.backend.snapshot().orders == []
    assert drive(b, ["Bas."], "hi-en") == ["Aapka order cancel ho gaya hai. Shukriya, namaste!"]


def test_item_swap_keeps_quantity_unless_given():
    a = agent()
    drive(a, ["Two paneer wraps, no onion.", "Actually make them chicken wraps."])
    cart = a.backend.snapshot().cart
    assert [(c.item_id, c.quantity, set(c.modifiers)) for c in cart] == [
        ("chicken_wrap", 2, {"no_onion"})
    ]
    b = agent()
    replies = drive(b, ["Two paneer wraps.", "Actually make it a chicken wrap."])
    assert replies[1] == "Changed that to 1 Chicken Wrap. Anything else?"
    assert [(c.item_id, c.quantity) for c in b.backend.snapshot().cart] == [("chicken_wrap", 1)]


def test_modifier_correction_replaces_same_exclusive_group():
    a = agent()
    drive(a, ["One chicken wrap, extra spicy, extra cheese.", "Actually make it mild."])
    (line,) = a.backend.snapshot().cart
    assert set(line.modifiers) == {"mild", "extra_cheese"}


def test_remove_by_item_and_remove_last():
    a = agent()
    replies = drive(
        a,
        ["Two samosas and one chai and one lassi.", "Remove the lassi.", "Actually, cancel that."],
    )
    assert replies[1] == "Removed Mango Lassi. Anything else?"
    assert replies[2] == "Removed Masala Chai. Anything else?"
    assert [(c.item_id, c.quantity) for c in a.backend.snapshot().cart] == [("samosa", 2)]
    b = agent()
    drive(b, ["Three samosas.", "Remove one samosa."])
    assert [(c.item_id, c.quantity) for c in b.backend.snapshot().cart] == [("samosa", 2)]


def test_readback_requests():
    a = agent()
    replies = drive(
        a, ["One chai.", "Can you repeat that?", "That's all.", "Did my order go through?"]
    )
    assert replies[1] == "So far: 1 Masala Chai. Anything else?"
    assert replies[3] == "Your order: 1 Masala Chai. It has been placed."
    assert len(a.backend.snapshot().active_orders()) == 1


def test_unknown_and_backend_error_replies_never_raise():
    a = agent()
    assert drive(a, ["A plate of chips please."]) == [
        "Sorry, I didn't catch that. Which item would you like?"
    ]
    assert drive(a, ["Make it two."]) == ["Sorry, I didn't catch that. Which item would you like?"]
    assert drive(a, ["Remove the lassi."]) == [
        "Sorry, I couldn't do that: Mango Lassi is not in your cart."
    ]
    # no_onion does not apply to a lassi -> BackendError from add_item -> apology
    (reply,) = drive(a, ["One mango lassi, no onion."])
    assert reply.startswith("Sorry, I couldn't do that: ")
    assert a.backend.snapshot().cart == []
    b = agent(language="hi-en")
    assert drive(b, ["Kuch bhi."], "hi-en") == [
        "Maaf kijiye, samajh nahi aaya. Konsa item chahiye?"
    ]


def test_backend_error_from_a_raising_backend_becomes_apology():
    class Boom(OrderBackend):
        def submit_order(self):  # type: ignore[override]
            raise BackendError("down", "kitchen offline")

    a = RuleBasedAgent(Boom(make_test_menu()), "en-IN")
    assert drive(a, ["One chai.", "That's all."])[1] == "Sorry, I couldn't do that: kitchen offline"


# -- bugs ------------------------------------------------------------------------


def test_bugs_tuple():
    assert BUGS == (
        "ignore_corrections",
        "drop_modifiers",
        "double_submit",
        "ignore_cancellation",
        "quantity_default_one",
    )


def test_bug_ignore_corrections_keeps_first_values_but_replies_the_same():
    correct, buggy = agent(), agent({"ignore_corrections"})
    assert drive(correct, DEMO) == drive(buggy, DEMO)
    assert lines(buggy.backend.snapshot()) == [("mango_lassi", 1, set()), ("paneer_wrap", 2, set())]
    assert [c.name for c in buggy.backend.snapshot().trace] == [
        "add_item",
        "add_item",
        "submit_order",
    ]


def test_bug_drop_modifiers_omits_modifiers_from_every_call():
    correct, buggy = agent(), agent({"drop_modifiers"})
    assert drive(correct, DEMO) == drive(buggy, DEMO)
    assert lines(buggy.backend.snapshot()) == [("mango_lassi", 1, set()), ("paneer_wrap", 1, set())]
    assert all(c.args.get("modifiers") in ([], None) for c in buggy.backend.snapshot().trace)


def test_bug_double_submit_resubmits_on_a_second_closing():
    flow = [*DEMO, "Yes, that's it."]
    correct, buggy = agent(), agent({"double_submit"})
    assert drive(correct, flow) == drive(buggy, flow)
    snap = buggy.backend.snapshot()
    assert len(snap.active_orders()) == 2 and lines(snap, 0) == lines(snap, 1)
    assert len(correct.backend.snapshot().active_orders()) == 1


def test_bug_ignore_cancellation_leaves_order_active():
    flow = [*DEMO, "Actually, cancel the order."]
    correct, buggy = agent(), agent({"ignore_cancellation"})
    assert drive(correct, flow) == drive(buggy, flow)
    assert correct.backend.snapshot().active_orders() == []
    assert len(buggy.backend.snapshot().active_orders()) == 1
    removed = agent({"ignore_cancellation"})
    drive(removed, ["Two samosas and one chai.", "Remove the chai.", "That's all."])
    assert lines(removed.backend.snapshot()) == [("masala_chai", 1, set()), ("samosa", 2, set())]


def test_bug_quantity_default_one_ignores_spoken_quantity():
    flow = ["Two paneer wraps and one mango lassi.", "That's all"]
    correct, buggy = agent(), agent({"quantity_default_one"})
    assert drive(correct, flow) == drive(buggy, flow)
    assert lines(buggy.backend.snapshot()) == [("mango_lassi", 1, set()), ("paneer_wrap", 1, set())]
    assert lines(correct.backend.snapshot()) == [
        ("mango_lassi", 1, set()),
        ("paneer_wrap", 2, set()),
    ]
    corrected = agent({"quantity_default_one"})
    drive(corrected, ["Three samosas.", "Actually make it two.", "That's all"])
    assert lines(corrected.backend.snapshot()) == [("samosa", 2, set())]


# -- spec grammar and factory -----------------------------------------------------------


def test_parse_agent_spec():
    assert parse_agent_spec("builtin:correct") == frozenset()
    assert parse_agent_spec("builtin:buggy") == frozenset(BUGS)
    assert parse_agent_spec("builtin:buggy:drop_modifiers,double_submit") == frozenset(
        {"drop_modifiers", "double_submit"}
    )
    assert parse_agent_spec(" builtin:buggy: drop_modifiers , double_submit ") == frozenset(
        {"drop_modifiers", "double_submit"}
    )
    with pytest.raises(ValueError, match="nonsense"):
        parse_agent_spec("builtin:buggy:nonsense")
    with pytest.raises(ValueError, match="quantity_default_one"):  # names the known bugs
        parse_agent_spec("builtin:buggy:nonsense")
    with pytest.raises(ValueError):
        parse_agent_spec("builtin:buggy:")
    with pytest.raises(ValueError):
        parse_agent_spec("builtin:other")
    assert parse_agent_spec("http://localhost:8080") is None
    assert parse_agent_spec("openai:gpt") is None


def test_unknown_bug_names_are_rejected():
    with pytest.raises(ValueError, match="nope"):
        RuleBasedAgent(OrderBackend(make_test_menu()), "en-IN", frozenset({"nope"}))
    with pytest.raises(ValueError, match="double_submit"):
        make_factory(frozenset({"nope", "drop_modifiers"}))


async def test_make_factory_through_inprocess_adapter():
    backend = OrderBackend(make_test_menu())
    session = SessionInfo(
        scenario_id="s",
        trial=1,
        language="hi-en",
        modality=Modality.TEXT,
        backend=backend,
        session_id="x",
    )
    adapter = InProcessAdapter(make_factory())
    await adapter.start(session)
    reply = await adapter.respond(utt("Do samosa.", "hi-en"))
    assert isinstance(reply, AgentReply) and reply.text == "2 Samosa add kar diya. Aur kuch?"
    reply = await adapter.respond(utt("Bas.", "hi-en"))
    assert reply.text.endswith("Order place ho gaya, shukriya!")
    await adapter.stop()
    assert len(backend.snapshot().active_orders()) == 1
    buggy = InProcessAdapter(make_factory(frozenset({"quantity_default_one"})))
    backend2 = OrderBackend(make_test_menu())
    await buggy.start(
        SessionInfo(
            scenario_id="s",
            trial=1,
            language="en-IN",
            modality=Modality.TEXT,
            backend=backend2,
            session_id="y",
        )
    )
    await buggy.respond(utt("Two samosas."))
    assert [c.quantity for c in backend2.snapshot().cart] == [1]


# -- audio ---------------------------------------------------------------------------


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    def transcribe(self, audio: bytes, language: str) -> str:
        self.calls.append((audio, language))
        return "two samosas"


def test_audio_only_utterance_uses_transcriber(tmp_path: Path):
    t = FakeTranscriber()
    assert isinstance(t, Transcriber)
    a = RuleBasedAgent(OrderBackend(make_test_menu()), "en-IN", transcriber=t)
    reply = a.handle(utt(None, audio=b"RIFF"))
    assert reply.text == "Added 2 Samosa. Anything else?" and t.calls == [(b"RIFF", "en-IN")]
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"RIFFpath")
    a.handle(CallerUtterance("t", None, clip, None, "en-IN"))
    assert t.calls[-1] == (b"RIFFpath", "en-IN")
    # text wins over audio when both are present
    a.handle(utt("one chai", audio=b"RIFF"))
    assert len(t.calls) == 2
    assert [c.item_id for c in a.backend.snapshot().cart] == ["samosa", "samosa", "masala_chai"]


def test_audio_only_without_transcriber_is_unknown_not_a_crash():
    a = agent()
    reply = a.handle(utt(None, audio=b"RIFF"))
    assert reply.text == "Sorry, I didn't catch that. Which item would you like?"
    assert a.backend.snapshot().cart == [] and a.backend.snapshot().trace == []
    assert a.handle(utt(None)).text == reply.text


def test_correction_naming_an_earlier_line_updates_that_line():
    a = agent()
    replies = drive(a, ["Two samosas and one chai.", "Wait, change the samosas to three."])
    assert replies[1] == "Changed Samosa to 3. Anything else?"
    assert [(c.item_id, c.quantity) for c in a.backend.snapshot().cart] == [
        ("samosa", 3),
        ("masala_chai", 1),
    ]
    b = agent()
    drive(b, ["One paneer wrap and one chai.", "Actually make the paneer wrap spicy."])
    assert [(c.item_id, set(c.modifiers)) for c in b.backend.snapshot().cart] == [
        ("paneer_wrap", {"spicy"}),
        ("masala_chai", set()),
    ]


def test_bug_ignore_cancellation_submits_a_cancelled_cart_at_closing():
    flow = ["Two samosas and one chai.", "Actually, cancel the whole order.", "That's all."]
    correct, buggy = agent(), agent({"ignore_cancellation"})
    assert drive(correct, flow) == drive(buggy, flow)
    assert correct.backend.snapshot().orders == []
    assert lines(buggy.backend.snapshot()) == [("masala_chai", 1, set()), ("samosa", 2, set())]


# -- review fixes ------------------------------------------------------------------


def test_negated_modifier_corrects_the_last_line_instead_of_removing_it():
    a = agent(language="hi-en")
    drive(a, ["Ek paneer wrap aur ek chai.", "Teekha nahi chahiye."], "hi-en")
    assert [c.item_id for c in a.backend.snapshot().cart] == ["paneer_wrap", "masala_chai"]
    b = agent(language="hi-en")
    replies = drive(b, ["Ek chicken wrap zyada teekha.", "Teekha nahi chahiye."], "hi-en")
    assert replies[1] == "Chicken Wrap se Spicy hata diya. Aur kuch?"
    assert [set(c.modifiers) for c in b.backend.snapshot().cart] == [set()]
    c = agent()
    replies = drive(c, ["One chicken wrap, extra spicy, extra cheese.", "Remove the extra cheese."])
    assert replies[1] == "Removed Extra cheese from Chicken Wrap. Anything else?"
    assert [set(ln.modifiers) for ln in c.backend.snapshot().cart] == [{"spicy"}]


def test_swap_adds_before_removing_and_drops_inapplicable_options():
    a = agent()
    replies = drive(a, ["One paneer wrap.", "Actually make it a mango lassi with no onion."])
    assert replies[1] == (
        "Changed that to 1 Mango Lassi. No onion does not apply to Mango Lassi. Anything else?"
    )
    assert [(c.item_id, c.quantity) for c in a.backend.snapshot().cart] == [("mango_lassi", 1)]
    assert [t.name for t in a.backend.snapshot().trace] == ["add_item", "add_item", "remove_line"]
    assert drive(a, ["That's all."]) == ["Your order: 1 Mango Lassi. Order placed, thank you!"]
    assert len(a.backend.snapshot().active_orders()) == 1


def test_article_in_a_modifier_correction_keeps_the_quantity():
    a = agent()
    drive(a, ["Two paneer wraps.", "Actually make them a bit less spicy."])
    assert [(c.quantity, set(c.modifiers)) for c in a.backend.snapshot().cart] == [(2, {"mild"})]


def test_rehne_do_removes_the_named_line():
    a = agent(language="hi-en")
    replies = drive(a, ["Ek chai aur do samosa.", "Chai rehne do."], "hi-en")
    assert replies[1] == "Masala Chai hata diya. Aur kuch?"
    assert [(c.item_id, c.quantity) for c in a.backend.snapshot().cart] == [("samosa", 2)]


# -- review round 2 ----------------------------------------------------------------


def test_group_subject_negation_sets_no_onion_on_the_wrap():
    cases = [
        ("Ek paneer wrap aur ek chai.", "Pyaaz wala nahi chahiye.", "hi-en"),
        ("One paneer wrap and one chai.", "I don't want onions.", "en-IN"),
        ("Ek paneer wrap aur ek chai.", "Onion mat do.", "hi-en"),
    ]
    for first, second, lang in cases:
        a = agent(language=lang)
        drive(a, [first, second], lang)
        cart = a.backend.snapshot().cart
        assert [(c.item_id, set(c.modifiers)) for c in cart] == [
            ("paneer_wrap", {"no_onion"}),
            ("masala_chai", set()),
        ], second


def test_named_item_negation_corrects_the_existing_line():
    a = agent(language="hi-en")
    replies = drive(
        a,
        ["Ek paneer wrap zyada teekha aur ek chai.", "Paneer wrap mein teekha nahi chahiye."],
        "hi-en",
    )
    assert replies[1] == "Paneer Wrap se Spicy hata diya. Aur kuch?"
    assert [(c.item_id, set(c.modifiers)) for c in a.backend.snapshot().cart] == [
        ("paneer_wrap", set()),
        ("masala_chai", set()),
    ]
    b = agent(language="hi-en")
    replies = drive(b, ["Ek paneer wrap.", "Paneer wrap mein pyaaz nahi chahiye."], "hi-en")
    assert replies[1] == "Paneer Wrap No onion kar diya. Aur kuch?"
    assert [(c.item_id, c.quantity, set(c.modifiers)) for c in b.backend.snapshot().cart] == [
        ("paneer_wrap", 1, {"no_onion"})
    ]


def test_rehne_do_with_a_quantity_keeps_that_many():
    a = agent(language="hi-en")
    replies = drive(a, ["Teen samosa.", "Nahi nahi, ek hi rehne do."], "hi-en")
    assert replies[1] == "Samosa 1 kar diya. Aur kuch?"
    assert [(c.item_id, c.quantity) for c in a.backend.snapshot().cart] == [("samosa", 1)]
    b = agent(language="hi-en")
    drive(b, ["Ek chai aur do samosa.", "Chai rehne do."], "hi-en")
    assert [(c.item_id, c.quantity) for c in b.backend.snapshot().cart] == [("samosa", 2)]


def test_negation_targets_the_most_recent_line_with_that_group():
    a = agent(language="hi-en")
    replies = drive(
        a, ["Ek chicken wrap zyada teekha aur ek chai.", "Teekha nahi chahiye."], "hi-en"
    )
    assert replies[1] == "Chicken Wrap se Spicy hata diya. Aur kuch?"
    assert [(c.item_id, set(c.modifiers)) for c in a.backend.snapshot().cart] == [
        ("chicken_wrap", set()),
        ("masala_chai", set()),
    ]
    b = agent()
    replies = drive(b, ["One chai.", "I don't want onions."])
    assert replies[1] == "No onion does not apply to Masala Chai. Anything else?"
    assert [t.name for t in b.backend.snapshot().trace] == ["add_item"]
    c = agent()
    replies = drive(c, ["One paneer wrap and one chai.", "Teekha nahi chahiye."])
    assert replies[1] == "Paneer Wrap has no Spicy. Anything else?"
    assert [t.name for t in c.backend.snapshot().trace] == ["add_item", "add_item"]


def test_inapplicable_only_correction_makes_no_backend_call():
    a = agent()
    replies = drive(a, ["One chai.", "Make it spicy."])
    assert replies[1] == "Spicy does not apply to Masala Chai. Anything else?"
    assert [t.name for t in a.backend.snapshot().trace] == ["add_item"]
    assert [set(c.modifiers) for c in a.backend.snapshot().cart] == [set()]


# -- review round 3 ----------------------------------------------------------------


def test_removing_an_item_described_by_a_modifier_removes_the_line():
    a = agent()
    replies = drive(a, ["One spicy chicken wrap and one samosa.", "Remove the spicy chicken wrap."])
    assert replies[1] == "Removed Chicken Wrap. Anything else?"
    assert [c.item_id for c in a.backend.snapshot().cart] == ["samosa"]
    assert [t.name for t in a.backend.snapshot().trace] == ["add_item", "add_item", "remove_line"]
    b = agent(language="hi-en")
    replies = drive(
        b, ["Ek chai kam cheeni aur do samosa.", "Kam cheeni wali chai hata do."], "hi-en"
    )
    assert replies[1] == "Masala Chai hata diya. Aur kuch?"
    assert [c.item_id for c in b.backend.snapshot().cart] == ["samosa"]
    c = agent()
    replies = drive(c, ["One paneer wrap and one chai.", "Remove the onion from the paneer wrap."])
    assert replies[1] == "Updated Paneer Wrap: No onion. Anything else?"
    assert [(ln.item_id, set(ln.modifiers)) for ln in c.backend.snapshot().cart] == [
        ("paneer_wrap", {"no_onion"}),
        ("masala_chai", set()),
    ]


def test_named_item_edit_for_an_item_not_in_the_cart_makes_no_call():
    hi = "Aapke order mein Paneer Wrap nahi hai. Aur kuch?"
    en = "There is no Paneer Wrap in your order. Anything else?"
    cases = [
        ("Ek chai.", "Paneer wrap mein teekha nahi chahiye.", "hi-en", hi),
        ("One chai.", "No onion in the paneer wrap.", "en-IN", en),
        ("Ek chai.", "Paneer wrap mein pyaaz mat do.", "hi-en", hi),
        ("Ek chai.", "Paneer wrap mein extra cheese daal do.", "hi-en", hi),
    ]
    for first, second, lang, expected in cases:
        a = agent(language=lang)
        replies = drive(a, [first, second], lang)
        assert replies[1] == expected, second
        snap = a.backend.snapshot()
        assert [(c.item_id, c.quantity, set(c.modifiers)) for c in snap.cart] == [
            ("masala_chai", 1, set())
        ], second
        assert [t.name for t in snap.trace] == ["add_item"], second
    b = agent()
    drive(b, ["One paneer wrap.", "Actually make it a chicken wrap."])
    assert [c.item_id for c in b.backend.snapshot().cart] == ["chicken_wrap"]


def test_quantity_is_applied_even_when_the_modifier_does_not_apply():
    a = agent()
    replies = drive(a, ["One chai.", "Make it two extra spicy."])
    assert replies[1] == (
        "Changed Masala Chai to 2. Spicy does not apply to Masala Chai. Anything else?"
    )
    trace = a.backend.snapshot().trace
    assert [t.name for t in trace] == ["add_item", "update_line"]
    assert trace[1].args["quantity"] == 2 and trace[1].args["modifiers"] is None
    assert [(c.quantity, set(c.modifiers)) for c in a.backend.snapshot().cart] == [(2, set())]
