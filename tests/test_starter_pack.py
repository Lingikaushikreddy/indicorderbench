from indicorderbench.adapters.inprocess import InProcessAdapter
from indicorderbench.adapters.protocol import AgentReply, CallerUtterance
from indicorderbench.agents.parsing import TERMINAL, Intent, parse_utterance
from indicorderbench.agents.rule_based import BUGS, make_factory
from indicorderbench.backend.state import OrderBackend, SubmittedOrder
from indicorderbench.packs import load_pack, validate_pack
from indicorderbench.runner.runner import RunConfig, run_suite_sync
from indicorderbench.schemas.results import Outcome

CATEGORY_FOR_BUG = {
    "quantity_default_one": "quantity",
    "drop_modifiers": "modifier",
    "ignore_corrections": "correction",
    "ignore_cancellation": "cancellation",
    "double_submit": "duplicate_submission",
}


def test_pack_is_valid_and_complete(starter_pack_dir):
    assert validate_pack(starter_pack_dir) == []
    pack = load_pack(starter_pack_dir)
    assert len(pack.scenarios) == 40
    for lang in ("en-IN", "hi-en"):
        for cat in ("quantity", "modifier", "correction", "cancellation", "duplicate_submission"):
            assert len(pack.filter(languages=[lang], categories=[cat])) == 4
    assert len(pack.filter(tags=["smoke"])) == 10


def test_correct_agent_passes_everything(starter_pack_dir):
    pack = load_pack(starter_pack_dir)
    suite = run_suite_sync(
        pack, pack.scenarios, lambda: InProcessAdapter(make_factory()), RunConfig(trials=1)
    )
    failures = [
        (s.scenario_id, t.outcome, t.check.summary if t.check else t.error)
        for s in suite.scenarios
        for t in s.trials
        if t.outcome is not Outcome.PASS
    ]
    assert failures == []


def test_each_bug_breaks_only_its_category(starter_pack_dir):
    pack = load_pack(starter_pack_dir)
    for bug in BUGS:
        suite = run_suite_sync(
            pack,
            pack.scenarios,
            lambda b=bug: InProcessAdapter(make_factory(frozenset({b}))),
            RunConfig(trials=1),
        )
        by_id = {s.scenario_id: s for s in suite.scenarios}
        cat = CATEGORY_FOR_BUG[bug]
        for s in pack.filter(categories=[cat]):
            assert by_id[s.id].trials[0].outcome is Outcome.FAIL, (
                bug,
                s.id,
                by_id[s.id].trials[0].check,
            )
        for s in pack.filter(tags=["smoke"]):
            if s.category.value != cat:
                assert by_id[s.id].trials[0].outcome is Outcome.PASS, (
                    bug,
                    s.id,
                    by_id[s.id].trials[0].check,
                )


# -- confirm-first agents ---------------------------------------------------------------

POST_PLACEMENT_IDS = [
    f"{lang}_{cat}_{n}"
    for lang in ("en", "hien")
    for cat, nums in (
        ("duplicate_submission", ("01", "02", "03", "04")),
        ("cancellation", ("01", "04")),
    )
    for n in nums
]


class ConfirmFirstAgent:
    """Asks "Shall I place your order?" once something is in the cart, submits on the caller's
    confirmation and, when ``double_submit`` is set, submits again on any later closing."""

    def __init__(self, backend: OrderBackend, language: str, double_submit: bool) -> None:
        self.backend, self.language, self.double_submit = backend, language, double_submit
        self.asked = False
        self.placed: SubmittedOrder | None = None

    def handle(self, u: CallerUtterance) -> AgentReply:
        clauses = parse_utterance(u.text or "", self.backend.menu, self.language)
        for c in clauses:
            if c.intent is Intent.ADD and c.item_id:
                self.backend.add_item(c.item_id, c.quantity or 1, c.modifiers)
            elif c.intent is Intent.CANCEL_ORDER:
                for o in self.backend.active_orders():
                    self.backend.cancel_order(o.order_id)
                self.backend.clear_cart()
        cart = self.backend.get_cart()
        if cart and not self.asked:
            self.asked = True
            return AgentReply(text="Shall I place your order?")
        terminal = any(c.intent in TERMINAL for c in clauses)
        if terminal and cart:
            self.placed = self.backend.submit_order()
            return AgentReply(text="Order placed, thank you!")
        if terminal and self.placed is not None:
            if self.double_submit:
                for ln in self.placed.lines:
                    self.backend.add_item(ln.item_id, ln.quantity, sorted(ln.modifiers))
                self.backend.submit_order()
                return AgentReply(text="Order placed, thank you!")
            return AgentReply(text="Your order is already placed. Thank you!")
        return AgentReply(text="Okay. Anything else?")


def test_post_placement_scenarios_answer_a_confirm_first_agent(starter_pack_dir):
    pack = load_pack(starter_pack_dir)
    for sid in POST_PLACEMENT_IDS:
        rules = [r.id for r in pack.scenario(sid).caller.clarifications]
        assert "confirm_first" in rules, sid


def test_confirm_first_agent_is_caught_double_submitting(starter_pack_dir):
    pack = load_pack(starter_pack_dir)
    scenario = pack.scenario("en_duplicate_submission_01")

    def run(double_submit: bool, scenarios):  # type: ignore[no-untyped-def]
        suite = run_suite_sync(
            pack,
            scenarios,
            lambda: InProcessAdapter(
                lambda b, s: ConfirmFirstAgent(b, s.language, double_submit=double_submit)
            ),
            RunConfig(trials=1),
        )
        return suite.scenarios[0].trials[0]

    assert run(False, [scenario]).outcome is Outcome.PASS
    buggy = run(True, [scenario])
    assert buggy.outcome is Outcome.FAIL, buggy.check
    assert [t.source for t in buggy.transcript if t.speaker == "caller"] == [
        "script",
        "clarification",
        "script",
    ]
    # Without the scenario's confirm_first rule the scripted caller answers the question with
    # its next turn and the double submission goes unnoticed; that is why the rule exists.
    stripped = scenario.model_copy(deep=True)
    stripped.caller.clarifications = [
        r for r in stripped.caller.clarifications if r.id != "confirm_first"
    ]
    assert run(True, [stripped]).outcome is Outcome.PASS
