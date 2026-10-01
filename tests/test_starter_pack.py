from indicorderbench.adapters.inprocess import InProcessAdapter
from indicorderbench.agents.rule_based import BUGS, make_factory
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
