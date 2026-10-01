from decimal import Decimal

import pytest

from indicorderbench.agents import lexicon
from indicorderbench.agents.parsing import Clause, Intent, parse_utterance, split_clauses
from indicorderbench.schemas.menu import Menu, MenuItem, Modifier, ModifierGroup


def make_test_menu() -> Menu:
    return Menu(
        id="t",
        name="Test menu",
        modifier_groups=[
            ModifierGroup(
                id="onion",
                name="Onion",
                default="with_onion",
                options=[
                    Modifier(id="with_onion", name="With onion", aliases=["with onion"]),
                    Modifier(
                        id="no_onion",
                        name="No onion",
                        aliases=[
                            "no onion",
                            "no onions",
                            "without onion",
                            "bina pyaaz",
                            "pyaaz nahi",
                        ],
                    ),
                ],
            ),
            ModifierGroup(
                id="spice",
                name="Spice level",
                default="medium",
                options=[
                    Modifier(id="mild", name="Mild", aliases=["mild", "less spicy", "kam teekha"]),
                    Modifier(id="medium", name="Medium", aliases=["medium"]),
                    Modifier(
                        id="spicy",
                        name="Spicy",
                        aliases=["spicy", "extra spicy", "teekha", "zyada teekha"],
                    ),
                ],
            ),
            ModifierGroup(
                id="extras",
                name="Extras",
                exclusive=False,
                options=[
                    Modifier(id="extra_cheese", name="Extra cheese", aliases=["extra cheese"]),
                    Modifier(id="extra_paneer", name="Extra paneer", aliases=["extra paneer"]),
                ],
            ),
            ModifierGroup(
                id="sugar",
                name="Sugar",
                default="normal_sugar",
                options=[
                    Modifier(id="normal_sugar", name="Normal sugar", aliases=["normal sugar"]),
                    Modifier(
                        id="less_sugar", name="Less sugar", aliases=["less sugar", "kam cheeni"]
                    ),
                    Modifier(
                        id="no_sugar",
                        name="No sugar",
                        aliases=["no sugar", "bina cheeni", "cheeni nahi"],
                    ),
                ],
            ),
        ],
        items=[
            MenuItem(
                id="paneer_wrap",
                name="Paneer Wrap",
                aliases=["paneer wrap", "paneer wraps", "paneer roll"],
                price=Decimal("180"),
                modifier_groups=["onion", "spice", "extras"],
            ),
            MenuItem(
                id="chicken_wrap",
                name="Chicken Wrap",
                aliases=["chicken wrap", "chicken wraps"],
                price=Decimal("200"),
                modifier_groups=["onion", "spice", "extras"],
            ),
            MenuItem(
                id="mango_lassi",
                name="Mango Lassi",
                aliases=["mango lassi", "lassi", "aam lassi"],
                price=Decimal("90"),
                modifier_groups=["sugar"],
            ),
            MenuItem(
                id="masala_chai",
                name="Masala Chai",
                aliases=["masala chai", "chai"],
                price=Decimal("40"),
                modifier_groups=["sugar"],
            ),
            MenuItem(
                id="samosa",
                name="Samosa",
                aliases=["samosa", "samosas", "samose"],
                price=Decimal("30"),
            ),
        ],
    )


MENU = make_test_menu()


def parse(text: str, language: str = "en-IN") -> list[Clause]:
    return parse_utterance(text, MENU, language)


def brief(clauses: list[Clause]) -> list[tuple[Intent, str | None, int | None, list[str]]]:
    return [(c.intent, c.item_id, c.quantity, sorted(c.modifiers)) for c in clauses]


# -- lexicon ---------------------------------------------------------------


def test_lexicon_number_words_are_language_specific():
    assert lexicon.NUMBER_WORDS["hi-en"]["do"] == 2
    assert "do" not in lexicon.NUMBER_WORDS["en-IN"]
    assert lexicon.NUMBER_WORDS["en-IN"]["a"] == 1 and lexicon.NUMBER_WORDS["en-IN"]["an"] == 1
    assert "a" not in lexicon.NUMBER_WORDS["hi-en"]
    for lang in ("en-IN", "hi-en"):
        assert all(lexicon.NUMBER_WORDS[lang][str(n)] == n for n in range(1, 11))


def test_lexicon_phrases_are_normalised():
    from indicorderbench.schemas.menu import normalise_text

    for name in (
        "CONNECTORS",
        "CORRECTION_MARKERS",
        "REMOVE_MARKERS",
        "CANCEL_ORDER_MARKERS",
        "CLOSING_MARKERS",
        "CONFIRM_MARKERS",
        "QUESTION_MARKERS",
        "READBACK_MARKERS",
        "IGNORED_CLAUSES",
    ):
        phrases = getattr(lexicon, name)
        assert phrases and all(p == normalise_text(p) for p in phrases), name
    assert lexicon.CONNECTORS == ["and also", "and", "aur", "also", "plus", "then", "phir"]


# -- split_clauses ---------------------------------------------------------


def test_split_on_punctuation_and_connectors():
    assert split_clauses(
        "Two paneer wraps... actually make it one. No onion. And one mango lassi."
    ) == [
        "two paneer wraps",
        "actually make it one",
        "no onion",
        "one mango lassi",
    ]
    assert split_clauses("do paneer wrap aur ek lassi, phir bas") == [
        "do paneer wrap",
        "ek lassi",
        "bas",
    ]
    assert split_clauses("a wrap and also a lassi plus a chai") == ["a wrap", "a lassi", "a chai"]
    assert split_clauses("   ") == []


def test_split_keeps_apostrophes_inside_a_clause():
    assert split_clauses("That's all, thanks.") == ["that s all", "thanks"]


def test_split_protects_closing_phrase_starting_with_a_connector():
    assert split_clauses("ek lassi aur kuch nahi") == ["ek lassi", "aur kuch nahi"]
    assert split_clauses("Bas, aur kuch nahi.") == ["bas", "aur kuch nahi"]


# -- parse_utterance: brief cases ----------------------------------------------


def test_english_add_with_quantity():
    assert brief(parse("Two paneer wraps please.")) == [(Intent.ADD, "paneer_wrap", 2, [])]
    assert brief(parse("I'll have 3 samosas and a masala chai")) == [
        (Intent.ADD, "samosa", 3, []),
        (Intent.ADD, "masala_chai", 1, []),
    ]


def test_hinglish_do_means_two():
    assert brief(parse("Do paneer wrap dena", "hi-en")) == [(Intent.ADD, "paneer_wrap", 2, [])]


def test_actually_ek_hi_karo_is_quantity_correction():
    assert brief(parse("actually ek hi karo", "hi-en")) == [(Intent.CORRECT, None, 1, [])]


def test_no_onion_merges_into_preceding_add():
    assert brief(parse("two paneer wraps, no onion, and one mango lassi")) == [
        (Intent.ADD, "paneer_wrap", 2, ["no_onion"]),
        (Intent.ADD, "mango_lassi", 1, []),
    ]
    assert brief(parse("Do paneer wrap dena, bina pyaaz. Aur ek mango lassi.", "hi-en")) == [
        (Intent.ADD, "paneer_wrap", 2, ["no_onion"]),
        (Intent.ADD, "mango_lassi", 1, []),
    ]


def test_standalone_modifier_with_correction_marker_is_correct():
    assert brief(parse("no onion, actually")) == [(Intent.CORRECT, None, None, ["no_onion"])]
    clauses = parse("two paneer wraps, no onion actually")
    assert brief(clauses) == [
        (Intent.ADD, "paneer_wrap", 2, []),
        (Intent.CORRECT, None, None, ["no_onion"]),
    ]
    assert clauses[1].is_correction


def test_standalone_modifier_without_marker_is_a_correction_too():
    assert brief(parse("No onion.")) == [(Intent.CORRECT, None, None, ["no_onion"])]
    assert brief(parse("Spicy.")) == [(Intent.CORRECT, None, None, ["spicy"])]


def test_cancel_order():
    assert brief(parse("Cancel the order.")) == [(Intent.CANCEL_ORDER, None, None, [])]
    assert brief(parse("Actually, cancel the whole order please")) == [
        (Intent.CANCEL_ORDER, None, None, [])
    ]
    assert brief(parse("Poora order cancel kar do", "hi-en")) == [
        (Intent.CANCEL_ORDER, None, None, [])
    ]


def test_closing():
    assert brief(parse("That's all.")) == [(Intent.CLOSING, None, None, [])]
    assert brief(parse("That's all, thanks.")) == [(Intent.CLOSING, None, None, [])]
    assert brief(parse("No, that's all.")) == [(Intent.CLOSING, None, None, [])]
    assert brief(parse("Nahi, bas itna hi.", "hi-en")) == [(Intent.CLOSING, None, None, [])]
    assert brief(parse("Bas, aur kuch nahi.", "hi-en")) == [(Intent.CLOSING, None, None, [])]


def test_confirm():
    assert brief(parse("haan", "hi-en")) == [(Intent.CONFIRM, None, None, [])]
    assert brief(parse("Haan, order place kar do.", "hi-en")) == [(Intent.CONFIRM, None, None, [])]
    assert brief(parse("Yes, please place the order.")) == [(Intent.CONFIRM, None, None, [])]
    assert brief(parse("Yes, that's right.")) == [(Intent.CONFIRM, None, None, [])]
    assert brief(parse("Please place the order now.")) == [(Intent.CONFIRM, None, None, [])]
    assert brief(parse("Ab order place kar do.", "hi-en")) == [(Intent.CONFIRM, None, None, [])]


def test_remove_item():
    assert brief(parse("remove the lassi")) == [(Intent.REMOVE, "mango_lassi", None, [])]
    assert brief(parse("Lassi nahi chahiye", "hi-en")) == [(Intent.REMOVE, "mango_lassi", None, [])]
    assert brief(parse("Actually, cancel that.")) == [(Intent.REMOVE, None, None, [])]


def test_do_in_english_is_not_a_number():
    assert brief(parse("do you have paneer wrap")) == [(Intent.ADD, "paneer_wrap", None, [])]


def test_unknown_item():
    assert brief(parse("a plate of chips")) == [(Intent.UNKNOWN, None, None, [])]
    assert brief(parse("hmm let me think")) == [(Intent.UNKNOWN, None, None, [])]


def test_item_swap():
    assert brief(parse("actually make it a chicken wrap")) == [
        (Intent.CORRECT, "chicken_wrap", 1, [])
    ]
    assert brief(parse("nahi nahi, chicken wrap kar do", "hi-en")) == [
        (Intent.CORRECT, "chicken_wrap", None, [])
    ]


# -- parse_utterance: extensions -------------------------------------------------


def test_bare_marker_clause_carries_correction_into_next_clause():
    assert brief(parse("Nahi nahi, ek hi karo.", "hi-en")) == [(Intent.CORRECT, None, 1, [])]
    assert brief(parse("Wait, make it two.")) == [(Intent.CORRECT, None, 2, [])]
    clauses = parse("Sorry, two paneer wraps.")
    assert brief(clauses) == [(Intent.CORRECT, "paneer_wrap", 2, [])] and clauses[0].is_correction


def test_hinglish_do_as_verb_is_not_a_quantity():
    assert brief(parse("Masala chai de do", "hi-en")) == [(Intent.ADD, "masala_chai", None, [])]
    assert brief(parse("Paneer wrap do", "hi-en")) == [(Intent.ADD, "paneer_wrap", None, [])]
    assert brief(parse("Mujhe do samosa de do", "hi-en")) == [(Intent.ADD, "samosa", 2, [])]
    assert brief(parse("Ek paneer wrap do", "hi-en")) == [(Intent.ADD, "paneer_wrap", 1, [])]


def test_option_alias_containing_marker_word_is_still_a_modifier():
    assert brief(parse("Do paneer wrap, pyaaz nahi.", "hi-en")) == [
        (Intent.ADD, "paneer_wrap", 2, ["no_onion"])
    ]
    assert brief(parse("Cheeni nahi chahiye", "hi-en")) == [
        (Intent.CORRECT, None, None, ["no_sugar"])
    ]


def test_longest_alias_wins_and_multiple_groups():
    assert brief(parse("one chicken wrap, extra spicy, extra cheese, no onions")) == [
        (Intent.ADD, "chicken_wrap", 1, ["extra_cheese", "no_onion", "spicy"])
    ]
    assert brief(parse("ek chai kam cheeni", "hi-en")) == [
        (Intent.ADD, "masala_chai", 1, ["less_sugar"])
    ]


def test_quantity_nearest_before_item_wins():
    assert brief(parse("two samosas for the 3 of us")) == [(Intent.ADD, "samosa", 2, [])]
    assert brief(parse("make it two chicken wraps")) == [(Intent.CORRECT, "chicken_wrap", 2, [])]
    assert brief(parse("make them spicy")) == [(Intent.CORRECT, None, None, ["spicy"])]


def test_remove_with_quantity_and_item():
    assert brief(parse("remove one samosa")) == [(Intent.REMOVE, "samosa", 1, [])]
    assert brief(parse("Ek samosa hata do", "hi-en")) == [(Intent.REMOVE, "samosa", 1, [])]


def test_terminal_clauses_are_collapsed_and_fillers_dropped():
    assert brief(parse("Okay, that's all.")) == [(Intent.CONFIRM, None, None, [])]
    assert brief(parse("That's it. Yes, place the order.")) == [(Intent.CLOSING, None, None, [])]
    assert brief(parse("Haan, do paneer wrap.", "hi-en")) == [(Intent.ADD, "paneer_wrap", 2, [])]
    assert brief(parse("Hmm, two samosas.")) == [(Intent.ADD, "samosa", 2, [])]
    assert brief(parse("Thanks.")) == []
    assert brief(parse("Hello? Are you there?")) == []


def test_readback_and_status_requests():
    assert brief(parse("Can you repeat my order?")) == [(Intent.READBACK, None, None, [])]
    assert brief(parse("Did my order go through?")) == [(Intent.READBACK, None, None, [])]
    assert brief(parse("Mera order ho gaya kya?", "hi-en")) == [(Intent.READBACK, None, None, [])]


def test_bas_with_a_number_is_a_quantity_not_a_closing():
    assert brief(parse("bas ek hi", "hi-en")) == [(Intent.CORRECT, None, 1, [])]
    assert brief(parse("bas ek lassi aur", "hi-en")) == [(Intent.ADD, "mango_lassi", 1, [])]


def test_an_item_only_clause_after_add_is_a_second_add():
    assert brief(parse("two paneer wraps and chicken wrap")) == [
        (Intent.ADD, "paneer_wrap", 2, []),
        (Intent.ADD, "chicken_wrap", None, []),
    ]


@pytest.mark.parametrize("language", ["en-IN", "hi-en"])
def test_raw_is_the_normalised_fragment(language: str):
    (c,) = parse("Two paneer wraps!", language)
    assert c.raw == "two paneer wraps"


# -- extensions needed by the starter pack -----------------------------------------


def test_cancel_that_order_and_confirm_the_order_phrases():
    assert brief(parse("Cancel that order please, I changed my mind.")) == [
        (Intent.CANCEL_ORDER, None, None, [])
    ]
    assert brief(parse("Yes, confirm the order please.")) == [(Intent.CONFIRM, None, None, [])]
    assert brief(parse("Perfect. Confirm it, please.")) == [(Intent.CONFIRM, None, None, [])]
    assert brief(parse("Theek hai, confirm kar do.", "hi-en")) == [(Intent.CONFIRM, None, None, [])]


def test_readback_with_a_number_word_inside():
    assert brief(parse("Ek baar order repeat karo.", "hi-en")) == [
        (Intent.READBACK, None, None, [])
    ]
    assert brief(parse("Hello? Sun rahe ho? Mera order ho gaya kya?", "hi-en")) == [
        (Intent.READBACK, None, None, [])
    ]
    assert brief(parse("Hello? Are you still there? Did my order go through?")) == [
        (Intent.READBACK, None, None, [])
    ]


def test_bare_modifier_with_hindi_particles_merges():
    assert brief(parse("Ek chicken wrap, extra cheese aur extra paneer daal ke.", "hi-en")) == [
        (Intent.ADD, "chicken_wrap", 1, ["extra_cheese", "extra_paneer"])
    ]
    assert brief(parse("A chicken wrap with extra cheese and extra paneer, please.")) == [
        (Intent.ADD, "chicken_wrap", 1, ["extra_cheese", "extra_paneer"])
    ]
    assert brief(parse("Aur 3 samosa bhi daal dena.", "hi-en")) == [(Intent.ADD, "samosa", 3, [])]


def test_correction_naming_an_item_keeps_its_item_and_quantity():
    assert brief(parse("Wait, change the chai to three.")) == [
        (Intent.CORRECT, "masala_chai", 3, [])
    ]
    assert brief(parse("Wait, chai teen kar do.", "hi-en")) == [
        (Intent.CORRECT, "masala_chai", 3, [])
    ]
    assert brief(parse("Sorry, veg biryani ki jagah chicken wrap kar do.", "hi-en")) == [
        (Intent.CORRECT, "chicken_wrap", None, [])
    ]


# -- review fixes ------------------------------------------------------------------


def test_connectors_split_after_number_words_except_ek_aur():
    assert brief(parse("Ek chai do aur do samose", "hi-en")) == [
        (Intent.ADD, "masala_chai", 1, []),
        (Intent.ADD, "samosa", 2, []),
    ]
    assert brief(parse("Chai do aur samosa bhi", "hi-en")) == [
        (Intent.ADD, "masala_chai", None, []),
        (Intent.ADD, "samosa", None, []),
    ]
    assert brief(parse("Lassi teen aur samosa do", "hi-en")) == [
        (Intent.ADD, "mango_lassi", 3, []),
        (Intent.ADD, "samosa", None, []),
    ]
    assert brief(parse("One paneer wrap, make it two and a mango lassi")) == [
        (Intent.ADD, "paneer_wrap", 1, []),
        (Intent.CORRECT, None, 2, []),
        (Intent.ADD, "mango_lassi", 1, []),
    ]
    assert brief(parse("Ek aur lassi", "hi-en")) == [(Intent.ADD, "mango_lassi", 1, [])]


def test_negated_modifier_is_a_correction_not_a_removal():
    (c,) = parse("Teekha nahi chahiye.", "hi-en")
    assert (c.intent, c.item_id, c.modifiers, c.negated) == (Intent.CORRECT, None, [], ["spicy"])
    (c,) = parse("Remove the extra cheese.")
    assert (c.intent, c.item_id, c.negated) == (Intent.CORRECT, None, ["extra_cheese"])
    assert brief(parse("Pyaaz nahi chahiye.", "hi-en")) == [
        (Intent.CORRECT, None, None, ["no_onion"])
    ]
    assert brief(parse("Lassi nahi chahiye.", "hi-en")) == [
        (Intent.REMOVE, "mango_lassi", None, [])
    ]


def test_article_counts_as_one_only_directly_before_an_item():
    assert brief(parse("actually make them a bit less spicy")) == [
        (Intent.CORRECT, None, None, ["mild"])
    ]
    assert brief(parse("a paneer wrap")) == [(Intent.ADD, "paneer_wrap", 1, [])]
    assert brief(parse("a spicy chicken wrap")) == [(Intent.ADD, "chicken_wrap", 1, ["spicy"])]
    assert brief(parse("an extra spicy chicken wrap")) == [
        (Intent.ADD, "chicken_wrap", 1, ["spicy"])
    ]
    assert brief(parse("make it a chicken wrap")) == [(Intent.CORRECT, "chicken_wrap", 1, [])]


def test_rehne_do_is_a_removal():
    assert "rehne do" in lexicon.REMOVE_MARKERS and "rehne do" not in lexicon.CORRECTION_MARKERS
    assert brief(parse("chai rehne do", "hi-en")) == [(Intent.REMOVE, "masala_chai", None, [])]
    assert brief(parse("rehne do", "hi-en")) == [(Intent.REMOVE, None, None, [])]


# -- review round 2 ----------------------------------------------------------------


def test_group_subject_negation_is_a_modifier_correction():
    for text, lang in [
        ("Pyaaz wala nahi chahiye", "hi-en"),
        ("I don't want onions", "en-IN"),
        ("Remove the onion", "en-IN"),
        ("Onion mat do", "hi-en"),
        ("Pyaaz mat do", "hi-en"),
    ]:
        assert brief(parse(text, lang)) == [(Intent.CORRECT, None, None, ["no_onion"])], text
    assert brief(parse("I don't want sugar")) == [(Intent.CORRECT, None, None, ["no_sugar"])]
    assert brief(parse("Spice nahi chahiye", "hi-en")) == [(Intent.CORRECT, None, None, ["medium"])]
    (c,) = parse("Cheese nahi chahiye", "hi-en")
    assert (c.intent, c.item_id, c.modifiers, c.negated) == (
        Intent.CORRECT,
        None,
        [],
        ["extra_cheese"],
    )
    (c,) = parse("I don't want the cheese")
    assert (c.intent, c.negated) == (Intent.CORRECT, ["extra_cheese"])
    # a plain line removal is still a removal
    assert brief(parse("Lassi nahi chahiye", "hi-en")) == [(Intent.REMOVE, "mango_lassi", None, [])]


def test_named_item_modifier_negation_targets_that_item():
    (c,) = parse("Paneer wrap mein teekha nahi chahiye", "hi-en")
    assert (c.intent, c.item_id, c.modifiers, c.negated) == (
        Intent.CORRECT,
        "paneer_wrap",
        [],
        ["spicy"],
    )
    assert brief(parse("Paneer wrap mein pyaaz nahi chahiye", "hi-en")) == [
        (Intent.CORRECT, "paneer_wrap", None, ["no_onion"])
    ]
    assert brief(parse("No onion in the paneer wrap")) == [
        (Intent.CORRECT, "paneer_wrap", None, ["no_onion"])
    ]
    assert brief(parse("Ek paneer wrap pyaaz nahi", "hi-en")) == [
        (Intent.ADD, "paneer_wrap", 1, ["no_onion"])
    ]


def test_rehne_do_with_a_quantity_is_a_quantity_correction():
    assert brief(parse("ek hi rehne do", "hi-en")) == [(Intent.CORRECT, None, 1, [])]
    assert brief(parse("Nahi nahi, ek hi rehne do.", "hi-en")) == [(Intent.CORRECT, None, 1, [])]
    assert brief(parse("sirf ek rehne do", "hi-en")) == [(Intent.CORRECT, None, 1, [])]
    assert brief(parse("do hi rehne do", "hi-en")) == [(Intent.CORRECT, None, 2, [])]
    assert brief(parse("chai rehne do", "hi-en")) == [(Intent.REMOVE, "masala_chai", None, [])]
    assert brief(parse("rehne do", "hi-en")) == [(Intent.REMOVE, None, None, [])]


def test_ek_aur_protection_only_when_ek_starts_the_fragment():
    assert brief(parse("Chai ek aur samosa do", "hi-en")) == [
        (Intent.ADD, "masala_chai", 1, []),
        (Intent.ADD, "samosa", None, []),
    ]
    assert brief(parse("Lassi ek aur samosa do", "hi-en")) == [
        (Intent.ADD, "mango_lassi", 1, []),
        (Intent.ADD, "samosa", None, []),
    ]
    assert brief(parse("Mango lassi ek aur do samosa", "hi-en")) == [
        (Intent.ADD, "mango_lassi", 1, []),
        (Intent.ADD, "samosa", 2, []),
    ]
    assert brief(parse("ek aur lassi", "hi-en")) == [(Intent.ADD, "mango_lassi", 1, [])]
    assert brief(parse("Do samosa aur ek aur lassi", "hi-en")) == [
        (Intent.ADD, "samosa", 2, []),
        (Intent.ADD, "mango_lassi", 1, []),
    ]
