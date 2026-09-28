"""Heuristic gates: answer-type flip and when the referent check must run."""

from evaluators.semantic_judge import (
    _check_answer_type,
    _needs_referent_check,
    answer_type,
)

WESTPHALIA = "In which year was the Peace of Westphalia signed?"
VERSAILLES = "In which year was the Treaty of Versailles signed?"
CIVIL_WAR = "In which year did the American Civil War end?"


def test_year_stays_year():
    assert answer_type(WESTPHALIA) == "year"
    assert _check_answer_type(
        "The treaties concluded at Osnabruck and Munster were signed in which year?",
        WESTPHALIA,
    ) is None


def test_type_flip_document():
    reason = _check_answer_type(
        "Which document signed in the year marking the end of a global conflict "
        "mandated a defeated nation to pay reparations?",
        VERSAILLES,
    )
    assert reason and "entity" in reason


def test_type_flip_event():
    reason = _check_answer_type(
        "In what epochal event did the United States shed its fratricidal bloodshed?",
        CIVIL_WAR,
    )
    assert reason and "entity" in reason


def test_when_is_still_year():
    assert answer_type("When did a treaty marking the decline of an empire become effective?") == "year"
    assert _check_answer_type(
        "When did a treaty marking the decline of an empire become effective?",
        WESTPHALIA,
    ) is None


def test_referent_needed_for_new_place():
    assert _needs_referent_check(
        "In which year did the treaty in Nuremberg that ended imperial rule occur?",
        WESTPHALIA,
    )


def test_referent_needed_for_dropped_name():
    assert _needs_referent_check(
        "The treaties concluded at Osnabruck and Munster were signed in which year?",
        WESTPHALIA,
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all gates passed")
