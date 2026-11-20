"""Score answers and hits without loading an embedding model."""

import pytest

from rag.eval_cases import CASES
from rag.scoring import case_passed, phrase_supported, source_supported
from rag.vector_store import RetrievalHit


def _case(case_id: str) -> dict[str, str]:
    for case in CASES:
        if case["id"] == case_id:
            return case
    raise AssertionError(case_id)


def _hit(source: str, text: str) -> RetrievalHit:
    return RetrievalHit(id=source, text=text, distance=0.0, metadata={"source": source})


@pytest.mark.parametrize(
    ("case_id", "answer"),
    [
        ("pto_carryover", "Employees may carry over 140 hours."),
        ("pto_carryover", "You may not carry over 40 hours."),
        (
            "expense_deadline",
            "Do not file within 30 calendar days; the deadline is 90 days.",
        ),
        ("expense_finance_approval", "Finance approval is not required above $2,000."),
        ("expense_receipt", "Receipts are not required for expenses of $25 or more."),
    ],
)
def test_phrase_supported_is_false_for_negations_and_near_misses(case_id, answer):
    case = _case(case_id)
    assert phrase_supported(answer, case["accept"], case["reject"]) is False


@pytest.mark.parametrize(
    ("case_id", "answer"),
    [
        ("pto_carryover", "up to **40 hours**"),
        ("expense_deadline", "within **30 calendar days**"),
        ("expense_finance_approval", "above $2,000"),
        ("expense_receipt", "expenses of $25 or more"),
        ("pto_carryover", "no more than 40 hours"),
        ("expense_deadline", "no later than 30 calendar days"),
        ("expense_meals", "$65 per person per day"),
        ("pto_allowance", "20 days (160 hours)"),
        ("pto_notice", "two weeks in advance"),
        ("remote_wifi", "company-approved secure access method"),
    ],
)
def test_phrase_supported_is_true_for_the_stated_fact(case_id, answer):
    case = _case(case_id)
    assert phrase_supported(answer, case["accept"], case["reject"]) is True


def test_expected_phrases_satisfy_the_accept_pattern():
    for case in CASES:
        stated = f"This policy states {case['expected_phrase']}."
        assert phrase_supported(stated, case["accept"], case["reject"]) is True


def test_source_supported_requires_the_fact_on_the_expected_file():
    case = _case("pto_carryover")
    filename = case["expected_source"]
    wrong_text = _hit(filename, "140 hours")
    other_file = _hit("other.md", "40 hours")

    assert source_supported([wrong_text], filename, case["accept"]) is False
    assert source_supported([other_file], filename, case["accept"]) is False
    assert source_supported([wrong_text, other_file], filename, case["accept"]) is False

    # Reject is an answer check. A negated chunk still supports the source.
    negated = _hit(filename, "You may not carry over 40 hours.")
    assert source_supported([negated], filename, case["accept"]) is True

    both = _hit(filename, "40 hours")
    assert source_supported([both], filename, case["accept"]) is True


def test_case_passed_returns_source_and_phrase():
    case = _case("pto_carryover")
    hit = _hit(case["expected_source"], "40 hours")

    assert case_passed(case, [hit], "up to **40 hours**") == (True, True)
    assert case_passed(case, [_hit("other.md", "40 hours")], "up to **40 hours**") == (False, True)
    assert case_passed(case, [hit], "Employees may carry over 140 hours.") == (True, False)
