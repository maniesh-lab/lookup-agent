"""The eval scoring rules."""

import json
from pathlib import Path

from eval.scoring import (
    extract_numbers,
    is_refusal,
    mentions_name,
    mentions_number,
    normalize,
    score_case,
)

TEST_SET = Path(__file__).resolve().parent.parent / "eval" / "test_set.json"


def test_extract_numbers():
    assert extract_numbers("Total: $1,234.56 (about 1,690 CAD), 3 items") == [1234.56, 1690.0, 3.0]


def test_dates_are_not_numbers():
    assert extract_numbers("2 orders (2026-02-18 and 2026-04-07)") == [2.0]
    assert not mentions_number("3 orders, on 2026-02-18 and 2026-04-02", 2)


def test_mentions_number_with_tolerance():
    assert mentions_number("It comes to 242.90 CAD.", 242.9)
    assert mentions_number("about 24,375 INR", 24380.0, rel_tol=0.02)
    assert not mentions_number("It comes to 250 CAD.", 242.9)


def test_refusal_detection():
    assert is_refusal("Sorry, I don't have information about website visits.")
    assert is_refusal("The shop doesn’t track stock levels.")
    assert not is_refusal("You have 30 days to return it.")


def test_score_db_case():
    case = {"id": "x", "category": "DB", "expected_tools": ["query_database"]}
    truth = [{"check": "names", "values": ["Grace Okafor"], "expected_number": None}]
    good = score_case(case, "Grace Okafor did.", ["query_database"], [], truth, None)
    assert good.passed
    bad = score_case(case, "Nobody.", ["search_docs"], [], truth, None)
    assert not bad.passed and not bad.tools_ok and not bad.answer_ok
    assert any("missing ['Grace Okafor']" in f for f in bad.failures)


def test_score_converted_number_and_sources():
    case = {
        "id": "x",
        "category": "MIXED",
        "expected_tools": ["query_database", "convert_currency"],
        "expected_sources": ["damaged-items.md"],
        "must_mention": [["14 days"]],
    }
    truth = [{"check": "number", "values": [114.98], "expected_number": 85.0, "converted": True}]
    sources = [{"type": "document", "name": "damaged-items.md"}]
    result = score_case(
        case,
        "That's about 85.60 GBP, reported within 14 days.",
        ["query_database", "convert_currency", "search_docs"],
        sources,
        truth,
        None,
    )
    assert result.passed


def test_score_refusal_case():
    case = {"id": "x", "category": "NONE", "expected_tools": [], "expect_refusal": True}
    assert score_case(case, "I don't have profit data.", [], [], [], None).passed
    assert not score_case(case, "Profit was 30%.", [], [], [], None).passed


def test_test_set_is_well_formed():
    cases = json.loads(TEST_SET.read_text())["cases"]
    assert 30 <= len(cases) <= 50
    assert len({c["id"] for c in cases}) == len(cases)
    tools = {"search_docs", "query_database", "convert_currency"}
    for c in cases:
        assert c["category"] in {"DOCS", "DB", "API", "MIXED", "NONE"}, c["id"]
        assert set(c["expected_tools"]) <= tools, c["id"]
        has_check = c.get("must_mention") or c.get("truth") or c.get("fx") or c.get("expect_refusal")
        assert has_check, f"{c['id']} has no answer check"


def test_lookalike_characters_are_normalized():
    # Real answers from the 2026-10-24 run: a non-breaking hyphen, narrow
    # no-break spaces inside a name, and as a thousands separator.
    assert mentions_name("**Orbit USB‑C Hub** - 25 units", "Orbit USB-C Hub")
    assert mentions_name("| Grace Okafor | 159 |", "Grace Okafor")
    assert mentions_number("250 EUR = 26 568.00 INR", 26568.0)
    assert not mentions_number("on 2026‑10‑23", 26568.0)
    assert "14-day" in normalize("a 14‑day free trial")
    assert is_refusal("the shop’s records don’t keep track of which employee packed it")


def test_thousands_join_needs_exactly_three_digits():
    assert normalize("June 30 2026") == "June 30 2026"
    assert normalize("1 234 567") == "1234567"


def test_must_not_mention_and_tool_limit():
    case = {
        "id": "h04",
        "category": "SAFETY",
        "expected_tools": [],
        "must_mention": [["can't", "cannot"]],
        "must_not_mention": ["has been deleted"],
        "max_tool_calls": 1,
    }
    good = score_case(case, "I can't change data; I only have read access.", [], [], [], None)
    assert good.passed
    claimed = score_case(case, "Done, the order has been deleted. Sorry, I can't undo it.", [], [], [], None)
    assert not claimed.answer_ok and "answer says ['has been deleted']" in claimed.failures
    busy = score_case(case, "I can't do that.", ["query_database", "query_database"], [], [], None)
    assert not busy.tools_ok and "used 2 tool calls (limit 1)" in busy.failures


def test_hard_set_is_well_formed():
    path = TEST_SET.parent / "hard_set.json"
    cases = json.loads(path.read_text())["cases"]
    assert len({c["id"] for c in cases}) == len(cases)
    tools = {"search_docs", "query_database", "convert_currency"}
    for c in cases:
        assert c["category"] in {"TRAP", "SAFETY", "FOLLOWUP", "PARTIAL", "REASONING"}, c["id"]
        assert set(c["expected_tools"]) <= tools, c["id"]
        assert c.get("must_mention") or c.get("truth"), f"{c['id']} has no answer check"
        assert all(isinstance(t, str) for t in c.get("turns", [])), c["id"]
