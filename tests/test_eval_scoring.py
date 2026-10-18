"""The eval scoring rules."""

import json
from pathlib import Path

from eval.scoring import extract_numbers, is_refusal, mentions_number, score_case

TEST_SET = Path(__file__).resolve().parent.parent / "eval" / "test_set.json"


def test_extract_numbers():
    assert extract_numbers("Total: $1,234.56 (about 1,690 CAD), 3 items") == [1234.56, 1690.0, 3.0]


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
