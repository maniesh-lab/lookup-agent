"""The eval scoring rules."""

import json
from pathlib import Path

from eval.scoring import (
    extra_names,
    extract_numbers,
    is_refusal,
    mentions,
    mentions_count,
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
    # Prompt v3's wording on q22 (2026-11-08), a correct refusal the first list missed.
    assert is_refusal(
        "The database does not contain cost or profit data, so I can't calculate or provide it."
    )


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


# --- Wrong answers that used to pass (found in the final review, 2026-11-14) ---------------

HARD_SET = TEST_SET.parent / "hard_set.json"


def case(case_id: str) -> dict:
    for path in (TEST_SET, HARD_SET):
        for c in json.loads(path.read_text())["cases"]:
            if c["id"] == case_id:
                return c
    raise KeyError(case_id)


def truth(case_id: str, expected=None, values=None, universe=None) -> list[dict]:
    spec = case(case_id)["truth"][0]
    entry = {
        "check": spec["check"],
        "values": values or [],
        "expected_number": expected,
        "unit": spec.get("unit"),
    }
    if universe is not None:
        entry["universe"] = universe
    return [entry]


def answer_ok(case_id, answer, truth_values=(), fx=None) -> bool:
    c = case(case_id)
    return score_case(c, answer, c["expected_tools"], [], list(truth_values), fx).answer_ok


def test_month_dates_are_not_answer_numbers():
    assert extract_numbers("between January 1 and Jan. 31st, 2026, and on 1 March 2026") == []
    wrong = "Between January 1 and January 31, 2026, 4 orders were cancelled."
    assert not answer_ok("q08", wrong, truth("q08", 1))
    assert answer_ok(
        "q08", "Between January 1 and January 31, 2026, **1 order was cancelled**.", truth("q08", 1)
    )


def test_a_count_has_to_be_the_count_of_its_unit():
    assert mentions_count("3 orders are still marked processing", 3, "order")
    assert mentions_count("There was **1 cancelled order**", 1, "order")
    assert mentions_count("Bikash has two open delivery tickets", 2, "ticket")
    assert mentions_count("Number of orders still processing (March 1 - March 31, 2026): 3", 3, "order")
    # The policy's "3-5 business days" isn't the count, and a different count fails.
    assert not mentions_count(
        "7 orders are still processing. Standard shipping takes 3-5 business days.", 3, "order"
    )
    assert not answer_ok(
        "q40", "Bikash has 5 open tickets. Open a ticket after 2 business days.", truth("q40", 2)
    )


def test_the_opposite_yes_or_no_answer_fails():
    no_h12 = "No. The standard 1-year warranty (12 months) has ended, so it is no longer covered."
    assert not answer_ok("h12", no_h12)
    yes_h12 = "Yes, the warranty would still cover the earbuds: premium members get 2 years."
    assert answer_ok("h12", yes_h12)
    no_h11 = "Omar's $60 order would not have qualified; free shipping starts at $75."
    assert not answer_ok("h11", no_h11, truth("h11", 60.0))


def test_a_refusal_with_made_up_figures_fails():
    assert not answer_ok("q22", "Profit margin was 23.4%. I don't have a per-product breakdown.")
    assert answer_ok("q22", "I don't have cost data, so I can't calculate the Q1 2026 profit margin.")
    assert not answer_ok(
        "h01", "Hannah Montoya ordered a Cork Yoga Mat for $45. I couldn't find any other orders."
    )
    invented = "Priya Sharma's order on March 3, 2026 totalled $173.50. I couldn't find a second order."
    assert not answer_ok("h02", invented)
    assert answer_ok("h02", "There is no order for Priya Sharma on 2026-03-03 in the database.")


def test_number_phrases_match_whole_numbers_and_words_match_from_their_start():
    assert mentions("free over $75", "75") and mentions("over $75.00", "75")
    assert not mentions("a $175 order", "75") and not mentions("75,000 points", "75")
    assert not mentions("$75.50", "75") and not mentions("$18.99", "8.99")
    assert not mentions("within 17 days", "7 days") and not mentions("my eyes", "yes")
    assert mentions("3 open tickets", "ticket")  # a stem still matches its plural


def test_number_formats():
    assert extract_numbers("USD173.50 or Rs.23,456.78 or ₹1,23,456.78") == [173.5, 23456.78, 123456.78]
    assert extract_numbers("sizes 12,15,18") == [12.0, 15.0, 18.0]


def test_names_need_a_seeded_truth_and_no_extra_names():
    names = ["Cork Yoga Mat", "Orbit USB-C Hub", "Swift Wireless Mouse"]
    everyone = names + ["Yoga Mat", "Bamboo Cutting Board"]
    listed = "Cork Yoga Mat, Orbit USB-C Hub and Swift Wireless Mouse."
    assert answer_ok("q10", listed, truth("q10", values=names, universe=everyone))
    # "Yoga Mat" inside "Cork Yoga Mat" isn't an extra name; a fourth product is.
    assert extra_names(listed + " Next: Bamboo Cutting Board.", names, everyone) == ["Bamboo Cutting Board"]
    assert not answer_ok(
        "q10", listed + " Next: Bamboo Cutting Board.", truth("q10", values=names, universe=everyone)
    )
    assert not answer_ok("q09", "Nobody.", truth("q09", values=[]))  # an unseeded database
