"""Re-scoring a saved eval run (no LLM, no database: Truth is faked)."""

import json

import httpx

import eval.rescore as rescore
from app.tools.external_api import convert


class FakeTruth:
    def __init__(self, rate_date=None):
        self.rate_date = rate_date

    def for_case(self, case):
        if case["id"] == "q10":
            names = ["Orbit USB-C Hub", "Swift Wireless Mouse", "Bamboo Cutting Board"]
            return [{"check": "names", "values": names, "expected_number": None}], None
        return [], None


CASES = [
    {"id": "q10", "category": "DB", "question": "top 3?", "expected_tools": ["query_database"]},
    {
        "id": "q05",
        "category": "DOCS",
        "question": "trial?",
        "expected_tools": ["search_docs"],
        "expected_sources": ["premium-membership.md"],
        "must_mention": [["14-day", "14 days"]],
    },
]


def old_record(case_id, category, answer, tools, sources_ok=True):
    return {
        "id": case_id,
        "category": category,
        "passed": False,
        "tools_ok": True,
        "sources_ok": sources_ok,
        "answer_ok": False,
        "failures": ["old failure"],
        "question": "?",
        "answer": answer,
        "tools_used": tools,
        "seconds": 1.0,
    }


def test_rescore_fixes_lookalike_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(rescore, "Truth", FakeTruth)
    monkeypatch.setattr(rescore, "load_cases", lambda ids, eval_set="main": CASES)
    monkeypatch.setattr(rescore, "RESULTS", tmp_path)
    import eval.run_eval as run_eval

    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)

    run = {
        "prompt": "v2",
        "model": "m",
        "provider": "groq",
        "tokens": 10,
        "records": [
            old_record(
                "q10",
                "DB",
                "1. Orbit USB‑C Hub\n2. Swift Wireless Mouse\n3. Bamboo Cutting Board",
                ["query_database"],
            ),
            # Old file without a saved sources list: its sources result is kept.
            old_record("q05", "DOCS", "A 14‑day free trial.", ["search_docs"]),
        ],
    }
    path = tmp_path / "v2-old.json"
    path.write_text(json.dumps(run))

    out = rescore.rescore_file(path, "2026-10-23")

    data = json.loads(out.read_text())
    assert data["rate_date"] == "2026-10-23" and data["rescored"] is True
    assert [r["passed"] for r in data["records"]] == [True, True]
    assert data["records"][0]["answer"] == run["records"][0]["answer"]  # answers untouched
    assert "(re-scored)" in (tmp_path / "summary.md").read_text()


def test_convert_can_use_a_past_date():
    def handler(request):
        assert request.url.path.endswith("/2026-10-23")
        return httpx.Response(200, json={"date": "2026-10-23", "rates": {"CAD": 1.3995}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = convert(100, "USD", "CAD", on_date="2026-10-23", client=client)
    assert result["converted"] == 139.95 and result["rate_date"] == "2026-10-23"
