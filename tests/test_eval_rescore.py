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


def use_fakes(tmp_path, monkeypatch):
    import eval.run_eval as run_eval

    monkeypatch.setattr(rescore, "Truth", FakeTruth)
    monkeypatch.setattr(rescore, "load_cases", lambda ids, eval_set="main": CASES)
    monkeypatch.setattr(rescore, "RESULTS", tmp_path)
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)


def test_rescore_skips_copies_that_are_already_rescored(tmp_path):
    path = tmp_path / "v2-old-rescored.json"
    path.write_text("{}")
    assert rescore.rescore_file(path, None) is None


def test_rescore_keeps_removed_cases_and_needs_no_rate_date_without_currency_cases(
    tmp_path, monkeypatch, capsys
):
    use_fakes(tmp_path, monkeypatch)
    run = {
        "prompt": "v3",
        "rate_date": None,
        "records": [
            old_record("q05", "DOCS", "A 14-day free trial.", ["search_docs"]),
            old_record("q99", "DB", "a case that was removed later", []),
        ],
    }
    path = tmp_path / "v3-old.json"
    path.write_text(json.dumps(run))

    out = rescore.rescore_file(path, None)

    data = json.loads(out.read_text())
    assert [r["id"] for r in data["records"]] == ["q05", "q99"]
    assert data["records"][1]["answer"] == "a case that was removed later"
    assert "q99: no longer in the eval set" in capsys.readouterr().out


def test_rescore_points_out_expected_values_that_changed(tmp_path, monkeypatch, capsys):
    use_fakes(tmp_path, monkeypatch)
    record = old_record("q10", "DB", "Orbit USB-C Hub, Swift Wireless Mouse, Bamboo Cutting Board", [])
    record["expected"] = {"truth": [{"check": "names", "values": ["Other"], "expected_number": None}]}
    path = tmp_path / "v3-old.json"
    path.write_text(json.dumps({"prompt": "v3", "records": [record]}))
    rescore.rescore_file(path, "2026-10-23")
    assert "q10: expected values differ" in capsys.readouterr().out


# --- The eval runner: errors fail a case, and a run that stops early is still saved ---------


class BrokenTruth:
    rate_date = None

    def for_case(self, case):
        raise RuntimeError("Currency API failed while scoring")


class FailingAgent:
    llm = None

    async def ask(self, question, thread_id=None):
        raise RuntimeError("provider down")


async def test_an_error_fails_the_case_even_with_only_must_not_checks():
    import eval.run_eval as run_eval

    only_must_not = {"id": "s1", "category": "SAFETY", "question": "delete it"}
    only_must_not["must_not_mention"] = ["deleted"]
    record = await run_eval.run_case(FailingAgent(), BrokenTruth(), only_must_not)
    assert record["passed"] is False and record["answer_ok"] is False
    assert record["failures"][0] == "RuntimeError: provider down"
    assert record["failures"][1].startswith("expected values unavailable")


async def test_a_run_that_stops_early_saves_what_it_has(tmp_path, monkeypatch):
    import asyncio

    import eval.run_eval as run_eval

    class StopsOnSecond:
        llm = None

        def __init__(self):
            self.asked = 0

        async def ask(self, question, thread_id=None):
            self.asked += 1
            if self.asked == 2:
                raise asyncio.CancelledError  # like Ctrl+C
            reply = {"answer": "A 14-day trial.", "tools_used": [], "sources": [], "thread_id": "t"}
            return type("Reply", (), reply)

    class EmptyTruth:
        rate_date = None

        def for_case(self, case):
            return [], None

    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(run_eval, "load_cases", lambda ids, eval_set="main": CASES[::-1])
    monkeypatch.setattr(run_eval, "create_agent", lambda **kwargs: StopsOnSecond())
    monkeypatch.setattr(run_eval, "Truth", EmptyTruth)

    try:
        await run_eval.run("v3", None, 0)
    except asyncio.CancelledError:
        pass
    [saved] = [p for p in tmp_path.glob("v3-*.json")]
    data = json.loads(saved.read_text())
    assert len(data["records"]) == 1 and data["partial"].startswith("1 of 2")
    assert "(partial, 1/2)" in (tmp_path / "summary.md").read_text()
