"""
Run the eval set through the real agent and score it.

    python -m eval.run_eval --prompt v1
    python -m eval.run_eval --prompt v2
    python -m eval.run_eval --prompt v2 --ids q18 q20      # just some cases
    python -m eval.run_eval --prompt v2 --delay 3          # pause between questions (rate limits)
    python -m eval.run_eval --prompt v2 --set hard         # the harder set (eval/hard_set.json)

Needs the database set up (python -m scripts.setup_db), an LLM key in
.env, and internet access for the currency API. Each case starts a new
conversation; a case with "turns" sends those messages first, in the same
conversation. Results go to eval/results/<prompt>-<time>.json (or
<prompt>-hard-<time>.json) and a one-line summary is added to
eval/results/summary.md (or summary-hard.md).
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.agent import create_agent
from app.core.config import settings
from app.core.logging import configure_logging
from app.db import connect
from app.tools.external_api import convert
from eval.scoring import score_case

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


# name -> (file, categories in display order, summary file)
SETS = {
    "main": ("test_set.json", ["DOCS", "DB", "API", "MIXED", "NONE"], "summary.md"),
    "hard": ("hard_set.json", ["TRAP", "SAFETY", "FOLLOWUP", "PARTIAL", "REASONING"], "summary-hard.md"),
}


def load_cases(ids: list[str] | None, eval_set: str = "main") -> list[dict]:
    cases = json.loads((HERE / SETS[eval_set][0]).read_text())["cases"]
    return [c for c in cases if not ids or c["id"] in ids]


class Truth:
    """Expected values, computed at run time from the database and the live API."""

    def __init__(self, rate_date: str | None = None):
        """rate_date: use that day's exchange rates (for re-scoring an old run);
        None means the latest rates, like the agent itself uses."""
        self.conn = connect()
        self.requested_date = rate_date
        self.rate_date: str | None = None  # the date of the rates actually used
        self.rates: dict[tuple[str, str], float] = {}

    def rate(self, src: str, dst: str) -> float:
        if (src, dst) not in self.rates:
            result = convert(1, src, dst, on_date=self.requested_date)
            if "error" in result:
                raise RuntimeError(f"Currency API failed while scoring: {result['error']}")
            self.rates[(src, dst)] = result["rate"]
            self.rate_date = self.rate_date or result.get("rate_date")
        return self.rates[(src, dst)]

    def for_case(self, case: dict) -> tuple[list[dict], float | None]:
        values = []
        for truth in case.get("truth", []):
            with self.conn.cursor() as cur:
                cur.execute(truth["sql"])
                rows = cur.fetchall()
            first_column = [r[0] for r in rows]
            entry = {
                "check": truth["check"],
                "values": [_plain(v) for v in first_column],
                "expected_number": None,
            }
            if truth["check"] == "number" and first_column and first_column[0] is not None:
                number = float(first_column[0])
                if truth.get("convert_to"):
                    number = round(number * self.rate("USD", truth["convert_to"]), 2)
                    entry["converted"] = True
                entry["expected_number"] = number
            values.append(entry)

        fx_expected = None
        if case.get("fx"):
            fx = case["fx"]
            fx_expected = round(fx["amount"] * self.rate(fx["from"], fx["to"]), 4)
        return values, fx_expected


async def run(prompt: str, ids: list[str] | None, delay: float, eval_set: str = "main") -> None:
    configure_logging("WARNING")
    cases = load_cases(ids, eval_set)
    agent = create_agent(checkpointer=InMemorySaver(), prompt_version=prompt)
    truth = Truth()

    records = []
    start_tokens = _total_tokens(agent)
    for i, case in enumerate(cases, start=1):
        started = time.perf_counter()
        try:
            thread_id = None
            for turn in case.get("turns", []):  # earlier messages in the same conversation
                thread_id = (await agent.ask(turn, thread_id=thread_id)).thread_id
            result = await agent.ask(case["question"], thread_id=thread_id)
            answer, tools, sources = result.answer, result.tools_used, result.sources
            error = None
        except Exception as exc:  # one failing case shouldn't stop the run
            answer, tools, sources, error = "", [], [], f"{type(exc).__name__}: {exc}"
        seconds = round(time.perf_counter() - started, 2)

        truth_values, fx_expected = truth.for_case(case)
        score = score_case(case, answer, tools, sources, truth_values, fx_expected)
        if error:
            score.failures.insert(0, error)
        mark = "PASS" if score.passed else "FAIL"
        print(
            f"[{i:>2}/{len(cases)}] {mark} {case['id']} {case['category']:<9} {seconds:>5}s "
            f"tools={tools} {'; '.join(score.failures)}"
        )
        records.append(
            {
                **asdict(score),
                "question": case["question"],
                "answer": answer,
                "tools_used": tools,
                "sources": sources,
                "expected": {"truth": truth_values, "fx": fx_expected},
                "seconds": seconds,
            }
        )
        if delay and i < len(cases):
            await asyncio.sleep(delay)

    tokens = _total_tokens(agent) - start_tokens
    write_report(prompt, records, tokens, rate_date=truth.rate_date, eval_set=eval_set)


def _plain(value):
    """Database values as JSON-friendly numbers or text."""
    if isinstance(value, (int, float, str)) or value is None:
        return value
    try:
        return float(value)  # Decimal
    except (TypeError, ValueError):
        return str(value)  # dates


def _total_tokens(agent) -> int:
    llm = getattr(agent, "llm", None)
    client = getattr(llm, "client", None)
    return client.total_usage.total_tokens if client else 0


def pct(values) -> str:
    return f"{100 * sum(values) / len(values):.0f}% ({sum(values)}/{len(values)})" if values else "-"


def print_summary(prompt: str, records: list[dict], tokens: int, eval_set: str = "main") -> dict:
    categories = SETS[eval_set][1]
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["passed"])
    calls = sum(len(r["tools_used"]) for r in records)
    summary = {
        "overall": pct([r["passed"] for r in records]),
        "tools": pct([r["tools_ok"] for r in records]),
        "tool_calls": str(calls),
        **{cat: pct(by_cat.get(cat, [])) for cat in categories},
    }
    print(
        f"\nPrompt {prompt} ({eval_set} set): overall {summary['overall']}, "
        f"tool choice {summary['tools']}, tool calls {calls}, tokens {tokens}"
    )
    for cat in categories:
        if cat in by_cat:
            print(f"  {cat:<9} {summary[cat]}")
    return summary


def append_summary_row(
    label: str, prompt: str, model_label: str, summary: dict, eval_set: str = "main"
) -> None:
    _, categories, filename = SETS[eval_set]
    path = RESULTS / filename
    if not path.exists():
        header = ["run", "prompt", "provider / model", "overall", "tool choice", *categories]
        path.write_text("| " + " | ".join(header) + " |\n" + "|---" * len(header) + "|\n")
    row = [label, prompt, model_label, summary["overall"], summary["tools"]]
    row += [summary[cat] for cat in categories]
    with path.open("a") as f:
        f.write("| " + " | ".join(row) + " |\n")


def write_report(
    prompt: str, records: list[dict], tokens: int, rate_date: str | None = None, eval_set: str = "main"
) -> None:
    model = settings.llm_model if settings.llm_provider == "groq" else settings.openai_model
    summary = print_summary(prompt, records, tokens, eval_set)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{prompt}-{stamp}" if eval_set == "main" else f"{prompt}-{eval_set}-{stamp}"
    out = RESULTS / f"{name}.json"
    report = {
        "set": eval_set,
        "prompt": prompt,
        "model": model,
        "provider": settings.llm_provider,
        "tokens": tokens,
        "rate_date": rate_date,
        "records": records,
    }
    out.write_text(json.dumps(report, indent=2))
    append_summary_row(stamp, prompt, f"{settings.llm_provider} / {model}", summary, eval_set)
    print(f"Saved {out.name} and updated {SETS[eval_set][2]} in {RESULTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--prompt", default=settings.prompt_version)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--set", dest="eval_set", choices=sorted(SETS), default="main")
    args = parser.parse_args()
    asyncio.run(run(args.prompt, args.ids, args.delay, args.eval_set))


if __name__ == "__main__":
    main()
