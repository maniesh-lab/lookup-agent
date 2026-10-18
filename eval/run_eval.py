"""
Run the eval set through the real agent and score it.

    python -m eval.run_eval --prompt v1
    python -m eval.run_eval --prompt v2
    python -m eval.run_eval --prompt v2 --ids q18 q20      # just some cases
    python -m eval.run_eval --prompt v2 --delay 3          # pause between questions (rate limits)

Needs the database set up (python -m scripts.setup_db), an LLM key in
.env, and internet access for the currency API. Each case starts a new
conversation. Results go to eval/results/<prompt>-<time>.json and a
one-line summary is added to eval/results/summary.md.
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


def load_cases(ids: list[str] | None) -> list[dict]:
    cases = json.loads((HERE / "test_set.json").read_text())["cases"]
    return [c for c in cases if not ids or c["id"] in ids]


class Truth:
    """Expected values, computed at run time from the database and the live API."""

    def __init__(self):
        self.conn = connect()
        self.rates: dict[tuple[str, str], float] = {}

    def rate(self, src: str, dst: str) -> float:
        if (src, dst) not in self.rates:
            result = convert(1, src, dst)
            if "error" in result:
                raise RuntimeError(f"Currency API failed while scoring: {result['error']}")
            self.rates[(src, dst)] = result["rate"]
        return self.rates[(src, dst)]

    def for_case(self, case: dict) -> tuple[list[dict], float | None]:
        values = []
        for truth in case.get("truth", []):
            with self.conn.cursor() as cur:
                cur.execute(truth["sql"])
                rows = cur.fetchall()
            first_column = [r[0] for r in rows]
            entry = {"check": truth["check"], "values": first_column, "expected_number": None}
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


async def run(prompt: str, ids: list[str] | None, delay: float) -> None:
    configure_logging("WARNING")
    cases = load_cases(ids)
    agent = create_agent(checkpointer=InMemorySaver(), prompt_version=prompt)
    truth = Truth()

    records = []
    start_tokens = _total_tokens(agent)
    for i, case in enumerate(cases, start=1):
        started = time.perf_counter()
        try:
            result = await agent.ask(case["question"])
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
            f"[{i:>2}/{len(cases)}] {mark} {case['id']} {case['category']:<5} {seconds:>5}s "
            f"tools={tools} {'; '.join(score.failures)}"
        )
        records.append(
            {
                **asdict(score),
                "question": case["question"],
                "answer": answer,
                "tools_used": tools,
                "seconds": seconds,
            }
        )
        if delay and i < len(cases):
            await asyncio.sleep(delay)

    write_report(prompt, records, _total_tokens(agent) - start_tokens)


def _total_tokens(agent) -> int:
    llm = getattr(agent, "llm", None)
    client = getattr(llm, "client", None)
    return client.total_usage.total_tokens if client else 0


def write_report(prompt: str, records: list[dict], tokens: int) -> None:
    model = settings.llm_model if settings.llm_provider == "groq" else settings.openai_model
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["passed"])

    def pct(values) -> str:
        return f"{100 * sum(values) / len(values):.0f}% ({sum(values)}/{len(values)})" if values else "-"

    overall = pct([r["passed"] for r in records])
    tools = pct([r["tools_ok"] for r in records])
    print(f"\nPrompt {prompt}: overall {overall}, tool choice {tools}, tokens {tokens}")
    for cat in ["DOCS", "DB", "API", "MIXED", "NONE"]:
        if cat in by_cat:
            print(f"  {cat:<6} {pct(by_cat[cat])}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"{prompt}-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "model": model,
                "provider": settings.llm_provider,
                "tokens": tokens,
                "records": records,
            },
            indent=2,
        )
    )
    summary = RESULTS / "summary.md"
    if not summary.exists():
        summary.write_text(
            "| run | prompt | provider / model | overall | tool choice | DOCS | DB | API "
            "| MIXED | NONE |\n|---|---|---|---|---|---|---|---|---|---|\n"
        )
    row = [
        stamp,
        prompt,
        f"{settings.llm_provider} / {model}",
        overall,
        tools,
    ] + [pct(by_cat.get(c, [])) for c in ["DOCS", "DB", "API", "MIXED", "NONE"]]
    with summary.open("a") as f:
        f.write("| " + " | ".join(row) + " |\n")
    print(f"Saved {out.relative_to(HERE.parent)} and updated eval/results/summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--prompt", default=settings.prompt_version)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    asyncio.run(run(args.prompt, args.ids, args.delay))


if __name__ == "__main__":
    main()
