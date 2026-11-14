"""
Score a saved eval run again, without asking the AI anything.

Use it after changing the scoring rules (eval/scoring.py) or the checks in
eval/test_set.json. The saved answers stay exactly as the agent wrote them;
only the scoring is redone.

    python -m eval.rescore eval/results/v2-20261024-131146.json
    python -m eval.rescore eval/results/v1-*.json --rate-date 2026-10-23

Exchange rates: the expected currency amounts use the same day's rates the
run used (saved in the file as rate_date). Older files don't have it, so pass
--rate-date with the date the answers mention (e.g. "rate date 2026-10-23").
It's only needed when the run has currency cases.

Needs the database (for the truth SQL) and internet (for that day's rates).
Each file gets a "-rescored" copy next to it and a line in summary.md (or summary-hard.md).
Files that are already "-rescored" copies are skipped, so a glob is safe.
Cases removed from the eval set since the run are kept as they were. A case
whose question changed, or whose expected values differ from the ones saved
with the run (the seed or the truth SQL changed), is pointed out.
"""

import argparse
import json
from pathlib import Path

from eval.run_eval import (
    RESULTS,
    SETS,
    Truth,
    append_summary_row,
    load_cases,
    print_summary,
    saved_expected,
)
from eval.scoring import score_case


def needs_rates(case: dict) -> bool:
    return bool(case.get("fx")) or any(t.get("convert_to") for t in case.get("truth", []))


def _comparable(truth_values: list[dict]) -> list[tuple]:
    return [(t["check"], t["values"], t["expected_number"]) for t in truth_values]


def rescore_file(path: Path, rate_date: str | None) -> Path | None:
    if path.stem.endswith("-rescored"):
        print(f"Skipping {path.name}: it's already a re-scored copy (re-score the original instead).")
        return None
    run = json.loads(path.read_text())
    eval_set = run.get("set", "main")  # older files are all from the main set
    cases = {c["id"]: c for c in load_cases(None, eval_set)}
    rate_date = rate_date or run.get("rate_date")
    if rate_date is None and any(needs_rates(cases[r["id"]]) for r in run["records"] if r["id"] in cases):
        raise SystemExit(f"{path.name} has currency cases but no rate_date; pass --rate-date YYYY-MM-DD")

    truth = Truth(rate_date=rate_date)
    records = []
    changed = 0

    for old in run["records"]:
        case = cases.get(old["id"])
        if case is None:
            print(f"  {old['id']}: no longer in the eval set; kept as it was")
            records.append(old)
            continue
        if old.get("question") and old["question"] != case["question"]:
            print(f"  {old['id']}: the question changed since this run; scored against the current checks")
        truth_values, fx_expected = truth.for_case(case)
        saved = (old.get("expected") or {}).get("truth")
        if saved is not None and _comparable(saved) != _comparable(truth_values):
            print(f"  {old['id']}: expected values differ from the ones saved with the run")
        # Older runs didn't save the sources list; reuse their sources result then.
        sources = old.get("sources")
        score = score_case(case, old["answer"], old["tools_used"], sources or [], truth_values, fx_expected)
        if sources is None and case.get("expected_sources"):
            score.sources_ok = old["sources_ok"]
            score.failures = [f for f in score.failures if not f.startswith("search didn't return")]
            if not old["sources_ok"]:
                score.failures.append(f"search didn't return {case['expected_sources']}")
            score.passed = score.tools_ok and score.sources_ok and score.answer_ok

        if score.passed != old["passed"]:
            changed += 1
            mark = "now PASS" if score.passed else "now FAIL"
            print(f"  {old['id']}: {mark} {'; '.join(score.failures)}")
        records.append({**old, **score.__dict__, "expected": saved_expected(truth_values, fx_expected)})

    print(f"{path.name}: {changed} case(s) changed")
    summary = print_summary(run["prompt"], records, run.get("tokens", 0), eval_set)

    out = path.with_name(path.stem + "-rescored.json")
    out.write_text(
        json.dumps({**run, "rate_date": rate_date, "rescored": True, "records": records}, indent=2)
    )
    model_label = f"{run.get('provider')} / {run.get('model')}"
    RESULTS.mkdir(exist_ok=True)
    append_summary_row(f"{path.stem} (re-scored)", run["prompt"], model_label, summary, eval_set)
    print(f"Saved {out.name} and updated {SETS[eval_set][2]}\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--rate-date", help="YYYY-MM-DD of the exchange rates the run used")
    args = parser.parse_args()
    for path in args.files:
        rescore_file(path, args.rate_date)


if __name__ == "__main__":
    main()
