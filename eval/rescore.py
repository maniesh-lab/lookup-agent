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

Needs the database (for the truth SQL) and internet (for that day's rates).
Each file gets a "-rescored" copy next to it and a line in summary.md.
"""

import argparse
import json
from pathlib import Path

from eval.run_eval import RESULTS, Truth, append_summary_row, load_cases, print_summary
from eval.scoring import score_case


def rescore_file(path: Path, rate_date: str | None) -> Path:
    run = json.loads(path.read_text())
    rate_date = rate_date or run.get("rate_date")
    if rate_date is None:
        raise SystemExit(f"{path.name} has no rate_date saved; pass --rate-date YYYY-MM-DD")

    cases = {c["id"]: c for c in load_cases(None)}
    truth = Truth(rate_date=rate_date)
    records = []
    changed = 0

    for old in run["records"]:
        case = cases[old["id"]]
        truth_values, fx_expected = truth.for_case(case)
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
        records.append({**old, **score.__dict__, "expected": {"truth": truth_values, "fx": fx_expected}})

    print(f"{path.name}: {changed} case(s) changed")
    summary = print_summary(run["prompt"], records, run.get("tokens", 0))

    out = path.with_name(path.stem + "-rescored.json")
    out.write_text(
        json.dumps({**run, "rate_date": rate_date, "rescored": True, "records": records}, indent=2)
    )
    model_label = f"{run.get('provider')} / {run.get('model')}"
    RESULTS.mkdir(exist_ok=True)
    append_summary_row(f"{path.stem} (re-scored)", run["prompt"], model_label, summary)
    print(f"Saved {out.name} and updated summary.md\n")
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
