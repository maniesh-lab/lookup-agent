"""
Measure search quality for each layer: vector, hybrid, hybrid + reranker.

    python -m eval.run_retrieval_eval          # k = 5
    python -m eval.run_retrieval_eval --k 3

Hit rate @k: share of queries where the expected policy file is in the top k.
MRR: average of 1/rank of the first correct result (1.0 = always first).

Needs the policies ingested (python -m scripts.ingest_docs). The first run
downloads the embedding and reranker models.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from app.db import connect
from app.retrieval.models import get_embedder, get_reranker
from app.retrieval.search import MODES, search

HERE = Path(__file__).resolve().parent


def evaluate(queries: list[dict], k: int) -> dict[str, dict]:
    embedder, reranker = get_embedder(), get_reranker()
    results = {}
    with connect(vector=True) as conn:
        for mode in MODES:
            hits, rr, misses = 0, 0.0, []
            for q in queries:
                sources = [
                    h.source
                    for h in search(
                        q["query"], mode=mode, k=k, conn=conn, embedder=embedder, reranker=reranker
                    )
                ]
                if q["expected_source"] in sources:
                    hits += 1
                    rr += 1 / (sources.index(q["expected_source"]) + 1)
                else:
                    misses.append(q["query"])
            results[mode] = {"hit_rate": hits / len(queries), "mrr": rr / len(queries), "misses": misses}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    queries = json.loads((HERE / "retrieval_set.json").read_text())["queries"]
    results = evaluate(queries, args.k)

    lines = [f"| mode | hit rate @{args.k} | MRR |", "|---|---|---|"]
    for mode, r in results.items():
        lines.append(
            f"| {mode} | {r['hit_rate']:.0%} ({round(r['hit_rate'] * len(queries))}/{len(queries)})"
            f" | {r['mrr']:.2f} |"
        )
    table = "\n".join(lines)
    print(table)
    for mode, r in results.items():
        for miss in r["misses"]:
            print(f"  missed ({mode}): {miss}")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    (out / "retrieval.md").write_text(
        f"# Search hit rate\n\nRun: {stamp}, {len(queries)} queries\n\n{table}\n"
    )
    print("Saved eval/results/retrieval.md")


if __name__ == "__main__":
    main()
