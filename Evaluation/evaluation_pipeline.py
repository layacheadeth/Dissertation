"""
Runs both evaluators over every answer file and writes one comparison table.

    python evaluation_pipeline.py --answers answers_*.json --benchmark bench.json

Each row is one cell: a chunking strategy, a model, and the token budget the
run used. Retrieval columns are identical across models within a strategy and
budget, because those runs share a retriever.
"""

import argparse
import json
import os

import pandas as pd

import evaluate_generation as gen
import evaluate_retrieval as ret
import score_filter


def budget_of(path):
    """The token budget an answer file was generated under, if any."""
    return ret.read_answers(path)[1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a grid of runs.")
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out-dir", default="Evaluation/results")
    parser.add_argument("--no-bert", action="store_true",
                        help="skip BERTScore (slow, downloads ~1.4GB)")
    parser.add_argument("--filter-score", action="store_true",
                        help="score only the chunks the reranker rated above "
                             "--score-threshold. Off by default: it changes the "
                             "numbers, so it is a condition to report, not a "
                             "setting. Questions left with no chunks are kept "
                             "and score zero.")
    parser.add_argument("--score-threshold", type=float,
                        default=score_filter.DEFAULT_THRESHOLD,
                        help="the score a chunk must beat to be counted "
                             "(default 0.0, only used with --filter-score)")
    args = parser.parse_args()

    # None when off, which both evaluators read as "use the chunks as they are".
    chunk_filter = (score_filter.make_filter(args.score_threshold)
                    if args.filter_score else None)
    if chunk_filter:
        print(f"filtering chunks scoring <= {args.score_threshold}")

    bench = ret.benchmark_map(ret.load_benchmark(args.benchmark))
    print(f"{len(bench)} benchmark questions")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    for path in args.answers:
        name = ret.cell_name(path)
        print(f"  scoring {name}")

        retrieval = ret.score_file(path, bench, chunk_filter=chunk_filter)
        generation = gen.score_file(path, bench, use_bert=not args.no_bert,
                                    chunk_filter=chunk_filter)

        row = {"cell": name, "budget": budget_of(path)}
        # What the filter removed, recorded next to the scores it produced.
        if chunk_filter:
            row.update(score_filter.describe(ret.read_answers(path)[0],
                                             args.score_threshold))
        row.update(gen.average(generation))
        row.update(ret.average(retrieval))
        rows.append(row)

        # Per-question detail, for breaking results down by question type.
        merged = [{"query_id": qid,
                   **{k: v for k, v in generation.get(qid, {}).items() if k != "answer"},
                   **retrieval.get(qid, {})}
                  for qid in generation]
        pd.DataFrame(merged).to_csv(
            os.path.join(args.out_dir, f"per_query_{name}.csv"), index=False)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "comparison.csv"), index=False)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + table.to_string(index=False))
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
