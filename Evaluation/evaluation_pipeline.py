"""
Runs both evaluators over every answer file and writes one report.

    python evaluation_pipeline.py --answers answers_*.json --benchmark bench.json

Each cell is one run: a chunking strategy, a model, and the token budget the
run used. Retrieval columns are identical across models within a strategy and
budget, because those runs share a retriever, so the strategy section reports
retrieval once and lists the models under it.

Three reports are written, each carrying the run conditions it was produced
under:

    summary.json      one object for the whole grid
    by_strategy.json  one entry per (strategy, budget), models nested inside
    by_query.json     one entry per question, cells nested inside
"""

import argparse
import json
import math
import os
import re

import evaluate_generation as gen
import evaluate_retrieval as ret
import score_filter

# A cell name is the strategy, then the model, then an optional budget suffix:
# 'exp1_bge_combo2' + '05b' + 'budget500'. Non-greedy so the model is the last
# field before the suffix rather than the first field after the experiment.
CELL_PATTERN = r"^(?P<strategy>.+?)_(?P<model>[^_]+)(?:_budget(?P<budget>\d+))?$"

# Totals rather than averages: averaging a question count across cells says
# nothing, while summing says how much evidence a row rests on.
COUNT_KEYS = ("n_queries", "n_abstained", "n_scored", "chunks_before",
              "chunks_after", "chunks_dropped", "questions_emptied")


# ==========================================================================
# 1. Cell Name Parsing
# ==========================================================================

def cell_parts(name, pattern=CELL_PATTERN):
    """'exp1_bge_combo2_05b_budget500' -> ('exp1_bge_combo2', '05b', 500).

    Falls back to the whole name as the strategy when the pattern does not
    match, so an unexpected name groups alone rather than merging elsewhere.
    """
    match = re.match(pattern, name)
    if not match:
        return name, None, None
    budget = match.groupdict().get("budget")
    return match.group("strategy"), match.group("model"), int(budget) if budget else None


def budget_of(path, from_name=None):
    """The token budget a run used, from the file if recorded, else its name."""
    return ret.read_answers(path)[1] or from_name


# ==========================================================================
# 2. Aggregation
# ==========================================================================

def is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and not math.isnan(value))


def combine(dicts, keys):
    """Mean of each metric and total of each count, over a list of rows."""
    out = {}
    for key in keys:
        values = [d[key] for d in dicts if is_number(d.get(key))]
        if not values:
            out[key] = None
        elif key in COUNT_KEYS:
            out[key] = sum(values)
        else:
            out[key] = round(sum(values) / len(values), 4)
    return out


def ordered_keys(dicts):
    """Every key across the rows, in the order they first appear."""
    keys = []
    for d in dicts:
        keys.extend(k for k in d if k not in keys)
    return keys


# ==========================================================================
# 3. Report Sections
# ==========================================================================

def strategy_section(cells):
    """One entry per (strategy, budget), with the models nested inside.

    Retrieval is reported once for the group: the models share a retriever, so
    averaging their identical scores would only obscure that.
    """
    groups, order = {}, []
    for cell in cells:
        key = (cell["strategy"], cell["budget"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(cell)

    section = []
    for strategy, budget in order:
        members = groups[(strategy, budget)]
        generations = [m["generation"] for m in members]
        retrievals = [m["retrieval"] for m in members]
        section.append({
            "strategy": strategy,
            "budget": budget,
            "n_models": len(members),
            "retrieval": combine(retrievals, ordered_keys(retrievals)),
            "generation": combine(generations, ordered_keys(generations)),
            "models": [{"model": m["model"], "cell": m["cell"],
                        "generation": m["generation"], "retrieval": m["retrieval"]}
                       for m in members],
        })
    return section


def query_section(per_query):
    """One entry per question, with the cells nested inside.

    The mean across cells comes first, so a question every run struggles with
    is visible without reading the breakdown.
    """
    section = []
    for qid in sorted(per_query):
        scores = per_query[qid]
        metrics = [{k: v for k, v in s.items() if k not in ("cell", "model", "abstained")}
                   for s in scores]
        section.append({
            "query_id": qid,
            "n_cells": len(scores),
            "n_abstained": sum(1 for s in scores if s.get("abstained")),
            "mean": combine(metrics, ordered_keys(metrics)),
            "cells": scores,
        })
    return section


# ==========================================================================
# 4. Main Execution
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate a grid of runs.")
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out-dir", default="Evaluation/results")
    parser.add_argument("--k", type=int, default=10,
                        help="cutoff for the Top-K retrieval metrics")
    parser.add_argument("--cell-pattern", default=CELL_PATTERN,
                        help="regex with 'strategy', 'model' and optional 'budget' "
                             "groups, matched against the cell name to decide "
                             "which cells share a retriever")
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

    cells, per_query = [], {}

    for path in args.answers:
        name = ret.cell_name(path)
        strategy, model, named_budget = cell_parts(name, args.cell_pattern)
        print(f"  scoring {name}")

        retrieval = ret.score_file(path, bench, k=args.k, chunk_filter=chunk_filter)
        generation = gen.score_file(path, bench, use_bert=not args.no_bert,
                                    chunk_filter=chunk_filter)

        cell = {"cell": name, "strategy": strategy, "model": model,
                "budget": budget_of(path, named_budget)}
        # What the filter removed, recorded next to the scores it produced.
        # Only the benchmarked questions: counting the rest would describe a
        # file larger than the one the metrics beside it come from.
        if chunk_filter:
            scored = [r for r in ret.read_answers(path)[0]
                      if str(r.get("question_id")) in bench]
            cell["filter"] = score_filter.describe(scored, args.score_threshold)
        cell["generation"] = gen.average(generation)
        cell["retrieval"] = ret.average(retrieval)
        cells.append(cell)

        # Per-question detail, collected across cells rather than per file.
        for qid, scores in generation.items():
            per_query.setdefault(qid, []).append(
                {"cell": name, "model": model,
                 **{k: v for k, v in scores.items() if k != "answer"},
                 **retrieval.get(qid, {})})

    flat = [{**c["generation"], **c["retrieval"]} for c in cells]
    # The run conditions travel with each file, so a report is readable on its
    # own without needing the other two beside it.
    run = {"benchmark": args.benchmark, "k": args.k,
           "bert": not args.no_bert,
           "score_threshold": args.score_threshold if chunk_filter else None,
           "n_questions": len(bench), "n_cells": len(cells)}

    reports = {
        "summary.json": {"run": run, "overall": combine(flat, ordered_keys(flat))},
        "by_strategy.json": {"run": run, "by_strategy": strategy_section(cells)},
        "by_query.json": {"run": run, "by_query": query_section(per_query)},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    for filename, report in reports.items():
        with open(os.path.join(args.out_dir, filename), "w") as f:
            json.dump(report, f, indent=2)

    print_report(reports["by_strategy.json"]["by_strategy"],
                 reports["summary.json"]["overall"])
    print(f"\nSaved {', '.join(reports)} to {args.out_dir}")


def print_report(by_strategy, overall):
    """A short console view; the JSON files hold the per-question detail."""
    for group in by_strategy:
        print(f"\n{group['strategy']} (budget {group['budget']})")
        print("  retrieval  " + "  ".join(
            f"{k}={v}" for k, v in group["retrieval"].items() if v is not None))
        for member in group["models"]:
            print(f"  {member['model']:>6}     " + "  ".join(
                f"{k}={v}" for k, v in member["generation"].items() if v is not None))
    print("\noverall      " + "  ".join(
        f"{k}={v}" for k, v in overall.items() if v is not None))


if __name__ == "__main__":
    main()
