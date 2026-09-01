"""
Retrieval scoring: did the retriever find the right pages?

Reads an answer file from the inference pipeline and scores each query.

    python evaluate_retrieval.py --answers answers_*.json --benchmark bench.json

This module also holds the helpers shared with evaluate_generation and
evaluation_pipeline: load_benchmark, benchmark_map, read_answers and cell_name.
"""

import argparse
import json
import math
import os
import re

import numpy as np

ROLE_GAIN = {"primary": 3.0, "supporting": 2.0, "background": 1.0}

# Top-level keys an answer file may wrap its records and token budget under.
RECORD_KEYS = ("results", "answers", "records")
BUDGET_KEYS = ("budget", "token_budget", "context_budget", "max_context_tokens")

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
    "this", "that", "these", "those", "then", "just", "so", "than", "such", "both",
    "through", "about", "for", "is", "of", "while", "during", "to", "from", "in", "on"
}


# --------------------------------------------------------------------------
# Metric Calculation Functions
# --------------------------------------------------------------------------

def compute_recall(top_k_pages, gold_pages):
    """Calculates Recall@K over gold pages."""
    if not gold_pages:
        return 0.0
    found = set(top_k_pages) & set(gold_pages)
    return len(found) / len(gold_pages)


def compute_hit_at_1(ranked_pages, gold_pages):
    """Calculates Hit@1 (Is the top ranked page a gold page?)."""
    if not ranked_pages:
        return 0.0
    return 1.0 if ranked_pages[0] in gold_pages else 0.0


def compute_mrr(ranked_pages, gold_pages):
    """Calculates Mean Reciprocal Rank (MRR)."""
    for i, page in enumerate(ranked_pages):
        if page in gold_pages:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(top_k_pages, gold_pages):
    """Calculates Normalized Discounted Cumulative Gain (NDCG@K)."""
    if not top_k_pages or not gold_pages:
        return 0.0
    gains = [gold_pages.get(p, 0.0) for p in top_k_pages]
    ideal = sorted(gold_pages.values(), reverse=True)[:len(gains)]

    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def compute_keyword_recall(chunks, question, k=10):
    """Calculates Keyword Recall@K (percentage of target keywords present in top K chunks)."""
    # Extract target keywords from benchmark question/keywords/gold text
    target_text = ""
    if "keywords" in question:
        kw = question["keywords"]
        target_text = " ".join(kw) if isinstance(kw, list) else str(kw)
    else:
        target_text = question.get("question", "") or question.get("query", "")

    target_words = set(re.findall(r"\b\w+\b", target_text.lower())) - STOP_WORDS
    if not target_words:
        return 0.0

    # Extract words from top K retrieved text chunks
    top_k_chunks = chunks[:k]
    retrieved_text = " ".join(chunk.get("text", "") for chunk in top_k_chunks)
    retrieved_words = set(re.findall(r"\b\w+\b", retrieved_text.lower()))

    # Calculate recall of target keywords
    found_keywords = target_words & retrieved_words
    return len(found_keywords) / len(target_words)


# --------------------------------------------------------------------------
# Data Preprocessing & Parsing Helpers
# --------------------------------------------------------------------------

def week_number(value):
    """The number in a week label: 'Week 3' and 3 both give 3, else None."""
    match = re.search(r"\d+", str(value))
    return int(match.group()) if value is not None and match else None


def page_number(value):
    """A page as an int, else None.

    Gold pages and retrieved pages are compared as tuples, so '12' and 12 have
    to land on the same key or the gold page silently never matches.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"\s*(\d+)\s*", str(value))
    return int(match.group(1)) if match else None


def extract_gold_pages(question):
    """Extracts {(week, page): importance} mapping for a benchmark question."""
    gold = {}
    for page in question.get("gold_pages", []):
        # A page that will not parse cannot be matched by any chunk, so counting
        # it would deflate recall against a target nothing can reach.
        key = (week_number(page.get("week")), page_number(page.get("page")))
        if key[1] is None:
            continue

        gain = page.get("utility") or ROLE_GAIN.get(page.get("role"), 2.0)
        gold[key] = max(float(gain), gold.get(key, 0))
    return gold


def flatten_chunks_to_pages(chunks):
    """Flattens chunks into a ranked list of unique (week, page) tuples."""
    ranked_pages, seen = [], set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        week_num = week_number(meta.get("week"))

        pages = meta.get("page_number")
        pages = [pages] if not isinstance(pages, list) else pages

        for p in sorted({(week_num, pg) for pg in map(page_number, pages) if pg is not None}):
            if p not in seen:
                seen.add(p)
                ranked_pages.append(p)
    return ranked_pages


# --------------------------------------------------------------------------
# File Loading Helpers
# --------------------------------------------------------------------------

def load_benchmark(path):
    """The benchmark questions as a list, whatever key they are wrapped in."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = next((data[key] for key in ("queries", "questions", "benchmark")
                     if key in data), [])
    return data


def benchmark_map(questions):
    """{query_id: question} for the loaded benchmark."""
    return {str(q["query_id"]): q for q in questions}


def read_answers(path):
    """(records, budget) for one answer file.

    The single reader for all three scripts. evaluate_generation.read_answers
    takes the records from here and drops the budget, so the two cannot come to
    disagree about what a malformed file means.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None

    # Raise rather than return nothing: an unrecognised wrapper would otherwise
    # score as a silent row of zeros in the comparison table.
    records = next((data[key] for key in RECORD_KEYS if key in data), None)
    if records is None:
        raise ValueError(f"{path}: no answer records found "
                         f"(looked for {', '.join(RECORD_KEYS)})")

    meta = data.get("config") or data.get("metadata") or {}
    budget = next((source[key] for source in (data, meta) for key in BUDGET_KEYS
                   if key in source), None)
    return records, budget


def cell_name(path):
    """'answers_exp1_bge_combo2_1b.json' -> 'exp1_bge_combo2_1b'."""
    return os.path.basename(path).replace("answers_", "").replace(".json", "")


# --------------------------------------------------------------------------
# Scoring Functions
# --------------------------------------------------------------------------

def score_query(record, question, k=10, chunk_filter=None):
    """Scores a single query record against a benchmark question."""
    gold = extract_gold_pages(question)
    if not gold:
        return None

    chunks = record.get("chunks_used", [])
    if chunk_filter:
        chunks = chunk_filter(chunks)
    ranked_pages = flatten_chunks_to_pages(chunks)
    top_k_pages = ranked_pages[:k]

    return {
        f"recall@{k}": compute_recall(top_k_pages, gold),
        f"keyword_recall@{k}": compute_keyword_recall(chunks, question, k),
        f"ndcg@{k}": compute_ndcg(top_k_pages, gold),
        # Cut at k like the rest: a gold page below the cutoff is one the
        # generator never saw, so it should not lift the rank score either.
        f"mrr@{k}": compute_mrr(top_k_pages, gold),
        "hit@1": compute_hit_at_1(ranked_pages, gold)
    }


def score_file(path, bench, k=10, chunk_filter=None):
    """{query_id: metrics} for one answer file."""
    rows = {}
    for record in read_answers(path)[0]:
        qid = str(record.get("question_id"))
        question = bench.get(qid)
        if question is None:
            continue

        scores = score_query(record, question, k=k, chunk_filter=chunk_filter)
        if scores:
            rows[qid] = scores
    return rows


def average(rows):
    """Mean of each metric, plus how many questions had gold pages to score."""
    if not rows:
        return {}

    out = {}
    for metric in next(iter(rows.values())):
        values = [r[metric] for r in rows.values() if r.get(metric) is not None]
        out[metric] = round(float(np.mean(values)), 4) if values else None
    out["n_scored"] = len(rows)
    return out


def score_runs(answers_paths, benchmark_path, k=10):
    """Evaluates multiple answer files against the benchmark."""
    bench = benchmark_map(load_benchmark(benchmark_path))

    summary = {}
    for path in answers_paths:
        summary[cell_name(path)] = average(score_file(path, bench, k=k))

    metrics_keys = (f"recall@{k}", f"keyword_recall@{k}", f"ndcg@{k}", f"mrr@{k}", "hit@1")
    return summary, metrics_keys


# --------------------------------------------------------------------------
# Main Flow
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score retrieval with Top-K and Keyword metrics.")
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out-dir", default="Evaluation/results")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for Top-K metrics")
    args = parser.parse_args()

    summary, metrics_keys = score_runs(args.answers, args.benchmark, k=args.k)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "retrieval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print results table
    width = max((len(n) for n in summary), default=10) + 2
    header = f"\n{'':{width}}" + "".join(f"{m:>18}" for m in metrics_keys)
    print(header)
    print("-" * len(header))
    for name, row in summary.items():
        cells = "".join(f"{row.get(m, 0.0):>18.3f}" if row.get(m) is not None else f"{'-':>18}" for m in metrics_keys)
        print(f"{name:{width}}{cells}")
    print(f"\nSaved to {args.out_dir}/retrieval_summary.json")


if __name__ == "__main__":
    main()
