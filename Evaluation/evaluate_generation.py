"""
Generation scoring: was the answer any good?

Reads an answer file from the inference pipeline and scores each answer.

    python evaluate_generation.py --answers answers_*.json --benchmark bench.json
"""

import argparse
import json
import math
import os
import re
from collections import Counter

import numpy as np

import evaluate_retrieval as ret

# The prompt emits this exact string when the material doesn't cover the
# question, so refusals can be counted separately from wrong answers.
REFUSAL = "the provided course material does not cover this"

# How much of a chunk's vocabulary an answer must share before that chunk
# counts as "used".
UTILISATION_THRESHOLD = 0.10

METRICS = ("token_f1", "rouge_l", "bert_f1", "groundedness", "context_utilisation")

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with without from as is
are was were be been being do does did have has had it its this that these
those there here he she they them we you your i not no so than too very can
will just should would could may might must what which who when where why how
all any both each few more most other some such only own same
""".split())


# ==========================================================================
# 1. Evaluation Metric Functions
# ==========================================================================

def words(text, keep_stopwords=False):
    """Lower-cased content words, with LaTeX and markdown stripped out."""
    text = (text or "").lower()
    text = re.sub(r"\$\$?.*?\$\$?", " ", text, flags=re.DOTALL)   # $...$
    text = re.sub(r"\\[a-z]+", " ", text)                         # \frac
    text = re.sub(r"[*_`#>|]+", " ", text)                        # markdown
    found = re.findall(r"[a-z0-9]+", text)
    return found if keep_stopwords else [w for w in found if w not in STOPWORDS]


def f1(overlap, in_answer, in_gold):
    if not overlap:
        return 0.0
    precision, recall = overlap / in_answer, overlap / in_gold
    return 2 * precision * recall / (precision + recall)


def token_f1(answer, gold):
    """Word overlap, ignoring order."""
    a, g = Counter(words(answer)), Counter(words(gold))
    if not a or not g:
        return float("nan")
    return f1(sum((a & g).values()), sum(a.values()), sum(g.values()))


def rouge_l(answer, gold):
    """Word overlap that also respects order (longest common subsequence)."""
    a, g = words(answer), words(gold)
    if not a or not g:
        return float("nan")

    previous = [0] * (len(g) + 1)
    for word in a:
        current = [0] * (len(g) + 1)
        for j, other in enumerate(g, 1):
            current[j] = (previous[j - 1] + 1 if word == other
                          else max(previous[j], current[j - 1]))
        previous = current
    return f1(previous[-1], len(a), len(g))


def bert_f1_all(answers, golds):
    """BERTScore for a whole file at once."""
    try:
        from bert_score import score
    except ImportError:
        return [float("nan")] * len(answers)
    _, _, scores = score(answers, golds, lang="en",
                         model_type="roberta-large", verbose=False)
    return [float(s) for s in scores]


def groundedness(answer, chunks):
    """Share of the answer's words that appear somewhere in the context."""
    a = words(answer)
    if not a or not chunks:
        return float("nan")
    context = set().union(*[set(words(c)) for c in chunks])
    return sum(1 for w in a if w in context) / len(a)


def context_utilisation(answer, chunks):
    """Share of the retrieved chunks the answer actually drew on."""
    a = set(words(answer))
    if not a or not chunks:
        return float("nan")
    used = 0
    for chunk in chunks:
        c = set(words(chunk))
        if c and len(a & c) / len(c) > UTILISATION_THRESHOLD:
            used += 1
    return used / len(chunks)


def is_refusal(answer):
    return REFUSAL in (answer or "").lower()


# ==========================================================================
# 2. Scoring Functions
# ==========================================================================

def read_answers(path):
    """The answer records, dropping the budget the retrieval reader also returns.

    Delegated so both scorers agree on what a malformed answer file means.
    """
    return ret.read_answers(path)[0]


def chunk_texts(record, chunk_filter=None):
    chunks = record.get("chunks_used", [])
    return [c.get("text", "") for c in (chunk_filter(chunks) if chunk_filter else chunks)]


def score_file(path, bench, use_bert=True, chunk_filter=None):
    """{query_id: metrics} for one answer file."""
    rows = {}
    for record in read_answers(path):
        qid = str(record["question_id"])
        question = bench.get(qid)
        if question is None:
            continue

        answer = record.get("answer", "")
        gold = question.get("qa_gold_standard", "")
        chunks = chunk_texts(record, chunk_filter)

        rows[qid] = {
            "token_f1": token_f1(answer, gold),
            "rouge_l": rouge_l(answer, gold),
            "groundedness": groundedness(answer, chunks),
            "context_utilisation": context_utilisation(answer, chunks),
            "abstained": is_refusal(answer),
            "answer": answer,
        }

    if use_bert and rows:
        ids = list(rows)
        answers = [rows[i]["answer"] for i in ids]
        golds = [bench[i].get("qa_gold_standard", "") for i in ids]
        for qid, score in zip(ids, bert_f1_all(answers, golds)):
            rows[qid]["bert_f1"] = score

    return rows


def average(rows):
    """Mean of each metric, plus how many questions refused to answer."""
    out = {}
    for metric in METRICS:
        values = [r[metric] for r in rows.values()
                  if isinstance(r.get(metric), float) and not math.isnan(r[metric])]
        out[metric] = round(float(np.mean(values)), 4) if values else None
    out["n_queries"] = len(rows)
    out["n_abstained"] = sum(1 for r in rows.values() if r.get("abstained"))
    return out


def cell_name(path):
    """'answers_exp1_bge_combo2_1b.json' -> 'exp1_bge_combo2_1b'."""
    return ret.cell_name(path)


# ==========================================================================
# 3. Main Execution
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Score generated answers.")
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--out-dir", default="Evaluation/results")
    parser.add_argument("--no-bert", action="store_true",
                        help="skip BERTScore (slow, downloads ~1.4GB)")
    args = parser.parse_args()

    bench = ret.benchmark_map(ret.load_benchmark(args.benchmark))
    print(f"{len(bench)} benchmark questions")

    summary = {}
    for path in args.answers:
        name = cell_name(path)
        print(f"  scoring {name}")
        rows = score_file(path, bench, use_bert=not args.no_bert)
        if not rows:
            print(f"  [!] {name}: no question ids matched the benchmark")
        summary[name] = average(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    width = max(len(n) for n in summary) + 2
    print(f"\n{'':{width}}" + "".join(f"{m:>21}" for m in METRICS))
    for name, row in summary.items():
        cells = "".join(f"{row[m]:>21.3f}" if row[m] is not None else f"{'-':>21}"
                        for m in METRICS)
        print(f"{name:{width}}{cells}")
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()