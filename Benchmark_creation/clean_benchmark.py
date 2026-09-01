#!/usr/bin/env python3
"""
build_benchmark.py

Converts the ORIGINAL benchmark_qa.json into latest_benchmark_qa.json.

Nothing about the gold pages is asserted by hand. Two inputs drive everything:

  CORRECTIONS        (below)             factual fixes to questions and answers
  CLAIMS / EVIDENCE  (evidence_tables)   answers decomposed into atomic claims,
                                         and every candidate slide judged
                                         against every claim

From EVIDENCE the script derives gold_pages by greedy set cover:

  rank 1 (primary) = slide entailing the most claims
  rank n           = slide adding the most claims not yet covered
  tie-break        = lower week, then lower page. Lower week means the
                     original teaching slide beats the week 13 revision slide
                     that restates it, which is the direction a student would
                     be sent.
  stop             = all claims covered, or MAX_GOLD_PAGES reached
  excluded         = any slide whose marginal gain is zero

CRITICAL DESIGN POINT
    EVIDENCE holds the COMPLETE judged pool, including slides that entail
    nothing (empty claim list) and slides that entail claims already covered.
    An earlier version recorded only the slides that survived pruning, which
    made every gold page look like the unique source for its claims - the
    `necessary` flag came out true 44 times out of 44, an artefact of the
    missing denominator rather than a property of the corpus. Uniqueness is
    only meaningful when counted over everything considered.

    Slides that entail a claim but are not needed for the cover are emitted
    as `alternate_pages`. They are NOT gold: a retriever should not be
    required to find them. They are also not noise: scoring code should
    exclude them from the precision denominator rather than count them wrong.

WHY ENTAILMENT AND NOT KEYWORDS OR COSINE
    Keyword overlap misses paraphrase - w12 p29 supports EXAM_001's
    attachment claim without ever containing the string "nmod".
    Cosine similarity with the retrieval embedder is circular: it lets
    MiniLM define the gold pages it is then scored against, so whichever
    slides MiniLM likes become "correct" by construction.
    Every verdict here carries a quoted span. No quotable span, no support.

PROVENANCE
    The verdicts come from a single LLM judge (Claude) reading each claim
    against each candidate slide. They are recorded so they can be audited,
    disputed, or regenerated. Before publishing results, hand-label a sample
    of (claim, slide) pairs and report Cohen's Kappa against this table.

USAGE
    python3 build_benchmark.py \
        --original benchmark_qa.json \
        --corpus   Data/All_extracted_text \
        --output   latest_benchmark_qa.json \
        --report   BUILD_REPORT.txt
"""

import argparse
import glob
import json
import os
import re
import sys

from Benchmark_creation.evidence_tables import CLAIMS, EVIDENCE, UNSUPPORTED

MAX_GOLD_PAGES = 5

# =========================================================================
# CORRECTIONS - factual fixes, applied before evidence is attached
# =========================================================================
DROP = {
    "EXAM_012": "The strings 'entity linking', 'candidate generation' and "
                "'candidate ranking' occur zero times across all 13 lectures. "
                "Unanswerable from the corpus.",
}

REWRITE = {
    "EXAM_009": {
        "reason": "The answer said word=row, document=column, inverting its own primary "
                  "slide (w1 p69: rows are documents, columns are vocabulary words).",
        "student_query": "In the document-word (bag-of-words) matrix used to represent a "
            "corpus, what do the rows and columns correspond to, and how is a single "
            "document represented as a vector?",
        "qa_gold_standard": "The matrix X is |D| x |V|: the rows are the documents in the "
            "corpus and the columns are the vocabulary words, and each cell counts how many "
            "times that word appears in that document. A document is therefore represented "
            "by its row, i.e. its counts over the whole vocabulary, and a word by its "
            "column, i.e. its counts across all the documents.",
        "keywords": ["document-word matrix", "bag-of-words", "rows and columns",
                     "vocabulary", "document vector", "term counts"],
    },
    "EXAM_013": {
        "reason": "'hyponym' has 0 corpus hits; 'antonym' appears only in w1 p83 about word "
                  "vectors, not WordNet. Rescoped to the relations w3 p3 actually teaches.",
        "student_query": "According to the traditional symbolic approach to representing "
            "word meaning (e.g. WordNet), which explicit lexical relations are used, and "
            "what are the stated properties of meaning under this approach?",
        "qa_gold_standard": "WordNet represents meaning through explicit symbolic relations "
            "between words, specifically synonymy (words with the same meaning) and "
            "hypernymy, which links a word to a more general category (for example dog to "
            "animal). Under this approach meaning is discrete, manually defined, and "
            "relational rather than learned from data.",
        "keywords": ["WordNet", "synonymy", "hypernymy", "lexical resource",
                     "symbolic representation", "word meaning"],
    },
    "EXAM_014": {
        "reason": "'semantic role' has 0 corpus hits; the Semantic Role Labelling claim was "
                  "imported from outside the corpus.",
        "qa_gold_standard": "Sequence labelling suits tasks that assign one label to each "
            "token in a sequence, in order. Named Entity Recognition is formulated this way, "
            "with each token tagged for whether it is inside an entity and which type, and "
            "Part-of-Speech tagging follows the same pattern, assigning one tag per token.",
    },
    "EXAM_017": {
        "reason": "'keeps the sequence length unchanged' is stated by no slide. Replaced "
                  "with the scaling and softmax steps w3 p43 does teach.",
        "qa_gold_standard": "Self-attention derives a query, key and value vector for each "
            "token by multiplying the input embeddings with learned weight matrices. "
            "Attention scores are the scaled dot products between queries and keys, passed "
            "through a softmax to give a distribution, which is then used to take a weighted "
            "sum of the value vectors. Each token can attend to every other token in the "
            "sequence, not just its neighbours, producing an updated contextualised "
            "representation.",
    },
    "EXAM_023": {
        "reason": "No dependency-level 'punct' relation exists in the corpus; PUNCT appears "
                  "only as a POS tag on w13 p18.",
        "qa_gold_standard": "The head of the sentence is the verb 'saw', and its direct "
            "dependents are 'boy' as the subject and 'man' as the object. 'telescope' is not "
            "a dependent of 'saw', because under this reading the prepositional phrase "
            "attaches to the noun 'man' instead.",
    },
    "EXAM_026": {
        "reason": "0.5555... was truncated rather than rounded.",
        "qa_gold_standard": "Kappa is about 0.556. Using Kappa = (observed - expected) / "
            "(1 - expected), this is (0.75 - 0.4375) / (1 - 0.4375) = 0.3125 / 0.5625 = "
            "0.5556, which rounds to 0.556.",
    },
    "EXAM_029": {
        "reason": "Arithmetic: the stated working 7 x 0.125 gives 0.875, not 0.874 (exact "
                  "7*log10(4/3) = 0.87457). Separately, 'term-document matrix', 'battle' and "
                  "the Shakespeare titles have 0 corpus hits - recast in the corpus's own "
                  "'document-word matrix' terminology. Arithmetic unchanged.",
        "student_query": "In a document-word matrix built from 4 documents, the term "
            "'battle' occurs 7 times in Document 2 and appears in 3 of the 4 documents. "
            "Using idf(t) = log10(N/df(t)) with raw term frequency tf, what is the tf-idf "
            "value of 'battle' in Document 2?",
        "qa_gold_standard": "tf-idf('battle', Document 2) = 0.875. With tf = 7 and df = 3 "
            "out of N = 4 documents, idf = log10(4/3) = 0.125, so 7 x 0.125 = 0.875.",
        "keywords": ["tf-idf", "idf", "document frequency", "term frequency",
                     "document-word matrix", "log10"],
    },
}

RETYPE = {
    "EXAM_019": ("procedural", "Asks how a quantity is computed without supplying numbers - "
                               "neither definition recall nor arithmetic."),
    "EXAM_020": ("procedural", "Asks how a quantity is computed without supplying numbers."),
}


# =========================================================================
# Corpus
# =========================================================================
_WEEK_RE = re.compile(r"Data_week(\d+)")


def load_corpus(root):
    """(week, page) -> cleaned slide text, from the raw lecture JSONs."""
    corpus = {}
    paths = [p for p in glob.glob(os.path.join(root, "Data_week*", "*.json"))
             if "_chunks.json" not in os.path.basename(p)]
    if not paths:
        paths = [p for p in glob.glob(os.path.join(root, "*.json"))
                 if "_chunks.json" not in os.path.basename(p)]
    for path in paths:
        m = _WEEK_RE.search(path)
        if not m:
            continue
        week = int(m.group(1))
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for page in data.get("pages", []):
            corpus[(week, page["page_number"])] = page["content"]
    if not corpus:
        raise SystemExit(f"No lecture JSONs found under {root}")
    return corpus


# =========================================================================
# Greedy set cover - this IS the ranking rule
# =========================================================================
def derive_pages(qid, corpus):
    claims = CLAIMS[qid]
    target = {c for c in claims if (qid, c) not in UNSUPPORTED}
    evidence = EVIDENCE[qid]

    # source count per claim, over the COMPLETE judged pool
    sources = {c: sum(1 for pc, _ in evidence.values() if c in pc) for c in claims}

    remaining = set(target)
    gold, pool = [], {k: v for k, v in evidence.items() if v[0]}

    while remaining and pool and len(gold) < MAX_GOLD_PAGES:
        (week, page), (page_claims, quote) = max(
            pool.items(),
            key=lambda kv: (len(set(kv[1][0]) & remaining),   # marginal gain
                            len(kv[1][0]),                    # total coverage
                            -kv[0][0],                        # lower week wins
                            -kv[0][1]))                       # lower page wins
        gain = set(page_claims) & remaining
        if not gain:
            break
        gold.append({
            "rank": len(gold) + 1,
            "week": week,
            "page": page,
            "role": "primary" if not gold else "supporting",
            "claims_supported": sorted(page_claims),
            "marginal_claims": sorted(gain),
            "necessary": any(sources[c] == 1 for c in gain),
            "evidence": quote,
            "why": ("Sole source in the corpus for " if any(sources[c] == 1 for c in gain)
                    else "Supplies ") + "; ".join(claims[c] for c in sorted(gain)),
            "content": corpus[(week, page)],
        })
        remaining -= gain
        del pool[(week, page)]

    chosen = {(g["week"], g["page"]) for g in gold}
    alternates = [
        {"week": w, "page": p, "claims_supported": sorted(pc), "evidence": q,
         "note": "Entails claims already covered by a gold page. Not required; "
                 "should not be scored as an error if retrieved."}
        for (w, p), (pc, q) in sorted(evidence.items())
        if pc and (w, p) not in chosen
    ]
    judged_no_support = sorted(f"w{w}p{p}" for (w, p), (pc, _) in evidence.items() if not pc)
    return gold, alternates, sorted(remaining), judged_no_support


# =========================================================================
# Build
# =========================================================================
def build(original_path, corpus_root):
    with open(original_path, "r", encoding="utf-8") as f:
        original = json.load(f)
    corpus = load_corpus(corpus_root)

    out, report = [], []
    for item in original:
        qid = item["query_id"]
        if qid in DROP:
            report.append(f"{qid}  DROPPED - {DROP[qid]}")
            continue

        q = dict(item)
        if qid in REWRITE:
            patch = dict(REWRITE[qid])
            report.append(f"{qid}  REWRITTEN - {patch.pop('reason')}")
            q.update(patch)
        if qid in RETYPE:
            new_type, reason = RETYPE[qid]
            report.append(f"{qid}  RETYPED {q['question_type']} -> {new_type} - {reason}")
            q["question_type"] = new_type

        gold, alternates, uncovered, no_support = derive_pages(qid, corpus)
        if not gold:
            raise SystemExit(f"{qid}: no entailing evidence recorded")

        q["claims"] = CLAIMS[qid]
        q["unsupported_claims"] = {c: UNSUPPORTED[(qid, c)]
                                   for c in CLAIMS[qid] if (qid, c) in UNSUPPORTED}
        q["gold_pages"] = gold
        q["alternate_pages"] = alternates
        out.append(q)

        line = (f"{qid}  {len(gold)} gold, {len(alternates)} alternate, "
                f"{len(no_support)} judged-no-support")
        if uncovered:
            line += f"  | UNCOVERED: {uncovered}"
        report.append(line)

    return out, report


def validate(benchmark, corpus):
    def norm(t):
        return " ".join(t.lower().split())
    problems = []
    for q in benchmark:
        gp = q["gold_pages"]
        if [g["rank"] for g in gp] != list(range(1, len(gp) + 1)):
            problems.append(f"{q['query_id']}: non-contiguous ranks")
        if sum(g["role"] == "primary" for g in gp) != 1 or gp[0]["role"] != "primary":
            problems.append(f"{q['query_id']}: primary must be exactly one, at rank 1")
        if len(gp) > MAX_GOLD_PAGES:
            problems.append(f"{q['query_id']}: exceeds {MAX_GOLD_PAGES} gold pages")
        seen = set()
        for g in gp:
            key = (g["week"], g["page"])
            if key in seen:
                problems.append(f"{q['query_id']}: duplicate {key}")
            seen.add(key)
            if key not in corpus:
                problems.append(f"{q['query_id']}: {key} absent from corpus")
            elif norm(corpus[key]) != norm(g["content"]):
                problems.append(f"{q['query_id']}: {key} content drift")
            if not g["marginal_claims"]:
                problems.append(f"{q['query_id']}: {key} has zero marginal gain")
        for a in q["alternate_pages"]:
            if (a["week"], a["page"]) in seen:
                problems.append(f"{q['query_id']}: alternate duplicates a gold page")
        covered = {c for g in gp for c in g["claims_supported"]}
        target = {c for c in q["claims"] if c not in q["unsupported_claims"]}
        if not target <= covered:
            problems.append(f"{q['query_id']}: claims uncovered {sorted(target - covered)}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default="benchmark_qa.json")
    ap.add_argument("--corpus", default="Data/All_extracted_text")
    ap.add_argument("--output", default="latest_benchmark_qa.json")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    benchmark, report = build(args.original, args.corpus)
    corpus = load_corpus(args.corpus)
    problems = validate(benchmark, corpus)

    for line in report:
        print(line)
    print()
    if problems:
        print("VALIDATION FAILED")
        for p in problems:
            print("  " + p)
        sys.exit(1)

    ngold = sum(len(q["gold_pages"]) for q in benchmark)
    nalt = sum(len(q["alternate_pages"]) for q in benchmark)
    nnec = sum(1 for q in benchmark for g in q["gold_pages"] if g["necessary"])
    print(f"validation OK | {len(benchmark)} questions | {ngold} gold pages "
          f"(mean {ngold / len(benchmark):.2f}) | {nalt} alternates | "
          f"{nnec}/{ngold} gold pages are the sole corpus source for a claim")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    print(f"written to {args.output}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")
        print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
