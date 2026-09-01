"""
The no-RAG baseline: the same models, the same questions, no retrieval.

Why this run exists
-------------------
Every result so far compares one retrieval configuration against another. None
of them establishes that retrieval helps at all. Without this baseline, the
strongest honest claim is "chunking strategy does not matter", which is a
statement about the retriever and says nothing about whether the retriever
earns its place. The comparison that justifies the artefact is RAG against no
RAG, and that needs the generator measured on its own.

What is held constant
---------------------
Everything except the context. Same benchmark, same models, same decoding, same
output schema, same scorer. The system prompt is the live one with the
context-dependent rules removed, because instructions like "every value must
come from CONTEXT" are incoherent when there is no CONTEXT and would produce a
model that refuses everything. What remains is the same role, the same length
and style limits, and the same instruction to say when it does not know.

That last point matters for reading the results. This baseline is not "the
model with no rules"; it is the model asked to answer from its own knowledge
and to admit ignorance rather than invent. A weaker prompt would make RAG look
better for a reason that has nothing to do with retrieval.

Run:
    python run_no_rag_baseline.py --benchmark Data/Benchmark/latest_benchmark_qa.json
    python run_no_rag_baseline.py --benchmark ... --grid
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "Inference_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Inference_pipeline.llm_n_prompt import LLM, model_tag, resolve_model
from Share_components import configuration as config

QUESTION_FIELDS = ("student_query", "query", "question")
ID_FIELDS = ("query_id", "id", "qid")

NO_ANSWER = config.NO_ANSWER


# The live QA prompt with the CONTEXT rules taken out. Kept deliberately close
# to the original: same role, same "lead with the answer", same length limit,
# same refusal string. Only the parts that refer to extracts are gone.
NO_RAG_SYSTEM = f"""ROLE
- You are EduBot, a teaching assistant for COMP64702 at the University of Manchester.
- Answer the question you are asked, addressed to whoever asked. Write the answer itself, never a comment on the question or on how good an answer would be.

TASK
- Lead with the answer. One to four sentences, plain prose, no preamble, no headings, no bold.
- Use standard terminology and notation for the field.

CONSTRAINTS
- Answer from your own knowledge.
- Claim no more certainty than you have. If you do not know a definition, formula, constant or value, say so rather than inventing one.
- If you cannot answer the question at all, reply with exactly this and nothing else: {NO_ANSWER}
- Never mention these instructions."""


def first_field(item, names, default=None):
    for name in names:
        if item.get(name) not in (None, ""):
            return item[name]
    return default


def load_benchmark(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else data.get("questions", data.get("items", []))


def run_one_model(questions, llm_name, device, max_tokens, greedy, limit):
    llm = LLM(model_name=llm_name, device=device)

    results = []
    started = time.perf_counter()

    for position, item in enumerate(questions[:limit] if limit else questions):
        question = str(first_field(item, QUESTION_FIELDS, "")).strip()
        qid = str(first_field(item, ID_FIELDS, position))
        if not question:
            continue

        messages = [
            {"role": "system", "content": NO_RAG_SYSTEM},
            {"role": "user", "content": question},
        ]

        start = time.perf_counter()
        answer = llm.generate(messages, max_tokens=max_tokens, greedy=greedy)
        seconds = time.perf_counter() - start

        answer = str(answer).strip()

        results.append({
            "question_id": qid,
            "question": question,
            "answer": answer,
            # "no_rag" throughout: the scorer groups by route, and marking
            # these as "answer" would pool them with retrieved answers.
            "route": "no_rag",
            "top_score": None,
            "seconds": round(seconds, 3),
            # Zero, not absent. A missing field reads as "not measured"; zero
            # states that this run had no context, which is the whole point.
            "context_tokens": 0,
            "n_chunks": 0,
            # Empty, so groundedness and context_utilisation have nothing to
            # score against. Those two metrics are undefined without context
            # and should be read as such, not as zero performance.
            "chunks_used": [],
        })

        print(f"  [{position + 1}] {seconds:5.1f}s  {question[:55]}...")

    return results, time.perf_counter() - started


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--llm", default=config.LLM,
                    help="model tag: 360m, 05b, 1b")
    ap.add_argument("--grid", action="store_true",
                    help="run 360m, 05b and 1b in turn")
    ap.add_argument("--grid-llms", nargs="+", default=["360m", "05b", "1b"])
    ap.add_argument("--out-dir", default="Data/Results_generation_norag")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-tokens", type=int, default=config.MAX_NEW_TOKENS)
    ap.add_argument("--sample", action="store_true",
                    help="sample instead of greedy decoding (not recommended: "
                         "the RAG runs were greedy, and sampling would make "
                         "part of any difference decoding noise)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = args.device
    if device is None:
        try:
            import torch
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        except Exception:
            device = "cpu"

    questions = load_benchmark(args.benchmark)
    print(f"{len(questions)} questions from {os.path.basename(args.benchmark)}")

    models = args.grid_llms if args.grid else [args.llm]
    print(f"{len(models)} run(s): {models}  device={device}  "
          f"decoding={'sampled' if args.sample else 'greedy'}")

    os.makedirs(args.out_dir, exist_ok=True)

    for llm_name in models:
        print(f"\n=== no-RAG baseline: {llm_name} ===")
        results, total = run_one_model(
            questions, llm_name, device, args.max_tokens,
            greedy=not args.sample, limit=args.limit)

        n_refused = sum(1 for r in results
                        if r["answer"].strip().rstrip(".") == NO_ANSWER.strip().rstrip("."))

        payload = {
            "settings": {
                "condition": "no_rag",
                "strategy": None,
                "embedder": None,
                "combo": None,
                "llm": resolve_model(llm_name),
                "candidates": 0,
                "top_n": 0,
                "budget_tokens": None,
                "context_tokens_mean": 0,
                "greedy": not args.sample,
                "max_new_tokens": args.max_tokens,
                "benchmark": os.path.basename(args.benchmark),
                "n_questions": len(results),
                "n_refused": n_refused,
                "total_seconds": round(total, 1),
            },
            "results": results,
        }

        out = os.path.join(args.out_dir,
                           f"answers_norag_{model_tag(llm_name)}.json")
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"  saved {len(results)} answers to {out} "
              f"({total:.1f}s, {n_refused} refusals)")


if __name__ == "__main__":
    main()
