"""
Answering one question, from the command line.

Run it:
    python run_inference_one_question.py "What is BM25?"
    python run_inference_one_question.py --interactive
    python run_inference_one_question.py "What is BM25?" --retrieve-only
    python run_inference_one_question.py "What can you do?"          # routed
    python run_inference_one_question.py "What is BM25?" --no-save

This file is now only the front of the system: it reads the flags, prints the
chunks, prints the answer and prints the timings. The system itself — the four
steps, build_pipeline, and the shared flags — lives in RAG_pipeline.py, which
run_inference_bigfile.py imports too, so the interactive tool and the benchmark
cannot end up running different systems.

Stage 4 of inference, and the only one that runs the assembled pipeline rather
than a component of it. Stages 1 to 3 show what each part hands the next; this
shows the router and the relevance gate, which sit above all three and appear
nowhere else. Its record is written in the same envelope, so the four can be
read in order.

Reads  Data/Database/chroma_db/
Writes Data/Results_stage4/<strategy>_<model>_<slug>.json

Defaults come from Share_components/configuration.py. The flags below override
them for one run.
"""

import argparse
import time

# RAG_pipeline puts the project root on sys.path when it loads, so nothing has
# to be done here.
from RAG_pipeline import add_pipeline_flags, build_pipeline

from Share_components import configuration as config_inference

STAGE = "4-pipeline"
STAGE_DIR = config_inference.ROOT / "Data" / "Results_stage4"


# ---------------------------------------------------------------------------
# Showing the result
# ---------------------------------------------------------------------------
def show_chunks(documents, scores=None):
    """Print the chunks that were found, before the answer.

    That order is deliberate. When an answer is wrong, the first question is
    whether the right chunk was even found. If it was, the model mishandled it;
    if it was not, retrieval is the problem. Mixing the two up wastes days.
    """
    print(f"\nFOUND {len(documents)} chunks:")
    for i, doc in enumerate(documents):
        meta = doc.metadata
        score = f" [{scores[i]:+.2f}]" if scores else ""
        preview = doc.page_content[:90].replace("\n", " ")
        print(f"  {i+1}.{score} {meta.get('week', '?')} "
              f"p{meta.get('page_number', [])} {meta.get('chunk_id', '?')}")
        if meta.get("section_title"):
            print(f"      section: {meta['section_title']}")
        print(f"      {preview}...")


def answer_question(pipeline, search, retriever, question, show_scores=False):
    """Answer one question and print the chunks, the answer and the timings.

    Returns a dictionary of what happened, so the caller can save it. The
    route is part of that: a refusal the gate produced and a refusal the model
    produced are different outcomes and must not be recorded as the same thing.
    """
    print(f"\n{'=' * 70}\nQ: {question}\n{'=' * 70}")

    # Searched once and reused below. Searching again for the answer would run
    # the reranker a second time and make the timings meaningless.
    start = time.perf_counter()

    if pipeline is None:
        # Retrieval-only: no pipeline, so no router and no gate to apply.
        if show_scores and retriever.active_combo == "combo2":
            scored = retriever.retrieve_with_scores(question, search.embed(question))
            documents = [doc for doc, _ in scored]
            scores = [score for _, score in scored]
            show_chunks(documents, scores)
        else:
            documents = retriever.retrieve(question, search.embed(question))
            scores = None
            show_chunks(documents)
        search_seconds = time.perf_counter() - start
        print(f"\n(search only, {search_seconds*1000:.0f}ms)")
        return {"route": "retrieve-only", "answer": None, "documents": documents,
                "scores": scores, "search_seconds": search_seconds,
                "answer_seconds": 0.0}

    # prepare() routes, searches and gates, but does not generate, so the
    # chunks can still be shown before the answer.
    plan = pipeline.prepare(question)
    search_seconds = time.perf_counter() - start

    if plan.route == "meta":
        print("\n(not a course question, answered without searching)")
        print(f"\nANSWER:\n{plan.answer}")
        print(f"\n[routed in {search_seconds*1000:.0f}ms]")
        return {"route": "meta", "answer": plan.answer, "documents": plan.documents,
                "scores": plan.scores, "search_seconds": search_seconds,
                "answer_seconds": 0.0}

    show_chunks(plan.documents,
                plan.scores if (show_scores and plan.scores) else None)

    if plan.route == "refused":
        # The gate stopped it, not the model. Worth saying which, because the
        # fix is different: a floor set too high against a model that ignores
        # its instructions.
        print(f"\n(best score {plan.top_score:+.2f} is below the floor "
              f"{pipeline.relevance_floor:+.2f}, so the model was not asked)")
        print(f"\nANSWER:\n{plan.answer}")
        print(f"\n[search {search_seconds*1000:.0f}ms]")
        return {"route": "refused", "answer": plan.answer,
                "documents": plan.documents, "scores": plan.scores,
                "search_seconds": search_seconds, "answer_seconds": 0.0}

    documents = plan.documents
    start = time.perf_counter()
    answer = pipeline.answer(question, documents)
    write_seconds = time.perf_counter() - start

    print(f"\nANSWER:\n{answer}")
    # Searching is usually well under 1% of the total. Writing the answer is
    # the slow part, and it is worth seeing that rather than blaming the database.
    print(f"\n[search {search_seconds*1000:.0f}ms | answer {write_seconds:.1f}s]")
    return {"route": "answer", "answer": answer, "documents": documents,
            "scores": plan.scores, "search_seconds": search_seconds,
            "answer_seconds": write_seconds}


# ---------------------------------------------------------------------------
# Saving what happened
# ---------------------------------------------------------------------------
def stage_settings(args, search, retriever, pipeline):
    """Everything that decides what this run produced."""
    return {
        "strategy": search.strategy,
        "embedder": search.embedder_name,
        "collection": search.collection_name,
        "combo": retriever.active_combo,
        "candidates": retriever.candidates,
        "top_n": retriever.top_n,
        "budget_tokens": retriever.budget_tokens,
        "model": None if pipeline is None else pipeline.llm.model_name,
        "route_meta": None if pipeline is None else pipeline.route_meta,
        "relevance_floor": None if pipeline is None else pipeline.relevance_floor,
        "retrieve_only": pipeline is None,
    }


def stage_hash(settings, pipeline):
    """A short code for these settings, with the prompt text when there is one.

    The prompt is included because Experiment 5 found it moved token-F1 by more
    than any other factor measured: two runs with the same model and different
    prompt wording are not the same run.
    """
    import hashlib
    parts = [f"{key}={settings[key]}" for key in sorted(settings)]
    if pipeline is not None:
        prompt = pipeline.prompt_builder
        parts += [prompt.qa_system, prompt.qa_few_shot]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def write_record(question, result, settings, hashed):
    """One JSON file, in the envelope stages 1 to 3 also use."""
    import json

    from llm_n_prompt import PromptTemplate

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in question.lower()).strip("_")[:40]
    model = (settings["model"] or "nollm").split("/")[-1].replace(".", "").lower()
    path = STAGE_DIR / f"{settings['strategy']}_{model}_{slug}.json"

    answer = result["answer"]
    record = {
        "stage": STAGE,
        "settings": settings,
        "settings_hash": hashed,
        "input": {"question": question},
        "output": {
            "route": result["route"],
            "answer": answer,
            "is_no_answer": bool(answer) and PromptTemplate.is_no_answer(answer),
            "is_out_of_scope": bool(answer) and PromptTemplate.is_out_of_scope(answer),
            "searched": result["route"] != "meta",
            "chunks": [
                {"rank": rank,
                 "chunk_id": doc.metadata.get("chunk_id"),
                 "week": doc.metadata.get("week"),
                 "page_number": doc.metadata.get("page_number"),
                 "token_count": doc.metadata.get("token_count"),
                 "score": (round(float(result["scores"][rank - 1]), 4)
                           if result.get("scores") else None)}
                for rank, doc in enumerate(result["documents"], 1)
            ],
            "context_tokens": sum(int(doc.metadata.get("token_count") or 0)
                                  for doc in result["documents"]),
        },
        "seconds": {
            "search": round(result["search_seconds"], 3),
            "answer": round(result["answer_seconds"], 2),
            "total": round(result["search_seconds"] + result["answer_seconds"], 2),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def interactive(pipeline, search, retriever):
    print("\nAsk away. Type :quit to leave.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in (":quit", ":q", "exit"):
            break
        try:
            answer_question(pipeline, search, retriever, question)
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ask the system a question.")
    parser.add_argument("question", nargs="?", help="the question to answer")
    add_pipeline_flags(parser)
    parser.add_argument("--retrieve-only", action="store_true",
                        help="skip the language model, much faster while "
                             "working on retrieval")
    parser.add_argument("--scores", action="store_true",
                        help="show the reranker's scores (combo2 only)")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--no-save", action="store_true",
                        help="print only, write no JSON")
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.error("give a question, or use --interactive")

    print("Loading...")
    pipeline, search, retriever = build_pipeline(
        strategy=args.strategy,
        combo=args.combo,
        embedder=args.embedder,
        llm_name=args.llm,
        candidates=args.candidates,
        top_n=args.top_n,
        budget_tokens=args.budget_tokens,
        device=args.device,
        load_llm=not args.retrieve_only,
        verify=not args.no_verify,
        route_meta=not args.no_router,
        relevance_floor=args.relevance_floor,
    )

    if args.interactive:
        # Not saved: an exploratory session is not a record, and writing one
        # file per question would bury the runs that were meant to be kept.
        interactive(pipeline, search, retriever)
        return

    result = answer_question(pipeline, search, retriever, args.question,
                             show_scores=args.scores)

    if args.no_save:
        print("\n(not saved)")
        return

    settings = stage_settings(args, search, retriever, pipeline)
    path = write_record(args.question, result, settings,
                        stage_hash(settings, pipeline))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
