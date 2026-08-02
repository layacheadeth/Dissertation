"""
inference.py — wires every component into a working RAG system.

This is the file ranking_n_retrieval.py's comment refers to ("active_combo is
set in main.py"). Nothing here implements retrieval, chunking, embedding or
generation; it only assembles the parts and hands them to RAGPipeline.

Everything is built ONCE and reused across questions. Loading MiniLM, the
cross-encoder and Qwen takes far longer than answering, so a script that
rebuilds them per question is mostly waiting.

Usage
    python inference.py "What is BM25?"
    python inference.py "What is BM25?" --mode socratic
    python inference.py --interactive
    python inference.py "What is cosine similarity?" --strategy exp3 --combo combo1
    python inference.py "What is BM25?" --retrieve-only     # no LLM, fast
"""

import argparse
import sys
import time

from ranking_n_retrieval import COMBOS, Retriever
from vector_search import VectorSearch

STRATEGIES = ("exp1", "exp2", "exp3", "exp4", "exp5")


# ---------------------------------------------------------------------------
def build_pipeline(strategy="exp5",
                   combo="combo2",
                   chroma_path=None,
                   k=20,
                   top_n=5,
                   mode="qa",
                   load_llm=True,
                   verify=True):
    """Assemble the full system. Returns (pipeline, vector_search, retriever).

    pipeline is None when load_llm=False — useful for retrieval evaluation,
    which does not need a generator and should not pay to load one.
    """
    kwargs = {"chroma_path": chroma_path} if chroma_path else {}
    vs = VectorSearch(strategy, **kwargs)
    print(f"  {vs}")

    if verify:
        # The one check that catches the silent failure: if the corpus was
        # embedded with a different model than the one loaded here, retrieval
        # returns plausible nonsense and nothing raises.
        score = vs.self_check()
        if score < 0.95:
            raise SystemExit(
                f"Self-check failed: {score:.3f} (expected ~1.0).\n"
                f"The collection '{vs.collection_name}' was embedded with a "
                f"different model than the one loaded now. Re-run ingestion:\n"
                f"  python Ingestion_pipeline_code/3-1-Ingest-to-ChromaDB.py --reset"
            )
        print(f"  self-check: {score:.3f}")

    retriever = Retriever(vs.store, vs.documents,
                          active_combo=combo, k=k, top_n=top_n)
    print(f"  {retriever}")

    if not load_llm:
        return None, vs, retriever

    # Imported here, not at module scope: retrieval-only runs should not pay
    # for torch and transformers.
    from llm_n_prompt import PromptTemplate, QwenLLM
    from rag_pipeline import RAGPipeline

    llm = QwenLLM()
    pipeline = RAGPipeline(
        chunker=None,              # corpus is already ingested; no chunking here
        embedder=vs.embedder,      # the SAME embedder VectorSearch loaded
        vectordb=vs.store,
        retriever=retriever,
        prompt_builder=PromptTemplate(),
        llm_qa=llm,
        llm_socratic=llm,          # one model serving both personas
    )
    return pipeline, vs, retriever


# ---------------------------------------------------------------------------
def show_retrieval(docs, scores=None):
    """Print what was retrieved BEFORE the answer.

    This ordering is deliberate. When an answer is wrong the first question is
    always 'was the right chunk retrieved?' — if yes it is a generation
    problem, if no it is a retrieval problem. Conflating the two wastes days.
    """
    print(f"\nRETRIEVED ({len(docs)}):")
    for i, doc in enumerate(docs):
        meta = doc.metadata
        score = f" [{scores[i]:+.2f}]" if scores else ""
        preview = doc.page_content[:90].replace("\n", " ")
        print(f"  {i+1}.{score} {meta.get('week', '?')} "
              f"p{meta.get('page_number', [])} {meta.get('chunk_id', '?')}")
        if meta.get("section_title"):
            print(f"      section: {meta['section_title']}")
        print(f"      {preview}...")


def answer_question(pipeline, vs, retriever, question, mode="qa", show_scores=False):
    """Answer one question and print retrieval, answer and timings."""
    print(f"\n{'=' * 70}\nQ: {question}\n{'=' * 70}")

    t0 = time.perf_counter()
    if show_scores and retriever.active_combo == "combo2":
        scored = retriever.retrieve_with_scores(question, vs.embed(question))
        docs = [d for d, _ in scored]
        show_retrieval(docs, [s for _, s in scored])
    else:
        docs = retriever.retrieve(question, vs.embed(question))
        show_retrieval(docs)
    t_retrieve = time.perf_counter() - t0

    if pipeline is None:
        print(f"\n(retrieval only, {t_retrieve*1000:.0f}ms)")
        return None, docs

    t0 = time.perf_counter()
    answer, docs = pipeline.query(question, mode=mode)
    t_total = time.perf_counter() - t0

    print(f"\nANSWER ({mode}):\n{answer}")
    # Retrieval is typically <1% of this. Generation is the bottleneck, and it
    # is worth seeing that split rather than assuming the vector DB is slow.
    print(f"\n[retrieval {t_retrieve*1000:.0f}ms | full query {t_total:.1f}s]")
    return answer, docs


# ---------------------------------------------------------------------------
def interactive(pipeline, vs, retriever, mode):
    print(f"\nInteractive mode ({mode}). Commands: :qa  :socratic  :quit")
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
        if question in (":qa", ":socratic"):
            mode = question[1:]
            print(f"mode -> {mode}")
            continue
        try:
            answer_question(pipeline, vs, retriever, question, mode=mode)
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Query the COMP64702 RAG system.")
    ap.add_argument("question", nargs="?", help="question to answer")
    ap.add_argument("--strategy", default="exp5", choices=STRATEGIES)
    ap.add_argument("--combo", default="combo2", choices=COMBOS)
    ap.add_argument("--mode", default="qa", choices=["qa", "socratic"])
    ap.add_argument("--k", type=int, default=20, help="candidates before fusion")
    ap.add_argument("--top-n", type=int, default=5, help="chunks sent to the LLM")
    ap.add_argument("--chroma-path", default=None)
    ap.add_argument("--retrieve-only", action="store_true",
                    help="skip the LLM entirely — much faster while tuning retrieval")
    ap.add_argument("--scores", action="store_true",
                    help="show cross-encoder scores (combo2 only)")
    ap.add_argument("--no-verify", action="store_true", help="skip the self-check")
    ap.add_argument("--interactive", "-i", action="store_true")
    args = ap.parse_args()

    if not args.question and not args.interactive:
        ap.error("give a question, or use --interactive")

    print("Building pipeline...")
    pipeline, vs, retriever = build_pipeline(
        strategy=args.strategy,
        combo=args.combo,
        chroma_path=args.chroma_path,
        k=args.k,
        top_n=args.top_n,
        load_llm=not args.retrieve_only,
        verify=not args.no_verify,
    )

    if args.interactive:
        interactive(pipeline, vs, retriever, args.mode)
    else:
        answer_question(pipeline, vs, retriever, args.question,
                        mode=args.mode, show_scores=args.scores)


if __name__ == "__main__":
    main()
