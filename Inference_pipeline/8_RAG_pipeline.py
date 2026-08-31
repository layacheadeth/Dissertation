"""
The system itself: what a question passes through, and how it is assembled.

This module holds everything both entry points need, and nothing either of
them alone needs. It has no command line and prints nothing except the
progress lines build_pipeline is asked for, so it can be imported by a script,
a notebook or a test without side effects.

    run_inference_one_question.py   asks one question and prints it nicely
    run_inference_bigfile.py        answers a whole benchmark file

Both import build_pipeline from here rather than assembling their own copy, so
the interactive tool and the benchmark cannot end up running different
systems. add_pipeline_flags lives here for the same reason: one definition of
--strategy, --combo, --llm and the rest, so their defaults cannot drift apart.

Four steps answer a question:

    1. turn the question into a vector
    2. find the chunks that match it
    3. put those chunks into a prompt
    4. ask the language model

The decisions taken around those steps — the meta router before the search and
the relevance gate after it — live in routing.py.

No chunking happens here. The slides were chunked and stored by the ingestion
pipeline long before any question is asked.

Everything is loaded once and reused. Loading the models takes far longer than
answering, so anything that reloads per question is mostly just waiting.

Defaults come from Share_components/configuration.py. The flags override them
for one run.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The root, for Share_components and Ingestion_pipeline; this folder, so the
# sibling modules import by bare name however this one was reached.
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.append(_p)

from ranking_n_retrieval import COMBOS, Retriever
from vector_search import SUFFIXES, VectorSearch

from routing import (GATE_COMBO, META_ANSWER, NO_ANSWER, RELEVANCE_FLOOR,
                     Plan, below_floor, check_floor_supported, gate_active,
                     is_meta_question, meta_plan, refusal_plan)

from Share_components import configuration as config_inference

from Ingestion_pipeline.ingestion_strategy import COLLECTIONS

STRATEGIES = tuple(COLLECTIONS)


# ---------------------------------------------------------------------------
# The four steps
# ---------------------------------------------------------------------------
class RAGPipeline:
    """Answers one question."""

    def __init__(self, embedder, retriever, prompt_builder, llm,
                 route_meta=True, relevance_floor=None):
        """
        embedder        : turns text into vectors, the model that built the chunks
        retriever       : finds the matching chunks
        prompt_builder  : puts the chunks and the question into a prompt
        llm             : the language model that writes the answer
        route_meta      : answer "what can you do" without searching
        relevance_floor : refuse when the best reranker score is below this.
                          None turns the gate off. combo2 only, because the
                          other modes have no calibrated relevance score.
        """
        self.embedder = embedder
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.route_meta = route_meta
        self.relevance_floor = relevance_floor

    @property
    def gate_active(self):
        return gate_active(self.relevance_floor, self.retriever.active_combo)

    def retrieve(self, question):
        """Steps 1 and 2. Separate, so a caller can show the chunks before the
        answer without searching twice."""
        return self.retriever.retrieve(question, self.embedder.embed_query(question))

    def prepare(self, question):
        """Everything up to generation: route, search, then gate. Returns a Plan.

        Split out from answering so the interactive tool can print the chunks
        first and the benchmark can record why a question was refused, neither
        of which should cost a second search.
        """
        # Not searched at all: there is nothing in the slides to find, and
        # sending it on spends the reranker and the model on a coin flip.
        if self.route_meta and is_meta_question(question):
            return meta_plan()

        vector = self.embedder.embed_query(question)

        # Under combo2 the reranker has already scored every candidate, so its
        # scores cost nothing extra to keep — and they are the only calibration
        # data there is for setting a floor later.
        if self.retriever.active_combo != GATE_COMBO:
            return Plan("answer", documents=self.retriever.retrieve(question, vector))

        scored = self.retriever.retrieve_with_scores(question, vector)
        documents = [doc for doc, _ in scored]
        scores = [score for _, score in scored]

        if self.gate_active and below_floor(scores, self.relevance_floor):
            return refusal_plan(documents, scores)

        return Plan("answer", documents=documents, scores=scores)

    def answer(self, question, documents=None):
        """Steps 3 and 4. Pass documents in if you already called retrieve()."""
        if documents is None:
            documents = self.retrieve(question)
        prompt = self.prompt_builder.build_prompt(question, documents)
        return self.llm.generate(prompt)

    def run(self, question):
        """Router, search, gate and generation. Returns a finished Plan.

        The Plan carries how the answer was reached, which the benchmark saves:
        a refusal the gate produced and a refusal the model produced are
        different results and should not be counted as the same thing.
        """
        plan = self.prepare(question)
        if plan.answer is None:
            plan.answer = self.answer(question, plan.documents)
        return plan

    def query(self, question):
        """All four steps, as (answer, chunks). Kept for callers that only want
        the two; run() is the fuller version."""
        plan = self.run(question)
        return plan.answer, plan.documents


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def get_device():
    """cuda if there is a GPU, mps on a Mac, otherwise cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_pipeline(strategy=config_inference.STRATEGY,
                   combo=config_inference.COMBO,
                   embedder=config_inference.EMBEDDER,
                   llm_name=config_inference.LLM,
                   candidates=config_inference.CANDIDATES,
                   top_n=config_inference.TOP_N,
                   budget_tokens=getattr(config_inference, "BUDGET_TOKENS", None),
                   device=None,
                   load_llm=True,
                   verify=True,
                   quiet=False,
                   route_meta=True,
                   relevance_floor=RELEVANCE_FLOOR):
    """Load everything. Returns (pipeline, search, retriever).

    pipeline is None when load_llm is False, which is what retrieval-only runs
    want: they do not need a language model and should not wait for one.
    """
    say = (lambda *a: None) if quiet else print

    check_floor_supported(relevance_floor, combo)

    search = VectorSearch(strategy, embedder=embedder)
    say(f"  {search}")

    if verify:
        # The one check that catches the silent failure. If the chunks were
        # embedded with a different model than the one loaded now, search
        # returns believable nonsense and nothing raises an error.
        score = search.self_check()
        if score < 0.95:
            raise SystemExit(
                f"Self-check failed: {score:.3f}, expected about 1.0.\n"
                f"Collection '{search.collection_name}' was built with a "
                f"different model than the one loaded now. Re-run:\n"
                f"  python Ingestion_pipeline/3-1-Ingest-to-ChromaDB-bge-Embed.py"
            )
        say(f"  self-check: {score:.3f}")

    # A budget only holds context equal if the pool is deep enough to fill it.
    # exp1 needs about 20 chunks for 1500 tokens, so the default 20 candidates
    # would cap it below budget while exp2 filled easily — reintroducing the
    # very imbalance the budget exists to remove.
    budget_candidates = getattr(config_inference, "BUDGET_CANDIDATES", 60)
    if budget_tokens and candidates < budget_candidates:
        say(f"  [warning] --budget-tokens {budget_tokens} with only {candidates} "
            f"candidates. Strategies with small chunks may not reach the budget. "
            f"Suggested: --candidates {budget_candidates}")

    retriever = Retriever(search.store, search.documents, combo=combo,
                          candidates=candidates, top_n=top_n,
                          budget_tokens=budget_tokens)
    say(f"  {retriever}")

    if not load_llm:
        return None, search, retriever

    # Imported here rather than at the top, so retrieval-only runs do not pay
    # to load torch and transformers.
    from llm_n_prompt import LLM, PromptTemplate, check_model_available, resolve_model

    model_id = resolve_model(llm_name)
    check_model_available(model_id)      # fails fast, before anything slow loads

    pipeline = RAGPipeline(
        embedder=search.embedder,        # the same model VectorSearch loaded
        retriever=retriever,
        prompt_builder=PromptTemplate(),
        llm=LLM(model_id, device=device or get_device()),
        route_meta=route_meta,
        relevance_floor=relevance_floor,
    )
    return pipeline, search, retriever


def add_pipeline_flags(parser):
    """The flags that say which pipeline to build.

    Shared by both entry points so the two cannot drift apart: changing a
    default in one place changes it everywhere.
    """
    parser.add_argument("--strategy", default=config_inference.STRATEGY, choices=STRATEGIES)
    parser.add_argument("--embedder", default=config_inference.EMBEDDER, choices=list(SUFFIXES))
    parser.add_argument("--combo", default=config_inference.COMBO, choices=COMBOS)
    parser.add_argument("--llm", default=config_inference.LLM,
                        help="a tag from llm_n_prompt.MODELS, or a full model name")
    parser.add_argument("--candidates", type=int, default=config_inference.CANDIDATES,
                        help="chunks each search method puts forward")
    parser.add_argument("--top-n", type=int, default=config_inference.TOP_N,
                        help="chunks sent to the language model")
    parser.add_argument("--budget-tokens", type=int,
                        default=getattr(config_inference, "BUDGET_TOKENS", None),
                        help="equal-context control: fill this many tokens of "
                             "context in rank order instead of taking a fixed "
                             "--top-n chunks, so strategies with different "
                             "chunk sizes are compared on the same amount of "
                             "text (suggested: 1500, with --candidates 60)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the self-check")
    parser.add_argument("--no-router", action="store_true",
                        help="send questions about the system itself "
                             "('what can you teach me') through retrieval "
                             "instead of answering them directly")
    parser.add_argument("--relevance-floor", type=float, default=RELEVANCE_FLOOR,
                        help=f"refuse when the best reranker score is below this "
                             f"({GATE_COMBO} only; off by default, calibrate before use)")
    return parser
