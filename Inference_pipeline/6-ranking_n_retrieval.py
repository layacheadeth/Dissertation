"""
Finding the chunks that answer a question.

There are two ways to search, and they fail differently, which is why both
exist:

  vector search   matches meaning. "how do we compare two documents" finds the
                  cosine similarity slide even though the word "cosine" is not
                  in the question.
  keyword search  matches words. It is the only thing that reliably finds the
                  slide that literally says "BM25", because vector search
                  smears similar-sounding technical terms together.

Four modes, so each part can be shown to earn its place:

    dense    vector search only        is keyword search worth having?
    sparse   keyword search only       is the embedding model worth having?
    combo1   both, merged              fast, good for development
    combo2   combo1 plus a reranker    best, and the default

WHY MERGE BY RANK, NOT BY SCORE
    Vector scores run about 0 to 1. Keyword scores are unbounded and depend on
    the corpus. Adding them together would be meaningless. So the scores are
    thrown away and only the positions are used: a chunk placed 1st, 2nd, 3rd
    scores 1/61, 1/62, 1/63, and the two lists' scores are summed. A chunk that
    does respectably in both beats one that tops a single list.

WHY THE RERANKER RUNS LAST
    It reads the question and one chunk together, which is much more accurate
    than comparing ready-made vectors, and much too slow to run over the whole
    corpus. So it only ever sees the survivors of the merge.
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent
# The root, for Share_components and Ingestion_pipeline; this folder, so the
# sibling modules import by bare name however this one was reached.
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.append(_p)

from Share_components import configuration as config_inference

COMBOS = ("dense", "sparse", "combo1", "combo2")

# Softens the rank curve when merging, so 1st place does not overwhelm 2nd.
# 60 is the value from the paper this method comes from.
RANK_SMOOTHING = 60


class Retriever:
    """Searches a collection four different ways."""

    def __init__(self,
                 store,
                 documents,
                 combo=config_inference.COMBO,
                 candidates=config_inference.CANDIDATES,
                 top_n=config_inference.TOP_N,
                 reranker_model=config_inference.RERANKER,
                 budget_tokens=getattr(config_inference, "BUDGET_TOKENS", None)):
        """
        store         : the collection, anything with a .search() method
        documents     : every chunk in it, needed to build the keyword index
        combo         : one of COMBOS
        candidates    : how many chunks each search method puts forward
        top_n         : how many chunks come out at the end
        budget_tokens : when set, chunks come out by total token count instead
                        of by count — see _fill_budget. top_n is then ignored.
        """
        if combo not in COMBOS:
            raise ValueError(f"Unknown combo {combo!r}. Choose from {COMBOS}.")
        if not documents:
            raise ValueError("Retriever needs every chunk to build the keyword "
                             "index. Pass store.all_documents().")

        self.store = store
        self.documents = documents
        self.active_combo = combo
        self.candidates = candidates
        self.top_n = top_n
        self.budget_tokens = budget_tokens

        # Token counts are expensive enough to be worth keeping, and a chunk is
        # measured again on every question that retrieves it.
        self._token_cache = {}
        # Set when a budget run ran out of candidates before it ran out of
        # budget: that strategy was handed less context than the others and the
        # comparison it is part of is no longer equal. Reported, not silent.
        self.budget_underfilled = 0

        # Keyword index. Splitting on spaces after lowercasing has to match how
        # questions are split below, or the scores mean nothing.
        self.bm25 = BM25Okapi([d.page_content.lower().split() for d in documents])

        self._reranker_model = reranker_model
        self._reranker = None

    # ------------------------------------------------------------------
    @property
    def reranker(self):
        """The reranking model, loaded the first time it is needed.

        It is about 90MB and takes a few seconds, and three of the four modes
        never touch it, so loading it up front would waste time on every run
        that does not use it.
        """
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "combo2 needs the reranker: pip install sentence-transformers\n"
                    "Or use combo1, dense or sparse instead."
                ) from e
            print(f"Loading reranker ({self._reranker_model})...")
            self._reranker = CrossEncoder(self._reranker_model)
        return self._reranker

    # ------------------------------------------------------------------
    # Equal-context budget
    # ------------------------------------------------------------------
    @property
    def budget_active(self):
        return bool(self.budget_tokens)

    def _tokens_in(self, doc):
        """How many tokens a chunk is, measured once and remembered.

        Counted with the same BGE tokenizer that measured the chunks during
        ingestion, so a budget of 1500 here means the same thing as MAX_TOKENS
        does there. Counting words instead would drift between strategies,
        because exp2 carries more punctuation and formula notation per word.
        """
        key = doc.metadata.get("chunk_id") or hash(doc.page_content)
        if key not in self._token_cache:
            from Share_components.chunking_tokenizer import count_tokens
            self._token_cache[key] = count_tokens(doc.page_content)
        return self._token_cache[key]

    def _fill_budget(self, ranked, budget_tokens=None):
        """Take chunks in rank order until the next one would not fit.

        Accepts either a list of Documents or a list of (Document, score), and
        gives back the same shape.

        STOPS at the first chunk that does not fit, rather than skipping it and
        carrying on down the list. Skipping would pack the budget fuller, but it
        would also let a low-ranked short chunk overtake a high-ranked long one,
        so the context would no longer be "the best material that fits" and the
        ranking under test would be partly undone by the packing. Stopping keeps
        rank order intact, which is the thing being measured.

        The best chunk is always kept, even when it alone exceeds the budget:
        returning nothing would make the question unanswerable for a reason
        that has nothing to do with retrieval quality.
        """
        budget = budget_tokens or self.budget_tokens
        if not budget or not ranked:
            return ranked

        kept, used = [], 0
        for item in ranked:
            doc = item[0] if isinstance(item, tuple) else item
            n = self._tokens_in(doc)
            if used + n > budget:
                break
            kept.append(item)
            used += n

        if not kept:
            kept = [ranked[0]]
            used = self._tokens_in(kept[0][0] if isinstance(kept[0], tuple) else kept[0])

        # Every candidate was consumed and there was still room: this question
        # got a smaller context than a strategy with bigger chunks would have.
        if len(kept) == len(ranked):
            self.budget_underfilled += 1

        self.last_budget_used = used
        return kept

    # ------------------------------------------------------------------
    # The four modes
    # ------------------------------------------------------------------
    def search_dense(self, question, question_vector, top_n=None):
        """Vector search on its own."""
        top_n = top_n or self.top_n
        return self.store.search(question_vector, k=max(self.candidates, top_n))[:top_n]

    def search_sparse(self, question, top_n=None):
        """Keyword search on its own."""
        top_n = top_n or self.top_n
        return self._keyword_search(question, max(self.candidates, top_n))[:top_n]

    def search_both(self, question, question_vector, top_n=None):
        """Both searches, merged by rank. No reranker."""
        top_n = top_n or self.top_n
        by_vector = self.store.search(question_vector, self.candidates)
        by_keyword = self._keyword_search(question, self.candidates)
        return self._merge(by_vector, by_keyword, top_n=top_n)

    def search_both_reranked(self, question, question_vector, top_n=None):
        """Both searches merged, then reranked. The default."""
        top_n = top_n or self.top_n
        # The reranker gets the whole merged pool, not a shortened list.
        # Cutting it first would throw away the chunks it exists to promote.
        pool = self.search_both(question, question_vector, top_n=self.candidates)
        return [doc for doc, _ in self._rerank(question, pool)[:top_n]]

    def retrieve(self, question, question_vector, top_n=None):
        """The entry point. Runs whichever mode is active.

        Under a token budget the mode is asked for the whole candidate pool
        rather than top_n, and the budget decides where the list is cut.
        """
        if self.budget_active and top_n is None:
            ranked = self._retrieve_ranked(question, question_vector,
                                           top_n=self.candidates)
            return self._fill_budget(ranked)
        return self._retrieve_ranked(question, question_vector, top_n)

    def _retrieve_ranked(self, question, question_vector, top_n=None):
        """The ranked list, before any budget is applied."""
        if self.active_combo == "dense":
            return self.search_dense(question, question_vector, top_n)
        if self.active_combo == "sparse":
            return self.search_sparse(question, top_n)
        if self.active_combo == "combo1":
            return self.search_both(question, question_vector, top_n)
        return self.search_both_reranked(question, question_vector, top_n)

    def retrieve_with_scores(self, question, question_vector, top_n=None):
        """combo2, with the reranker's scores attached, for debugging.

        A top score near zero or below means the reranker found nothing really
        relevant. That tells "retrieval missed" apart from "retrieval was fine
        and the model mishandled it".
        """
        pool = self.search_both(question, question_vector, top_n=self.candidates)
        ranked = self._rerank(question, pool)

        # The budget is applied after reranking, not before: the reranker has
        # to see the whole pool or it cannot promote the chunk it exists to
        # promote, and cutting first would spend the budget on the merge order.
        if self.budget_active and top_n is None:
            return self._fill_budget(ranked)

        return ranked[:(top_n or self.top_n)]

    # ------------------------------------------------------------------
    # The pieces
    # ------------------------------------------------------------------
    def _keyword_search(self, question, k):
        scores = self.bm25.get_scores(question.lower().split())
        best = np.argsort(scores)[::-1][:k]
        # Chunks sharing no words with the question score zero and are returned
        # in arbitrary order, so they are dropped rather than merged in as noise.
        return [self.documents[i] for i in best if scores[i] > 0]

    @staticmethod
    def _identity(doc, position):
        """A stable name for a chunk while merging.

        Its chunk_id, when it has one. Chunks without one used to all collapse
        into a single merged entry, quietly shrinking the pool, so they get a
        unique fallback instead.
        """
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id and chunk_id != "?":
            return str(chunk_id)
        return f"__no_id_{position}_{hash(doc.page_content) & 0xFFFFFFFF}"

    def _merge(self, *ranked_lists, top_n=None):
        """Merge lists by position: a chunk at rank r scores 1/(r + 60).

        Ranks count from 1, as in the paper this method comes from, so the best
        chunk in a list scores 1/61.
        """
        scores, by_id = {}, {}
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                chunk_id = self._identity(doc, rank)
                by_id[chunk_id] = doc
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank + RANK_SMOOTHING)

        best_first = sorted(scores, key=scores.get, reverse=True)
        merged = [by_id[chunk_id] for chunk_id in best_first]
        return merged[:top_n] if top_n else merged

    def _rerank(self, question, chunks):
        """Score every chunk against the question, best first."""
        if not chunks:
            return []
        scores = self.reranker.predict([[question, c.page_content] for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked]

    def __repr__(self):
        cut = (f"budget_tokens={self.budget_tokens}" if self.budget_active
               else f"top_n={self.top_n}")
        return (f"Retriever({self.active_combo}, {len(self.documents)} chunks, "
                f"candidates={self.candidates}, {cut})")

"""
---------------------------------------------------------------------------
Running this stage on its own
---------------------------------------------------------------------------
Stage 2 of inference, runnable the way the numbered ingestion scripts are.

  python Inference_pipeline/ranking_n_retrieval.py --question "What is BM25?"
  python Inference_pipeline/ranking_n_retrieval.py --question "What is BM25?" --strategy exp1
  python Inference_pipeline/ranking_n_retrieval.py --question "What is BM25?" --budget-tokens 1500 --candidates 60
  python Inference_pipeline/ranking_n_retrieval.py --from-stage1 Data/Results_stage1/exp3_what_is_bm25_.json

Reads  Data/Database/chroma_db/, and optionally a stage 1 record
Writes Data/Results_stage2/<strategy>_<combo>_<slug>.json

This stage takes a question and hands stage 3 the chunks it will answer from.
Every mode is shown, not only the configured one, because the point of the
four modes is that each part can be seen to earn its place. It is an
illustration of the pipeline, not a result: the numbers reported in the
dissertation come from Data/Results_evaluation/.

"""

STAGE = "2-ranking-retrieval"
STAGE_DIR = config_inference.ROOT / "Data" / "Results_stage2"


def _ids(retriever, docs):
    """Chunk ids in list order, using the same identity the merge uses."""
    return [retriever._identity(doc, position)
            for position, doc in enumerate(docs, start=1)]


def _row(retriever, rank, doc, score=None, was=None):
    """One printed line: rank, score, where it came from, what it says."""
    chunk_id = retriever._identity(doc, rank)
    pages = doc.metadata.get("page_number", [])
    pages = ",".join(str(p) for p in pages) if isinstance(pages, list) else str(pages)
    moved = "" if was is None else (f"  (was {was})" if was else "  (new)")
    shown = f"{score:<9.3f}" if score is not None else " " * 9
    preview = " ".join(doc.page_content.split())[:44]
    return (f"  {rank:<5}{shown}{str(doc.metadata.get('week','?')):<9}"
            f"{pages:<9}{str(chunk_id):<17}{preview}{moved}")


def _header(with_score=True):
    score = "score    " if with_score else " " * 9
    return f"  {'rank':<5}{score}{'week':<9}{'pages':<9}{'chunk id':<17}text"


def _record_list(retriever, docs, scores=None):
    """A list of chunks as plain data, for the JSON."""
    out = []
    for position, doc in enumerate(docs, start=1):
        entry = {
            "rank": position,
            "chunk_id": retriever._identity(doc, position),
            "week": doc.metadata.get("week"),
            "page_number": doc.metadata.get("page_number"),
            "token_count": retriever._tokens_in(doc),
        }
        if scores is not None:
            entry["score"] = round(float(scores[position - 1]), 4)
        out.append(entry)
    return out


def stage_settings(strategy, embedder, collection, combo, candidates,
                   top_n, budget_tokens):
    return {
        "strategy": strategy,
        "embedder": embedder,
        "collection": collection,
        "combo": combo,
        "candidates": candidates,
        "top_n": top_n,
        "budget_tokens": budget_tokens,
        "rank_smoothing": RANK_SMOOTHING,
        "reranker": config_inference.RERANKER,
    }


def stage_hash(settings):
    """A short code for these settings.

    Not run_inference_bigfile.settings_hash, which covers the prompt and the
    generator: this stage never loads a language model and should not pay to
    hash one.
    """
    import hashlib
    parts = [f"{key}={settings[key]}" for key in sorted(settings)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def run_stage(question, strategy=config_inference.STRATEGY,
              embedder=config_inference.EMBEDDER,
              combo=config_inference.COMBO,
              candidates=config_inference.CANDIDATES,
              top_n=config_inference.TOP_N,
              budget_tokens=None,
              from_stage1=None,
              save=True):
    """Every phase of retrieval, printed as it happens."""
    import json
    import time

    from vector_search import VectorSearch

    print(f"\n[1] Loading the collection and building the keyword index")
    search = VectorSearch(strategy, embedder=embedder)
    retriever = Retriever(search.store, search.documents, combo=combo,
                          candidates=candidates, top_n=top_n,
                          budget_tokens=budget_tokens)
    print(f"  {search}")
    print(f"  {retriever}")
    print(f"  question    {question!r}")
    if from_stage1:
        print(f"  following   {from_stage1}")

    vector = search.embed(question)

    print(f"\n[2] Dense search: nearest {candidates} by meaning")
    started = time.time()
    dense = search.store.search(vector, candidates)
    dense_seconds = time.time() - started
    print(_header(with_score=False))
    for rank, doc in enumerate(dense[:10], start=1):
        print(_row(retriever, rank, doc))
    if len(dense) > 10:
        print(f"  ... {len(dense) - 10} more")
    print(f"  {len(dense)} candidates in {dense_seconds:.2f}s")

    print(f"\n[3] Keyword search: best {candidates} by BM25")
    started = time.time()
    sparse = retriever._keyword_search(question, candidates)
    sparse_seconds = time.time() - started
    print(_header(with_score=False))
    for rank, doc in enumerate(sparse[:10], start=1):
        print(_row(retriever, rank, doc))
    if len(sparse) > 10:
        print(f"  ... {len(sparse) - 10} more")
    print(f"  {len(sparse)} candidates in {sparse_seconds:.2f}s")
    if len(sparse) < candidates:
        print(f"  {candidates - len(sparse)} dropped: no word in common with the question")

    dense_ids, sparse_ids = _ids(retriever, dense), _ids(retriever, sparse)
    shared = set(dense_ids) & set(sparse_ids)
    print(f"\n  the two methods agree on {len(shared)} of {len(set(dense_ids) | set(sparse_ids))} chunks")

    print(f"\n[4] Merge by rank, 1/(rank + {RANK_SMOOTHING})")
    merged = retriever._merge(dense, sparse, top_n=candidates)
    merged_ids = _ids(retriever, merged)
    print(_header(with_score=False))
    for rank, doc in enumerate(merged[:10], start=1):
        chunk_id = merged_ids[rank - 1]
        where = []
        if chunk_id in dense_ids:
            where.append(f"dense {dense_ids.index(chunk_id) + 1}")
        if chunk_id in sparse_ids:
            where.append(f"bm25 {sparse_ids.index(chunk_id) + 1}")
        print(_row(retriever, rank, doc, was=", ".join(where)))
    print(f"  {len(merged)} chunks after merging")

    print(f"\n[5] Rerank: the question read against each chunk")
    started = time.time()
    reranked = retriever._rerank(question, merged)
    rerank_seconds = time.time() - started
    reranked_docs = [doc for doc, _ in reranked]
    reranked_scores = [score for _, score in reranked]
    print(_header())
    for rank, (doc, score) in enumerate(reranked[:10], start=1):
        chunk_id = retriever._identity(doc, rank)
        was = f"merge {merged_ids.index(chunk_id) + 1}" if chunk_id in merged_ids else ""
        print(_row(retriever, rank, doc, score=score, was=was))
    print(f"  {len(reranked)} chunks rescored in {rerank_seconds:.2f}s")

    if budget_tokens:
        print(f"\n[6] Cut at {budget_tokens} tokens, in rank order")
        kept = retriever._fill_budget(reranked)
        used = getattr(retriever, "last_budget_used", 0)
        print(f"  kept {len(kept)} chunks, {used} tokens")
        if retriever.budget_underfilled:
            print(f"  [warning] candidates ran out before the budget did. This "
                  f"strategy was handed less context than one with larger chunks.")
    else:
        print(f"\n[6] Cut at top {top_n}")
        kept = reranked[:top_n]

    final_docs = [item[0] if isinstance(item, tuple) else item for item in kept]
    final_scores = [item[1] for item in kept if isinstance(item, tuple)]
    context_tokens = sum(retriever._tokens_in(doc) for doc in final_docs)

    print(f"\n  what stage 3 receives")
    print(_header())
    for rank, doc in enumerate(final_docs, start=1):
        score = final_scores[rank - 1] if len(final_scores) == len(final_docs) else None
        print(_row(retriever, rank, doc, score=score))
    print(f"\n  {len(final_docs)} chunks, {context_tokens} tokens of context")

    print(f"\n[7] What each mode would have returned, top {top_n}")
    modes = {
        "dense": search.store.search(vector, max(candidates, top_n))[:top_n],
        "sparse": retriever._keyword_search(question, max(candidates, top_n))[:top_n],
        "combo1": retriever._merge(dense, sparse, top_n=top_n),
        "combo2": reranked_docs[:top_n],
    }
    for name, docs in modes.items():
        marker = " <- configured" if name == combo else ""
        print(f"  {name:<8}{', '.join(_ids(retriever, docs))}{marker}")

    settings = stage_settings(search.strategy, search.embedder_name,
                              search.collection_name, combo, candidates,
                              top_n, budget_tokens)

    if not save:
        print(f"\n[8] Not saved (--no-save)")
        return final_docs

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in question.lower()).strip("_")[:40]
    path = STAGE_DIR / f"{search.strategy}_{combo}_{slug}.json"

    record = {
        "stage": STAGE,
        "settings": settings,
        "settings_hash": stage_hash(settings),
        "input": {"question": question, "from_stage1": from_stage1},
        "stages": {
            "dense": _record_list(retriever, dense),
            "sparse": _record_list(retriever, sparse),
            "merged": _record_list(retriever, merged),
            "reranked": _record_list(retriever, reranked_docs, reranked_scores),
        },
        "output": {
            "chunks": [
                {
                    "rank": rank,
                    "chunk_id": retriever._identity(doc, rank),
                    "week": doc.metadata.get("week"),
                    "page_number": doc.metadata.get("page_number"),
                    "token_count": retriever._tokens_in(doc),
                    "content": doc.page_content,
                }
                for rank, doc in enumerate(final_docs, start=1)
            ],
            "context_tokens": context_tokens,
            "budget_underfilled": bool(retriever.budget_underfilled),
            "modes_top_n": {name: _ids(retriever, docs) for name, docs in modes.items()},
        },
        "seconds": {
            "dense": round(dense_seconds, 3),
            "sparse": round(sparse_seconds, 3),
            "rerank": round(rerank_seconds, 3),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"\n[8] Written to {path}")
    return final_docs


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Stage 2 of inference: search, merge, rerank, cut.")
    parser.add_argument("--question", default="What is BM25?")
    parser.add_argument("--from-stage1", default=None,
                        help="a Data/Results_stage1/*.json file to take the question from")
    parser.add_argument("--strategy", default=config_inference.STRATEGY)
    parser.add_argument("--embedder", default=config_inference.EMBEDDER)
    parser.add_argument("--combo", default=config_inference.COMBO, choices=COMBOS)
    parser.add_argument("--candidates", type=int, default=config_inference.CANDIDATES)
    parser.add_argument("--top-n", type=int, default=config_inference.TOP_N)
    parser.add_argument("--budget-tokens", type=int, default=None,
                        help="cut by total tokens instead of by chunk count")
    parser.add_argument("--no-save", action="store_true",
                        help="print only, write no JSON")
    args = parser.parse_args()

    question = args.question
    if args.from_stage1:
        with open(args.from_stage1, encoding="utf-8") as f:
            question = json.load(f)["input"]["question"]

    if args.budget_tokens and args.candidates < 60:
        print(f"[warning] --budget-tokens with only {args.candidates} candidates. "
              f"Strategies with small chunks may not reach the budget. "
              f"Suggested: --candidates 60")

    run_stage(question, args.strategy, args.embedder, args.combo,
              args.candidates, args.top_n, args.budget_tokens,
              from_stage1=args.from_stage1, save=not args.no_save)
