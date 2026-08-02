"""
ranking_n_retrieval.py — hybrid retrieval over an ingested collection.

Four retrieval modes, so the design can be justified with numbers rather than
asserted. Each isolates one component:

    dense     dense vector search only          (is BM25 earning its place?)
    sparse    BM25 only                         (is the embedder earning its?)
    combo1    dense + BM25, fused with RRF      (fast; development)
    combo2    combo1 + cross-encoder rerank     (best; matches the flowchart)

Default is combo2. The previous default was combo1, which silently skipped the
reranker while the flowchart showed it — the running system and the diagram
disagreed.

WHY DENSE AND SPARSE BOTH EXIST
    Dense search matches meaning: "how do we compare two documents" finds the
    cosine similarity slide without the word "cosine". BM25 matches words: it
    is the only path that reliably finds the slide that literally says "BM25"
    or "RRF", because a 384-dim embedding smears near-synonymous technical
    terms into the same neighbourhood. Fusion pays off precisely because the
    two fail differently.

WHY RRF AND NOT SCORE ADDITION
    Cosine similarities (~0 to 1) and BM25 scores (unbounded, corpus-dependent)
    are not on a comparable scale. RRF discards the scores and keeps only the
    RANKS, so no calibration is needed.

WHY THE CROSS-ENCODER RUNS LAST
    It scores each (query, chunk) pair jointly, which is far more accurate than
    comparing pre-computed vectors — and far too slow to run over the corpus.
    So it only sees the k survivors of fusion.
"""

from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

COMBOS = ("dense", "sparse", "combo1", "combo2")


class Retriever:

    def __init__(self,
                 vectordb,
                 documents: List[Document],
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 active_combo: str = "combo2",
                 k: int = 20,
                 top_n: int = 5):
        """
        vectordb     : ChromaVectorDB or FAISSVectorDB — anything with .search()
        documents    : EVERY chunk in the collection, for the BM25 index.
                       Get it from vectordb.all_documents().
        active_combo : one of COMBOS. Defaults to combo2 (full pipeline).
        k            : candidates each retriever contributes before fusion
        top_n        : chunks finally returned to the prompt builder
        """
        if active_combo not in COMBOS:
            raise ValueError(f"Unknown combo: {active_combo}. Choose from {COMBOS}.")

        self.vectordb = vectordb
        self.documents = documents
        self.active_combo = active_combo
        self.k = k
        self.top_n = top_n

        if not documents:
            raise ValueError(
                "Retriever needs the full document list to build BM25. "
                "Pass vectordb.all_documents()."
            )

        # BM25 index over raw text. Lowercase whitespace tokenisation matches
        # how queries are tokenised below — the two MUST agree or scores are
        # meaningless.
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        self._reranker_model = reranker_model
        self._reranker = None          # loaded lazily — see reranker property

    # ------------------------------------------------------------------
    @property
    def reranker(self):
        """Cross-encoder, loaded on first use.

        Lazy because it is ~90MB and several seconds to load, and the dense,
        sparse and combo1 modes never touch it. Eager loading made an ablation
        run pay for a model it did not use.
        """
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "combo2 needs a cross-encoder: pip install sentence-transformers\n"
                    "Or run with combo1 (RRF, no rerank) / dense / sparse."
                ) from e
            print(f"Loading cross-encoder reranker ({self._reranker_model})...")
            self._reranker = CrossEncoder(self._reranker_model)
        return self._reranker

    # ------------------------------------------------------------------
    # Retrieval modes
    # ------------------------------------------------------------------
    def retrieve_dense(self, query, query_embedding, k=None, top_n=None):
        """Ablation arm: dense vector search alone."""
        top_n = top_n or self.top_n
        return self.vectordb.search(query_embedding, k=max(k or self.k, top_n))[:top_n]

    def retrieve_sparse(self, query, query_embedding=None, k=None, top_n=None):
        """Ablation arm: BM25 alone. query_embedding is ignored, kept for a
        uniform signature across modes."""
        top_n = top_n or self.top_n
        return self._bm25_search(query, max(k or self.k, top_n))[:top_n]

    def retrieve_hybrid_rrf(self, query, query_embedding, k=None, top_n=None):
        """COMBO 1 — dense + BM25, fused by RRF. Fast, no reranker."""
        k = k or self.k
        top_n = top_n or self.top_n
        dense_results = self.vectordb.search(query_embedding, k)
        sparse_results = self._bm25_search(query, k)
        return self._rrf_fusion(dense_results, sparse_results, top_n=top_n)

    def retrieve_hybrid_reranked(self, query, query_embedding, k=None, top_n=None):
        """COMBO 2 — combo1, then cross-encoder rerank of the survivors."""
        k = k or self.k
        top_n = top_n or self.top_n
        # top_n=k here on purpose: give the reranker the full candidate pool,
        # not an already-truncated list. Truncating first would throw away the
        # chunks the reranker exists to promote.
        candidates = self.retrieve_hybrid_rrf(query, query_embedding, k, top_n=k)
        return self._rerank(query, candidates, top_n)

    # ------------------------------------------------------------------
    def retrieve(self, query, query_embedding, k=None, top_n=None) -> List[Document]:
        """Entry point called by RAGPipeline."""
        if self.active_combo == "dense":
            return self.retrieve_dense(query, query_embedding, k, top_n)
        if self.active_combo == "sparse":
            return self.retrieve_sparse(query, query_embedding, k, top_n)
        if self.active_combo == "combo1":
            return self.retrieve_hybrid_rrf(query, query_embedding, k, top_n)
        if self.active_combo == "combo2":
            return self.retrieve_hybrid_reranked(query, query_embedding, k, top_n)
        raise ValueError(f"Unknown combo: {self.active_combo}. Choose from {COMBOS}.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _bm25_search(self, query: str, k: int = 20) -> List[Document]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        # Drop zero-score hits: with no term overlap BM25 returns arbitrary
        # documents, and feeding those into fusion adds noise rather than signal.
        return [self.documents[i] for i in top_k_indices if scores[i] > 0]

    @staticmethod
    def _doc_key(doc: Document, position: int) -> str:
        """Stable identity for fusion.

        chunk_id when present. The old code did doc.metadata["chunk_id"]
        directly, which raised KeyError on any chunk lacking one — and worse,
        several chunks defaulting to the same placeholder would COLLIDE into a
        single fusion entry and silently shrink the candidate pool.
        """
        key = doc.metadata.get("chunk_id")
        if key and key != "?":
            return str(key)
        return f"__nokey_{position}_{hash(doc.page_content) & 0xFFFFFFFF}"

    def _rrf_fusion(self, *ranked_lists, rrf_k: int = 60, top_n=None) -> List[Document]:
        """Reciprocal Rank Fusion: score by 1/(rank + rrf_k), summed.

        rrf_k=60 is the value from the original paper. It flattens the curve so
        rank 1 does not dominate rank 2 outright — appearing respectably in both
        lists beats topping one and missing the other.
        """
        scores, doc_map = {}, {}
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                key = self._doc_key(doc, rank)
                doc_map[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank + rrf_k)

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = [doc_map[key] for key in sorted_keys]
        return result[:top_n] if top_n is not None else result

    def _rerank(self, query: str, chunks: List[Document], top_n: int = 5) -> List[Document]:
        if not chunks:
            return []
        pairs = [[query, chunk.page_content] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_n]]

    # ------------------------------------------------------------------
    def retrieve_with_scores(self, query, query_embedding,
                             k=None, top_n=None) -> List[Tuple[Document, float]]:
        """combo2 with the cross-encoder scores attached, for debugging.

        A negative or near-zero top score means the reranker found nothing
        genuinely relevant — useful for telling 'retrieval missed' apart from
        'the generator mishandled good context'.
        """
        k = k or self.k
        top_n = top_n or self.top_n
        candidates = self.retrieve_hybrid_rrf(query, query_embedding, k, top_n=k)
        if not candidates:
            return []
        pairs = [[query, c.page_content] for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_n]]

    def __repr__(self) -> str:
        return (f"Retriever(combo={self.active_combo!r}, corpus={len(self.documents)}, "
                f"k={self.k}, top_n={self.top_n})")
