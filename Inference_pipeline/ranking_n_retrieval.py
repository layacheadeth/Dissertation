import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class Retriever:

    def __init__(self, vectordb, documents, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):

        self.vectordb     = vectordb
        self.documents    = documents
        self.active_combo = "combo1"
        # BM25 index built from raw text chunks
        tokenized  = [doc.page_content.lower().split() for doc in documents]
        self.bm25  = BM25Okapi(tokenized)

        # cross-encoder loaded once, reused across every query
        print("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder(reranker_model)


    # ------------------------------------------------------------------
    # COMBO 1 — Hybrid (Dense + BM25) + RRF
    # fast, good quality — use during development
    # ------------------------------------------------------------------

    def retrieve_hybrid_rrf(self, query, query_embedding, k=20, top_n=5):

        dense_results  = self.vectordb.search(query_embedding, k)
        sparse_results = self._bm25_search(query, k)

        return self._rrf_fusion(dense_results, sparse_results, top_n=top_n)


    # ------------------------------------------------------------------
    # COMBO 2 — Hybrid + RRF + Cross-encoder reranker
    # best quality — use in production
    # ------------------------------------------------------------------

    def retrieve_hybrid_reranked(self, query, query_embedding, k=20, top_n=5):
        candidates = self.retrieve_hybrid_rrf(query, query_embedding, k, top_n=k)

        return self._rerank(query, candidates, top_n)


    # ------------------------------------------------------------------
    # Entry point — called by RAGPipeline
    # active_combo is set in main.py config
    # ------------------------------------------------------------------

    def retrieve(self, query, query_embedding):

        if self.active_combo == "combo1":
            return self.retrieve_hybrid_rrf(query, query_embedding)

        elif self.active_combo == "combo2":
            return self.retrieve_hybrid_reranked(query, query_embedding)

        else:
            raise ValueError(f"Unknown combo: {self.active_combo}. Choose 'combo1' or 'combo2'")


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bm25_search(self, query, k=20):

        tokenized_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenized_query)
        top_k_indices   = np.argsort(scores)[::-1][:k]

        return [self.documents[i] for i in top_k_indices]


    def _rrf_fusion(self, *ranked_lists, rrf_k=60, top_n=None):

        scores  = {}
        doc_map = {}
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                key = doc.metadata["chunk_id"]
                doc_map[key] = doc
                scores[key]  = scores.get(key, 0) + 1 / (rank + rrf_k)

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        result = [doc_map[k] for k in sorted_keys]
        return result[:top_n] if top_n is not None else result


    def _rerank(self, query, chunks, top_n=5):

        pairs  = [[query, chunk.page_content] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in ranked[:top_n]]