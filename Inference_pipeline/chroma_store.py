"""
ChromaDB vector store for the COMP64702 RAG pipeline.

Exposes the SAME interface as FAISSVectorDB, so RAGPipeline never learns which
backend it is talking to:

    store = ChromaVectorDB(collection_name="exp5_section_aware")
    store.add_documents(embeddings, docs)
    docs = store.search(query_embedding, k=5)        -> List[Document]
    store = ChromaVectorDB.load(collection_name="exp5_section_aware")

Two things this file exists to get right:

1. EMBEDDINGS ARE PASSED IN, NEVER COMPUTED BY CHROMA.
   If you call collection.upsert() without an `embeddings=` argument, Chroma
   silently embeds the text with its own default model. Your queries would then
   be embedded with MiniLM and your corpus with something else — two different
   vector spaces, no error raised, quietly meaningless retrieval.

2. METADATA IS FLATTENED ON WRITE AND REHYDRATED ON READ.
   Chroma metadata values must be str/int/float/bool. Our chunkers emit
   page_number as a LIST, so it is stored as "1,2,3" and parsed back to
   [1, 2, 3] on the way out. Downstream code (RRF, evaluation) therefore sees
   the same Document shape it would get from FAISS.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import chromadb
import numpy as np
from langchain_core.documents import Document

DEFAULT_PATH = "Data/chroma_db"

# Metadata keys whose values are lists and must survive the round trip.
_LIST_KEYS = ("page_number",)


# ---------------------------------------------------------------------------
# Metadata translation
# ---------------------------------------------------------------------------
def flatten_metadata(meta: Dict) -> Dict:
    """Document metadata -> Chroma-safe scalars."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out[k] = ",".join(str(x) for x in v)
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def rehydrate_metadata(meta: Dict) -> Dict:
    """Chroma scalars -> Document metadata, restoring list-valued fields."""
    out = dict(meta or {})
    for key in _LIST_KEYS:
        raw = out.get(key)
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            out[key] = [int(p) if p.lstrip("-").isdigit() else p for p in parts]
        elif raw is None:
            out[key] = []
        elif not isinstance(raw, list):
            out[key] = [raw]
    return out


# ---------------------------------------------------------------------------
class ChromaVectorDB:
    """Persistent Chroma collection, one per chunking strategy."""

    def __init__(self,
                 collection_name: str,
                 path: str = DEFAULT_PATH,
                 description: str = "",
                 reset: bool = False):
        self.collection_name = collection_name
        self.path = path
        self.client = chromadb.PersistentClient(path=str(path))

        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass  # did not exist

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # Cosine, to match MiniLM's normalised vectors and the FAISS
            # IndexFlatIP store. Chroma's default is L2 — leaving it would
            # make the two backends rank differently on unnormalised input.
            metadata={"hnsw:space": "cosine", "description": description},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _as_list(vectors) -> List[List[float]]:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.tolist()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def add_documents(self,
                      embeddings,
                      documents: Sequence[Document],
                      batch_size: int = 500) -> None:
        """Upsert vectors and Documents together. Additive and idempotent:
        re-running with the same chunk_ids overwrites rather than duplicates."""
        if len(embeddings) != len(documents):
            raise ValueError(
                f"add_documents got {len(embeddings)} embeddings but "
                f"{len(documents)} documents — these must correspond 1:1."
            )
        if not len(documents):
            return

        vectors = self._as_list(embeddings)
        texts, metas, ids = [], [], []
        seen = set()

        for i, doc in enumerate(documents):
            meta = flatten_metadata(doc.metadata)
            # chunk_id is the stable identity; fall back to position if absent.
            raw_id = str(doc.metadata.get("chunk_id") or f"auto_{i}")
            doc_id, n = raw_id, 1
            while doc_id in seen:            # guard against duplicate chunk_ids
                doc_id = f"{raw_id}_dup{n}"
                n += 1
            seen.add(doc_id)

            texts.append(doc.page_content)
            metas.append(meta)
            ids.append(doc_id)

        for i in range(0, len(texts), batch_size):
            self.collection.upsert(
                documents=texts[i:i + batch_size],
                embeddings=vectors[i:i + batch_size],   # <- never omit this
                metadatas=metas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def search(self,
               query_embedding,
               k: int = 5,
               where: Optional[Dict] = None) -> List[Document]:
        """Return the k most similar Documents, most similar first.

        `where` is a Chroma metadata filter, e.g. {"week": "Week 3"} — the
        capability FAISS does not have. Optional, so the signature stays
        interface-compatible.
        """
        return [doc for doc, _ in self.search_with_scores(query_embedding, k, where)]

    def search_with_scores(self,
                           query_embedding,
                           k: int = 5,
                           where: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Same, but with COSINE SIMILARITY scores (higher = better).

        Chroma returns cosine DISTANCE (lower = better). We convert with
        similarity = 1 - distance so scores are directly comparable with the
        FAISS inner-product store and so any score-based logic downstream does
        not silently invert.
        """
        total = self.collection.count()
        if total == 0:
            return []

        res = self.collection.query(
            query_embeddings=self._as_list(query_embedding),
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        for text, meta, dist in zip(res["documents"][0],
                                    res["metadatas"][0],
                                    res["distances"][0]):
            out.append((
                Document(page_content=text, metadata=rehydrate_metadata(meta)),
                1.0 - float(dist),
            ))
        return out

    def all_documents(self) -> List[Document]:
        """Every Document in the collection, in stable id order.

        The Retriever needs this to build its BM25 index. BM25 is lexical, not
        vector-based: it scores by term frequency across the whole corpus, so
        it needs all the text in memory rather than a similarity search. At
        ~2,000 chunks that is trivial; at millions you would use a dedicated
        sparse index instead.

        Sorted by id so the ordering is deterministic across runs — BM25 scores
        do not depend on order, but reproducible debugging does.
        """
        got = self.collection.get(include=["documents", "metadatas"])
        rows = sorted(
            zip(got["ids"], got["documents"], got["metadatas"]),
            key=lambda r: str(r[0]),
        )
        return [
            Document(page_content=text, metadata=rehydrate_metadata(meta))
            for _id, text, meta in rows
        ]

    # ------------------------------------------------------------------
    # Persistence — Chroma persists on write, so these are thin.
    # ------------------------------------------------------------------
    def save(self, *args, **kwargs) -> None:
        """No-op. PersistentClient writes to disk as it goes. Present so the
        interface matches FAISSVectorDB and callers need no special case."""
        return None

    @classmethod
    def load(cls,
             collection_name: str,
             path: str = DEFAULT_PATH) -> "ChromaVectorDB":
        store = cls(collection_name=collection_name, path=path)
        if store.collection.count() == 0:
            raise ValueError(
                f"Collection '{collection_name}' at {path} is empty. "
                "Run ingest_to_chroma.py first."
            )
        return store

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.collection.count()

    def __repr__(self) -> str:
        return f"ChromaVectorDB(collection={self.collection_name!r}, vectors={len(self)})"
