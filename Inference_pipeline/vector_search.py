"""
vector_search.py — the inference-side entry point to the vector store.

3-1-Ingest-to-ChromaDB.py WRITES collections. This module READS them, and it
imports 3-1's STRATEGIES dict rather than restating the collection names, so
inference can never point at a collection name that ingestion does not produce.
That mapping has drifted once already (exp2_sliding_window vs
exp2_fixed_overlap); importing it makes drift impossible.

Nothing here writes. Loading is read-only and fails loudly on an empty or
missing collection.

Usage
    from vector_search import VectorSearch

    search = VectorSearch("exp5")                 # loads the store + embedder
    docs   = search.query("What is BM25?", k=5)
    docs   = search.query("What is BM25?", k=5, week="Week 2")

    # for the Retriever, which needs the whole corpus to build BM25:
    store, documents = search.store, search.documents
"""

import importlib.util
import os
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from chroma_store import ChromaVectorDB
from embeddings import MiniLMEmbedding

# Where the ingestion scripts live, relative to this file.
INGESTION_DIR = os.environ.get(
    "INGESTION_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "Ingestion_pipeline_code"),
)

DEFAULT_CHROMA_PATH = os.environ.get(
    "CHROMA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "Data", "Database", "chroma_db"),
)

_MODULE_CACHE: Dict[str, object] = {}


# ---------------------------------------------------------------------------
def load_ingestion_module(filename: str, directory: Optional[str] = None):
    """Load a module from the ingestion folder BY FILE PATH.

    '3-1-Ingest-to-ChromaDB.py' starts with a digit and contains hyphens, so it
    is not a legal Python identifier and `import` cannot reach it. importlib
    does not care about the name.

    Requires 3-1 to have its module-level work under `if __name__ == "__main__"`
    — otherwise importing it runs the ingestion.
    """
    directory = directory or INGESTION_DIR
    path = os.path.abspath(os.path.join(directory, filename))

    if path in _MODULE_CACHE:
        return _MODULE_CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {filename} in {directory}. Set INGESTION_DIR."
        )

    spec = importlib.util.spec_from_file_location(
        "ingestion_" + re.sub(r"\W", "_", filename), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[path] = module
    return module


def _strategies() -> Dict[str, dict]:
    """3-1's STRATEGIES dict — the single source of truth for collection names."""
    return load_ingestion_module("3-1-Ingest-to-ChromaDB.py").STRATEGIES


def collection_for(strategy: str) -> str:
    """'exp5' -> 'exp5_section_aware', as named by the ingestion script."""
    strategies = _strategies()
    key = strategy.lower()
    if key in strategies:
        return strategies[key]["collection_name"]
    # allow a full collection name to be passed straight through
    if strategy in {c["collection_name"] for c in strategies.values()}:
        return strategy
    raise ValueError(
        f"Unknown strategy {strategy!r}. Choose from {list(strategies)} "
        f"or pass a collection name directly."
    )


def list_collections(chroma_path: str = DEFAULT_CHROMA_PATH) -> List[Tuple[str, int]]:
    """(collection_name, vector_count) for everything actually on disk.

    Useful for checking ingestion landed before debugging retrieval.
    """
    import chromadb
    client = chromadb.PersistentClient(path=chroma_path)
    out = []
    for col in sorted(client.list_collections(), key=lambda c: c.name):
        out.append((col.name, col.count()))
    return out


# ---------------------------------------------------------------------------
class VectorSearch:
    """Read-only dense search over one ingested collection.

    Holds the embedder and the store together, because the ONE invariant that
    matters is that queries are embedded with the same model that embedded the
    corpus. Keeping them in one object means no call site can get that wrong.
    """

    def __init__(self,
                 strategy: str = "exp5",
                 chroma_path: str = DEFAULT_CHROMA_PATH,
                 embedder=None):
        self.strategy = strategy
        self.collection_name = collection_for(strategy)
        self.chroma_path = chroma_path

        # Accept an existing embedder so a process running several strategies
        # does not load MiniLM once per collection.
        self.embedder = embedder if embedder is not None else MiniLMEmbedding()

        self.store = ChromaVectorDB.load(self.collection_name, path=chroma_path)
        self._documents: Optional[List[Document]] = None

    # ------------------------------------------------------------------
    @property
    def documents(self) -> List[Document]:
        """Every chunk in the collection, loaded once and cached.

        The Retriever needs this to build its BM25 index: BM25 is lexical, not
        vector-based, so it needs the whole corpus text in memory rather than a
        similarity search. Fine at this scale; a real sparse index would be
        needed in the millions.
        """
        if self._documents is None:
            self._documents = self.store.all_documents()
        return self._documents

    # ------------------------------------------------------------------
    def embed(self, question: str):
        return self.embedder.embed_query(question)

    def query(self, question: str, k: int = 5,
              week: Optional[str] = None) -> List[Document]:
        """Dense search only. Hybrid retrieval is the Retriever's job."""
        where = {"week": week} if week else None
        return self.store.search(self.embed(question), k=k, where=where)

    def query_with_scores(self, question: str, k: int = 5,
                          week: Optional[str] = None):
        """Same, with cosine similarity (higher = better).

        Worth using when debugging: a top hit around 0.3 means nothing in the
        corpus really matched, which is a different problem from ranking badly.
        """
        where = {"week": week} if week else None
        return self.store.search_with_scores(self.embed(question), k=k, where=where)

    # ------------------------------------------------------------------
    def self_check(self) -> float:
        """Query the collection with the exact text of one of its own chunks.

        Same embedder on both sides returns ~1.0. A materially lower score means
        the corpus was embedded with a DIFFERENT model than the one loaded here
        — the silent failure mode where retrieval returns plausible nonsense and
        nothing raises. Run this once after every ingest.
        """
        got = self.store.collection.get(limit=1, include=["documents"])
        if not got["documents"]:
            raise ValueError(f"Collection {self.collection_name} is empty.")
        hits = self.store.search_with_scores(self.embed(got["documents"][0]), k=1)
        return hits[0][1] if hits else 0.0

    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        return (f"VectorSearch(strategy={self.strategy!r}, "
                f"collection={self.collection_name!r}, vectors={len(self)})")
