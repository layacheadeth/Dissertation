"""
The vector database.

Two things this file exists to get right:

1. We hand ChromaDB the vectors ourselves. If you call upsert() without them,
   ChromaDB quietly embeds the text with its own model, and searching then
   compares vectors from two different models. No error is raised, the results
   are just meaningless.

2. ChromaDB only stores single values in metadata, not lists. Our page_number
   is a list, so it is saved as "1,2,3" and turned back into [1, 2, 3] when
   read out again.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import chromadb
import numpy as np
from langchain_core.documents import Document

from Share_components import configuration


# Metadata fields that are lists and must survive being saved and reloaded.
_LIST_FIELDS = ("page_number",)


def flatten_metadata(metadata: Dict) -> Dict:
    """Turn metadata into values ChromaDB accepts."""
    out = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            out[key] = ",".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def rehydrate_metadata(metadata: Dict) -> Dict:
    """Turn the saved values back into what they were, lists included."""
    out = dict(metadata or {})
    for key in _LIST_FIELDS:
        value = out.get(key)
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            out[key] = [int(p) if p.lstrip("-").isdigit() else p for p in parts]
        elif value is None:
            out[key] = []
        elif not isinstance(value, list):
            out[key] = [value]
    return out


class ChromaVectorDB:
    """One collection of chunks and their vectors."""

    def __init__(self, collection_name: str, reset: bool = False, path=None):
        self.collection_name = collection_name
        self.path = str(path or configuration.CHROMA_DIR)
        self.client = chromadb.PersistentClient(path=self.path)

        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass                      # it did not exist yet

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # Cosine similarity, to match our normalised vectors. ChromaDB's
            # default is straight-line distance, which would rank differently.
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _as_list(vectors) -> List[List[float]]:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array.tolist()

    def add_documents(self, vectors, documents: Sequence[Document],
                      batch_size: int = 500) -> None:
        """Save chunks and their vectors together.

        Running this twice with the same chunk ids overwrites rather than
        duplicating.
        """
        if len(vectors) != len(documents):
            raise ValueError(
                f"got {len(vectors)} vectors but {len(documents)} chunks — "
                f"there must be one vector per chunk."
            )
        if not documents:
            return

        rows = self._as_list(vectors)
        texts, metadatas, ids = [], [], []

        for position, doc in enumerate(documents):
            texts.append(doc.page_content)
            metadatas.append(flatten_metadata(doc.metadata))
            ids.append(str(doc.metadata.get("chunk_id") or f"auto_{position}"))

        # Written in batches because ChromaDB rejects very large single writes.
        for start in range(0, len(texts), batch_size):
            stop = start + batch_size
            self.collection.upsert(
                documents=texts[start:stop],
                embeddings=rows[start:stop],     # never leave this out
                metadatas=metadatas[start:stop],
                ids=ids[start:stop],
            )

    def search(self, query_vector, k: int = 5,
               where: Optional[Dict] = None) -> List[Document]:
        """The k closest chunks, closest first."""
        return [doc for doc, _ in self.search_with_scores(query_vector, k, where)]

    def search_with_scores(self, query_vector, k: int = 5,
                           where: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """The same, with a similarity score where higher means closer.

        ChromaDB reports distance, where lower means closer, so we flip it.
        """
        total = self.collection.count()
        if total == 0:
            return []

        result = self.collection.query(
            query_embeddings=self._as_list(query_vector),
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return [
            (Document(page_content=text, metadata=rehydrate_metadata(meta)),
             1.0 - float(distance))
            for text, meta, distance in zip(result["documents"][0],
                                            result["metadatas"][0],
                                            result["distances"][0])
        ]

    def all_documents(self) -> List[Document]:
        """Every chunk in the collection, in a fixed order.

        The keyword search (BM25) needs all the text at once, because it scores
        by how often words appear across the whole corpus rather than by
        vector distance.
        """
        got = self.collection.get(include=["documents", "metadatas"])
        rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                      key=lambda row: str(row[0]))
        return [Document(page_content=text, metadata=rehydrate_metadata(meta))
                for _id, text, meta in rows]

    @classmethod
    def load(cls, collection_name: str, path=None) -> "ChromaVectorDB":
        store = cls(collection_name, path=path)
        if len(store) == 0:
            raise ValueError(
                f"Collection '{collection_name}' is empty. "
                f"Run the 3-x ingestion scripts first."
            )
        return store

    def save(self) -> None:
        """Does nothing. ChromaDB writes to disk as it goes."""
        return None

    def __len__(self) -> int:
        return self.collection.count()

    def __repr__(self) -> str:
        return f"ChromaVectorDB({self.collection_name!r}, {len(self)} vectors)"
