import json
import faiss
import numpy as np
from langchain_core.documents import Document


class FAISSVectorDB:

    def __init__(self, dim):
        print("Initializing FAISS...")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.documents = []

    def add_documents(self, embeddings, documents):
        embeddings = np.array(embeddings).astype("float32")

        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch on add: index expects {self.dim}, "
                f"got {embeddings.shape[1]}"
            )

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding, k=5):
        query_embedding = np.array([query_embedding]).astype("float32")

        if query_embedding.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch on search: index expects {self.dim}, "
                f"got {query_embedding.shape[1]}"
            )

        scores, indices = self.index.search(query_embedding, k)
        results = [self.documents[i] for i in indices[0]]
        return results

    def save(self, index_path, docs_path):
        faiss.write_index(self.index, index_path)
        with open(docs_path, "w") as f:
            json.dump(
                [{"page_content": d.page_content, "metadata": d.metadata}
                 for d in self.documents], f
            )
        print(f"FAISS saved → {index_path}, {docs_path}")

    def load(self, index_path, docs_path):
        self.index = faiss.read_index(index_path)
        self.dim = self.index.d  # always sync dim from the saved index
        with open(docs_path, "r") as f:
            raw = json.load(f)
        self.documents = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in raw
        ]
        print(f"FAISS loaded ← {index_path} ({len(self.documents):,} chunks, dim={self.dim})")