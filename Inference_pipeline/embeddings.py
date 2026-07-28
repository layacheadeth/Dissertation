from sentence_transformers import SentenceTransformer


class MiniLMEmbedding:

    def __init__(self):
        print("Loading MiniLM embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query):
        return self.model.encode(query, normalize_embeddings=True)
