"""
The embedding model, which turns text into vectors.

One model is used for everything: BGE (bge-small-en-v1.5).

    embed_documents(list of texts) -> one vector per text
    embed_query(a question)        -> one vector

BGE needs a short instruction added to questions but not to chunks. That is
handled inside embed_query, so as long as you use embed_documents for chunks
and embed_query for questions, you never have to think about it. Getting it
the wrong way round loses most of the benefit of the model.

The model is loaded once and shared, so nothing loads it twice and wastes a
minute and a gigabyte.
"""

from sentence_transformers import SentenceTransformer


class BGEEmbedding:
    """bge-small-en-v1.5. 384 numbers per vector, reads up to 512 tokens."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    DIMENSIONS = 384
    MAX_TOKENS = 512       # anything longer is cut off by the model
    BATCH_SIZE = 64

    def __init__(self):
        print(f"  {self.MODEL_NAME} ({self.DIMENSIONS} numbers per vector, "
              f"max {self.MAX_TOKENS} tokens)")
        self.model = SentenceTransformer(self.MODEL_NAME)

    def embed_documents(self, texts):
        """Vectors for chunks, sentences, or anything that is not a question."""
        return self.model.encode(
            texts,
            batch_size=self.BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )

    def embed_query(self, question):
        """Vector for a question, with BGE's retrieval instruction attached."""
        return self.model.encode(
            self.QUERY_PREFIX + question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def __repr__(self):
        return f"<BGEEmbedding {self.MODEL_NAME}>"


_SHARED = None


def get_embedder() -> BGEEmbedding:
    """The one BGE instance, loaded the first time it is asked for."""
    global _SHARED
    if _SHARED is None:
        _SHARED = BGEEmbedding()
    return _SHARED


# Kept so older code that looked the model up by name still works.
EMBEDDERS = {"bge": BGEEmbedding}
