import argparse
import json
import os
import re
import math
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Token counting (same convention as experiments 2 and 3)
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4


# ---------------------------------------------------------------------------
# Embedding backend selection
#   1. sentence-transformers  -> true dense semantic embeddings (preferred)
#   2. scikit-learn TF-IDF    -> sparse lexical-semantic approximation
#   3. pure-python bag-of-words -> always available fallback
# ---------------------------------------------------------------------------
def build_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """
    Returns (embed_fn, backend_name).
    embed_fn: List[str] -> List[List[float]] (L2-normalised vectors)
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)

        def embed(texts: List[str]) -> List[List[float]]:
            vecs = model.encode(texts, normalize_embeddings=True)
            return [list(map(float, v)) for v in vecs]

        return embed, f"sentence-transformers:{model_name}"
    except Exception:
        pass

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        def embed(texts: List[str]) -> List[List[float]]:
            vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            matrix = vectorizer.fit_transform(texts)
            dense = matrix.toarray()
            out = []
            for row in dense:
                norm = math.sqrt(sum(x * x for x in row)) or 1.0
                out.append([float(x / norm) for x in row])
            return out

        return embed, "sklearn-tfidf"
    except Exception:
        pass

    def embed(texts: List[str]) -> List[List[float]]:
        # Bag-of-words over a shared vocabulary, L2-normalised
        tokenised = [re.findall(r"[a-z0-9]+", t.lower()) for t in texts]
        vocab = {}
        for toks in tokenised:
            for tok in toks:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        out = []
        for toks in tokenised:
            vec = [0.0] * len(vocab)
            for tok in toks:
                vec[vocab[tok]] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

    return embed, "bag-of-words-fallback"


def cosine(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Sentence segmentation with page provenance
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def split_sentences(text: str, min_len: int = 3) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return [p for p in parts if len(p) >= min_len]


def collect_units(pages: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """Flatten pages into (sentence, page_number) units, in reading order."""
    units = []
    for page in pages:
        content = page.get("content", "").strip()
        page_num = page.get("page_number")
        if not content:
            continue
        for sent in split_sentences(content):
            units.append((sent, page_num))
    return units


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


# ---------------------------------------------------------------------------
# Experiment 4
# ---------------------------------------------------------------------------
def run_experiment_4(
    input_json_path: str,
    breakpoint_percentile: float = 25.0,
    max_tokens: int = 250,
    min_tokens: int = 40,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Experiment 4: Semantic-Aware Chunking
    Segments text into sentences, embeds each sentence, and cuts the stream
    wherever the similarity between adjacent sentences drops below an adaptive
    threshold (a low percentile of all observed similarities). Boundaries
    therefore follow meaning shifts rather than page or token positions.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = data.get("filename", "")
    week = data.get("week", "")
    pages = data.get("pages", [])

    units = collect_units(pages)
    if not units:
        return []

    sentences = [u[0] for u in units]
    page_nums = [u[1] for u in units]

    embed, backend = build_embedder(model_name)
    vectors = embed(sentences)

    # Similarity between each adjacent sentence pair
    similarities = [cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    threshold = percentile(similarities, breakpoint_percentile) if similarities else 0.0

    chunks = []
    chunk_idx = 1
    current_sents: List[str] = []
    current_pages: List[int] = []

    def flush():
        nonlocal current_sents, current_pages, chunk_idx
        if not current_sents:
            return
        text = " ".join(current_sents).strip()
        if text:
            seen_pages = sorted({p for p in current_pages if p is not None})
            chunks.append({
                "experiment_id": "exp4_semantic_aware",
                "chunk_id": f"{week.replace(' ', '')}_sem_{chunk_idx}",
                "filename": filename,
                "week": week,
                "page_start": seen_pages[0] if seen_pages else None,
                "page_end": seen_pages[-1] if seen_pages else None,
                "source_pages": seen_pages,
                "sentence_count": len(current_sents),
                "token_count": count_tokens(text),
                "embedding_backend": backend,
                "breakpoint_percentile": breakpoint_percentile,
                "similarity_threshold": round(threshold, 4),
                "content": text,
            })
            chunk_idx += 1
        current_sents = []
        current_pages = []

    for i, sent in enumerate(sentences):
        current_sents.append(sent)
        current_pages.append(page_nums[i])
        candidate = " ".join(current_sents)

        # Hard cap: never let a semantic chunk exceed the token budget
        if count_tokens(candidate) >= max_tokens:
            flush()
            continue

        # Semantic breakpoint: similarity to the next sentence is unusually low
        if i < len(similarities):
            if similarities[i] <= threshold and count_tokens(candidate) >= min_tokens:
                flush()

    flush()

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 4: Semantic-Aware Chunking on slide JSON data."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to input JSON file"
    )

    parser.add_argument(
        "-o", "--output",
        required=False,
        type=str,
        default=None,
        help="Path to save output JSON file"
    )

    parser.add_argument(
        "-p", "--percentile",
        required=False,
        type=float,
        default=25.0,
        help="Similarity percentile used as the breakpoint threshold (default: 25). Lower = fewer, larger chunks."
    )

    parser.add_argument(
        "-m", "--max-tokens",
        required=False,
        type=int,
        default=250,
        help="Hard upper bound on tokens per chunk (default: 250)"
    )

    parser.add_argument(
        "-n", "--min-tokens",
        required=False,
        type=int,
        default=40,
        help="Minimum tokens before a semantic breakpoint may be honoured (default: 40)"
    )

    parser.add_argument(
        "--model",
        required=False,
        type=str,
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (ignored if the library is unavailable)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    results = run_experiment_4(
        args.input,
        breakpoint_percentile=args.percentile,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        model_name=args.model,
    )
    backend = results[0]["embedding_backend"] if results else "n/a"
    print(f"[Exp 4] Generated {len(results)} semantic chunks (backend: {backend}).")

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Exp 4] Saved results to: {args.output}")
    else:
        print("\n--- Sample Output Chunk ---")
        if results:
            print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
