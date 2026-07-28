"""
============================================================================
 RETRIEVAL EVALUATOR  --  RAG-Socratic Teaching Assistant (COMP64702)
============================================================================

Scores the three chunking strategies (exp1 page-level, exp2 fixed-overlap,
exp3 structure-level) on benchmark_qa.json.

For every benchmark question it embeds student_query (all-MiniLM-L6-v2,
384-dim), queries each Chroma collection for the top-k chunks, and reports
six metrics:

  PAGE-ANCHOR metrics (a chunk is "relevant" if its (week, page) is in the
  question's gold page set = page_number + also_pages):
    - Precision@k : fraction of the top-k retrieved chunks that are relevant
    - Recall@k    : fraction of gold pages covered within the top-k
    - MRR         : 1 / rank of the first relevant chunk
    - Hit@1       : 1 if the top-1 chunk is relevant, else 0

  TEXT-based metrics (use the gold answer text, not the page anchor):
    - Keyword recall : gold keywords (extracted from gold_context /
                       qa_gold_standard) that appear in the retrieved chunk
                       text, over all gold keywords. Keywords are extracted
                       ONLY from the gold side; the retrieved side is just
                       tokenised and presence-checked.
    - Semantic sim   : max cosine similarity between any top-k retrieved
                       chunk and the gold_context (both embedded with the
                       same MiniLM model).

Page resolution:
  exp1 / exp3 chunks store 'page_number' in metadata -> read directly.
  exp2 (sliding-window) chunks span slides (metadata page = 'N/A'); the
  slides they cover are written INLINE as [Slide N] markers -> parsed out.

Outputs:
  - retrieval_summary.csv     : one row per strategy (averaged metrics)
  - retrieval_per_query.csv   : one row per (query, strategy) for sig-testing

Setup:  pip install chromadb sentence-transformers
Run:    python Evaluation/evaluate_retrieval.py
============================================================================
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

import csv
import json
import re
import sys

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


# ===========================================================================
#  CONFIGURATION
# ===========================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Backend: "chroma" (default) or "faiss". Override via CLI: python evaluate_retrieval.py faiss
BACKEND = (sys.argv[1].lower() if len(sys.argv) > 1 else "chroma")

CHROMA_PATH = os.path.join(_ROOT, "Data", "chroma_db")
FAISS_DIR   = os.path.join(_ROOT, "faiss_store")   # written by ingest_to_faiss.py
BENCHMARK   = os.path.join(_ROOT, "Data", "Benchmark", "benchmark_qa.json")

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTIONS = ["exp1_page_level", "exp2_fixed_overlap", "exp3_structure_level"]

K_VALUES = [1, 3, 5]
TOP_K = max(K_VALUES)
TEXT_K = TOP_K

# Output filenames are suffixed by backend so chroma/faiss runs don't overwrite.
SUMMARY_OUT  = os.path.join(_ROOT, "Data", "Benchmark", f"retrieval_summary_{BACKEND}.csv")
PERQUERY_OUT = os.path.join(_ROOT, "Data", "Benchmark", f"retrieval_per_query_{BACKEND}.csv")


# ===========================================================================
#  Page resolution
# ===========================================================================

_SLIDE_RE = re.compile(r"\[Slide\s+(\d+)\]", re.IGNORECASE)


def _week_num(metadata):
    for key in ("week_number", "week"):
        val = metadata.get(key)
        if val is None:
            continue
        if isinstance(val, int):
            return val
        m = re.search(r"\d+", str(val))
        if m:
            return int(m.group())
    return None


def pages_of_chunk(metadata, document):
    week = _week_num(metadata)
    pages = set()
    pn = metadata.get("page_number")
    if pn not in (None, "", "N/A"):
        try:
            pages.add((week, int(pn)))
        except (TypeError, ValueError):
            pass
    if not pages and document:
        for m in _SLIDE_RE.finditer(document):
            pages.add((week, int(m.group(1))))
    return pages


def gold_pages_of(entry):
    week = entry["week_number"]
    gold = {(week, entry["page_number"])}
    for p in entry.get("also_pages", []):
        gold.add((week, p))
    return gold


# ===========================================================================
#  Keyword extraction (gold side) + tokenisation (retrieved side)
# ===========================================================================

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "by", "for", "with", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "we", "you", "they", "he", "she", "i", "them", "us", "our", "your",
    "their", "from", "into", "than", "so", "such", "not", "no", "can", "will",
    "would", "should", "could", "may", "might", "do", "does", "did", "done",
    "has", "have", "had", "which", "what", "when", "where", "who", "how",
    "why", "each", "any", "all", "some", "more", "most", "other", "there",
    "here", "about", "up", "out", "over", "under", "again", "also", "very",
    "just", "only", "same", "between", "because", "while", "usually", "often",
    "different", "example", "using", "used", "use", "eg", "ie",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return [t for t in _WORD_RE.findall((text or "").lower()) if len(t) >= 3]


def extract_gold_keywords(entry):
    text = entry.get("gold_context") or entry.get("qa_gold_standard") or ""
    seen, keywords = set(), []
    for tok in _tokens(text):
        if tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        keywords.append(tok)
    return keywords


def keyword_recall(gold_keywords, retrieved_texts):
    if not gold_keywords:
        return 0.0
    bag = set()
    for txt in retrieved_texts:
        bag.update(_tokens(txt))
    found = sum(1 for kw in gold_keywords if kw in bag)
    return found / len(gold_keywords)


# ===========================================================================
#  Metrics (page-anchor)
# ===========================================================================

def hits_at_ranks(retrieved_page_sets, gold_pages):
    return [bool(pset & gold_pages) for pset in retrieved_page_sets]


def hit_rate_at_k(hit_flags, k):
    return 1.0 if any(hit_flags[:k]) else 0.0


def precision_at_k(hit_flags, k):
    topk = hit_flags[:k]
    return (sum(1 for h in topk if h) / len(topk)) if topk else 0.0


def reciprocal_rank(hit_flags):
    for i, hit in enumerate(hit_flags, start=1):
        if hit:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved_page_sets, gold_pages, k):
    covered = set()
    for pset in retrieved_page_sets[:k]:
        covered |= (pset & gold_pages)
    return len(covered) / len(gold_pages) if gold_pages else 0.0


# ===========================================================================
#  Metric: semantic similarity (text)
# ===========================================================================

def semantic_similarity(model, gold_context, retrieved_texts):
    texts = [t for t in retrieved_texts if t and t.strip()]
    if not gold_context or not gold_context.strip() or not texts:
        return 0.0
    vecs = model.encode([gold_context] + texts, normalize_embeddings=True)
    gold_vec, chunk_vecs = vecs[0], vecs[1:]
    sims = chunk_vecs @ gold_vec
    return float(np.max(sims))


# ===========================================================================
#  Backends: each returns {strategy_name: searcher}, where
#  searcher(query_vec, k) -> (list_of_metadata_dicts, list_of_doc_texts)
# ===========================================================================

def open_chroma_backend():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    searchers = {}
    for name in COLLECTIONS:
        col = client.get_collection(name)

        def make(col):
            def search(qvec, k):
                res = col.query(query_embeddings=[qvec], n_results=k,
                                include=["metadatas", "documents"])
                return res["metadatas"][0], res["documents"][0]
            return search
        searchers[name] = make(col)
    return searchers


def open_faiss_backend():
    # Import here so a chroma-only run doesn't require faiss installed.
    sys.path.insert(0, _ROOT)
    from vectore_store import FAISSVectorDB

    searchers = {}
    for name in COLLECTIONS:
        index_path = os.path.join(FAISS_DIR, f"{name}.index")
        docs_path = os.path.join(FAISS_DIR, f"{name}_docs.json")
        if not (os.path.exists(index_path) and os.path.exists(docs_path)):
            raise FileNotFoundError(
                f"Missing FAISS files for '{name}'. Run: python ingest_to_faiss.py")
        store = FAISSVectorDB(1)          # dim is overwritten by load()
        store.load(index_path, docs_path)

        def make(store):
            def search(qvec, k):
                docs = store.search(qvec, k)
                metas = [d.metadata for d in docs]
                texts = [d.page_content for d in docs]
                return metas, texts
            return search
        searchers[name] = make(store)
    return searchers


def open_backend():
    if BACKEND == "chroma":
        return open_chroma_backend()
    if BACKEND == "faiss":
        return open_faiss_backend()
    raise ValueError(f"Unknown backend '{BACKEND}'. Use 'chroma' or 'faiss'.")


# ===========================================================================
#  Evaluation
# ===========================================================================

def evaluate():
    benchmark = json.load(open(BENCHMARK, encoding="utf-8"))
    print(f"Loaded {len(benchmark)} benchmark questions from {BENCHMARK}")

    print(f"Loading embedding model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Backend: {BACKEND}")
    searchers = open_backend()
    print(f"Opened strategies: {list(searchers)}\n")

    queries = [e["student_query"] for e in benchmark]
    query_vecs = model.encode(queries, normalize_embeddings=True).tolist()

    metric_cols = (
        [f"precision@{k}" for k in K_VALUES] +
        [f"recall@{k}" for k in K_VALUES] +
        ["mrr", "hit@1", f"keyword_recall@{TEXT_K}", f"semantic_sim@{TEXT_K}"]
    )

    per_query_rows = []
    acc = {name: {mc: [] for mc in metric_cols} for name in COLLECTIONS}

    for entry, qvec in zip(benchmark, query_vecs):
        gold = gold_pages_of(entry)
        gold_keywords = extract_gold_keywords(entry)
        gold_context = entry.get("gold_context") or entry.get("qa_gold_standard") or ""

        for name, search in searchers.items():
            metas, docs = search(qvec, TOP_K)

            retrieved = [pages_of_chunk(m, d) for m, d in zip(metas, docs)]
            hit_flags = hits_at_ranks(retrieved, gold)

            row = {"query_id": entry["query_id"], "collection": name,
                   "gold_week": entry["week_number"], "gold_page": entry["page_number"]}

            for k in K_VALUES:
                row[f"precision@{k}"] = round(precision_at_k(hit_flags, k), 4)
                row[f"recall@{k}"] = round(recall_at_k(retrieved, gold, k), 4)
            row["mrr"] = round(reciprocal_rank(hit_flags), 4)
            row["hit@1"] = hit_rate_at_k(hit_flags, 1)

            topk_docs = docs[:TEXT_K]
            row[f"keyword_recall@{TEXT_K}"] = round(keyword_recall(gold_keywords, topk_docs), 4)
            row[f"semantic_sim@{TEXT_K}"] = round(semantic_similarity(model, gold_context, topk_docs), 4)

            for mc in metric_cols:
                acc[name][mc].append(row[mc])
            per_query_rows.append(row)

    os.makedirs(os.path.dirname(PERQUERY_OUT) or ".", exist_ok=True)
    fieldnames = ["query_id", "collection", "gold_week", "gold_page"] + metric_cols
    with open(PERQUERY_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_query_rows)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["collection"] + metric_cols)
        for name in COLLECTIONS:
            w.writerow([name] + [mean(acc[name][mc]) for mc in metric_cols])

    width = 18
    line = "=" * (24 + width * len(metric_cols))
    print(line)
    print(f"{'collection':22} " + " ".join(f"{mc:>{width}}" for mc in metric_cols))
    print("-" * (24 + width * len(metric_cols)))
    for name in COLLECTIONS:
        print(f"{name:22} " +
              " ".join(f"{mean(acc[name][mc]):{width}.4f}" for mc in metric_cols))
    print(line)
    print(f"\nWrote summary   -> {SUMMARY_OUT}")
    print(f"Wrote per-query -> {PERQUERY_OUT}  (use this for significance testing)")


if __name__ == "__main__":
    evaluate()
