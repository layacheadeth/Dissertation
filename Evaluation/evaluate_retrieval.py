"""
Retrieval evaluator -- RAG-Socratic Teaching Assistant (COMP64702)

Scores five chunking strategies on benchmark_qa_pruned.json.

    STAGE 1  load benchmark        -> student queries + gold anchors
    STAGE 2  encode queries        -> one vector per question (MiniLM, normalised)
    STAGE 3  search 5 collections  -> top-5 chunks each, same query vector
    STAGE 4  score by page anchor  -> precision, recall, MRR, hit@1, nDCG
    STAGE 5  score by text         -> keyword recall, semantic similarity
    STAGE 6  write JSON            -> per-query records, then per-strategy means

Benchmark formats accepted (in priority order):
    1. "gold_pages": [{"week": w, "page": p, "rank": r, "utility": 1-3, ...}, ...]
    2. "gold_anchors": [{"week": w, "page": p}, ...]
    3. flat "week_number" + "page_number" (+ optional "also_pages")

Format 1 carries graded relevance. Pages below MIN_UTILITY are dropped from the
gold set, BUT every question always keeps at least one gold page -- if the filter
would empty a question, its best-rated page is kept regardless. A question with no
gold anchor is unscoreable and would silently drag every strategy's mean to zero,
which is worse than scoring it against an imperfect anchor.

Setup: pip install chromadb sentence-transformers
Run:   python Evaluation/evaluate_retrieval.py [chroma|faiss]
"""

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

import json
import math
import re
import sys

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


# ===========================================================================
# CONFIGURATION
# ===========================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

BACKEND = sys.argv[1].lower() if len(sys.argv) > 1 else "chroma"

CHROMA_PATH = os.path.join(_ROOT, "Data", "chroma_db")
FAISS_DIR = os.path.join(_ROOT, "faiss_store")
# BENCHMARK = os.path.join(_ROOT, "Data", "Benchmark", "benchmark_qa.json")

BENCHMARK = os.path.join(_ROOT, "Data", "Benchmark", "benchmark_qa_pruned.json")


EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTIONS = [
    "exp1_page_level",
    "exp2_fixed_overlap",
    "exp3_structure_level",
    "exp4_semantic_aware",
    "exp5_section_aware",
]

K_VALUES = [1, 3, 5]
TOP_K = max(K_VALUES)
TEXT_K = TOP_K

# How much of each retrieved chunk to echo into the per-query JSON. Set to 0 to
# omit snippets entirely if the file gets unwieldy.
SNIPPET_CHARS = 200

# Gold pages rated below this are dropped -- unless dropping them would leave a
# question with no gold at all, in which case the best-rated page is kept.
MIN_UTILITY = 2
DEFAULT_UTILITY = 3  # assumed grade for legacy formats that carry no ratings

SUMMARY_OUT = os.path.join(_ROOT, "Data", "Benchmark", f"retrieval_summary_{BACKEND}.json")
PERQUERY_OUT = os.path.join(_ROOT, "Data", "Benchmark", f"retrieval_per_query_{BACKEND}.json")

METRIC_COLS = (
    [f"precision@{k}" for k in K_VALUES]
    + [f"recall@{k}" for k in K_VALUES]
    + [f"ndcg@{k}" for k in K_VALUES]
    + ["mrr", "hit@1", f"keyword_recall@{TEXT_K}", f"semantic_sim@{TEXT_K}"]
)


# ===========================================================================
# STAGE 3 -- SEARCH BACKENDS
# Each backend returns {strategy: search(query_vec, k) -> (metadatas, texts)}
# ===========================================================================

def open_chroma_backend():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )

    def make(col):
        def search(qvec, k):
            res = col.query(query_embeddings=[qvec], n_results=k,
                            include=["metadatas", "documents"])
            return res["metadatas"][0], res["documents"][0]
        return search

    return {name: make(client.get_collection(name)) for name in COLLECTIONS}


def open_faiss_backend():
    sys.path.insert(0, _ROOT)
    from vectore_store import FAISSVectorDB

    def make(store):
        def search(qvec, k):
            docs = store.search(qvec, k)
            return [d.metadata for d in docs], [d.page_content for d in docs]
        return search

    searchers = {}
    for name in COLLECTIONS:
        index_path = os.path.join(FAISS_DIR, f"{name}.index")
        docs_path = os.path.join(FAISS_DIR, f"{name}_docs.json")
        if not (os.path.exists(index_path) and os.path.exists(docs_path)):
            raise FileNotFoundError(
                f"Missing FAISS files for '{name}'. Run: python ingest_to_faiss.py")
        store = FAISSVectorDB(1)
        store.load(index_path, docs_path)
        searchers[name] = make(store)
    return searchers


def open_backend():
    if BACKEND == "chroma":
        return open_chroma_backend()
    if BACKEND == "faiss":
        return open_faiss_backend()
    raise ValueError(f"Unknown backend '{BACKEND}'. Use 'chroma' or 'faiss'.")


# ===========================================================================
# STAGE 4 -- PAGE-ANCHOR SCORING
# A chunk is relevant if its (week, page) set intersects the gold set.
# ===========================================================================

_SLIDE_RE = re.compile(r"\[Slide\s+(\d+)\]", re.IGNORECASE)


def _as_int(val):
    if val in (None, "", "N/A"):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _parse_page_list(val):
    if val in (None, "", "N/A"):
        return []
    raw = val if isinstance(val, (list, tuple, set)) else str(val).split(",")
    return [p for p in (_as_int(str(i).strip()) for i in raw) if p is not None]


def _week_num(metadata):
    for key in ("week_number", "week"):
        val = metadata.get(key)
        if isinstance(val, int):
            return val
        if val is not None:
            m = re.search(r"\d+", str(val))
            if m:
                return int(m.group())
    return None


def pages_of_chunk(metadata, document):
    """exp1/exp3: page_number. exp4/exp5: source_pages, else page_start..page_end.
    exp2: page metadata is 'N/A', slides appear inline as [Slide N]."""
    week = _week_num(metadata)

    pn = _as_int(metadata.get("page_number"))
    if pn is not None:
        return {(week, pn)}

    pages = {(week, p) for p in _parse_page_list(metadata.get("source_pages"))}
    if pages:
        return pages

    start, end = _as_int(metadata.get("page_start")), _as_int(metadata.get("page_end"))
    if start is not None:
        end = start if (end is None or end < start) else end
        return {(week, p) for p in range(start, end + 1)}

    if document:
        return {(week, int(m.group(1))) for m in _SLIDE_RE.finditer(document)}

    return set()


def gold_grades_of(entry):
    """{(week, page): utility_grade} for one question. Never returns empty when
    the entry declares any gold page at all."""

    # --- format 1: graded gold_pages ---
    if entry.get("gold_pages"):
        graded = {}
        for p in entry["gold_pages"]:
            key = (p["week"], p["page"])
            grade = _as_int(p.get("utility"))
            if grade is None:
                grade = DEFAULT_UTILITY
            # a page listed twice keeps its highest grade
            graded[key] = max(grade, graded.get(key, 0))

        kept = {k: g for k, g in graded.items() if g >= MIN_UTILITY}
        if kept:
            return kept

        # Filter emptied the question -> keep its single best page rather than
        # leaving it unscoreable. Ties break on rank order in the source file.
        best_key = max(graded, key=lambda k: graded[k])
        return {best_key: graded[best_key]}

    # --- format 2: ungraded anchors ---
    if entry.get("gold_anchors"):
        return {(a["week"], a["page"]): DEFAULT_UTILITY for a in entry["gold_anchors"]}

    # --- format 3: flat week_number / page_number ---
    week = entry.get("week_number")
    p_num = entry.get("page_number")
    pages = set()
    if isinstance(p_num, list):
        pages |= {(week, p) for p in p_num}
    elif p_num is not None:
        pages.add((week, p_num))
    pages |= {(week, p) for p in entry.get("also_pages", [])}
    return {k: DEFAULT_UTILITY for k in pages}


def gold_pages_of(entry):
    return set(gold_grades_of(entry))


def hits_at_ranks(retrieved_page_sets, gold_pages):
    return [bool(pset & gold_pages) for pset in retrieved_page_sets]


def precision_at_k(hit_flags, k):
    topk = hit_flags[:k]
    return sum(topk) / len(topk) if topk else 0.0


def recall_at_k(retrieved_page_sets, gold_pages, k):
    covered = set()
    for pset in retrieved_page_sets[:k]:
        covered |= pset & gold_pages
    return len(covered) / len(gold_pages) if gold_pages else 0.0


def ndcg_at_k(retrieved_page_sets, gold_grades, k):
    """Graded ranking quality. A chunk's gain is the best grade among the gold
    pages it covers; 0 if it covers none."""
    if not gold_grades:
        return 0.0

    gains = [
        max((gold_grades[p] for p in pset if p in gold_grades), default=0)
        for pset in retrieved_page_sets[:k]
    ]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    ideal = sorted(gold_grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))

    return dcg / idcg if idcg else 0.0


def reciprocal_rank(hit_flags):
    for i, hit in enumerate(hit_flags, start=1):
        if hit:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(hit_flags, k):
    return 1.0 if any(hit_flags[:k]) else 0.0


# ===========================================================================
# STAGE 5 -- TEXT SCORING
# Keywords come from the gold side only; retrieved chunks are presence-checked.
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


def gold_text_of(entry):
    return entry.get("gold_context") or entry.get("qa_gold_standard") or ""


def extract_gold_keywords(entry):
    seen, keywords = set(), []
    for tok in _tokens(gold_text_of(entry)):
        if tok not in _STOPWORDS and tok not in seen:
            seen.add(tok)
            keywords.append(tok)
    return keywords


def keyword_recall(gold_keywords, retrieved_texts):
    if not gold_keywords:
        return 0.0
    bag = set()
    for txt in retrieved_texts:
        bag.update(_tokens(txt))
    return sum(1 for kw in gold_keywords if kw in bag) / len(gold_keywords)


def semantic_similarity(model, gold_context, retrieved_texts):
    texts = [t for t in retrieved_texts if t and t.strip()]
    if not gold_context.strip() or not texts:
        return 0.0
    vecs = model.encode([gold_context] + texts, normalize_embeddings=True)
    return float(np.max(vecs[1:] @ vecs[0]))


# ===========================================================================
# STAGES 1-6 -- PIPELINE
# ===========================================================================

def score_one(entry, metas, docs, model, gold_grades=None):
    """All metrics for one (question, strategy) pair."""
    if gold_grades is None:
        gold_grades = gold_grades_of(entry)
    gold = set(gold_grades)

    retrieved = [pages_of_chunk(m, d) for m, d in zip(metas, docs)]
    hit_flags = hits_at_ranks(retrieved, gold)
    topk_docs = docs[:TEXT_K]

    row = {}
    for k in K_VALUES:
        row[f"precision@{k}"] = round(precision_at_k(hit_flags, k), 4)
        row[f"recall@{k}"] = round(recall_at_k(retrieved, gold, k), 4)
        row[f"ndcg@{k}"] = round(ndcg_at_k(retrieved, gold_grades, k), 4)
    row["mrr"] = round(reciprocal_rank(hit_flags), 4)
    row["hit@1"] = hit_rate_at_k(hit_flags, 1)
    row[f"keyword_recall@{TEXT_K}"] = round(
        keyword_recall(extract_gold_keywords(entry), topk_docs), 4)
    row[f"semantic_sim@{TEXT_K}"] = round(
        semantic_similarity(model, gold_text_of(entry), topk_docs), 4)
    return row


def write_json(per_query_records, acc):
    """Two files: nested per-query records, and per-strategy means."""
    os.makedirs(os.path.dirname(PERQUERY_OUT) or ".", exist_ok=True)

    per_query_doc = {
        "backend": BACKEND,
        "benchmark": os.path.basename(BENCHMARK),
        "top_k": TOP_K,
        "k_values": K_VALUES,
        "min_utility": MIN_UTILITY,
        "strategies": COLLECTIONS,
        "metrics": METRIC_COLS,
        "queries": per_query_records,
    }
    with open(PERQUERY_OUT, "w", encoding="utf-8") as f:
        json.dump(per_query_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    per_strategy = {
        name: {mc: mean(acc[name][mc]) for mc in METRIC_COLS}
        for name in COLLECTIONS
    }
    ranked = sorted(COLLECTIONS,
                    key=lambda n: per_strategy[n].get("mrr", 0.0), reverse=True)

    summary_doc = {
        "backend": BACKEND,
        "benchmark": os.path.basename(BENCHMARK),
        "n_questions": len(per_query_records),
        "top_k": TOP_K,
        "k_values": K_VALUES,
        "min_utility": MIN_UTILITY,
        "embed_model": EMBED_MODEL,
        "metrics": METRIC_COLS,
        "ranking_by_mrr": ranked,
        "per_strategy": per_strategy,
    }
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def print_summary(acc):
    width = 18
    line = "=" * (24 + width * len(METRIC_COLS))
    print(line)
    print(f"{'collection':22} " + " ".join(f"{mc:>{width}}" for mc in METRIC_COLS))
    print("-" * (24 + width * len(METRIC_COLS)))
    for name in COLLECTIONS:
        print(f"{name:22} " +
              " ".join(f"{mean(acc[name][mc]):{width}.4f}" for mc in METRIC_COLS))
    print(line)
    print(f"\nWrote summary   -> {SUMMARY_OUT}")
    print(f"Wrote per-query -> {PERQUERY_OUT}  (use this for significance testing)")


def evaluate():
    # STAGE 1 -- load benchmark
    benchmark = json.load(open(BENCHMARK, encoding="utf-8"))
    print(f"Loaded {len(benchmark)} benchmark questions from {BENCHMARK}")

    # Resolve gold once per question, and report anything unusual up front.
    gold_by_query = {}
    for entry in benchmark:
        grades = gold_grades_of(entry)
        gold_by_query[entry["query_id"]] = grades
        if not grades:
            print(f"  WARNING: {entry['query_id']} has no gold page -- "
                  f"it will score 0 for every strategy.")
        elif max(grades.values()) < MIN_UTILITY:
            print(f"  NOTE: {entry['query_id']} has no page rated >= {MIN_UTILITY}; "
                  f"kept its best page (utility {max(grades.values())}) as anchor.")

    total_gold = sum(len(g) for g in gold_by_query.values())
    print(f"Gold pages: {total_gold} across {len(benchmark)} questions "
          f"(mean {total_gold / max(len(benchmark), 1):.2f} per question)")

    # STAGE 2 -- encode every question once, reused across all strategies
    print(f"Loading embedding model '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)
    query_vecs = model.encode([e["student_query"] for e in benchmark],
                              normalize_embeddings=True).tolist()

    print(f"Backend: {BACKEND}")
    searchers = open_backend()
    print(f"Opened strategies: {list(searchers)}\n")

    per_query_records = []
    acc = {name: {mc: [] for mc in METRIC_COLS} for name in COLLECTIONS}

    for entry, qvec in zip(benchmark, query_vecs):
        gold_grades = gold_by_query[entry["query_id"]]

        record = {
            "query_id": entry["query_id"],
            "student_query": entry["student_query"],
            "question_type": entry.get("question_type"),
            "difficulty": entry.get("difficulty"),
            "gold": [
                {"week": w, "page": p, "utility": gold_grades[(w, p)]}
                for w, p in sorted(gold_grades, key=lambda wp: (wp[0] or 0, wp[1]))
            ],
            "n_gold": len(gold_grades),
            "results": {},
        }

        for name, search in searchers.items():
            # STAGE 3 -- retrieve
            metas, docs = search(qvec, TOP_K)

            # STAGES 4 & 5 -- score
            scores = score_one(entry, metas, docs, model, gold_grades)

            record["results"][name] = {
                "metrics": scores,
                "retrieved": [
                    {
                        "rank": i,
                        "pages": [
                            {"week": w, "page": p}
                            for w, p in sorted(pset, key=lambda wp: (wp[0] or 0, wp[1]))
                        ],
                        "hit": bool(pset & set(gold_grades)),
                        "gain": max((gold_grades[x] for x in pset if x in gold_grades),
                                    default=0),
                        "snippet": (doc or "")[:SNIPPET_CHARS].replace("\n", " "),
                    }
                    for i, (pset, doc) in enumerate(
                        ((pages_of_chunk(m, d), d) for m, d in zip(metas, docs)), start=1)
                ],
            }
            for mc in METRIC_COLS:
                acc[name][mc].append(scores[mc])

        per_query_records.append(record)

    # STAGE 6 -- write and report
    write_json(per_query_records, acc)
    print_summary(acc)


if __name__ == "__main__":
    evaluate()