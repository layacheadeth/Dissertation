"""
Every setting for the pipeline. If you want to change something, change it
here. Nothing else in the codebase holds a number or a path.

One embedding model is used everywhere: BGE (bge-small-en-v1.5). It embeds the
chunks and the questions, so the same vector space is used from end to end.
"""

from pathlib import Path

# ROOT is the folder that contains Share_components/ and Ingestion_pipeline/.
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------
LECTURES_DIR = ROOT / "Data" / "All_lectures"          # the input PDFs
EXTRACTED_DIR = ROOT / "Data" / "All_extracted_text"   # stage 1 and 2 output
CHROMA_DIR = ROOT / "Data" / "Database" / "chroma_db"  # the vector database
ANALYSIS_DIR = ROOT / "Data" / "Analysis"              # statistics and manifest

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Token counts are WordPiece tokens, counted with BGE's own tokenizer. BGE
# reads 512 tokens at a time and spends 2 on its own markers, so 510 is the
# hard ceiling: anything above it is silently thrown away when embedded.
#
# This is set to that ceiling, so a chunk is only ever cut because the model
# cannot hold more, never because we chose a smaller number. chunking_tokenizer
# refuses to run if it is ever raised past what the model can read.
#
# Worth knowing: bigger chunks are not automatically better to retrieve. One
# vector has to stand for everything in the chunk, so a 510-token chunk covering
# four topics sits in the average of all four and may lose to a shorter chunk
# that is about one of them. Most chunks will come out well under this anyway,
# because slides and sections are usually shorter; the ceiling only decides when
# a long one gets cut. If retrieval looks vague, try 250-300 here and re-run the
# chunkers and the ingestion.
MAX_TOKENS = 500

# How much text two neighbouring sliding-window chunks share (exp2 only).
# Kept at roughly a fifth of MAX_TOKENS, so a point made at the join is whole
# in one of the two chunks.
OVERLAP_TOKENS = 100

# Slides shorter than this are title cards and section dividers, so they are
# skipped (exp1 and exp3 only).
MIN_SLIDE_CHARS = 25

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
# The embedding model's own tokenizer, so the counts here are the counts the
# model will make. Replace with a local folder path to run without internet.
TOKENIZER = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Answering questions
# ---------------------------------------------------------------------------
STRATEGY = "exp2"        # which chunking strategy to search: exp1/exp2/exp3
EMBEDDER = "bge"         # the only embedding model; kept so runs record it
COMBO = "combo2"         # dense, sparse, combo1 or combo2 (see ranking_n_retrieval.py)

CANDIDATES = 20          # chunks each search method puts forward
TOP_N = 5                # chunks finally sent to the language model

# ---------------------------------------------------------------------------
# Equal-context control
# ---------------------------------------------------------------------------
# TOP_N holds the NUMBER of chunks constant across strategies. That is not a
# fair comparison, because a chunk is not the same size in each: five exp1
# chunks are about 250 tokens of context, five exp2 chunks about 1500. So a
# strategy can win simply by handing the model more material, and the effect
# of the chunking itself cannot be separated from the effect of context size.
#
# Setting BUDGET_TOKENS holds the AMOUNT OF TEXT constant instead. Chunks are
# taken in rank order until the next one would not fit, so every strategy
# delivers roughly the same number of tokens and only the composition of those
# tokens differs — which is what the chunking strategy actually controls.
#
# None keeps the old behaviour (fixed TOP_N), so existing results stay
# reproducible. Switch it on per-run with --budget-tokens; both conditions are
# meant to be reported side by side.
BUDGET_TOKENS = None     # e.g. 1500. None = use TOP_N instead.

# With a budget on, TOP_N no longer decides anything; the pool does. exp1 needs
# roughly 20 chunks to fill 1500 tokens, so a 20-candidate pool would leave the
# reranker nothing to discard and would cap exp1 below its budget while exp2
# fills easily. Raise it with --candidates when running a budget condition.
BUDGET_CANDIDATES = 60   # suggested --candidates for budget runs

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# The language model that writes the answers
# ---------------------------------------------------------------------------
LLM = "0.5b"             # a tag from llm_n_prompt.MODELS, or a full HuggingFace id
MAX_NEW_TOKENS = 256     # the answer should be a few sentences, so this is plenty
TEMPERATURE = 0.4        # above 0, so answers vary slightly between runs

# The exact sentence the model must emit when the slides do not answer the
# question. Kept as one constant so the evaluation can spot it by equality
# instead of guessing at a dozen ways of saying "I don't know".
NO_ANSWER = "The provided course material does not cover this."

# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------
RESULTS_DIR = ROOT / "Data" / "Results_generation"
