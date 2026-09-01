# Architecture

A RAG system that answers questions from COMP64702 lecture slides, built to
compare three chunking strategies against three language models.

## The three folders

```
Share_components/     used by both pipelines: settings, chunking, embedding, storage
Inference_pipeline/   what happens when a question is asked
Evaluation/           scoring the answers afterwards
```

They depend in one direction only. `Share_components` knows nothing about
questions; `Inference_pipeline` imports it; `Evaluation` reads the files
`Inference_pipeline` wrote and imports `Share_components` for one thing only
(the fixed refusal sentence, so the two cannot disagree about its wording).

## The path a question takes

```
question
   │
   ├─ router ─────────────► "what can you do?" answered here, no search
   │
   ├─ 1. embed             vector_search/          question → vector
   ├─ 2. retrieve          ranking_n_retrieval/    vector + words → chunks
   │      dense ─┐
   │      sparse ─┴─ merge ─ rerank ─ cut (top_n or token budget)
   │
   ├─ gate ───────────────► best score below the floor? refuse, no generation
   │
   ├─ 3. prompt            llm_n_prompt/           chunks + question → messages
   └─ 4. generate          llm_n_prompt/           messages → answer
```

Chunking is **not** in this path. The slides were chunked and embedded by the
ingestion pipeline long before any question is asked;
`Share_components/chunking_strategies/` is what did it.

## The experiment

Two factors are varied and everything else is held fixed:

| Factor | Levels | Where it lives |
|---|---|---|
| chunking strategy | exp1, exp2, exp3 | `Share_components/chunking_strategies/` |
| generator | 360m, 0.5b, 1b | `Inference_pipeline/llm_n_prompt/models.py` |

Held fixed: BGE as the only embedder, combo2 as the only retrieval mode in the
benchmark grid, and one prompt. Each of those has a comment saying why, in the
file that pins it.

## Why the structure looks like this

Three ideas recur, and recognising them explains most of the layout.

**One definition, imported everywhere.** The fixed refusal sentence, the
pipeline flags, `build_pipeline`, the tokenizer. Anything defined twice can
drift, and the failure mode is a results file that quietly mixes two versions
of the system. `configuration/generation.py` holds the refusal sentence that
both the prompt and the evaluation use, for exactly this reason.

**Silent failures get an explicit check.** Several things here fail without
raising: a collection embedded by a different model still returns five ranked
chunks, just wrong ones. So `VectorSearch.self_check()` runs before every
session, `chunking_tokenizer/limits.py` refuses to import if `MAX_TOKENS` is
too high, and the evaluation warns when every retrieval metric is exactly zero.

**The reason is written down next to the code.** Much of this codebase is
commentary recording which failure a line prevents — the ordering of the
few-shot examples, why rank fusion discards scores, why the token budget stops
rather than skips. Those comments are the most valuable thing here and were
carried across unchanged.

## Running things

```bash
# one question
python Inference_pipeline/run_inference_one_question.py "What is BM25?"
python Inference_pipeline/run_inference_one_question.py --interactive

# a stage on its own, to see what it hands the next one
python Inference_pipeline/5-vector_search.py --question "What is BM25?"
python Inference_pipeline/6-ranking_n_retrieval.py --question "What is BM25?"
python Inference_pipeline/7-llm_n_prompt.py --question "What is BM25?"

# the whole benchmark
python Inference_pipeline/run_inference_bigfile.py --benchmark bench.json --grid

# scoring (from inside Evaluation/)
python -m evaluation_pipeline --grid ../Data/Results_generation --benchmark bench.json
```

---

# What changed in this rewrite

Behaviour is unchanged. Every module was split into smaller files, and the
public names were re-exported so existing imports keep working.

## How the split works

Each oversized module became a **package of the same name** whose `__init__.py`
re-exports its public API. So this still works, unchanged:

```python
from Share_components.chunking_tokenizer import count_tokens
from vector_search import VectorSearch
from ranking_n_retrieval import Retriever, COMBOS
from llm_n_prompt import LLM, PromptTemplate, MODELS
from RAG_pipeline import build_pipeline, add_pipeline_flags
```

The code moved; the import path did not.

## One thing that was broken and is now fixed

The numbered filenames could not be imported. `8-RAG_pipeline-orchestration.py`
was imported elsewhere as `from RAG_pipeline import build_pipeline`, and
`5-vector_search.py` as `from vector_search import VectorSearch` — but a Python
module name cannot begin with a digit or contain a hyphen. The stale
`__pycache__` entries (`rag_pipeline.cpython-311.pyc`,
`vector_search.cpython-311.pyc`) show these once had unnumbered names.

Now the packages carry the importable names and the numbered scripts are
three-line launchers that call into them, so both conventions hold at once.
`8-RAG_pipeline-orchestration.py` has no launcher because it had no command
line — it is now the `RAG_pipeline/` package.

## Duplication removed

The stage-record code (`stage_settings`, `stage_hash`, question slug, JSON
writing) was written out four times, once per stage, with near-identical
bodies. It is now `Inference_pipeline/stage_record.py`. The per-stage
differences that were real — stages 3 and 4 hash the prompt text, stages 1 and
2 deliberately do not, because they never load a language model — are preserved
and documented in that file.

## Where each old file went

### Share_components

| Was | Is now |
|---|---|
| `configuration.py` | `configuration/` → `paths`, `chunking`, `retrieval`, `generation` |
| `embeddings.py` | `embeddings/` → `bge`, `registry` |
| `chroma_store.py` | `chroma_store/` → `metadata`, `store` |
| `chunking_tokenizer.py` | `chunking_tokenizer/` → `limits`, `loader`, `counting`, `splitting`, `audit` |
| `chunking_strategies.py` | `chunking_strategies/` → `common`, `page_level`, `fixed_window`, `headings`, `section_aware`, `runner` |

exp3's heading detection is now its own file (`headings.py`), separate from the
chunker that uses it, because it is the part most likely to need tuning on a
different slide deck.

### Inference_pipeline

| Was | Is now |
|---|---|
| `5-vector_search.py` | `vector_search/` → `collections`, `search`, `stage` + launcher |
| `6-ranking_n_retrieval.py` | `ranking_n_retrieval/` → `keyword`, `fusion`, `reranker`, `budget`, `retriever`, `stage` + launcher |
| `7-llm_n_prompt.py` | `llm_n_prompt/` → `models`, `prompt_text`, `examples`, `prompt`, `cleanup`, `generator`, `stage` + launcher |
| `8-RAG_pipeline-orchestration.py` | `RAG_pipeline/` → `router`, `plan`, `pipeline`, `builder`, `cli_flags` |
| `run_inference_one_question.py` | `single_question/` → `display`, `record`; the file is now the CLI |
| `run_inference_bigfile.py` | `benchmark/` → `questions`, `records`, `fingerprint`, `progress`, `runner`, `cli`; the file is now the launcher |
| *(new)* | `stage_record.py` — the shared JSON envelope |

The `Retriever` class was doing four separate jobs — BM25, rank fusion,
cross-encoder reranking, and token budgeting — so each became its own module
and `retriever.py` is now mostly wiring. The old private methods
(`_merge`, `_rerank`, `_keyword_search`, `_fill_budget`, `_tokens_in`,
`_identity`) are kept as thin delegates so nothing that called them breaks.

### Evaluation

| Was | Is now |
|---|---|
| `evaluate_retrieval.py` | `evaluate_retrieval/` → `config`, `pages`, `gold`, `ranking_metrics`, `text_metrics`, `scoring`, `backends`, `benchmark_io`, `reporting`, `cli` |
| `evaluate_generation.py` | `evaluate_generation/` → `config`, `text`, `reference_metrics`, `grounding_metrics`, `targeted_metrics`, `scoring`, `cli` |
| `evaluation_pipeline.py` | `evaluation_pipeline/` → `config`, `labels`, `cells`, `statistics`, `grid`, `winners`, `reporting`, `cli` |

`evaluate_generation` is split along the three metric families its own
docstring already named: reference-based, reference-free, targeted.

**One command change.** These are packages now, so a directory and a `.py` file
cannot share a name. Run them with `-m`:

```bash
python evaluate_retrieval.py  --answers a.json --benchmark b.json   # before
python -m evaluate_retrieval  --answers a.json --benchmark b.json   # now
```

Same flags, same outputs. Everything else is unchanged.

`NO_ANSWER` in `evaluate_generation` was a hardcoded copy of the sentence in
`configuration`. It now imports it, so editing the sentence the model is told
to emit cannot leave the evaluation searching for the old wording.

## How this was checked

The rewrite was tested differentially: the same inputs through the old code and
the new, comparing output exactly.

- **Chunkers** — a synthetic lecture through all three strategies: byte-identical JSON.
- **Prompt** — `build_prompt`, the system prompt, `format_context`, `qa_few_shot`, `FORMAT_VERSION`: identical.
- **Router** — seven boundary cases including the ones that must *not* match ("how does BM25 work", "hello world example of tokenisation"): identical verdicts.
- **Cleanup** — `strip_scaffolding` and the fixed-reply classifiers: identical.
- **Rank fusion** — merge ordering and chunk identity: identical.
- **Evaluation** — a four-question benchmark covering all three gold formats, a question with no gold, an exp2 chunk with `page_number: "N/A"` and inline `[Slide N]` markers, and an abstention. Gold resolution, the audit, per-query rows, cell summaries, grid statistics and winners: all identical, including bootstrap CIs and permutation p-values (the fixed seed reproduces exactly).

Everything also byte-compiles, and all five command-line entry points build
their parsers and respond to `--help`.
