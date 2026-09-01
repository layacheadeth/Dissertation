# How to run this project

Run everything from the project root, in this order.

## 0. Install dependencies

```
pip install -r requirements.txt
```

Python 3.11. A Hugging Face token is needed for the models: `export HF_TOKEN=...`.

## 1. Ingestion

```
python Ingestion_pipeline/Ingestion-pipeline-automation-allweek.py
```

Builds the database from the PDFs in `Data/All_lectures/`. Run once.

## 2. Inference

```
python Inference_pipeline/run_inference_one_question.py "What is BM25?"
```

Answers a single question and prints it. It also saves the run to
`Data/Results_stage4/<strategy>_<model>_<slug>.json`; add `--no-save` to skip
that. `--retrieve-only` skips the language model and just shows the chunks,
`--scores` shows the reranker scores, `-i` starts an interactive session.

## 3. Batch generation

```
python Inference_pipeline/run_inference_bigfile.py \
    --benchmark Data/Benchmark/latest_benchmark_qa.json \
    --budget-tokens 500 --candidates 60

python Inference_pipeline/run_inference_bigfile.py \
    --benchmark Data/Benchmark/latest_benchmark_qa.json \
    --budget-tokens 1500 --candidates 60
```

Answers the whole benchmark and saves one file per cell to
`Data/Results_generation/answers_<strategy>_<embedder>_<combo>_<model>_budget<N>.json`.
Both budgets are needed because step 4 scores them together. Use `--out-dir` to
put them somewhere else.

Pass `--budget-tokens` explicitly. The shipped default in
`Share_components/configuration.py` is `BUDGET_TOKENS = None`, which falls back
to `TOP_N = 5` — a top-*k* run, not a budget run. `--candidates 60` matters too:
the default of 20 is not enough to fill a 1500-token budget, and the script only
warns about it.

Add `--grid` to run every chunking strategy against every model; it saves
progress as it goes, so it is safe to stop and restart. `--limit N` cuts the
number of questions, `--overwrite` ignores existing results.

## 4. Evaluation

```
python Evaluation/evaluation_pipeline.py \
    --answers Data/Results_generation/*_budget500.json \
              Data/Results_generation/*_budget1500.json \
    --benchmark Data/Benchmark/latest_benchmark_qa.json \
    --out-dir Evaluation/results
```

Use `--no-bert` for a fast pass. Outputs are `summary.json`, `by_strategy.json`
and `by_query.json`.

`evaluate_retrieval.py` and `evaluate_generation.py` also run on their own, with
the same `--answers` and `--benchmark` flags.

## The browser front end

```
python app.py --share
```

Serves a Gradio interface on `localhost:7860` and prints a public
`*.gradio.live` link beside it, so the app can be opened from another machine.
The link is a tunnel to your process and expires after 72 hours. `--port` moves
the local port; the pipeline flags from step 3 work here too.

It needs step 1 done first, since it reads the same database, and it builds its
pipeline through the same `build_pipeline()` as the scripts above — so the
answers it shows match the ones in the benchmark. It is optional; steps 1–4 do
not need it.

---

# How each pipeline works

## Ingestion

```
Data/All_lectures/*.pdf   (13 lecture PDFs)
   │
   ├─ 1. extract + clean      strip bullets, collapse whitespace, drop
   │                          number-only lines, remove image-only pages,
   │                          drop progressive-reveal duplicates
   │                             615 raw -> 590 with text -> 529 kept
   │
   ├─ 2. chunk (three ways, all from the same clean text)
   │       2-1  exp1  page-level        516 chunks,  79 tok mean
   │       2-2  exp2  fixed 500/100     104 chunks, 481 tok mean
   │       2-3  exp3  section-aware     296 chunks, 138 tok mean
   │
   ├─ 3. corpus statistics    run BEFORE the database is loaded, so a corpus
   │       (4-Corpus-...)     problem surfaces before embedding spends time on it
   │
   ├─ 4. embed + load         BGE bge-small-en-v1.5, 384-dim, reset=True
   │       (3-1-Ingest-...)   3 collections, 916 vectors, in
   │                          Data/Database/chroma_db
   │
   └─ 5. run manifest         versions, PDF hashes, chunker params
```

The numbered filenames do not match the run order: `4-Corpus-statistics.py`
runs third, before `3-1-Ingest-to-ChromaDB-bge-Embed.py`.

Written output:

| file | holds |
|---|---|
| `Data/All_extracted_text/Data_weekN/<Lecture>.json` | cleaned slides + page accounting |
| `.../<Lecture>_exp1_chunks.json` (and `_exp2_`, `_exp3_`) | flat list of chunks |
| `Data/Analysis/corpus_stats.json` / `.md` | counts and tables |
| `Data/Analysis/pages_by_week.csv`, `chunks_by_week.csv`, `chunk_length_stats.csv`, `tables.tex` | report tables |
| `Data/Analysis/run_manifest.json` / `.md` | versions, GPU, git state, PDF hashes, vector counts |
| `Data/Analysis/chunker_params.json` | configured chunker settings, written before the run |
| `Data/Database/chroma_db` | `exp1_page_level_bge`, `exp2_fixed_overlap_bge`, `exp3_section_aware_bge` |

A chunk record, exp3:

```json
{
  "experiment_id": "exp3_section_aware",
  "chunk_id": "Week1_sec1_part1",
  "filename": "Week1_Intro_to_vector.pdf",
  "week": "Week 1",
  "page_number": [1],
  "section_index": 1,
  "section_title": "COMP64702: Transforming Text Into Meaning",
  "part_index": 1,
  "part_count": 1,
  "slide_count": 1,
  "token_count": 44,
  "char_count": 207,
  "content": "COMP64702: Transforming Text Into Meaning\n..."
}
```

`page_number` is **always a list**, even for exp1's single pages. The
strategy-specific fields differ: exp1 adds `was_split`, `part_index`,
`word_count`; exp2 adds `chunk_size_tokens`, `overlap_tokens`,
`window_token_count`. And exp1/exp3 write `token_count` while exp2 writes
`actual_token_count`.

## Inference

```
question
   │
   ├─ router ──────────────> "what can you do?" answered here, no search
   │
   ├─ 1. embed              question -> vector
   ├─ 2. retrieve           vector + words -> chunks
   │       dense  ─┐
   │       sparse ─┴─ merge ─ rerank ─ fill the token budget
   │
   ├─ gate ────────────────> best score below the floor? refuse, no generation
   │
   ├─ 3. prompt             chunks + question -> messages
   └─ 4. generate           messages -> answer
```

Chunking is not in this path — it happened during ingestion. Every entry point
(`run_inference_one_question.py`, `run_inference_bigfile.py`, `app.py`) goes
through the same `build_pipeline()` in `rag_pipeline.py`, so the answers agree.

`run_inference_one_question.py` writes one record per question to
`Data/Results_stage4/`. `run_inference_bigfile.py` writes one file per cell to
`Data/Results_generation/`, shaped like this (abridged, from a real run):

```json
{
  "settings": {
    "hash": "dbcd58f02bb4",
    "strategy": "exp1", "embedder": "bge", "combo": "combo2",
    "llm": "Qwen/Qwen2.5-0.5B-Instruct",
    "device": "cpu", "candidates": 60, "top_n": 5, "budget_tokens": 500,
    "context_tokens_mean": 446.3,
    "route_meta": true, "relevance_floor": null,
    "collection": "exp1_page_level_bge",
    "routes": {"meta": 0, "refused": 0, "answer": 29},
    "n_questions": 29, "total_seconds": 1703.0, "seconds_per_question": 58.72
  },
  "results": [
    {
      "question_id": "EXAM_001",
      "question": "In a dependency analysis ...",
      "answer": "In a dependency analysis ...",
      "route": "answer",
      "top_score": 2.6079,
      "seconds": 33.814,
      "context_tokens": 431,
      "n_chunks": 6,
      "chunks_used": [
        {"chunk_id": "Week12_p23", "text": "...", "metadata": {}, "score": 2.6079}
      ]
    }
  ]
}
```

Nothing here is scored. `chunks_used` is what the evaluation reads, which is why
the scores describe the run that actually happened rather than a fresh search.

## Evaluation

```
answer files (chunks_used + answer)     benchmark (gold pages + gold answer)
        │                                          │
        └───────────────┬──────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
  evaluate_retrieval             evaluate_generation
  flatten chunks -> ranked       answer vs gold answer,
  pages, split tokens evenly     plus groundedness against
  across the pages covered       the context it was given
        │                               │
        └───────────────┬───────────────┘
                        │
               evaluation_pipeline
                        │
     summary.json · by_strategy.json · by_query.json
```

Two inputs only, and nothing is re-retrieved. Scoring is per **page**, not per
chunk, so a 7-page exp2 window cannot collect credit for all seven pages while
occupying a single rank slot.

| file | holds |
|---|---|
| `Evaluation/results/summary.json` | run conditions plus the overall numbers |
| `Evaluation/results/by_strategy.json` | one block per cell — strategy x model x budget |
| `Evaluation/results/by_query.json` | per-question detail |

Each file repeats the run conditions (`benchmark`, `k`, whether BERTScore ran,
the score threshold, question and cell counts), so a report is readable on its
own. A short table is also printed to the console.

Two reading rules worth keeping. `budget_fill` well below 1.0 means that run
could not fill its window, so its recall is capped by how much text was
retrieved rather than by ranking quality — compare only runs that filled. And
the retrieval columns must be **identical across the three models within a
strategy and budget**, since they share a retriever; if they differ, something
is wrong.
