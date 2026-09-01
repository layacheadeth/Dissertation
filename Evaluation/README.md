# RAG Evaluation

Three files. Each does one thing.

| File | Does |
|---|---|
| `evaluate_retrieval.py` | Did the retriever find the gold pages? |
| `evaluate_generation.py` | Was the answer any good? |
| `evaluation_pipeline.py` | Runs both over every file, writes one table. |

## Install

    pip install numpy pandas
    pip install bert-score sacrebleu     # optional: bert_f1 and bleu

## Run

    python evaluation_pipeline.py \
        --answers Results_generation_bigfile/*_budget500.json \
                  Results_generation_bigfile/*_budget1500.json \
        --benchmark latest_benchmark_qa.json \
        --out-dir Evaluation/results

Add `--no-bert` for a fast pass. Outputs `comparison.csv` (one row per run),
`summary.json`, and `per_query_<cell>.csv` for breakdowns by question type.

## Inputs

Two, and only two: the answer files the inference pipeline wrote, and the
benchmark. Retrieval is scored from `chunks_used` inside each answer file —
the chunks the model was actually given — not from a fresh search.

## Metrics

Retrieval:

| | |
|---|---|
| `recall` | How many gold pages were found. |
| `ndcg` | Were they near the top, weighted by importance. |
| `gold_density` | Share of the context window spent on gold pages. |
| `mrr` | How high the first gold page came. |
| `hit@1` | Was the top chunk a gold page. |
| `n_chunks`, `tokens`, `budget_fill` | Diagnostics: did the window fill? |

`recall` and `gold_density` pull against each other. High recall with low
density means the strategy found the right pages but padded the window to do
it.

Generation:

| | |
|---|---|
| `token_f1`, `rouge_l`, `bleu`, `bert_f1` | Answer vs the gold answer. |
| `groundedness` | Share of the answer's words present in the context. Catches fabrication, not wrongness. |
| `context_utilisation` | Share of chunks the answer drew on. Low + high groundedness = retriever oversupplying. |
| `keyword_coverage` | Did the answer name the expected concepts. |
| `n_abstained` | How many questions the model refused. |

## Reading the table

`budget_fill` well below 1.0 means that run could not fill its window, so its
recall is capped by how much text was retrieved rather than by ranking
quality. Compare only runs that filled.

Retrieval columns are identical across the three models within a strategy and
budget — they share a retriever. If they ever differ, something is wrong.

## Known limitation

Gold pages are labelled by page, so page-level chunking is scored in its own
unit and hits gold almost by construction. `gold_density` is what exposes
this. Worth stating if page-level wins.
