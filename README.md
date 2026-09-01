# How to run this project

Run everything from the project root, in this order.

## 0. Install dependencies

```
pip install -r requirements.txt
pip install bert-score sacrebleu        # optional, for bert_f1 and bleu in step 4
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

Answers a single question and prints it.

## 3. Batch generation

```
python Inference_pipeline/run_inference_bigfile.py \
    --benchmark Data/Benchmark/latest_benchmark_qa.json \
    --budget-tokens 500 --candidates 60

python Inference_pipeline/run_inference_bigfile.py \
    --benchmark Data/Benchmark/latest_benchmark_qa.json \
    --budget-tokens 1500 --candidates 60
```

Answers the whole benchmark and saves the results to
`Results_generation_bigfile/answers_..._budget<N>.json`. Both budgets are needed
because step 4 scores them together.

Pass `--budget-tokens` explicitly. The shipped default is `BUDGET_TOKENS = None`,
which silently falls back to top-*k* instead of a budget run. `--candidates 60`
matters too: the default of 20 is not enough to fill a 1500-token budget.

Add `--grid` to run every chunking strategy against every model; it saves
progress as it goes, so it is safe to stop and restart.

## 4. Evaluation

```
python evaluation_pipeline.py \
    --answers Data/Results_generation_bigfile/*_budget500.json \
              Data/Results_generation_bigfile/*_budget1500.json \
    --benchmark Data/Benchmark/latest_benchmark_qa.json \
    --out-dir Evaluation/results
```

Use `--no-bert` for a fast pass. Outputs are `comparison.csv`, `summary.json`,
and `per_query_<cell>.csv`.
