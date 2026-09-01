# How this system works

Turns lecture PDFs into a searchable database, then answers student questions
from it. The slides are chunked three different ways so the three ways can be
compared.

There are three parts. **Ingestion** builds the database and is run once.
**Inference** asks it questions and is run as often as you like. **Benchmark
creation** builds the question set that inference is measured on, and is also
run once.

```
PDFs  ->  clean text  ->  chunks  ->  database  ->  answers
       \________ ingestion ________/   \__ inference __/

mock exam  ->  claims  ->  judged slides  ->  gold pages
            \_________ benchmark creation _________/
```

## Folders

```
Data/All_lectures/          the 13 lecture PDFs you put in
Data/All_extracted_text/    clean text and chunks come out here
Data/Database/chroma_db/    the searchable database
Data/Analysis/              tables and numbers for the report
Data/Benchmark_intial_data/ screenshots of the mock exam, transcribed by hand
Data/Benchmark/             the question set and its gold pages
Data/Results/               benchmark answers
Data/Results_generation_norag/   answers from the no-RAG baseline

Share_components/           shared code, nothing here is run directly
Ingestion_pipeline/         building the database
Inference_pipeline/         asking it questions
Benchmark_creation/         building the question set
Evaluation/                 scoring the answers
Evaluation/results/         comparison.csv, summary.json, per-question CSVs
```

The dependencies run one way only. `Share_components` knows nothing about
questions; `Inference_pipeline` imports it; `Evaluation` reads the files
inference wrote and imports `Share_components` for one thing only — the fixed
refusal sentence, so the two cannot disagree about its wording.

Everything is run from the project root. This is not a stylistic preference:
`4-Corpus-statistics.py` and `5-Run-manifest.py` resolve `Data/Analysis` relative
to the working directory, which is why the automation script explicitly launches
them with `cwd` set to the root. Run them from inside `Ingestion_pipeline/` and
they will write their tables into the wrong place, or fail to find the corpus.

## Changing settings

All settings live in `Share_components/configuration.py` — chunk size, overlap,
folder paths, tokenizer, which strategy and model to search with, the token
budget, the default LLM, `MAX_NEW_TOKENS`, and the exact refusal string
`NO_ANSWER`. **Nowhere else in the code holds a number.** It is conventionally
imported as `from Share_components import configuration as config`, so a
reference to `config.X` anywhere in the codebase means this file.

Ingestion scripts take no arguments at all; they read config and go. Inference
scripts read the same config, but also accept flags so you can try a different
strategy or model for one run without editing anything.

---

# Part 1: Ingestion

## Running it

```
python Ingestion_pipeline/Ingestion-pipeline-automation-allweek.py
```

That does the whole thing. It also writes `chunker_params.json` before anything
runs, which is what the manifest later compares against. Note that statistics
are generated **before** ingestion, not after: the tables describe the chunk
files, and computing them first means a corpus problem surfaces before the
embedding model spends time on it.

To do one step at a time instead, run them in this order:

```
python Ingestion_pipeline/1-Extract_clean_format_text.py
python Ingestion_pipeline/2-1-PageLevel-chunking.py
python Ingestion_pipeline/2-2-FixedSizeOverlapping-chunking.py
python Ingestion_pipeline/2-3-SectionAware-chunking.py
python Ingestion_pipeline/4-Corpus-statistics.py
python Ingestion_pipeline/3-1-Ingest-to-ChromaDB-bge-Embed.py
python Ingestion_pipeline/5-Run-manifest.py
```

After changing a setting, re-run from step 2 onwards. Step 3-1 rebuilds its
collections from scratch every time, so nothing stale is left behind.

## What the corpus actually contains

13 lecture PDFs, COMP64702. Extraction throws away two kinds of page:

| | pages |
|---|---|
| raw slide pages | 615 |
| with extractable text | 590 (25 image-only or blank removed) |
| after deduplication | **529** (61 progressive reveals removed) |

That is a 14% reduction overall, but it is very unevenly spread — Week 1 loses
34 of 88 pages to build-up slides while Weeks 3, 5, 9 and 11 lose none, and
Week 12 loses 13 pages to image-only slides and none to dedup. Per-week figures
are in `Data/Analysis/pages_by_week.csv`, with a SHA-256 of each source PDF.

The whole corpus is small: about 174k characters, 41k tokens. Nothing here needs
to be clever about scale.

## What extraction actually does

`1-Extract_clean_format_text.py` reads every PDF in `Data/All_lectures/` and
writes one JSON per lecture. Cleaning removes bullet glyphs, collapses runs of
spaces and blank lines, and **drops any line that is only a number**, on the
assumption it is a slide number. Files whose names contain no week number are
skipped with a warning rather than filed under "Unknown".

Deduplication removes progressive reveals. An animated slide comes out of the
PDF as several pages that each add a line to the one before, so a page is
dropped when the *next* page contains everything it says, and the last page of a
lecture is always kept. That forward-looking substring test is what the w2
p27/p28 defect slips through — see "Known corpus defects" below.

The three counts are kept apart on purpose. `empty_pages_removed` and
`pages_removed_by_dedup` are different losses — picture-only slides are one
thing, animation duplicates another — and collapsing them into a single "pages
removed" figure would make it impossible to say where a lecture's pages went.

## What the three chunking strategies do

| | | chunks | pages/chunk | mean tokens | max tokens |
|---|---|---|---|---|---|
| exp1, page-level | one chunk per slide | 516 | 1.0 | 79.3 | 484 |
| exp3, section-aware | slides sharing a heading are merged into one section, and the heading is put at the top of every chunk | 296 | 3.0 | 138.5 | 492 |
| exp2, fixed-size overlapping | a 500-token window slides across the lecture with 100 tokens of overlap, ignoring slide boundaries | 104 | 7.1 | 481.4 | 501 |

Ordered by granularity; the spread is 5× between finest and coarsest. exp3 found
273 sections and emitted 296 chunks; exp1's oversized-page guard fired twice,
splitting 2 pages out of 516. **No chunk in any strategy exceeds the 510 usable
tokens** — see the note on tokens below for why that matters.

All three read the same clean text, so any difference in the search results
comes from the chunking and nothing else.

Two further strategies were explored and dropped: exp4 (semantic-aware) and an
earlier structure-level variant, exp5. exp5 prepended the section title to every
part of a split section, which replicated single slides across up to 20 chunks
and wrecked its page precision. `Inference_pipeline/PROMPT_HISTORY.md` keeps the
record of what they scored.

One embedding model is used throughout, BGE (bge-small-en-v1.5, 384 dimensions),
giving three collections in one database — `exp1_page_level_bge`,
`exp3_section_aware_bge`, `exp2_fixed_overlap_bge` — 916 vectors in total. An
earlier version also ingested MiniLM; it was dropped because changing the
embedding model moved the retrieval metrics far less than changing the chunking
strategy did.

## How the retrieved list is cut: fixed token budget

Chunks are not the same size in each strategy, so "the top 5 chunks" is not the
same amount of text in each. The numbers above make the size of the problem
concrete: an exp2 chunk carries 481 tokens and touches 7 slides; an exp1 chunk
carries 79 tokens and touches one. At a fixed *k*, exp2 hands the model six times
the material and gets seven times the chances to intersect the gold set — so it
would win for being coarse, not for retrieving well.

**Top-k is therefore not used anywhere in this system.** Every run cuts the
retrieved list the same way: candidates are taken in rank order until
`BUDGET_TOKENS` is reached, and the rest are discarded. Every strategy hands the
model the same amount of text, so only the *composition* of the context differs,
and that is the only thing being compared.

The budget is set in `configuration.py` and can be overridden per run:

```
--budget-tokens 1500 --candidates 60
```

`--candidates` is how many chunks are pulled out of the database before the
budget is applied; it needs to be comfortably larger than the number that will
fit, so the budget rather than the candidate pool is what does the cutting. A
1500-token budget is about three exp2 chunks but nearly twenty exp1 chunks —
that asymmetry is the point.

> **The shipped defaults do not match this policy.** `configuration.py` still has
> `BUDGET_TOKENS = None` and `TOP_N = 5`, so a run launched without
> `--budget-tokens` silently falls back to top-*k* — the exact comparison this
> project rejects. `CANDIDATES` defaults to 20, which is below the 60 a budget
> run needs: exp1 takes roughly 20 chunks to fill 1500 tokens, so a 20-candidate
> pool would leave the reranker nothing to discard and would cap exp1 under its
> budget while exp2 filled easily. `BUDGET_CANDIDATES = 60` exists in the config
> but is documented as "suggested" and is not wired to anything.
>
> Every result in `Evaluation/results/` was produced with the flags passed
> explicitly, so the published numbers are budget runs. But the defaults are a
> trap for the next person, and for you in six months. Set `BUDGET_TOKENS` to a
> real number, make `CANDIDATES` default to 60, and either delete `TOP_N` or
> make the pipeline refuse to run without a budget.

## Which ingestion file does what

**You run these** (`Ingestion_pipeline/`)

- `1-Extract_clean_format_text.py` — PDFs to clean text
- `2-1`, `2-2`, `2-3` — the three chunking strategies. These are 20-line
  launchers: each calls `chunk_all("expN")` and the real work is in
  `Share_components/chunking_strategies.py`, so a change to chunking behaviour
  is never made here.
- `3-1` — load the chunks into the database
- `4-Corpus-statistics.py` — counts and tables for the report
- `5-Run-manifest.py` — records what this run used, so it can be repeated
- `Ingestion-pipeline-automation-allweek.py` — runs all of the above in order
- `ingestion_strategy.py` — not run directly; holds `ingest_all`, the collection
  names, and the strategy-to-script mapping used in error messages
- `export_chroma_to_json.py` — a standalone inspection tool, not part of the
  pipeline. Dumps every record from every collection to JSON, deterministically
  ordered and stamped with the chromadb version and a UTC timestamp. Reads
  either an on-disk path or a running Chroma server; if a local read fails with
  `KeyError: '_type'` the database was written by a newer chromadb than the one
  installed.

## Three things the ingestion step gets right

**The text is embedded here, not by ChromaDB.** Chroma will happily embed
documents with its own default model if you let it, and then questions embedded
with BGE would be searched against vectors built by something else — returning
believable nonsense with no error. Embedding explicitly with the same model that
inference uses is what makes the self-check at query time meaningful.

**Collections are deleted and rebuilt every run** (`reset=True`), so chunks from
an older run cannot linger after a chunker changes. Related: a chunk file whose
lecture JSON no longer exists is skipped with a warning, so deleting a PDF does
not leave orphaned chunks in the database.

**Chunk ids are made unique across all lectures**, not just within one file, by
appending `_dup` until no collision remains. Two lectures both producing
`Week1_p1` would otherwise silently overwrite each other in Chroma.

One operational warning from the source, worth repeating: **do not run two
ingestion scripts at once.** Chroma is SQLite underneath and the two runs will
fight over the file.

## What ingestion writes

Per lecture, in `Data/All_extracted_text/Data_weekN/`:

- `<Lecture>.json` — the cleaned slides. Top level carries the page accounting
  (`raw_total_pages`, `empty_pages_removed`, `pages_removed_by_dedup`, …); each
  page is `{page_number: int, content: str}`.
- `<Lecture>_exp1_chunks.json`, `_exp2_`, `_exp3_` — a flat list of chunks.

Every chunk carries `experiment_id`, `chunk_id`, `week`, `page_number`,
`content` and its token/char counts, plus fields specific to its strategy:
`was_split` and `part_index` for exp1, `chunk_size_tokens` and `overlap_tokens`
for exp2, `section_title`, `section_index` and `slide_count` for exp3.
`chunk_id` is readable and stable — `Week1_p1`, `Week1_sw_1`,
`Week1_sec1_part1` — so a chunk in a result set can be traced back by eye.

**`page_number` is always a list, even for a single page.** Code that assumes an
integer works on exp1 and breaks on exp2.

`4-Corpus-statistics.py` writes `corpus_stats.json`/`.md`, `pages_by_week.csv`,
`chunks_by_week.csv`, `chunk_length_stats.csv` and `tables.tex` into
`Data/Analysis/`. Nothing is recomputed from the PDFs, so its figures describe
exactly the corpus that was ingested.

Two design decisions in it are worth knowing:

- **It counts tokens with the same tokenizer the chunkers budgeted with, and
  there is no character-count fallback.** If the tokenizer cannot be loaded the
  script stops. Counting in a different tokenizer would make the truncation
  audit unfalsifiable — the report could show zero overflows while the encoder
  was quietly truncating, or the reverse. `--embed-tokenizers` cross-checks
  against BGE's own tokenizer; the two should agree to within the two special
  tokens, and a wider gap means stage 2 ran under a different vocabulary than
  this report.
- **Limits are expressed in content tokens.** `count_tokens` excludes
  `[CLS]`/`[SEP]`, so comparing raw counts against 512 would under-report
  overflows by exactly two tokens per chunk.

It also records a SHA-256 of every JSON it reads, counts duplicate chunk ids per
strategy, and reports how many source slides each strategy's output actually
covers.

`5-Run-manifest.py` describes the *run* rather than the data: Python and package
versions, GPU presence, git state, a SHA-256 for every source PDF, the chunker
parameters, and the vector counts per collection.

Its best idea is recording chunker parameters **twice**. The configured values
come from `chunker_params.json`, which the automation script writes before it
starts; the observed values are read back out of the emitted chunk records, so
they cannot drift from what was really run. exp2 stores its window and overlap
on every record, exp1 records whether the oversized-page guard fired, exp3
reports how many sections it detected.

The console summary flags anything that weakens reproducibility rather than
burying it in the JSON: uncommitted files, a missing parameter block, an absent
git repository, and — the useful one — **exp2 chunk files that disagree with
each other on window size**, which means some weeks were chunked under different
settings and stage 2 needs re-running for all of them.

`transformers` and `tokenizers` are in the tracked-package list for a reason
that is easy to miss: they supply the WordPiece vocabulary that *budgets* the
chunkers, not just the one that embeds. A vocabulary change would move every
chunk boundary in the corpus.

## The week field, which has three formats

This one bites. The same fact is written three different ways:

| where | format |
|---|---|
| chunk files and Chroma metadata | `"Week 1"` — string, with a space |
| directory names | `Data_week1` |
| benchmark gold pages | `1` — integer |

Anything joining retrieved chunks to gold pages must normalise. The safest
source is the directory name, which is why the evaluation harness parses that
rather than trusting either metadata field.

---

# Part 2: Inference

## Running it

```
python Inference_pipeline/run_inference_one_question.py "What is BM25?"
python Inference_pipeline/run_inference_bigfile.py --benchmark Data/Benchmark/latest_benchmark_qa.json
python Inference_pipeline/run_inference_bigfile.py --benchmark Data/Benchmark/latest_benchmark_qa.json --grid
python app.py
```

The first answers one question at a time and prints it. The next two answer a
whole file of questions and save the results. The last is the browser front end.

`--grid` runs every chunking strategy against every model, which takes a long
time but saves progress as it goes, so it is safe to stop and restart.

All of them go through the same `build_pipeline()`, so the answers you check by
hand, the answers in the browser, and the answers in the benchmark come from the
same system.

## The path a question takes

```
question
   │
   ├─ router ──────────────► "what can you do?" answered here, no search
   │
   ├─ 1. embed              question → vector
   ├─ 2. retrieve           vector + words → chunks
   │       dense ─┐
   │       sparse ─┴─ merge ─ rerank ─ fill the token budget
   │
   ├─ gate ────────────────► best score below the floor? refuse, no generation
   │
   ├─ 3. prompt             chunks + question → messages
   └─ 4. generate           messages → answer
```

**Chunking is not in this path.** The slides were chunked and embedded by the
ingestion pipeline long before any question is asked.

Two of these steps are short circuits rather than stages, and both live in
`routing.py`, which loads no model and touches no store — so it imports cheaply
and tests without fixtures.

**The meta router** answers questions about the system rather than the course,
without searching. These have no answer in the slides, so retrieval would return
its nearest neighbours regardless and leave the model guessing between refusing
and listing whatever came back. The patterns are anchored and narrow on purpose:
a false positive refuses a real question, which is worse than letting an odd
greeting through. `^hello$` matches; "hello world example of tokenisation" must
not, and neither must "what does week 3 cover", "how does BM25 work", or "what
do you know about cosine similarity".

**The relevance gate** refuses after searching, when nothing cleared a floor.
Because the reranker has read the question against each chunk, a low best score
is evidence about the corpus rather than a guess made before looking at it. Three
things about it:

- **Off by default.** It changes the answers, so it is a condition to switch on
  and report, not a setting.
- **combo2 only.** It reads reranker scores, which no other mode produces. A
  floor set under `dense` would look applied and do nothing, so
  `check_floor_supported` fails when the pipeline is built rather than silently
  at query time.
- **The number is a raw cross-encoder logit** — unbounded and corpus-dependent,
  so no universal value exists. Calibrate before trusting it: run the benchmark
  with `--save-scores` and look at where the top scores of covered and uncovered
  questions separate.

Both produce a `Plan`, which carries one of three routes — `meta`, `refused`,
`answer` — plus the chunks and scores. Keeping the plan separate from generation
lets a caller print the chunks before the answer without searching twice, and
lets the benchmark record *why* a question was refused. Route counts appear in
every answer file's settings block.

## Two fixed sentences, not one

Two different declines, deliberately kept distinct so the evaluation can tell
them apart by equality rather than by reading the text:

| | when | |
|---|---|---|
| `NO_ANSWER` | no extract bears on the question | *The provided course material does not cover this.* |
| `OUT_OF_SCOPE` | the question is about how the course is run — exams, marking, deadlines, rooms | *That is a question for the course staff rather than for the lecture material.* |

Declining an exam-contents question is correct behaviour; refusing a question the
slides do answer is a failure. Counting both as "refused" would hide which is
happening, so `is_no_answer`, `is_out_of_scope` and `is_declined` are three
separate checks.

**Only `NO_ANSWER` actually lives in `configuration.py`.** `OUT_OF_SCOPE`,
`META_ANSWER` and `RELEVANCE_FLOOR` are read with `getattr` fallbacks, so their
real definitions sit in the modules that use them — the out-of-scope sentence in
`llm_n_prompt.py`, the meta answer in `routing.py`. The fallbacks keep the code
running, but they defeat the "one definition, imported everywhere" rule for
exactly the constants that most need it: an evaluator importing `OUT_OF_SCOPE`
from `configuration` gets nothing, and the two halves of the system can end up
matching different sentences. Move all four into the configuration and drop the
`getattr` defaults.

Note that refusing is framed as the *last resort*, not the safe default: if any
extract bears on the question the model is told to answer from it, even partly,
and name the gap in one clause.

## The models

| tag | model | |
|---|---|---|
| `360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` | |
| `0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | filenames use `05b` |
| `1b` | `meta-llama/Llama-3.2-1B-Instruct` | gated — accept the licence and set `HF_TOKEN` |

One table decides what a tag means, so the interactive tool and the benchmark
cannot disagree. `check_model_available` pings the Hub before anything slow
loads; without it a mistyped name is only discovered after the embedder, the
corpus and the reranker have all loaded.

Decoding is greedy by default and should stay that way in anything being
compared. The benchmark was previously run with sampling, which means two runs
of the same grid produce different answers and any gap between two cells is
partly sampling noise — at n=29 that noise is comparable to the effects being
measured. Sampling stays available for the interactive demo, where variety is
the point.

Budget on time: about 49 seconds a question on CPU for the 1b model, so roughly
24 minutes per cell and the better part of a day for the full 18-cell grid.

## The four ways of finding chunks

| | |
|---|---|
| dense | vector search only — matches meaning |
| sparse | keyword search only — matches words |
| combo1 | both, merged by rank |
| combo2 | combo1, then a reranker reads each chunk against the question |

Both searches exist because they fail differently. Vector search finds the
cosine similarity slide from "how do we compare two documents" without the word
"cosine" appearing. Keyword search is the only thing that reliably finds the
slide that literally says "BM25", because vector search smears similar
technical terms together. combo2 is the default.

The four modes exist so each part can be shown to earn its place: `dense` asks
whether keyword search is worth having, `sparse` asks whether the embedding
model is, `combo1` is fast enough for development, `combo2` is best.

**Merging is by rank, not by score.** Vector scores run roughly 0 to 1; BM25
scores are unbounded and corpus-dependent, so adding them would be meaningless.
The scores are discarded and only positions used: a chunk at rank *r* scores
1/(*r* + 60), and the two lists' contributions are summed. A chunk that does
respectably in both beats one that tops a single list. The 60 softens the curve
so first place does not overwhelm second.

**The reranker runs last** because it reads the question and one chunk together
— much more accurate than comparing ready-made vectors, and far too slow to run
over the whole corpus. It only ever sees the survivors of the merge, and it is
loaded lazily: about 90 MB and a few seconds, which three of the four modes
never pay.

## How the budget is filled

Chunks are taken in rank order, and filling **stops at the first chunk that does
not fit** rather than skipping it and carrying on down the list.

Skipping would pack the budget fuller, but it would also let a low-ranked short
chunk overtake a high-ranked long one — so the context would no longer be "the
best material that fits", and the ranking under test would be partly undone by
the packing. Stopping keeps rank order intact, which is the thing being
measured.

Two guards around that:

- **The best chunk is always kept**, even when it alone exceeds the budget.
  Returning nothing would make the question unanswerable for a reason unrelated
  to retrieval quality. (This is why exp2 shows `n_chunks` of exactly 1.0 at
  budget 500.)
- **`budget_underfilled` counts questions that ran out of candidates before
  running out of budget.** Those questions were handed less context than a
  strategy with bigger chunks would have received, so the comparison they belong
  to is no longer equal. It is reported, not silent — and it is the run-time
  counterpart of the `budget_fill` column in the results table.

When a budget is set, `top_n` is ignored. It is still recorded in the answer
file's settings block, which can be confusing: `"top_n": 5` alongside
`"budget_tokens": 500` means the budget governed and the 5 did nothing.

## The prompt

One prompt, in `llm_n_prompt.py`: four labelled sections (ROLE, TASK,
CONSTRAINTS, RESPONSIBILITY) and four worked examples, about 600 words. Written
short deliberately — on CPU the whole prompt is re-read before every answer, so
its length is a fixed cost paid per question, measured at roughly 12 seconds a
question at about 1,100 words.

Every bullet stops a failure that can be named, and bullets were not added for
symmetry: a small model has a fixed attention budget, so a fourth bullet in TASK
is paid for out of the first three.

| section | the failure it stops |
|---|---|
| ROLE | answering as an examiner grading a student ("That's correct. The student answered correctly…") |
| TASK | padding openers, restating the question, forcing every retrieved extract into the answer, all-or-nothing on partly covered questions |
| CONSTRAINTS | inventing formulas and constants — and the opposite failure, refusing a question whose formula *was* in the material because the arithmetic was not literally quoted |
| RESPONSIBILITY | answering "is this on the exam?" fluently and inventively |

### Facts versus reasoning

The boundary is not "is it written in CONTEXT" but "does it follow from
CONTEXT". Facts — definitions, formulas, constants, values — are premises and
must be in the extracts. Deriving a conclusion from them is the job: applying a
rule to the case in the question, or joining two extracts that only answer it
together. Most conceptual questions in an exam-style benchmark need two or more
extracts joined, so forbidding that would refuse questions the material does
answer.

That is the useful line between deduction and hallucination, and it is about
premises, not conclusions: deduction derives a new conclusion from premises that
are present; hallucination invents a premise that is not. *Never supply a
missing formula, constant or value* is the rule that enforces it.

One caveat when reading conclusions off this: some course rules are hedged ("a
phrase describing an entity *usually* attaches to the noun"), so what follows is
defeasible rather than strictly deductive. The conclusion should inherit the
hedge rather than drop it, which is what the "claim no more certainty than the
material gives" clause is for.

### Things that look arbitrary and are not

This is the most expensive knowledge in the repository. Each item is a
regression someone paid for in run time.

**The examples are replayed as chat turns, not pasted into one user message.**
They used to be four `EXAMPLE n / CONTEXT: / QUESTION: / ANSWER:` blocks inside a
single message, in the same shape as the live task. A 1B model read that as one
document with a repeating pattern and continued it: **7 of 29 benchmark answers
opened by regurgitating example 1 verbatim**, whatever had been asked. Those 7
scored token_f1 0.091 against 0.244 for the rest, and ran 172 words against 49.
Held as `(context_lines, question, answer)` and replayed as real turns, the
strings "EXAMPLE" and "ANSWER:" never appear in the prompt at all — a model
cannot echo scaffolding that was never sent. Token cost is within a few per cent.

**The partial-coverage example goes last, and both decline-last orderings have
now failed.** Out-of-scope last made a 360M model refuse ordinary course
questions with the staff-referral sentence, because the nearest thing to copy was
the wrong one. No-answer last made a 1B model refuse EXAM_006 and EXAM_010 *with
the gold page retrieved at rank 1*. Recency is the whole mechanism and turn
structure sharpens it, so the last exchange must demonstrate the behaviour wanted
most — which is answering the supported part and naming the unsupported one, the
behaviour a refusal displaces.

**The examples carry no preamble at all.** "Study these examples" read like an
exam script and the model started marking it. Replacing it with "four of your own
past replies, answer in the same voice" was worse: the model echoed the
instruction back as its answer, eighty-eight seconds of "The student should
answer the task in the same voice as the example." Any sentence placed there is a
sentence a small model may copy *instead of* the examples, so there is none.

**RESPONSIBILITY avoids the words marker, marking and marks**, and is phrased as
what to do rather than what not to do. A rule saying "do not describe a student's
ability" supplies the exact frame it forbids, and a small model drops the
negation before it drops the frame.

**Example 2 uses precision and recall, which the benchmark does not test.** An
example worked in tf-idf or Kappa would hand the model a solved version of
questions it is about to be scored on.

**Extracts are numbered `[1] (Week 3 — Section)`.** Running them together as one
wall of text is what made answers blend two unrelated slides into a single false
claim.

**`cite_sources` is off by default.** Asking a small model to add "(Week 3)"
markers puts them into the answer text that the evaluation compares against the
gold answer, costing accuracy for something the evaluation does not score. On for
a demo, off for the benchmark. It also lives in TASK rather than being appended
after "ANSWER:" — concatenated onto the end, the model saw an instruction sitting
exactly where its answer was supposed to begin, and sometimes continued the
instruction instead of answering.

### Belt and braces on the output

Three independent defences, because a single leaked answer is worth about 0.005
on a 29-question token_f1 mean and is indistinguishable from a real answer in the
summary file:

1. turn-structured examples, so the scaffolding is never in the prompt
2. `STOP_STRINGS` passed to `generate`, halting decoding if the model starts
   writing the next fake exchange
3. `strip_scaffolding`, which drops everything from the first marker onwards and
   returns whether it fired

That last one increments `n_truncated`. **Print it at the end of a run.** Anything
above zero means the prompt is drifting back toward the failure all this was
built to stop, and it is worth seeing in the log rather than silently absorbing.

Answer length is the model's problem, not the prompt's: the rules ask for one to
four sentences and a 360M model routinely generates until the token cap.
`MAX_NEW_TOKENS` bounds it; no wording can.

## Resuming safely

`settings_hash` is a 12-character code identifying the settings that produced a
result file, so a half-finished run can be resumed but a *different* system's
answers cannot be silently reused. Without it, editing the prompt and re-running
gives a results file quietly mixing two versions of the system — invisible in the
file, and only surfacing when the numbers make no sense.

It covers everything that changes an answer or the set of questions asked:
strategy, embedder, combo, model, candidates, `top_n`, budget, floor, the
benchmark filename *and its modification time*, `--limit`, and the full text of
both the system prompt and the examples. It deliberately **excludes the device**,
so a run started on a laptop resumes on a GPU instead of starting over.

The examples are hashed through `qa_few_shot`, which renders them from
`self.examples` rather than storing a copy, so the two cannot drift. That
rendering carries a `FORMAT_VERSION` marker (`chat-turns-v3-answer-last`) — bump
it whenever the delivery mechanism changes, or the hash will reproduce the
pre-turn-structure text byte for byte and a stale answer file will look current.

## What an answer file contains

`Results_generation_bigfile/answers_<strategy>_<embedder>_<combo>_<model>_budget<N>.json`.

The `settings` block records the hash, every pipeline setting, the resolved model
name, the device, `context_tokens_mean`, the route counts (`meta` / `refused` /
`answer`), and timing. Each result carries `question_id`, `question`, `answer`,
`route`, `top_score`, `seconds`, `context_tokens`, `n_chunks`, and `chunks_used`
— each chunk with its `chunk_id`, `text`, `metadata` and reranker `score`.

Those saved reranker scores are what `score_filter.py` in the evaluation reads,
and what a relevance floor would be calibrated against.

## Which inference file does what

- `RAG_pipeline.py` — the four steps, plus `build_pipeline`. **Start here.**
- `routing.py` — the meta router, the relevance gate, and `Plan`
- `vector_search.py` — reads the database, and runs `self_check`
- `ranking_n_retrieval.py` — the four ways of finding chunks, rank fusion, the
  reranker, and the token budget
- `llm_n_prompt.py` — the prompt, the model list, the model wrapper
- `pipeline_progress.py` — progress and resume bookkeeping for long runs
- `run_inference_one_question.py` — asks one question and prints it nicely
- `run_inference_bigfile.py` — runs a whole benchmark and saves the answers
- `app.py` (project root) — the browser front end, built from the same
  `build_pipeline` as everything else

Each stage also runs on its own, which is the fastest way to see what it hands
the next one:

```
python Inference_pipeline/vector_search.py --question "What is BM25?"
python Inference_pipeline/vector_search.py --list
python Inference_pipeline/ranking_n_retrieval.py --question "What is BM25?"
python Inference_pipeline/llm_n_prompt.py --question "What is BM25?"
```

`run_inference_bigfile.py` does not score anything. It saves the answers and the
chunks they came from; scoring is a separate step, in `Evaluation/`.

## When an answer is wrong

Run the question with `--retrieve-only`. It skips the language model, so it is
fast, and it shows you the chunks that were found.

- The right chunk is **not** in the list → retrieval is the problem
- The right chunk **is** in the list → the model mishandled it

Those are different problems, and mixing them up wastes days.

---

# The no-RAG baseline

```
python run_no_rag_baseline.py --benchmark Data/Benchmark/latest_benchmark_qa.json
python run_no_rag_baseline.py --benchmark Data/Benchmark/latest_benchmark_qa.json --grid
```

Every other run in this project compares one retrieval configuration against
another, and none of them shows that retrieval helps at all. Without this
baseline the strongest honest claim available is "chunking strategy does not
matter much" — a statement about the retriever that says nothing about whether
the retriever earns its place. This run measures the generator on its own, so
the comparison that justifies the whole artefact, RAG against no RAG, can
actually be made.

**Everything is held constant except the context**: same benchmark, same three
models, same greedy decoding, same output schema, same scorer.

The one thing that has to change is the prompt, and how it changes matters.
Rules like "every value must come from CONTEXT" are incoherent when there is no
context and would produce a model that refuses everything. So those clauses are
removed and nothing else is: same role, same lead-with-the-answer instruction,
same length limit, same refusal string, same instruction to admit ignorance
rather than invent. **This is not "the model with no rules".** A weaker prompt
here would make RAG look better for a reason that has nothing to do with
retrieval.

Results go to `Data/Results_generation_norag/answers_norag_<model>.json`, in the
same schema as the RAG runs so the scorer can read both. Three fields are set
deliberately rather than left out:

| field | value | why |
|---|---|---|
| `route` | `"no_rag"` | the scorer groups by route; marking these `"answer"` would pool them with retrieved answers |
| `context_tokens`, `n_chunks` | `0` | zero, not absent — a missing field reads as "not measured", zero states that this run had no context |
| `chunks_used` | `[]` | groundedness and context utilisation are *undefined* without context, and should be read that way, not as zero performance |

---

# Part 3: Benchmark creation

Retrieval can only be scored against a question set that says which slides
*should* have been found. This part builds that set from a past mock exam.

The output is `Data/Benchmark/latest_benchmark_qa.json`.

## What is in the finished benchmark

| | |
|---|---|
| questions | 29 — 20 conceptual, 7 calculation, 2 procedural |
| difficulty | 19 medium, 10 easy |
| gold pages | 42, mean 1.45 per question — 16 questions have 1, 13 have 2 |
| distinct gold slides | 27 |
| alternate pages | 41 |
| sole-source gold pages | 24 of 42 are the only slide in the corpus supporting one of their claims |
| claims | 67, of which 1 is marked unsupported |

Each item carries `student_query`, `qa_gold_standard`, `question_type`,
`difficulty`, `keywords`, `claims`, `unsupported_claims`, `gold_pages` and
`alternate_pages`. Each gold page carries its `rank`, `role`,
`claims_supported`, `marginal_claims`, a `necessary` flag, the quoted `evidence`
span, a `why` line, and the full slide `content`.

Mean 1.45 gold pages per question is the number to notice. The first pass asked
a model for up to five and got padding; deriving them from claim coverage
instead gives one or two, because that is how many slides a question actually
needs.

## Running it

```
python Benchmark_creation/build_benchmark_from_exam.py

python Benchmark_creation/clean_benchmark.py \
    --original Data/Benchmark/benchmark_qa.json \
    --corpus   Data/All_extracted_text \
    --output   Data/Benchmark/latest_benchmark_qa.json \
    --report   Data/Benchmark/BUILD_REPORT.txt
```

Stage 1 needs `NEBIUS_API_KEY` in `.env`; it calls
`Qwen/Qwen3-235B-A22B-Instruct-2507`. Stage 2 makes no API calls at all — every
judgement it needs is already written down in `evidence_tables.py`.

## The two stages

**Stage 1 — `build_benchmark_from_exam.py`.** Reads the 30 hand-transcribed
questions in `exam_source.py`, builds a table of contents from every slide across
the 13 lecture JSONs, and asks the LLM, per question, for candidate gold pages,
3–7 keywords, a Socratic hint, and a difficulty label. Writes `benchmark_qa.json`
and `benchmark_socratic.json`.

Two rules keep this stage honest. Any anchor the model returns that does not
exist in the real slide index is **dropped**, not emitted with empty content; and
a question left with no anchor is flagged `needs_review` rather than being given
an invented fallback page. The run prints a histogram of gold pages per question,
because a model asked for "up to 5" will pad if you let it.

**Stage 2 — `clean_benchmark.py`.** Discards the model's page choices entirely
and re-derives them from evidence. This is the file that produces the benchmark
actually used; only `difficulty` and `keywords` survive from stage 1.

## Where the questions come from

`Data/Benchmark_intial_data/` holds screenshots of the mock exam — 14 images
covering questions 1 to 23. `exam_source.py` is those transcribed by hand, then
rewritten so every answer is **self-contained**. The exam showed students a table
or matrix on screen and asked about it; an answer that quotes those numbers
without the data looks hallucinated to someone using the assistant, who never saw
them. So:

- questions with no numbers get method-only answers, with no invented values
- where a concrete example is essential, the example is moved *into* the question
- "which of these statements is correct" is reframed as an open conceptual question

Four duplicates were removed during transcription, leaving 30 items. The week
each question belongs to is **not** recorded here — it is derived by matching
against the real slides, so the source of truth stays the corpus.

## How gold pages are derived

The answer to each question is broken into **atomic claims** (`CLAIMS` in
`evidence_tables.py` — 67 claims across the 29 questions). Every candidate slide
is then judged against every claim, and the verdict is recorded with a quoted
span from that slide. No quotable span, no support.

Gold pages fall out of that table by **greedy set cover**:

| | |
|---|---|
| rank 1 (primary) | the slide entailing the most claims |
| rank n | the slide adding the most claims not yet covered |
| tie-break | lower week, then lower page |
| stop | all claims covered, or 5 gold pages reached |
| excluded | any slide whose marginal gain is zero |

The tie-break matters: lower week means the original teaching slide beats the
week 13 revision slide that merely restates it, which is where a student should
actually be sent. It does real work here — the week 13 deck restates Kappa,
dependency parsing, BIO tagging, TF-IDF and attention, so it competes for most
questions and wins only twice.

### Why entailment, and not keywords or cosine

Keyword overlap misses paraphrase — w12 p29 supports EXAM_001's attachment claim
without ever containing the string "nmod". Cosine similarity with the retrieval
embedder is worse than imprecise, it is circular: it lets the embedding model
define the gold pages it is then scored against, so whichever slides the model
likes become "correct" by construction.

### The complete judged pool

`EVIDENCE` records **every slide considered**, including the 52 rows (of 135)
that were judged and entail nothing. That is the point of the table, not clutter.

An earlier version stored only the slides that survived pruning, and the
`necessary` flag — "this slide is the sole corpus source for one of its claims" —
then came out true 44 times out of 44. That was an artefact of a missing
denominator, not a property of the corpus. Counted properly over everything
looked at, it is 24 of 42.

### alternate_pages

A slide that entails a claim but is not needed for the cover is emitted as an
**alternate page** — 41 of them. Alternates are not gold: a retriever should not
be required to find them. They are also not wrong: scoring code should exclude
them from the precision denominator rather than count them as errors.

### unsupported claims

One claim is marked `UNSUPPORTED`: EXAM_020's "select the candidate with the
largest product". No slide states it — it is an inferential step, not retrievable
content. It is excluded from the cover target rather than forcing a gold page
that cannot exist.

## Corrections applied

`clean_benchmark.py` applies factual fixes before evidence is attached. Each
carries its reason in the source, and each is printed in the build report.

| | |
|---|---|
| dropped | EXAM_012 (entity linking) — "entity linking", "candidate generation" and "candidate ranking" occur zero times in the 13 lectures. Unanswerable from the corpus. |
| rewritten | EXAM_009 (rows and columns inverted against its own primary slide), EXAM_013 and EXAM_014 (claims imported from outside the corpus), EXAM_017 (a property no slide states), EXAM_023 (a relation that does not exist in the corpus), EXAM_026 (0.5555 truncated instead of rounded), EXAM_029 (arithmetic: 7 × 0.125 = 0.875, not 0.874; also recast from a Shakespeare example with zero corpus hits into the corpus's own terminology) |
| retyped | EXAM_019, EXAM_020 → `procedural`: they ask how a quantity is computed without supplying numbers, so they are neither definition recall nor arithmetic |

The recurring theme is worth stating plainly: a question is only a fair test of
*retrieval* if the corpus contains its answer. Several exam questions did not,
and were fixed or dropped rather than left in to be failed.

## Validation

`clean_benchmark.py` exits non-zero if any of these fail, so a bad benchmark
cannot be written:

- gold ranks are contiguous from 1, with exactly one `primary`, at rank 1
- no duplicate pages, and no more than 5 gold pages
- every gold page exists in the corpus, and its stored `content` still matches
  the corpus text (catches silent drift after re-extraction)
- every gold page has non-zero marginal gain
- no alternate page duplicates a gold page
- every claim not marked unsupported is covered

## Provenance

The entailment verdicts come from a single LLM judge reading each claim against
each candidate slide. They are written down rather than computed on the fly so
they can be audited, disputed, or regenerated. **Before publishing results,
hand-label a sample of (claim, slide) pairs and report Cohen's Kappa against this
table.** A benchmark judged by one model, scoring systems built on models, needs
that number to be worth anything.

## Which benchmark file does what

- `exam_source.py` — the 30 hand-transcribed, self-contained exam questions
- `build_benchmark_from_exam.py` — stage 1, LLM proposes anchors, keywords,
  hints, difficulty
- `evidence_tables.py` — `CLAIMS`, `EVIDENCE`, `UNSUPPORTED`: the complete judged
  pool, with quoted spans. **The heart of the benchmark.**
- `clean_benchmark.py` — stage 2, corrections plus greedy set cover, then
  validation

---

# Part 4: Evaluation

Three files in `Evaluation/`, each doing one thing.

| | |
|---|---|
| `evaluate_retrieval.py` | did the retriever find the gold pages? |
| `evaluate_generation.py` | was the answer any good? |
| `evaluation_pipeline.py` | runs both over every answer file and writes one table |

```
pip install numpy pandas
pip install bert-score sacrebleu        # optional, for bert_f1 and bleu

python evaluation_pipeline.py \
    --answers Data/Results_generation_bigfile/*_budget500.json \
              Data/Results_generation_bigfile/*_budget1500.json \
    --benchmark Data/Benchmark/latest_benchmark_qa.json \
    --out-dir Evaluation/results
```

`--no-bert` for a fast pass; BERTScore downloads about 1.4 GB the first time and
is slow on CPU. Outputs are `comparison.csv` (one row per cell), `summary.json`,
and `per_query_<cell>.csv` for breakdowns by question type.

**Two inputs, and only two**: the answer files inference wrote, and the
benchmark. Retrieval is scored from `chunks_used` inside each answer file — the
chunks the model was actually handed — not from a fresh search. Nothing is
re-retrieved at scoring time, so the scores describe the run that happened.

## Scoring is per page, not per chunk

The benchmark's unit is a page; a chunk is not one page. Scoring over chunks
would give a big exp2 window one slot in the ranking while it collects credit
for all seven pages it happens to span. So chunks are flattened to a ranked list
of pages first, and the ordinary metrics computed over that:

- a chunk's tokens are **split evenly across the pages it covers**, so a window
  touching one gold page and eight others contributes a ninth of its length to
  `gold_density`, not all of it
- pages inside one chunk arrived together and have no order among themselves, so
  they **share that chunk's rank** rather than being numbered off arbitrarily
- a page already seen higher up is not counted twice

The nDCG ideal is built from the gold gains alone. Building it from achieved
plus gold gains counts every page found twice and inflates the denominator once
the list is longer than the gold set — which would score two runs of identical
ranking quality differently purely because the budget gave one of them more
chunks.

One more trap the code handles: **the token-count field has a different name per
strategy.** exp1 and exp3 write `token_count`, exp2 writes `actual_token_count`.
Reading only the first would measure exp2's entire context as zero tokens.

## The metrics

Retrieval:

| | |
|---|---|
| `recall` | how many gold pages were found |
| `ndcg` | were they near the top, weighted by importance |
| `gold_density` | share of the context window spent on gold pages |
| `mrr` | how high the first gold page came |
| `hit@1` | was the top chunk a gold page |
| `n_chunks`, `tokens`, `budget_fill` | diagnostics: did the window fill? |

`recall` and `gold_density` pull against each other, and that tension is the
most informative thing in the table. High recall with low density means the
strategy found the right pages but padded the window to do it.

Generation:

| | |
|---|---|
| `token_f1`, `rouge_l`, `bleu`, `bert_f1` | answer against the gold answer |
| `groundedness` | share of the answer's words present in the context — catches fabrication, not wrongness |
| `context_utilisation` | share of chunks the answer drew on |
| `keyword_coverage` | did the answer name the expected concepts |
| `n_abstained` | how many questions the model refused |

`groundedness` is a fabrication check, not a quality check: an answer copied out
of an irrelevant chunk still scores 1.0. Low `context_utilisation` with high
`groundedness` means the retriever is oversupplying. Text is normalised before
comparison — LaTeX and markdown are stripped, so two answers that agree on the
maths but differ in markup do not score as different. Multi-word keywords must
appear contiguously, so "nominal modifier" does not match "modifier of the
nominal".

## Reading the table

**`budget_fill` well below 1.0 means that run could not fill its window.** Its
recall is then capped by how much text was retrieved rather than by ranking
quality, so compare only runs that filled.

**Retrieval columns are identical across the three models within a strategy and
budget** — they share a retriever. If they ever differ, something is wrong. This
is the cheapest sanity check in the project; run your eye down those columns
before believing anything else in the table.

## The score filter

`score_filter.py` drops chunks the cross-encoder rated below zero. Under a token
budget the list is cut by length rather than by score, so a run can carry a lot
of negatively-scored material — 81% of exp1's chunks at budget 1500. The filter
answers "what would the metrics look like if only positively-scored chunks
counted?" without regenerating anything.

It is **off by default** and switched on with `--filter-score`, because it
changes the numbers: it is a condition to report, not a setting.

The caveat is important and asymmetric. The model already saw the full context,
so filtering changes what is measured, not what was given. Filtered
*generation* numbers score the answer against a context the model never had, and
should be reported as a diagnostic only. Filtered *retrieval* numbers are not
open to that objection — whether the gold pages were retrieved and ranked well
is a property of the retriever. Chunks with no score are kept, since only combo2
attaches scores and dropping unscored chunks would silently empty every other
run. Questions left with no chunks are kept too, so they score zero rather than
quietly shrinking the denominator.

## What the current results say

`Evaluation/results/comparison.csv` holds 18 cells: three strategies × three
models × two budgets (500 and 1500), all combo2, all BGE, 29 questions each.

Retrieval, per strategy and budget:

| | recall | nDCG | gold_density | hit@1 | chunks | budget_fill |
|---|---|---|---|---|---|---|
| exp1 @500 | 0.741 | 0.580 | 0.206 | 0.414 | 5.0 | 0.89 |
| exp2 @500 | 0.603 | 0.307 | 0.118 | 0.724 | 1.0 | 0.99 |
| exp3 @500 | 0.672 | 0.590 | 0.260 | 0.655 | 2.8 | 0.76 |
| exp1 @1500 | 0.897 | 0.636 | 0.080 | 0.414 | 16.9 | 0.97 |
| exp2 @1500 | 0.810 | 0.366 | 0.055 | 0.724 | 3.0 | 0.97 |
| exp3 @1500 | **0.914** | **0.651** | 0.080 | 0.655 | 8.6 | 0.92 |

Four things fall out of this.

**exp2 at budget 500 is a one-chunk system.** `n_chunks` is exactly 1.0 — a
single 495-token window fills the entire budget. Its recall and nDCG are
structurally capped there, so its budget-500 row is not really a ranking result
and should not be read as one. This is the clearest vindication of the token
budget: at a fixed *k* of 5 the same strategy would have been handed five
windows and roughly 2,400 tokens, and would have looked strong for no reason
connected to retrieval quality.

**exp3 could not fill its window at 500** (`budget_fill` 0.76). Its sections are
lumpy, so the next chunk in rank order often does not fit and packing stops
short. Its budget-500 recall is therefore capped too, though less severely.

**exp3 wins at 1500, where every strategy fills.** Best recall and best nDCG,
with a third of exp1's chunk count. That is the honest headline comparison.

**Tripling the budget raises recall and craters density.** Mean recall goes
0.672 → 0.874 while mean gold_density goes 0.195 → 0.072. The extra tokens are
mostly not gold. Generation barely moves in return — mean `token_f1` 0.2587 →
0.2605 — so the best cell in the whole grid is a *small*-budget one,
`exp1_1b_budget500` at 0.334. More retrieved text is not more answer quality.

By model, averaged across cells:

| | token_f1 | bert_f1 | groundedness | refusals |
|---|---|---|---|---|
| 360m | 0.214 | 0.845 | 0.670 | 8 |
| 05b | 0.258 | 0.850 | 0.591 | 3 |
| 1b | **0.307** | **0.869** | **0.767** | 1 |

Generator size dominates every generation metric, and the effect is larger than
anything chunking does. The 360m model also refuses eight times against the 1b
model's once, mostly at budget 500 — small models abstain when the context is
thin. Note that refusals are counted, not scored, so a run with many abstentions
has its text metrics computed over the questions it did answer.

## The limitation to state if page-level wins

Gold pages are labelled by page, so exp1 is scored in its own native unit and
hits gold almost by construction. The page-flattening above removes most of the
advantage, but not the fact that the label granularity matches one strategy
exactly. `gold_density` is what exposes it — and in the current results exp1
does *not* win at 1500, which is itself worth saying, because it means exp3's
result is not an artefact of the labelling.

---

# What the benchmark does not cover

Read alongside any result, not as an afterthought.

**Weeks 4–9 are untested.** Gold pages fall in weeks 1, 2, 3, 10, 11, 12 and 13
only. No question touches LLM pretraining, SFT/alignment, in-context learning,
multimodal or RAG — roughly half the syllabus, and the prose-heavier half, which
is exactly where chunking strategies would be expected to separate differently.
This is the single biggest gap.

**The 29 questions are not 29 independent trials.** w1 p77 is the primary page
for six tf-idf questions, w2 p27 for three bigram questions, and three of the
Kappa slides are primary for two each. Roughly 18 independent retrieval targets
are represented, so per-question scores should not be treated as independent
samples: a strategy that wins the tf-idf cluster wins six items at once.
EXAM_027 and EXAM_028 are worded identically apart from the target term —
consider dropping one.

**Supporting-page ordering below rank 1 is not audited.** Rank 1 is derived and
verified; rank 2 is whatever the set cover picked next. Weight nDCG accordingly.

---

# Known corpus defects

These are in the extracted text and affect results for reasons unrelated to
chunking.

**Formula extraction is damaged throughout.** PDF extraction mangles the
mathematics. w1 p77's IDF definition comes out as `idfw =log 10 N dfw` and w2
p27's bigram product as `P(x) = NY n=1 P(x n|xn−1)`. Retrieval finds these slides
fine — the surrounding prose is intact — but a generator cannot reliably
reconstruct the formula from them, so QA scores understate answer quality.
Separate retrieval metrics from generation metrics when reporting.

**w3 p3 lost a word.** The slide reads `synonymy (dog ↔)` — the target of the
synonymy relation vanished in extraction. EXAM_013 is worded so it does not
depend on it.

**Deduplication misses mid-string insertions.** w2 p27 and p28 are the same
"Bigram Language Models" slide; p28 inserts one intermediate term *in the middle
of the formula*. `remove_animation_duplicates` tests whether the earlier page is
a substring of the later one, which catches build-ups that append but not ones
that insert. Both pages survived into the extracted text and into all three
collections, so one slide can score a hit twice.

A token-set Jaccard threshold (> 0.9 on consecutive pages), in addition to the
existing subset test, would catch it. **Not applied**: it changes page numbering,
which invalidates every gold page, so it needs a coordinated re-label of the
benchmark in the same commit.

---

# Three ideas that explain the layout

Recognising these accounts for most of how the code is arranged.

**One definition, imported everywhere.** The refusal sentence, the pipeline
flags, `build_pipeline`, the tokenizer. Anything defined twice can drift, and
the failure mode is a results file that quietly mixes two versions of the
system. The refusal sentence in particular is used by both the prompt and the
evaluation, so it lives in one place and both import it.

**Silent failures get an explicit check.** Several things here fail without
raising: a collection embedded by the wrong model still returns five ranked
chunks, just wrong ones. So `VectorSearch.self_check()` runs before every
session, the tokenizer's limits module refuses to import if `MAX_TOKENS` is too
high, and the evaluation warns when every retrieval metric comes out exactly
zero.

**The reason is written down next to the code.** Much of this codebase is
commentary recording which failure a given line prevents — why rank fusion
discards scores, why the token budget stops rather than skips, why the nDCG
ideal is built from gold gains alone. Those comments are the most valuable thing
in the repository. Preserve them through any refactor.

# The package rewrite

The oversized modules were each split into a **package of the same name** whose
`__init__.py` re-exports the public API, so existing imports still work
unchanged:

```python
from Share_components.chunking_tokenizer import count_tokens
from vector_search import VectorSearch
from ranking_n_retrieval import Retriever, COMBOS
from llm_n_prompt import LLM, PromptTemplate, MODELS
from RAG_pipeline import build_pipeline, add_pipeline_flags
```

The code moved; the import path did not. Behaviour is unchanged, and was checked
differentially — the same inputs through old and new code, compared exactly:
byte-identical chunker output, identical prompts and router verdicts, and
identical evaluation output down to the bootstrap CIs and permutation p-values.

This also fixed something that was broken. The numbered filenames could not be
imported — a Python module name cannot begin with a digit or contain a hyphen —
yet `8-RAG_pipeline-orchestration.py` was imported elsewhere as
`from RAG_pipeline import build_pipeline`. The packages now carry the importable
names and the numbered scripts are thin launchers, so both conventions hold at
once.

**One command change.** A directory and a `.py` file cannot share a name, so the
evaluation entry points run with `-m`:

```bash
python evaluate_retrieval.py  --answers a.json --benchmark b.json   # before
python -m evaluate_retrieval  --answers a.json --benchmark b.json   # now
```

Same flags, same outputs.

Stage-record code (`stage_settings`, `stage_hash`, the question slug, JSON
writing) had been written out four times with near-identical bodies; it is now
`Inference_pipeline/stage_record.py`. The real per-stage differences are
preserved and documented there: stages 3 and 4 hash the prompt text, stages 1
and 2 deliberately do not, because they never load a language model.

Two consequences worth knowing. exp3's heading detection is now its own file,
separate from the chunker that uses it, because it is the part most likely to
need tuning on a different slide deck. And `Retriever` was doing four jobs —
BM25, rank fusion, cross-encoder reranking, token budgeting — so each is now its
own module with `retriever.py` as wiring; the old private methods survive as
thin delegates so nothing that called them breaks.

# Shared code

Nothing in `Share_components/` is run directly. Both halves import from it.

- `configuration.py` — **every setting**, the one file you edit
- `chunking_strategies.py` — the three strategies themselves
- `chunking_tokenizer.py` — counting and splitting text by tokens
- `chroma_store.py` — talking to the database
- `embeddings.py` — the embedding model
- `ingestion_strategy.py` — lives in `Ingestion_pipeline/`, not here, though it
  is imported like shared code

## The settings that matter most

| | default | |
|---|---|---|
| `MAX_TOKENS` | 500 | the chunk ceiling, just under BGE's usable 510 |
| `OVERLAP_TOKENS` | 100 | exp2 only, about a fifth of `MAX_TOKENS`, so a point made at a join survives whole in one of the two chunks |
| `MIN_SLIDE_CHARS` | 25 | slides shorter than this are title cards and dividers |
| `STRATEGY` | `exp2` | which collection to search |
| `COMBO` | `combo2` | |
| `CANDIDATES` | 20 | chunks each search method puts forward |
| `BUDGET_TOKENS` | `None` | see the warning below |
| `TEMPERATURE` | 0.4 | unused when decoding is greedy, which it is for every benchmark run |

`MIN_SLIDE_CHARS` applies to exp1 and exp3 but **not** exp2, which slides a
window over the raw token stream and cannot skip a slide. So exp2's chunks
include title cards and section dividers that the other two strategies drop.
That is a real difference in what each strategy sees, not a bug, but it belongs
in any discussion of why their page coverage differs.

The comment on `MAX_TOKENS` is worth reading in full before changing it. Bigger
chunks are not automatically better to retrieve: one vector has to stand for
everything in the chunk, so a 500-token chunk covering four topics sits at the
average of all four and can lose to a shorter chunk that is about one of them.
If retrieval looks vague, try 250–300 and re-run the chunkers and the ingestion.

## How the three chunkers actually work

**exp1** takes each slide, skips it if under `MIN_SLIDE_CHARS`, and splits it
only when it exceeds `MAX_TOKENS`. Unsplit slides keep a plain `Week1_p12` id;
split ones get `_part1`, `_part2`. That guard fired twice in the whole corpus.

**exp2** encodes the entire lecture into one token stream, remembering for each
token which slide it came from and *where in that slide's original text it sits*.
The window then cuts the original strings at those offsets. This matters:
decoding tokens back into text would lowercase everything and break the maths
notation, so the chunk would no longer be the slide's own words. It is also why
exp2 chunks carry both `window_token_count` and `actual_token_count` — the
window is a fixed 500 tokens, but the reassembled text re-tokenises slightly
differently once slide boundaries are stitched with newlines.

**exp3** decides what a heading is, which is the part most likely to need
tuning on a different deck. A first line counts as a heading if it is 12 words
or fewer, does not end like a sentence, and is either numbered (`2.3`, `IV.`),
one of a list of section words (agenda, overview, summary, recap…), mostly
capitals, or at least half title-case. Titles are normalised before comparison,
so "Attention (cont.)", "Attention 2/3" and "Attention part 2" all merge into
the same section as "Attention". A slide with no heading joins whichever section
is open.

Two details in exp3 that are easy to get wrong and are handled: the heading's
tokens are **subtracted from the budget before splitting**, so re-adding the
heading to each part cannot push a part over the limit; and a heading long
enough to fill the budget on its own is dropped rather than crowding out the
content it was meant to label.

## The tokenizer module

`split_text` cuts at the coarsest separator that works, falling back in order:
blank line → newline → full stop → space → individual characters. Sentences stay
whole whenever they can.

The `MAX_TOKENS` guard is a **module-level raise**, so it fires on import rather
than at the moment a chunk is embedded. Anything importing the tokenizer with a
bad setting stops immediately, instead of producing a corpus whose endings were
silently discarded three stages later.

`audit_chunks` prints the count, mean and longest chunk after every chunking
run, and warns on overflow. It is the same check the statistics script performs
later, but it runs at the point where you could still act on it.

## The embedding model

BGE needs a retrieval instruction prepended to **questions but not to chunks**.
`embed_query` adds it, `embed_documents` does not. Getting these the wrong way
round loses most of the model's benefit and raises no error, which is why the
distinction is buried inside the two methods rather than left to callers. One
shared instance is loaded lazily, so nothing loads it twice.

## The database wrapper

Two things `chroma_store.py` exists to get right:

**Vectors are always passed in explicitly.** Call `upsert` without them and
Chroma quietly embeds the text with its own model, after which searching
compares vectors from two different models — no error, just meaningless results.
The `embeddings=` argument carries a "never leave this out" comment for a
reason.

**Chroma metadata cannot hold lists.** `page_number` is flattened to `"1,2,3"`
on write and parsed back to `[1, 2, 3]` on read. This is why the raw SQLite
shows page numbers as comma-joined strings, and why anything reading the
database directly rather than through this wrapper must do its own parsing.

Also worth knowing: the collection is created with `hnsw:space` set to cosine,
because Chroma's default is straight-line distance and would rank differently
against normalised vectors. Scores are returned as `1 - distance`, so higher
means closer. `all_documents` sorts by id so BM25 builds over a deterministic
order, and writes go in batches of 500 because Chroma rejects very large single
writes.

# Two things worth knowing

**Tokens.** A "token" is a piece of a word. BGE reads 512 at a time and spends 2
on its own markers, so 510 is the hard ceiling: anything past it is silently
thrown away when the chunk is embedded, which would make long chunks look worse
than they are for the wrong reason. `MAX_TOKENS` is 500, just under that ceiling,
and `chunking_tokenizer` refuses to run if it is ever raised above what the model
can read — a module-level raise, so it fires the moment anything imports the
tokenizer rather than at the point a chunk is embedded. The guard works: the
longest chunk in the corpus is 501 tokens, and `corpus_stats.json` reports zero
truncations in all three strategies. Check that field after any change to
chunking — it is the only thing standing between you and silent data loss.

Token counts everywhere are WordPiece counts from the embedding model's own
tokenizer, excluding `[CLS]` and `[SEP]`. A count from any other tokenizer is not
comparable.

**Matching models.** A question must be turned into a vector by the same model
that turned the chunks into vectors. If they differ, search returns believable
nonsense and nothing raises an error — the worst kind of bug, because it looks
like it is working. Every inference run therefore starts with a self-check that
searches using the text of one of the database's own chunks: the right model
scores about 1.0, and anything much lower stops the run.

# Running it in Docker

```
export HF_TOKEN=...
docker compose build
docker compose run --rm rag-pipeline python Ingestion_pipeline/Ingestion-pipeline-automation-allweek.py
docker compose up          # serves the front end on localhost:7860
```

`python:3.11-slim`, CPU only, working directory `/workspace`. Compose mounts the
project root over `/workspace`, so `Data/` is written straight back to your real
disk — the database and every output survive the container. This is also why
paths in `run_manifest.json` read `/workspace/Data/...`: the recorded runs were
containerised.

Three things to know before relying on it:

- **`NEBIUS_API_KEY` is not passed through.** Compose forwards only `HF_TOKEN`,
  so benchmark stage 1 cannot reach the API from inside the container. Add it to
  the `environment:` block, or run that one script on the host.
- **The `CMD` is `python3 main.py`, and there is no `main.py`.** `docker compose
  up` therefore fails unless you override the command. Point it at `app.py`, or
  pass an explicit command as above.
- **Ports do not line up.** The Dockerfile exposes 8888 (Jupyter); compose maps
  7860 (Gradio). Mapping is what actually matters, so the front end works and
  the `EXPOSE` line is just misleading.

# Reproducing a run

`Data/Analysis/run_manifest.json` pins everything except the code: Python
3.11.16, torch 2.13.0, transformers 4.46.3, sentence-transformers 5.7.0,
chromadb 0.5.23, CPU only, plus a SHA-256 for each of the 13 PDFs and the chunker
parameters read back out of the chunk files.

Two gaps stop that from being reproducibility rather than a record of it.

**The code is not versioned.** The manifest records `git: not available`, so runs
cannot be tied to a commit. **Put the pipeline in a git repo.** Everything else
is already in place, and this is the piece that makes it worth having.

**The dependencies are not pinned.** `requirements.txt` pins exactly one package,
`chromadb==0.5.23`; everything else floats. So the manifest faithfully records
that a run used torch 2.13.0 and transformers 4.46.3, but rebuilding the image
tomorrow gets whatever is current, and the numbers may move for reasons nothing
to do with the pipeline. Freeze the versions the manifest already captured. The
Dockerfile comment claiming it installs "PyTorch 2.4.0" is stale in the same
way — nothing in the build pins any torch version.

While in there: `requirements.txt` carries both `gradio` and `streamlit`, and
`faiss-cpu` alongside `chromadb`. Only one of each pair is in use. Dropping the
unused ones cuts build time and removes the question of which front end or
vector store is really running.

# Stale documents

Two files under `Data/Benchmark/` describe an earlier lineage of the benchmark
and now contradict it:

- `DOCUMENTATION.md` — an audit of `benchmark_qa_v3.json`: 135 gold pages,
  reporting fixed-*k* alongside a character budget, across four strategies
  including exp5, scored with a BM25 baseline.
- `CHANGELOG.md` — the per-question change log from `benchmark_qa.json` through
  `_v2`, 107 gold pages.

Both predate the claim-and-set-cover derivation, which is why their gold page
counts are two to three times the current 42, and both predate the decision to
drop top-*k*. The reasoning in them is still worth reading — most of the
corrections listed above originate there, and the metric-bias argument in
DOCUMENTATION.md §1 is the reason this system uses a token budget. Read them as
history, not as a description of the current benchmark, and consider moving them
to an `archive/` folder so nobody scores against the numbers in them.

# Known inconsistencies in the code

Not behaviour, just things that will trip up the next person:

- `build_benchmark_from_exam.py` imports `from Evaluation.exam_source import
  EXAM`, but `exam_source.py` ships in `Benchmark_creation/`. One of the two
  needs to move.
- `clean_benchmark.py`'s docstring and usage block call the file
  `build_benchmark.py`.
- The stage-1 defaults write to `Data/Benchmark/benchmark_qa.json`, while stage 2
  defaults to a bare `benchmark_qa.json` in the working directory. Pass
  `--original` explicitly, as in the command above.
- `Data/Benchmark_intial_data/` is misspelled ("intial").
- The Dockerfile's `CMD` runs `main.py`, which does not exist in the project.
- The Dockerfile comment says it installs PyTorch 2.4.0; `requirements.txt` does
  not pin torch at all, and the recorded run used 2.13.0.
- `configuration.py`'s comments still describe the equal-k / equal-context
  choice as a pair of conditions to report side by side, and call `TOP_N` "the
  old behaviour" that is kept for reproducibility. Since top-*k* is no longer
  used, those comments now argue for something the project rejected. Rewrite
  them at the same time as the defaults.
- `RESULTS_DIR` in the configuration points at `Data/Results_generation`, but
  the runs wrote to `Inference_pipeline/Results_generation_bigfile/` and the
  no-RAG baseline to `Data/Results_generation_norag/`. Three conventions, none
  of them the configured one.
- `configuration.py` defines `TEMPERATURE = 0.4` while every benchmark run
  decodes greedily. Harmless, but it invites the assumption that runs were
  sampled.
- `2-3-SectionAware-chunking.py`'s docstring calls itself "Step 2d", left over
  from when a fourth chunker sat between it and 2-2. There is no 2-4.
- `Ingestion-pipeline-automation-allweek.py` imports the extractor via
  `import_module("1-Extract_clean_format_text")` because the filename starts
  with a digit, then runs `4-` and `5-` as subprocesses for the same reason —
  three different invocation styles in one 97-line file. The launcher pattern
  used by the 2-x scripts would make all of them importable.
- **`Inference_pipeline/` ships five files twice, byte for byte.** `4_routing.py`
  and `routing.py`, `5-vector_search.py` and `vector_search.py`,
  `6-ranking_n_retrieval.py` and `ranking_n_retrieval.py`,
  `7-llm_n_prompt.py` and `llm_n_prompt.py`, `8_RAG_pipeline.py` and
  `RAG_pipeline.py`. The numbered ones cannot be imported — a module name
  cannot start with a digit or contain a hyphen — so the unnumbered ones are
  what actually runs and the numbered copies are dead weight that will
  eventually be edited by mistake. Delete them, or make them the three-line
  launchers the package rewrite describes.
- `Inference_pipeline/Results_generation_bigfile/` and
  `Inference_pipeline/Output_eachStage_old/` hold run outputs inside the code
  folder, while everything else writes to `Data/`. `Output_eachStage_old` is
  also stale: its stage-5 filenames use full model names
  (`exp1_llama-32-1b-instruct_…`) rather than the current short tags.
- `Inference_pipeline_automation.py` (845 lines) is not referenced anywhere in
  the documented workflow. Either document it or remove it.
- `run_no_rag_baseline.py` lives at the project root and manipulates `sys.path`
  to import from `Inference_pipeline/`. It works, but it means the file cannot
  be moved without editing it.
- **Two copies of `Evaluation/` are in circulation.** The flat one
  (`evaluate_retrieval.py` and friends as single files) predates the package
  rewrite described above. You can tell them apart by one line: the flat
  `evaluate_generation.py` hardcodes `REFUSAL = "the provided course material
  does not cover this"`, whereas the rewritten version imports it from
  `configuration`. That hardcoded copy is exactly the drift the "one definition"
  rule exists to prevent — edit the sentence the model is told to emit and the
  old evaluator silently keeps searching for the previous wording, counting
  every refusal as an attempted answer. Delete the flat copy.
- `Evaluation/README.md` documents `--answers` paths without the `Data/` prefix
  and shows the pre-rewrite invocation without `-m`.
- `Evaluation/results.zip` sits next to the unpacked `results/` directory.
