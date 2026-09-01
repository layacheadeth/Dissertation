# Benchmark Audit & Evaluation — Documentation

**Inputs:** `benchmark_qa.json` (30 questions, 78 gold pages), `All_extracted_text.zip` (13 lecture JSONs + chunk files for exp1/2/3/5)
**Outputs:** `benchmark_qa_v3.json` (29 questions, 135 gold pages), `evaluate_retrieval.py`, `CHANGELOG.md`, this file

---

## 1. What was wrong, and why it mattered

The audit found **eleven distinct defects** in four classes. Two were serious enough to invalidate a strategy comparison on their own.

| # | Class | Defect | Consequence |
|---|---|---|---|
| 1 | Grounding | 3 answers asserted facts absent from the entire corpus | Unwinnable questions; retrieval blamed for a labelling error |
| 2 | Arithmetic | 2 calculation answers wrong | Correct system marked incorrect |
| 3 | Ordering | 2 questions had the wrong primary page | Correct retrieval penalised |
| 4 | Relevance | 1 gold page unrelated to its question | Noise in the relevance judgments |
| 5 | Contradiction | 2 answers contradicted their own cited slide | Grounded generation scored as wrong |
| 6 | Completeness | Week 13 cited zero times; weeks 1–12 under-cited | **Systematic false negatives across all strategies** |
| 7 | Corpus | Deduplication missed a near-duplicate slide | One slide scored a hit twice |
| 8 | Metric | Page-overlap rewards wide chunks | **Would have ranked strategies by chunk width** |
| 9 | Labels | 2 questions mistyped as `conceptual` | Minor; distorts per-type breakdowns |
| 10 | Redundancy | 6 questions share one primary page; one pair is verbatim identical | Correlated scores presented as independent |
| 11 | Extraction | Formulas mangled in the source PDFs | Depresses QA scores for reasons unrelated to chunking |

### The two that mattered most

**Defect 6 — completeness.** `Week13_revision_lecture_exam_focused.json` is a 25-slide exam revision deck covering Cohen's Kappa, dependency parsing, BIO tagging, TF–IDF, positional encoding and attention. It appeared in **zero** of the original 78 gold pages. A retriever surfacing w13 p23 for a Kappa question was returning ideal material and being scored as a miss. The same gap, less concentrated, existed across weeks 1–12.

**Defect 8 — metric bias.** The natural metric is:

```python
hit = bool(set(chunk["page_number"]) & set(gold_pages))
```

A chunk spanning 11 slides has eleven chances to intersect the gold set; a chunk spanning one slide has one. Measured on your actual chunk files:

| Strategy | chunks | mean pages/chunk | chunks per gold page |
|---|---|---|---|
| exp1 | 515 | 1.00 | 1.00 |
| exp2 | 175 | **4.60** | 1.57 |
| exp3 | 563 | 1.00 | 1.11 |
| exp5 | 336 | **3.97** | **4.87** (max 20) |

exp5's figure comes from section-title prepending replicating one slide across every part of a split section.

---

## 2. What was done

### 2.1 Benchmark corrections (v1 → v3)

**Removed:** EXAM_012 (Named Entity Linking). The strings *entity linking*, *candidate generation* and *candidate ranking* have **zero** occurrences across all 13 lectures. Unanswerable from the corpus.

**Rewritten questions** — where an answer contradicted its own source, editing only the answer would leave it not answering the question:

| Item | Was | Now |
|---|---|---|
| EXAM_009 | "word = row, document = column" | Matches w1 p69: rows are documents, columns are words |
| EXAM_013 | Asked about hyponymy/antonymy (0 corpus hits) | Asks about synonymy/hypernymy, which w3 p3 teaches |
| EXAM_029 | Shakespeare term-document example (0 corpus hits) | Corpus's own "document-word matrix"; arithmetic unchanged |

**Rewritten answers:** EXAM_014 (removed Semantic Role Labelling — 0 corpus hits), EXAM_017 (removed unsupported "sequence length unchanged"; added the softmax/scaling steps w3 p43 and p48 do teach), EXAM_023 (removed the punctuation-dependent claim; no dependency-level `punct` in corpus).

**Arithmetic:** EXAM_029 `0.874 → 0.875` (its own working, `7 × 0.125`, already gave 0.875). EXAM_026 `0.555 → 0.556` (0.5555… was truncated, not rounded).

**Ordering:** EXAM_001 and EXAM_022 both ask about `nmod`. Both had w12 p21 as primary — a slide that never mentions nmod. w12 p24, the only slide defining it, was demoted to supporting. Ranks swapped in both.

**Gold pages added — 57 in total:**
- 25 from Week 13 across 22 questions
- 30 from weeks 1–3 and 10–12 across 22 questions
- w2 p33 to the three bigram questions (the only slide using `<s>`/`</s>`; the cited w2 p32 computes the same example *without* boundary tokens, contradicting the gold answer)
- w11 p16 to EXAM_018 (supports the `[CLS]`-feeds-classification claim)
- w3 p48 to EXAM_017 (self-correction: round 2 added scaling to the answer without citing its slide)

All additions are **supporting, never primary** — the original teaching slide remains the target a retriever should rank first.

**Gold pages removed:** w12 p46 from EXAM_004 (NER agenda slide, no bearing on Cohen's Kappa); w2 p28 from three bigram questions (near-duplicate of w2 p27).

**Labels:** EXAM_019/020 retyped `conceptual → procedural` (they ask how a quantity is computed without supplying numbers).

### 2.2 Corpus defect found

`deduplicate_pages` in `1-Extract_clean_format_text.py` missed w2 p27 / p28 — the same "Bigram Language Models" slide, where p28 inserts one intermediate term *mid-formula*. The subset test only catches build-ups that append:

```python
if norm_current and norm_current in norm_next and len(norm_current) < len(norm_next):
```

A mid-string insertion breaks containment. Both survived into the extracted JSON and into all four Chroma collections. Given w1 dropped 88→54 and w12 66→53, other insertion-type build-ups are likely still present. **Suggested fix:** replace strict containment with a token-set Jaccard threshold (> 0.9 on consecutive pages) in addition to the existing subset test. Not applied — it changes page numbering and would invalidate every gold page, so it needs a coordinated re-label.

### 2.3 Evaluation harness

`evaluate_retrieval.py` reports the biased metric alongside three corrections so the bias stays visible:

1. **Page precision** — gold slides ÷ all slides pulled in. Wide chunks are charged for their filler.
2. **Equal-context budget** — retrieve to a fixed character budget instead of fixed *k*, so "more pages per chunk" stops being free.
3. **Deduplicated page recall** — a gold page counts once regardless of how many chunks touch it.

Plus MRR, nDCG (weighted by gold `rank`, so the primary counts most) and primary-hit rate. It takes `week` from the **directory name**, sidestepping the `12` vs `"Week 12"` vs `"Data_week12"` mismatch entirely.

A dependency-free BM25 baseline is included so the harness runs standalone; swap in your Chroma retriever to score the real system.

---

## 3. Results on your actual chunk files

Reachability first: **all 72 distinct gold pages are covered in all four collections.** No strategy is structurally handicapped; the comparison is fair.

```
### fixed k=5
strategy   ceiling   naive  P_page  R_page     F1    MRR   nDCG   prim  pages   chars
exp1        72/72    0.931   0.448   0.571  0.475  0.794  0.592  0.655    5.0    1973
exp2        72/72    1.000   0.181   0.748  0.284  0.878  0.609  0.828   19.4    5737
exp3        72/72    0.931   0.466   0.577  0.486  0.794  0.597  0.655    4.9    1886
exp5        72/72    0.931   0.127   0.630  0.192  0.818  0.606  0.655   22.9    2873

### budget=4000 chars
exp1        72/72    1.000   0.300   0.718  0.403  0.803  0.641  0.759   10.4    3813
exp2        72/72    0.931   0.246   0.659  0.342  0.862  0.620  0.793   11.8    3401
exp3        72/72    1.000   0.296   0.726  0.400  0.803  0.648  0.828   10.7    3762
exp5        72/72    1.000   0.119   0.701  0.193  0.828  0.637  0.724   25.8    3590
```

**Read this way:** at fixed *k*=5 exp2 scores a perfect 1.000 on the naive metric — apparently the best strategy — while pulling **19.4 slides and 5,737 characters** per query against exp1's 5 slides and 1,973 characters. Its page precision is 0.181, the second worst. Give every strategy the same 4,000-character budget and exp2's naive score falls to 0.931 while exp3 takes the nDCG lead.

**The naive metric would have selected exp2 for chunking coarsely, not for retrieving well.**

exp5's page precision (0.119–0.127) is the lowest throughout, reflecting the up-to-20× chunk replication from title prepending. Note these numbers come from the BM25 baseline, not your MiniLM retriever — treat them as a demonstration of the metric's behaviour, not as your final result.

---

## 4. Verification performed

- All 135 gold pages exist in the extracted corpus and match its text **byte-for-byte** after whitespace normalisation.
- Ranks contiguous `1..N` per question; exactly one `primary`, always at rank 1.
- No duplicate `(week, page)` within a question; no pair with token-set Jaccard > 0.85.
- Every factual assertion in every `qa_gold_standard` grepped against all 13 lectures.
- Schema unchanged from the original — existing evaluation code needs no modification.

---

## 5. Known limitations

**Not defects, but they shape how results should be read.**

- **Weeks 4–9 untested.** No question touches LLM pretraining, SFT/alignment, in-context learning, multimodal or RAG — roughly half the syllabus. These lectures are prose-heavier than the slide decks that are covered, so they would discriminate between chunking strategies differently. This is the single biggest gap remaining.
- **Redundancy.** EXAM_027 and EXAM_028 have **Jaccard 1.00** on question text (identical wording, different target term). Six questions share w1 p77 as primary. 29 questions represent roughly 18 independent retrieval targets; a strategy winning the tf-idf cluster wins six items at once. Consider dropping one of 027/028.
- **Week field format.** Gold pages use integer `week`; chunkers emit `"Week N"`; `3-1-Ingest` can inject `Data_weekN`. `evaluate_retrieval.py` avoids this by reading the directory, but your own code must normalise.
- **Formula extraction is damaged corpus-wide.** w1 p77's IDF renders as `idfw =log 10 N dfw`; w2 p27 as `P(x) = NY n=1 P(x n|xn−1)`. Retrieval finds these slides, but a generator cannot reliably reconstruct the formula — so QA scores understate answer quality for reasons unrelated to chunking. w3 p3 is worse: `synonymy (dog <->)` lost its target word (EXAM_013 is worded to avoid depending on it).
- **Supporting-page rank order beyond rank 1 is unaudited.** Rank 1 is verified correct everywhere; ranks 2–9 were largely inherited or appended. This matters if you weight nDCG heavily.
- **Relevance judgments for the 57 added pages are mine.** They are defensible and each carries a `why`, but they have not been reviewed by a second party.

---

## 6. Files

| File | Purpose |
|---|---|
| `benchmark_qa_v3.json` | Corrected benchmark — 29 questions, 135 gold pages |
| `evaluate_retrieval.py` | Span-normalised scoring harness |
| `CHANGELOG.md` | Itemised per-question change log across all four rounds |
| `DOCUMENTATION.md` | This file |

### Quick start

```bash
# compare all four strategies, both fixed-k and equal-budget
python3 evaluate_retrieval.py \
    --benchmark benchmark_qa_v3.json \
    --chunk-root Data/All_extracted_text \
    --all --top-k 5 --char-budget 4000

# one strategy, per-question detail
python3 evaluate_retrieval.py \
    --benchmark benchmark_qa_v3.json \
    --chunks "Data/All_extracted_text/Data_week*/*_exp5_chunks.json" \
    --strategy exp5 --out exp5_rows.json
```

**When reporting results, report page precision next to the naive hit rate.** The gap between them is the size of the chunk-width bias, and a reader needs to see it to trust the comparison.
