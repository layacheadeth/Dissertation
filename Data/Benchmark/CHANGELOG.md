# benchmark_qa — corrections

`benchmark_qa.json` (30 items, 78 gold pages) -> `benchmark_qa_v2.json` (29 items, 107 gold pages).

Every gold page in v2 was re-verified byte-for-byte against `All_extracted_text.zip` after editing.
Schema is unchanged, so existing evaluation code needs no modification.

## Round 1 — grounding, arithmetic, ordering

- EXAM_012 REMOVED — 'entity linking'/'candidate generation'/'candidate ranking' absent from all 13 lectures; unanswerable from corpus.
- EXAM_013 QUESTION+ANSWER REWRITTEN — 'hyponym' has 0 corpus hits; 'antonym' appears only in w1 p83 re: word vectors, not WordNet. Rescoped to synonymy/hypernymy, which w3 p3 does teach.
- EXAM_014 ANSWER REWRITTEN — 'semantic role' has 0 corpus hits. NER and POS tagging retained; SRL removed.
- EXAM_029 VALUE CORRECTED 0.874 -> 0.875 — the stated working (7 x 0.125) already gave 0.875; exact 7*log10(4/3) = 0.87457.
- EXAM_026 ROUNDING CORRECTED 0.555 -> 0.556 (0.5555... truncated rather than rounded).
- EXAM_001 / EXAM_022 RANKS SWAPPED — w12 p24 promoted to primary (only slide defining nmod); w12 p21 demoted. Old 'why' falsely claimed p21 defined nmod.
- EXAM_004 GOLD PAGE REMOVED — w12 p46 (NER module agenda) has no bearing on Cohen's Kappa.
- EXAM_019 / EXAM_020 / EXAM_030 GOLD PAGE ADDED — w2 p33 at rank 2. The cited w2 p32 computes the same example WITHOUT boundary tokens (5 factors, 0.00006255), contradicting the gold answer; p33 is the only slide using <s>/</s>.

## Round 2 — remaining issues and Week 13 audit

- EXAM_009 QUESTION+ANSWER CORRECTED — the old answer said word=row, document=column, inverting its own primary slide (w1 p69: 'rows are documents ... columns are vocabulary words'). Reworded to the corpus's document-word convention.
- EXAM_029 PREMISE REWRITTEN — 'term-document matrix', 'Julius Caesar', 'Twelfth Night', 'As You Like It' have 0 corpus hits (imported from Jurafsky & Martin). Recast with the corpus's own 'document-word matrix' terminology; arithmetic (tf=7, df=3, N=4) unchanged.
- EXAM_023 ANSWER TRIMMED — removed the claim that the full stop is a punctuation dependent; no dependency-level 'punct' relation exists in the corpus (PUNCT appears only as a POS tag, w13 p18).
- EXAM_017 ANSWER TRIMMED — removed 'keeps the sequence length unchanged', which no cited slide states; added the softmax/scaling step that w3 p43 and p48 do teach.
- WEEK 13 ADDED — 27 supporting gold pages across 23 questions, from the previously uncited revision lecture. Added as supporting, never primary: the original teaching slide stays the primary target.
- EXAM_018 GOLD PAGE ADDED — w11 p16, which supports the answer's claim that [CLS] feeds classification; previously only w11 p13 (NSP training) was cited for it.

## Verification performed

- All 107 gold pages exist in the extracted corpus and match its text exactly (whitespace-normalised).
- Ranks are contiguous 1..N per question; exactly one `primary` per question, always at rank 1.
- No duplicate (week, page) within a question.
- Every asserted fact in each `qa_gold_standard` was grepped against all 13 lectures.

## Known limitations (not defects)

- **Week field format.** Gold pages use integer `week`; the chunkers emit `"Week N"` strings and `3-1`
  can inject `Data_weekN`. Normalise on both sides in the eval harness.
- **Weeks 4-9 remain untested.** No question touches LLM pretraining, SFT/alignment, in-context
  learning, multimodal, or RAG. Adding items there would broaden the benchmark.
- **w3 p3 extraction is damaged.** The slide reads `synonymy (dog <->)` — the target word was lost
  during PDF extraction. EXAM_013 is worded to avoid depending on it.
- **Page reuse.** w1 p77 is primary for 6 tf-idf items and w10 p23-adjacent pages for 6 agreement
  items; scores on those clusters are correlated, not independent.

## Round 3 — self-correction

- EXAM_017 GOLD PAGE ADDED — w3 p48. Round 2 added the scaling step to this answer but did not cite the slide that teaches it, creating exactly the ungrounded-claim defect this audit exists to remove.

## Round 4 — completeness audit, near-duplicates, labels

- COMPLETENESS AUDIT — 30 supporting gold pages added across 21 questions from weeks 1-3, 10-12, closing the same false-negative gap previously fixed for week 13.
- EXAM_019 / EXAM_020 / EXAM_030 GOLD PAGE REMOVED — w2 p28 is a near-duplicate of w2 p27 (the same slide with one extra intermediate term). Keeping both let one slide score a hit twice.
- EXAM_019 / EXAM_020 TYPE RELABELLED conceptual -> procedural. They ask how a quantity is computed without supplying numbers, which is neither a definition recall nor an arithmetic item.
- EXAM_027 / EXAM_028 — flagged, not merged: identical wording, different target term ('bark' in Doc1 vs 'dog' in Doc3). They test the same skill on the same slide; consider dropping one.

See DOCUMENTATION.md for rationale, results and known limitations.
