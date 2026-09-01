# Run manifest

Generated: 2026-08-19T17:41:35+00:00

## Code version

- **Not available**: git not usable: FileNotFoundError
- Fix: commit the pipeline to a git repo so runs are traceable

## Environment

- Python 3.11.16 on Linux-6.12.54-linuxkit-x86_64-with-glibc2.41
- GPU: none detected (CPU inference)

| Package | Version |
|---|---|
| torch | 2.13.0 |
| transformers | 4.46.3 |
| sentence-transformers | 5.7.0 |
| tokenizers | 0.20.3 |
| huggingface-hub | 0.36.2 |
| chromadb | 0.5.23 |
| langchain-core | 1.5.6 |
| pypdf | 6.16.1 |
| numpy | 2.4.6 |

## Source documents

- 13 PDF(s) in `/workspace/Data/All_lectures`
- SHA-256 of each is recorded in run_manifest.json

## Chunker parameters

| Strategy | Parameters |
|---|---|
| exp1 | max_tokens=500, min_slide_chars=25 |
| exp2 | max_tokens=500, overlap_tokens=100 |
| exp3 | max_tokens=500, min_slide_chars=25 |

Read back from the chunk files: exp2 window [500], overlap [100].
exp1 oversized-page guard: fired (2 of 516 chunks came from split pages).
exp3 detected 273 section(s), emitted as 296 chunk(s).

## Vector store

- 3 collection(s), 916 vectors total, at `/workspace/Data/Database/chroma_db`

| Collection | Vectors | Dims |
|---|---|---|
| exp1_page_level_bge | 516 | ? |
| exp2_fixed_overlap_bge | 104 | ? |
| exp3_section_aware_bge | 296 | ? |

