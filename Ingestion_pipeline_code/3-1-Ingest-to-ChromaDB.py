"""
3-1-Ingest-to-ChromaDB.py

Ingest the exp*_chunks.json files produced by the 2-x chunking scripts into
ChromaDB, one collection per strategy.

CHANGES FROM THE PREVIOUS VERSION
  1. Embeddings are computed here with MiniLMEmbedding and passed explicitly.
     Previously collection.upsert() was called without an `embeddings=`
     argument, which makes Chroma embed the text with ITS OWN default model
     while queries at inference use MiniLM — two different vector spaces, no
     error raised, silently meaningless retrieval.
  2. page_number is preserved as a list. The old extract_text_and_meta kept
     only scalar values, so list-valued page_number was dropped — and page
     provenance is exactly what retrieval evaluation needs.
  3. Writes through ChromaVectorDB instead of a raw collection handle, so the
     writer and the reader share one metadata convention by construction
     rather than by two matching-but-separate implementations.
  4. Module-level work moved under main(). A script with import-time side
     effects cannot be imported by anything else.

Usage
    python 3-1-Ingest-to-ChromaDB.py                    # all five strategies
    python 3-1-Ingest-to-ChromaDB.py --strategies exp5
    python 3-1-Ingest-to-ChromaDB.py --reset            # rebuild from scratch

Collections are written to Data/Database/chroma_db/ by default.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from langchain_core.documents import Document

# The store and embedder live with the inference pipeline; add it to the path
# so both halves of the project share one copy of each.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE.parent / "Inference_pipeline", _HERE / "Inference_pipeline"):
    if _candidate.is_dir():
        sys.path.insert(0, str(_candidate))
        break

from chroma_store import ChromaVectorDB          # noqa: E402
from embeddings import MiniLMEmbedding           # noqa: E402


# ------------------------------------------------------------------
# 1. Path & Strategy Configurations
# ------------------------------------------------------------------
DATA_DIR = Path("Data/")
CHROMA_PATH = Path("Data/Database/chroma_db")

STRATEGIES = {
    "exp1": {
        "file_pattern": "exp1_chunks.json",
        "collection_name": "exp1_page_level",
        "description": "Page-Level Chunking Strategy",
    },
    "exp2": {
        "file_pattern": "exp2_chunks.json",
        "collection_name": "exp2_fixed_overlap",
        "description": "Fixed-Size Overlapping Chunking Strategy",
    },
    "exp3": {
        "file_pattern": "exp3_chunks.json",
        "collection_name": "exp3_structure_level",
        "description": "Structure-Level Chunking Strategy",
    },
    "exp4": {
        "file_pattern": "exp4_chunks.json",
        "collection_name": "exp4_semantic_aware",
        "description": "Semantic-Aware Chunking Strategy",
    },
    "exp5": {
        "file_pattern": "exp5_chunks.json",
        "collection_name": "exp5_section_aware",
        "description": "Section-Aware Chunking Strategy",
    },
}


# ------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------
def extract_text_and_meta(item, default_source, default_week, strategy_key):
    """Extract chunk text and metadata from one chunk-JSON record.

    Metadata is NOT scalarised here any more. ChromaVectorDB.flatten_metadata
    converts lists to strings on write and rehydrates them on read, so
    page_number can stay a list all the way through and evaluation code sees
    the same shape it would get from the FAISS store.
    """
    text = (
        item.get("content")
        or item.get("text")
        or item.get("chunk")
        or item.get("page_content")
        or ""
    )
    if isinstance(text, dict):
        text = json.dumps(text)
    text = (text or "").strip()

    metadata = {
        k: v for k, v in item.items()
        if k not in ("content", "text", "chunk", "page_content")
    }

    # page_number is ALWAYS a list. exp2/exp4/exp5 legitimately span several
    # slides, so a scalar cannot represent them; evaluation tests overlap:
    #     hit = bool(set(meta["page_number"]) & set(gold_pages))
    pages = metadata.get("page_number", [])
    if not isinstance(pages, list):
        pages = [pages] if pages is not None else []
    metadata["page_number"] = [p for p in pages if p is not None]

    metadata.setdefault("week", default_week)
    metadata.setdefault("source_file", default_source)
    metadata["chunking_strategy"] = strategy_key

    return text, metadata


def _keep_current_chunk_files(folder_path, chunk_files):
    """Drop chunk files with no matching raw lecture JSON in the same folder.

    Re-running a 2-x script after renaming a lecture leaves the OLD chunk file
    behind, and it still matches the *exp2_chunks.json glob. Those stale files
    carry the same chunk_ids as the current ones but different text, so both
    get ingested and the collection ends up with two conflicting copies of that
    week. (Real example: Week5_LLM-pretraining_exp2_chunks.json alongside
    Week5_TRIM_LLM-pretraining_exp2_chunks.json.)

    A chunk file is current if its name starts with the stem of a raw lecture
    JSON that actually exists here.
    """
    stems = [
        os.path.basename(p)[:-len(".json")]
        for p in glob.glob(os.path.join(folder_path, "*.json"))
        if "_chunks.json" not in os.path.basename(p)
    ]
    if not stems:
        return chunk_files                     # no raw files to check against

    kept = []
    for path in chunk_files:
        name = os.path.basename(path)
        if any(name.startswith(stem + "_") for stem in stems):
            kept.append(path)
        else:
            print(f"  ! skipping orphaned chunk file (no matching lecture): {name}")
    return kept


def collect_documents(strat_key, config):
    """Walk Data/Data_week*/ and load every chunk file for this strategy."""
    pattern = config["file_pattern"]
    documents = []
    seen_ids = set()

    week_folders = sorted(glob.glob(os.path.join(DATA_DIR, "Data_week*")))
    if not week_folders:                       # flat layout fallback
        week_folders = [str(DATA_DIR)]

    for folder_path in week_folders:
        week_name = os.path.basename(folder_path.rstrip("/")) or "unknown"
        target_files = sorted(glob.glob(os.path.join(folder_path, f"*{pattern}*")))
        target_files = _keep_current_chunk_files(folder_path, target_files)

        for json_path in target_files:
            file_name = os.path.basename(json_path)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  ! error loading {json_path}: {e}")
                continue

            for idx, item in enumerate(data if isinstance(data, list) else [data]):
                text, meta = extract_text_and_meta(item, file_name, week_name, strat_key)
                if not text:
                    continue

                # Stable unique id. ChromaVectorDB keys on chunk_id, and
                # _rrf_fusion matches on it too, so it must be unique across
                # files — not just within one.
                raw_id = str(item.get("chunk_id") or item.get("id") or f"{week_name}_{idx}")
                doc_id, dup = raw_id, 1
                while doc_id in seen_ids:
                    doc_id = f"{raw_id}_dup{dup}"
                    dup += 1
                seen_ids.add(doc_id)
                meta["chunk_id"] = doc_id

                documents.append(Document(page_content=text, metadata=meta))

    return documents


# ------------------------------------------------------------------
# 3. Strategy-by-Strategy Ingestion
# ------------------------------------------------------------------
def ingest_strategy(strat_key, config, embedder, reset=False):
    collection_name = config["collection_name"]

    print(f"\n{'=' * 55}")
    print(f"Ingesting [{strat_key.upper()}] -> collection '{collection_name}'")
    print(f"{'=' * 55}")

    documents = collect_documents(strat_key, config)
    if not documents:
        print(f"  no chunks found matching pattern: {config['file_pattern']}")
        return

    lengths = [len(d.page_content) for d in documents]
    print(f"  {len(documents)} chunks | mean {sum(lengths) // len(lengths)} chars, "
          f"max {max(lengths)}")

    # THE CRITICAL LINE: vectors come from OUR embedder, the same one used for
    # queries at inference. Never let Chroma embed for us.
    embeddings = embedder.embed_documents([d.page_content for d in documents])

    store = ChromaVectorDB(
        collection_name=collection_name,
        path=str(CHROMA_PATH),
        description=config["description"],
        reset=reset,
    )
    store.add_documents(embeddings, documents)   # batches internally
    print(f"  collection now holds {len(store)} vectors")


# ------------------------------------------------------------------
# 4. Main Execution Routine
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ingest chunked slide data into ChromaDB."
    )
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                        choices=list(STRATEGIES))
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--chroma-path", default=str(CHROMA_PATH))
    parser.add_argument("--reset", action="store_true",
                        help="delete each collection before writing. ALWAYS use "
                             "this after changing a chunker — upsert only "
                             "overwrites matching ids, so chunks whose ids no "
                             "longer exist would linger forever.")
    args = parser.parse_args()

    globals()["DATA_DIR"] = Path(args.data)
    globals()["CHROMA_PATH"] = Path(args.chroma_path)

    if not Path(args.data).is_dir():
        raise SystemExit(f"DATA_DIR does not exist: {args.data}")

    print("Loading embedding model...")
    embedder = MiniLMEmbedding()        # built ONCE, shared across strategies

    for strat_key in args.strategies:
        ingest_strategy(strat_key, STRATEGIES[strat_key], embedder, args.reset)

    print(f"\nAll {len(args.strategies)} collection(s) written to {CHROMA_PATH}/")


if __name__ == "__main__":
    main()
