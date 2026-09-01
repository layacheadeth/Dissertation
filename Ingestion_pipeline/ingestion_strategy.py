"""
Loading chunks into ChromaDB.

There is one embedding model, BGE, so there is one set of collections. The name
still ends in "_bge" so a collection can never be mistaken for one built with a
different model if you ever add one back.

Do not run two ingestion scripts at the same time. ChromaDB uses SQLite
underneath and the two runs will fight over the file.
"""

import json
from typing import Dict, List

from langchain_core.documents import Document

from Share_components import configuration
from Share_components.chroma_store import ChromaVectorDB
from Share_components.chunking_tokenizer import USABLE_TOKENS, count_tokens
from Share_components.embeddings import BGEEmbedding, get_embedder

config_ingestion = configuration

MODEL_TAG = "bge-small-en-v1.5"
COLLECTION_SUFFIX = "_bge"

# One collection per chunking strategy.
COLLECTIONS = {
    "exp1": "exp1_page_level",
    "exp2": "exp2_fixed_overlap",
    "exp3": "exp3_section_aware",
}

# Which 2-x script produces each strategy, for the "nothing found" message.
SCRIPT_FOR = {
    "exp1": "2-1-PageLevel-chunking.py",
    "exp2": "2-2-FixedSizeOverlapping-chunking.py",
    "exp3": "2-3-SectionAware-chunking.py",
}


def _build_document(record: Dict, source_file: str, week: str,
                    strategy: str, model_tag: str, used_ids: set) -> Document:
    """Turn one saved chunk into a Document ready for the database."""
    text = record.get("content", "").strip()

    # Everything except the text becomes metadata we can filter and report on.
    metadata = {k: v for k, v in record.items() if k != "content"}

    # page_number is always a list, because exp2 and exp3 chunks can cover
    # several slides at once.
    pages = metadata.get("page_number", [])
    if not isinstance(pages, list):
        pages = [pages] if pages is not None else []
    metadata["page_number"] = [p for p in pages if p is not None]

    metadata.setdefault("week", week)
    metadata.setdefault("source_file", source_file)
    metadata["chunking_strategy"] = strategy
    metadata["embedding_model"] = model_tag

    # Ids must be unique across all lectures, not just within one file.
    chunk_id = str(record.get("chunk_id") or f"{week}_{len(used_ids)}")
    while chunk_id in used_ids:
        chunk_id += "_dup"
    used_ids.add(chunk_id)
    metadata["chunk_id"] = chunk_id

    return Document(page_content=text, metadata=metadata)


def collect_documents(strategy: str, model_tag: str = MODEL_TAG) -> List[Document]:
    """Load every week's chunk file for one strategy."""
    documents = []
    used_ids = set()

    for path in sorted(config_ingestion.EXTRACTED_DIR.glob(f"Data_week*/*_{strategy}_chunks.json")):
        week = path.parent.name

        # Skip leftovers from a lecture PDF that is no longer in the folder.
        lecture_stem = path.name.split(f"_{strategy}_chunks.json")[0]
        if not (path.parent / f"{lecture_stem}.json").exists():
            print(f"  ! skipping {path.name} — its lecture file is gone")
            continue

        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            doc = _build_document(record, path.name, week, strategy,
                                  model_tag, used_ids)
            if doc.page_content:
                documents.append(doc)

    return documents


def warn_if_too_long(texts: List[str]) -> None:
    """Say so if any chunk is longer than the model can read."""
    counts = [count_tokens(t) for t in texts]
    too_long = sum(1 for c in counts if c > USABLE_TOKENS)

    print(f"  longest chunk: {max(counts)} tokens of {USABLE_TOKENS} allowed")
    if too_long:
        print(f"  !! {too_long} chunk(s) are too long and will be cut off. "
              f"Lower MAX_TOKENS in configuration.py and re-run the chunkers.")


def ingest_all(embedder_class=BGEEmbedding, model_tag: str = MODEL_TAG,
               collection_suffix: str = COLLECTION_SUFFIX) -> None:
    """Load every strategy's chunks into ChromaDB with BGE.

    Each collection is deleted and rebuilt, so chunks from an older run cannot
    linger in the database after you change a chunker. The arguments have
    defaults, so ingest_all() on its own does the right thing.
    """
    print(f"Loading embedding model: {model_tag}")
    embedder = get_embedder() if embedder_class is BGEEmbedding else embedder_class()

    for strategy, base_name in COLLECTIONS.items():
        collection_name = base_name + collection_suffix

        print(f"\n=== {strategy} -> {collection_name} ===")

        documents = collect_documents(strategy, model_tag)
        if not documents:
            print(f"  no chunks found. Run {SCRIPT_FOR[strategy]} first.")
            continue

        texts = [d.page_content for d in documents]
        print(f"  {len(documents)} chunks")
        warn_if_too_long(texts)

        # We embed the text ourselves with the same model used for questions
        # later. If ChromaDB embedded it instead, it would use its own model
        # and searching would return nonsense with no error shown.
        vectors = embedder.embed_documents(texts)

        store = ChromaVectorDB(collection_name, reset=True)
        store.add_documents(vectors, documents)
        print(f"  stored {len(store)} vectors")

    print(f"\nAll collections written to {config_ingestion.CHROMA_DIR}")
