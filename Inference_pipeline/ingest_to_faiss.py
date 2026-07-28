"""
Ingest chunked experiment files into FAISS (replaces the ChromaDB ingest).

For each strategy it globs the matching chunk JSONs under Data/, embeds the
chunk text with MiniLM (same model used at query time), and writes a FAISS
index + a docs sidecar per strategy:

    faiss_store/exp1_page_level.index      + exp1_page_level_docs.json
    faiss_store/exp2_fixed_overlap.index   + exp2_fixed_overlap_docs.json
    faiss_store/exp3_structure_level.index + exp3_structure_level_docs.json

Chunk JSON records may carry text under text/chunk/content/page_content and
metadata week / page_number / chunk_id — all handled below.

Run:  python ingest_to_faiss.py            # all strategies
      python ingest_to_faiss.py exp2       # one strategy
"""

import os
import sys
import glob
import json

from langchain_core.documents import Document
from embeddings import MiniLMEmbedding
from vectore_store import FAISSVectorDB

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
OUT_DIR = os.path.join(SCRIPT_DIR, "faiss_store")

STRATEGIES = {
    "exp1": {"pattern": "*exp1_chunks.json", "name": "exp1_page_level"},
    "exp2": {"pattern": "*exp2_chunks.json", "name": "exp2_fixed_overlap"},
    "exp3": {"pattern": "*exp3_chunks.json", "name": "exp3_structure_level"},
}


def _week_num(week):
    import re
    m = re.search(r"\d+", str(week))
    return int(m.group()) if m else None


def load_documents(pattern):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**", pattern), recursive=True))
    docs = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data["chunks"] if isinstance(data, dict) and "chunks" in data else data
        for rec in records:
            text = rec.get("text") or rec.get("chunk") or rec.get("content") or rec.get("page_content") or ""
            if isinstance(text, dict):
                text = json.dumps(text)
            text = (text or "").strip()
            if not text:
                continue
            week = rec.get("week", "")
            docs.append(Document(page_content=text, metadata={
                "chunk_id": str(rec.get("chunk_id", "?")),
                "week": week,
                "week_number": _week_num(week),
                "page_number": rec.get("page_number"),
            }))
        print(f"  loaded {len(records):>4} records from {os.path.relpath(fp, SCRIPT_DIR)}")
    return docs


def ingest(strategy, embedder):
    cfg = STRATEGIES[strategy]
    print(f"\n=== {strategy} -> {cfg['name']} ===")
    docs = load_documents(cfg["pattern"])
    if not docs:
        print(f"  no chunks found for pattern {cfg['pattern']} — skipping")
        return
    embeddings = embedder.embed_documents([d.page_content for d in docs])
    dim = len(embeddings[0])

    store = FAISSVectorDB(dim)
    store.add_documents(embeddings, docs)

    os.makedirs(OUT_DIR, exist_ok=True)
    index_path = os.path.join(OUT_DIR, f"{cfg['name']}.index")
    docs_path = os.path.join(OUT_DIR, f"{cfg['name']}_docs.json")
    store.save(index_path, docs_path)
    print(f"  {len(docs)} chunks indexed (dim={dim})")


def main():
    which = sys.argv[1:] or list(STRATEGIES)
    embedder = MiniLMEmbedding()
    for s in which:
        if s not in STRATEGIES:
            print(f"Unknown strategy '{s}'. Choose from {list(STRATEGIES)}.")
            continue
        ingest(s, embedder)


if __name__ == "__main__":
    main()
