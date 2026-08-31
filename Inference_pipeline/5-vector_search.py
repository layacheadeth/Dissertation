"""
Reading the database at question time.

The 3-x ingestion scripts write the collections. This file reads them. Nothing
here writes, and it stops with a clear message if a collection is missing.

One thing this file exists to get right: a question must be turned into a
vector by the same model that turned the chunks into vectors. Only BGE is
used, so every collection ends "_bge", but the model and the collection are
still held together in one object so no caller can pair them up wrongly.

Usage
    from vector_search import VectorSearch

    search = VectorSearch()                          # exp3 + BGE, from config
    search = VectorSearch("exp1")
    docs = search.query("What is BM25?", k=5)
"""

import sys
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
# The root, for Share_components and Ingestion_pipeline; this folder, so the
# sibling modules import by bare name however this one was reached.
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.append(_p)

from langchain_core.documents import Document

from Share_components import configuration as config_inference

from Share_components.chroma_store import ChromaVectorDB
from Share_components.embeddings import EMBEDDERS
from Ingestion_pipeline.ingestion_strategy import COLLECTIONS

# Which name ending goes with which model, matching what 3-1 writes. One
# model, so one entry — kept as a mapping so the collection name is still
# derived rather than hard-coded in several places.
SUFFIXES = {
    "bge": "_bge",
}


def collection_for(strategy, embedder):
    """'exp3' plus 'bge' gives 'exp3_section_aware_bge'."""
    if embedder not in SUFFIXES:
        raise ValueError(f"Unknown embedder {embedder!r}. Choose {list(SUFFIXES)}.")

    if strategy.lower() in COLLECTIONS:
        return COLLECTIONS[strategy.lower()] + SUFFIXES[embedder]

    # A full collection name can be passed straight through as well.
    full_names = {name + suffix
                  for name in COLLECTIONS.values()
                  for suffix in SUFFIXES.values()}
    if strategy in full_names:
        return strategy

    raise ValueError(f"Unknown strategy {strategy!r}. Choose {list(COLLECTIONS)}.")


def list_collections():
    """What is actually in the database, as (name, number of vectors).

    Worth checking first if retrieval behaves oddly, in case ingestion never ran.
    """
    import chromadb
    client = chromadb.PersistentClient(path=str(config_inference.CHROMA_DIR))
    return sorted((c.name, c.count()) for c in client.list_collections())


class VectorSearch:
    """Searches one collection, using the model that built it."""

    def __init__(self,
                 strategy=config_inference.STRATEGY,
                 embedder=config_inference.EMBEDDER,
                 loaded_model=None):
        self.strategy = strategy
        self.embedder_name = embedder
        self.collection_name = collection_for(strategy, embedder)

        # A model that is already in memory can be passed in, so a script
        # running several strategies does not load the same one repeatedly.
        self.embedder = loaded_model or EMBEDDERS[embedder]()

        self.store = ChromaVectorDB.load(self.collection_name)
        self._documents: Optional[List[Document]] = None

    @property
    def documents(self):
        """Every chunk in the collection, loaded once and kept.

        The keyword search needs all the text at once, because it scores by how
        often words appear across the whole corpus.
        """
        if self._documents is None:
            self._documents = self.store.all_documents()
        return self._documents

    def embed(self, question):
        return self.embedder.embed_query(question)

    def query(self, question, k=config_inference.TOP_N, week=None):
        """Vector search only. Adding keyword search is the Retriever's job."""
        where = {"week": week} if week else None
        return self.store.search(self.embed(question), k=k, where=where)

    def query_with_scores(self, question, k=config_inference.TOP_N, week=None):
        """The same, with scores where higher means closer.

        A best score around 0.3 means nothing in the corpus really matched,
        which is a different problem from matching but ranking badly.
        """
        where = {"week": week} if week else None
        return self.store.search_with_scores(self.embed(question), k=k, where=where)

    def self_check(self):
        """Search using the exact text of one of the collection's own chunks.

        The right model scores about 1.0. Much lower means the chunks were
        embedded with a different model than the one loaded now. That is the
        silent failure: search returns believable nonsense and nothing raises
        an error, so this runs before every session.
        """
        got = self.store.collection.get(limit=1, include=["documents"])
        if not got["documents"]:
            raise ValueError(f"Collection {self.collection_name} is empty.")
        hits = self.store.search_with_scores(self.embed(got["documents"][0]), k=1)
        return hits[0][1] if hits else 0.0

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return (f"VectorSearch({self.strategy}, {self.embedder_name}, "
                f"{self.collection_name}, {len(self)} vectors)")

"""
---------------------------------------------------------------------------
Running this stage on its own
---------------------------------------------------------------------------
Stage 1 of inference, runnable the way the numbered ingestion scripts are.

  python Inference_pipeline/vector_search.py --question "What is BM25?"
  python Inference_pipeline/vector_search.py --question "What is BM25?" --strategy exp1 --k 10
  python Inference_pipeline/vector_search.py --list

Reads  Data/Database/chroma_db/
Writes Data/Results_stage1/<strategy>_<slug>.json

Dense search only. Keyword search, fusion and reranking are stage 2. The file
written here records what this stage was given and what it handed on, so the
three stages can be read end to end for one question. It is an illustration
of the pipeline, not a result: the numbers reported in the dissertation come
# from Data/Results_evaluation/.

"""

STAGE = "1-vector-search"
STAGE_DIR = config_inference.ROOT / "Data" / "Results_stage1"


def stage_settings(strategy, embedder, collection, k, week):
    """The settings that decide what this stage returns."""
    return {
        "strategy": strategy,
        "embedder": embedder,
        "collection": collection,
        "k": k,
        "week": week,
    }


def stage_hash(settings):
    """A short code for these settings.

    Deliberately not run_inference_bigfile.settings_hash: that one covers the
    prompt and the generator, and computing it would load the language model
    this stage exists to do without.
    """
    import hashlib
    parts = [f"{key}={settings[key]}" for key in sorted(settings)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def write_record(settings, question, hits, seconds, self_check_score):
    """One JSON file, in the envelope stages 2 and 3 also use.

    stage / settings / settings_hash / input / output / seconds. Stage 2 takes
    this output as its input, so the hand-off is on disk rather than only in
    the prose.
    """
    import json

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in question.lower()).strip("_")[:40]
    path = STAGE_DIR / f"{settings['strategy']}_{slug}.json"

    record = {
        "stage": STAGE,
        "settings": settings,
        "settings_hash": stage_hash(settings),
        "self_check": round(float(self_check_score), 4),
        "input": {"question": question},
        "output": {
            "hits": [
                {
                    "rank": rank,
                    "score": round(float(score), 4),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "week": doc.metadata.get("week"),
                    "page_number": doc.metadata.get("page_number"),
                    "token_count": doc.metadata.get("token_count"),
                    "content": doc.page_content,
                }
                for rank, (doc, score) in enumerate(hits, 1)
            ],
            "context_tokens": sum(int(doc.metadata.get("token_count") or 0)
                                  for doc, _ in hits),
        },
        "seconds": round(seconds, 3),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def run_stage(question, strategy, embedder, k, week, save=True):
    """Every phase, printed as it happens, the way the ingestion scripts report."""
    import time
    import numpy as np

    print(f"\n[1] Collections in the database")
    for name, count in list_collections():
        print(f"  {name:<34} {count:>5} vectors")

    print(f"\n[2] Loading")
    search = VectorSearch(strategy, embedder=embedder)
    print(f"  strategy    {search.strategy}")
    print(f"  collection  {search.collection_name}")
    print(f"  vectors     {len(search)}")

    print(f"\n[3] Encoder self-check")
    score = search.self_check()
    print(f"  a chunk searched for itself scores {score:.3f}")
    if score < 0.95:
        raise SystemExit(
            f"  Expected about 1.000. Collection '{search.collection_name}' was\n"
            f"  built with a different model than the one loaded now. Re-run:\n"
            f"    python Ingestion_pipeline/3-1-Ingest-to-ChromaDB-bge-Embed.py")
    print(f"  the collection was built with the model now loaded")

    print(f"\n[4] The question as a vector")
    vector = search.embed(question)
    print(f"  question    {question!r}")
    print(f"  prefix      {search.embedder.QUERY_PREFIX!r}")
    print(f"  dimensions  {len(vector)}")
    print(f"  length      {float(np.linalg.norm(vector)):.3f}  (1.000 means normalised)")
    print(f"  first five  {', '.join(f'{x:+.3f}' for x in vector[:5])}")

    print(f"\n[5] Nearest {k} chunks" + (f" in {week}" if week else ""))
    started = time.time()
    hits = search.query_with_scores(question, k=k, week=week)
    seconds = time.time() - started

    if not hits:
        print("  nothing came back. Is the collection empty?")
        return None

    print(f"  {'rank':<6}{'score':<8}{'week':<9}{'pages':<10}{'chunk id':<18}text")
    for rank, (doc, hit_score) in enumerate(hits, 1):
        pages = doc.metadata.get("page_number", [])
        pages = ",".join(str(p) for p in pages) if isinstance(pages, list) else str(pages)
        preview = " ".join(doc.page_content.split())[:52]
        print(f"  {rank:<6}{hit_score:<8.3f}{str(doc.metadata.get('week', '?')):<9}"
              f"{pages:<10}{str(doc.metadata.get('chunk_id', '?')):<18}{preview}")

    context_tokens = sum(int(doc.metadata.get("token_count") or 0) for doc, _ in hits)
    print(f"\n  {context_tokens} tokens of context in {seconds:.2f}s")

    best = hits[0][1]
    if best < 0.4:
        print(f"  best score {best:.3f}: nothing in the corpus really matched, which "
              f"is a different problem from matching but ranking badly.")
    else:
        print(f"  best score {best:.3f}: something in the corpus matched the question.")

    settings = stage_settings(search.strategy, search.embedder_name,
                              search.collection_name, k, week)
    if save:
        path = write_record(settings, question, hits, seconds, score)
        print(f"\n[6] Written to {path}")
    else:
        print(f"\n[6] Not saved (--no-save)")

    return hits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 1 of inference: dense search only.")
    parser.add_argument("--question", default="What is BM25?")
    parser.add_argument("--strategy", default=config_inference.STRATEGY)
    parser.add_argument("--embedder", default=config_inference.EMBEDDER,
                        choices=list(SUFFIXES))
    parser.add_argument("--k", type=int, default=config_inference.TOP_N)
    parser.add_argument("--week", default=None, help='one week only, e.g. "Week 3"')
    parser.add_argument("--no-save", action="store_true",
                        help="print only, write no JSON")
    parser.add_argument("--list", action="store_true",
                        help="show the collections and stop")
    args = parser.parse_args()

    if args.list:
        for name, count in list_collections():
            print(f"{name:<34} {count:>5} vectors")
        raise SystemExit(0)

    run_stage(args.question, args.strategy, args.embedder,
              args.k, args.week, save=not args.no_save)
