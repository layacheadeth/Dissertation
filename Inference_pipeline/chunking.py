"""
Chunking adapter for the inference pipeline.

This file contains NO chunking logic. The five strategies live in
Ingestion_pipeline_code/2-*.py and are imported from there, so there is exactly
one implementation of each and it cannot drift from what built the collections.

All this module does is:
  1. load those scripts (their filenames are not valid module names, so plain
     `import` cannot reach them — see load_ingestion_module below)
  2. wrap run_experiment_N output in langchain Documents
  3. normalise metadata so page_number is always a list

Point INGESTION_DIR at the ingestion folder, or set the env var:
    export INGESTION_DIR=/path/to/Ingestion_pipeline_code

Usage
    from chunking import get_chunker
    chunker = get_chunker("exp5")
    docs = chunker.chunk_files(["Data/Data_week1/Week1.json"])   # preferred
    docs = chunker.chunk(corpus_list_of_dicts)                   # RAGPipeline path
"""

import glob
import importlib.util
import json
import os
import re
import tempfile
from typing import List, Optional

from langchain_core.documents import Document

# Where the ingestion scripts live. Default assumes this layout:
#     project/Inference_pipeline/chunking.py
#     project/Ingestion_pipeline_code/2-*.py
INGESTION_DIR = os.environ.get(
    "INGESTION_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "Ingestion_pipeline_code"),
)

# Collection names are NOT listed here on purpose. They belong to
# 3-1-Ingest-to-ChromaDB.py's STRATEGIES dict, which is the single source of
# truth for where chunks are stored. This module only produces chunks.
SCRIPTS = {
    "exp1": ("2-1-PageLevel-chunking.py",            "run_experiment_1"),
    "exp2": ("2-2-FixedSizeOverlapping-chunking.py", "run_experiment_2"),
    "exp3": ("2-3-StructureLevel-chunking.py",       "run_experiment_3"),
    "exp4": ("2-4-SemanticAware-chunking.py",        "run_experiment_4"),
    "exp5": ("2-5-SectionAware-chunking.py",         "run_experiment_5"),
}

_MODULE_CACHE = {}


# ---------------------------------------------------------------------------
def load_ingestion_module(filename: str, directory: Optional[str] = None):
    """Load a module from the ingestion folder BY FILE PATH.

    Needed because '2-4-SemanticAware-chunking.py' and '3-1-Ingest-to-ChromaDB.py'
    start with digits and contain hyphens — neither is a legal Python identifier,
    so `import` cannot reach them. importlib does not care about the name.

    Cached: exp4 loads a real embedding model at module scope in some versions,
    and reloading per lecture file would reload the model.

    Shared with ingest_to_chroma.py so the loading trick exists in one place.
    """
    directory = directory or INGESTION_DIR
    path = os.path.abspath(os.path.join(directory, filename))

    if path in _MODULE_CACHE:
        return _MODULE_CACHE[path]

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {filename} in {directory}.\n"
            f"Set INGESTION_DIR to your Ingestion_pipeline_code folder."
        )

    spec = importlib.util.spec_from_file_location(
        "ingestion_" + re.sub(r"\W", "_", filename), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[path] = module
    return module


# ---------------------------------------------------------------------------
def _week_num(week):
    m = re.search(r"\d+", str(week))
    return int(m.group()) if m else None


def record_to_document(rec: dict, experiment_id: str) -> Optional[Document]:
    """One run_experiment_N record -> Document.

    Metadata is passed through as-is apart from page_number, which is forced to
    a list. exp2/exp4/exp5 legitimately span several slides, so a scalar cannot
    represent them; evaluation therefore tests overlap, not equality:
        hit = bool(set(doc.metadata["page_number"]) & set(gold_pages))
    """
    text = (rec.get("content") or rec.get("text")
            or rec.get("chunk") or rec.get("page_content") or "")
    if isinstance(text, dict):
        text = json.dumps(text)
    text = (text or "").strip()
    if not text:
        return None

    meta = {k: v for k, v in rec.items()
            if k not in ("content", "text", "chunk", "page_content")}

    pages = meta.get("page_number", [])
    if not isinstance(pages, list):
        pages = [pages] if pages is not None else []
    meta["page_number"] = [p for p in pages if p is not None]

    meta.setdefault("experiment_id", experiment_id)
    meta.setdefault("chunk_id", "?")
    meta["week_number"] = _week_num(meta.get("week", ""))

    return Document(page_content=text, metadata=meta)


# ---------------------------------------------------------------------------
class IngestionChunker:
    """Wraps one run_experiment_N function from the ingestion scripts."""

    def __init__(self, strategy: str, **kwargs):
        filename, funcname = SCRIPTS[strategy]
        self.strategy = strategy
        self.kwargs = kwargs                       # forwarded to run_experiment_N
        module = load_ingestion_module(filename)
        self._run = getattr(module, funcname)
        # experiment_id comes from the records themselves; the scripts already
        # stamp it (exp1_page_level, exp5_section_aware, ...).
        self.experiment_id = strategy

    # -- primary entry point: the scripts are path-based, so give them paths --
    def chunk_files(self, paths) -> List[Document]:
        if isinstance(paths, str):
            paths = [paths]
        out = []
        for p in paths:
            for rec in self._run(p, **self.kwargs):
                doc = record_to_document(rec, self.experiment_id)
                if doc is not None:
                    out.append(doc)
        return out

    def chunk_dir(self, data_dir: str, pattern: str = "**/*.json") -> List[Document]:
        paths = sorted(glob.glob(os.path.join(data_dir, pattern), recursive=True))
        paths = [p for p in paths if "chroma_db" not in p and "_chunks.json" not in p]
        return self.chunk_files(paths)

    # -- compatibility shim for RAGPipeline.index_data, which passes dicts -----
    def chunk(self, corpus) -> List[Document]:
        """Accepts what RAGPipeline hands over: a lecture dict or list of dicts.

        The ingestion scripts read from disk, so in-memory corpora are written
        to a temp file and handed over. Slightly wasteful, but it keeps ONE
        implementation of each strategy rather than a second in-memory copy
        that could silently diverge. Prefer chunk_files() when you have paths.
        """
        if isinstance(corpus, dict):
            corpus = [corpus]

        out = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, lecture in enumerate(corpus):
                if not isinstance(lecture, dict) or "pages" not in lecture:
                    continue
                p = os.path.join(tmp, f"lecture_{i}.json")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(lecture, f)
                out.extend(self.chunk_files(p))
        return out


# ---------------------------------------------------------------------------
def get_chunker(strategy: str = "exp5", **kwargs) -> IngestionChunker:
    """Factory. exp5 (section-aware) is the default — it won on retrieval.

    kwargs are forwarded to the underlying run_experiment_N, e.g.
        get_chunker("exp2", chunk_size=512, overlap=64)
        get_chunker("exp3", max_tokens=150)

    The other four are kept reachable because retrieval accuracy is only half
    the question: a strategy that retrieves well but hands the 0.5B generator
    300 tokens of context may still lose end to end.
    """
    if strategy.lower() not in SCRIPTS:
        raise ValueError(f"Unknown strategy: {strategy}. Choose {list(SCRIPTS)}.")
    return IngestionChunker(strategy.lower(), **kwargs)
