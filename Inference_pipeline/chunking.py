"""
Chunking strategies for the COMP64702 slide corpus.

Input corpus shape (per lecture file):
    {"filename": ..., "week": "Week 1", "pages": [{"page_number": 1, "content": "..."}, ...]}

Three strategies, matching the ingestion experiments:
    - PageLevelChunker      (exp1) : one chunk per slide page
    - SlidingWindowChunker  (exp2) : fixed-token window with overlap, ignores page bounds
    - StructureChunker      (exp3) : per-page recursive split down to a token limit

Every chunk is returned as a langchain Document carrying at least:
    chunk_id, week, week_number, page_number, content
so the retriever (RRF needs chunk_id) and the evaluator (needs week/page) both work.
Use get_chunker("exp1" | "exp2" | "exp3").
"""

import re
from typing import List
from langchain_core.documents import Document

# Token helpers — tiktoken if available, else ~4 chars/token fallback.
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def _tokenize(t): return _ENC.encode(t)
    def _decode(t):   return _ENC.decode(t)
    def _count(t):    return len(_ENC.encode(t))
except ImportError:
    def _tokenize(t): return t.split()
    def _decode(t):   return " ".join(t)
    def _count(t):    return len(t) // 4


def _week_num(week):
    m = re.search(r"\d+", str(week))
    return int(m.group()) if m else None


def _as_files(corpus):
    """Normalise input to a list of lecture-file dicts (each with a 'pages' key)."""
    if isinstance(corpus, dict):
        return [corpus]
    return corpus


def _doc(content, week, chunk_id, page_number=None):
    return Document(
        page_content=content.strip(),
        metadata={
            "chunk_id": str(chunk_id),
            "week": week,
            "week_number": _week_num(week),
            "page_number": page_number,
        },
    )


class PageLevelChunker:
    """exp1: keep each slide page as a single chunk."""

    def __init__(self, min_chars: int = 25):
        self.min_chars = min_chars

    def chunk(self, corpus) -> List[Document]:
        out = []
        for data in _as_files(corpus):
            week = data.get("week", "")
            wk = str(week).replace(" ", "")
            for page in data.get("pages", []):
                content = (page.get("content") or "").strip()
                pn = page.get("page_number")
                if len(content) < self.min_chars:
                    continue
                out.append(_doc(content, week, f"{wk}_p{pn}", pn))
        return out


class SlidingWindowChunker:
    """exp2: flatten all pages (with [Slide N] tags) and window over tokens."""

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, corpus) -> List[Document]:
        out = []
        for data in _as_files(corpus):
            week = data.get("week", "")
            wk = str(week).replace(" ", "")
            flat = ""
            for page in data.get("pages", []):
                content = (page.get("content") or "").strip()
                if content:
                    flat += f"\n[Slide {page.get('page_number')}]\n" + content
            tokens = _tokenize(flat)
            step = self.chunk_size - self.overlap
            idx = 1
            for i in range(0, len(tokens), step):
                text = _decode(tokens[i:i + self.chunk_size]).strip()
                if not text:
                    continue
                # page_number stays None; slides are recoverable from [Slide N] markers.
                out.append(_doc(text, week, f"{wk}_sw_{idx}", None))
                idx += 1
        return out


class StructureChunker:
    """exp3: respect page boundaries, recursively split long pages to a token limit."""

    def __init__(self, max_tokens: int = 150):
        self.max_tokens = max_tokens
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _recursive_split(self, text, separators):
        text = text.strip()
        if not text:
            return []
        if _count(text) <= self.max_tokens or not separators:
            return [text]
        sep, rest = separators[0], separators[1:]
        splits = text.split(sep) if sep != "" else list(text)
        chunks, cur = [], []
        for s in splits:
            candidate = sep.join(cur + [s]) if cur else s
            if _count(candidate) <= self.max_tokens:
                cur.append(s)
            else:
                if cur:
                    chunks.append(sep.join(cur))
                cur = [s] if _count(s) <= self.max_tokens else None
                if cur is None:
                    chunks.extend(self._recursive_split(s, rest))
                    cur = []
        if cur:
            chunks.append(sep.join(cur))
        return [c.strip() for c in chunks if c.strip()]

    def chunk(self, corpus) -> List[Document]:
        out = []
        for data in _as_files(corpus):
            week = data.get("week", "")
            wk = str(week).replace(" ", "")
            for page in data.get("pages", []):
                content = (page.get("content") or "").strip()
                pn = page.get("page_number")
                if not content:
                    continue
                for sub, text in enumerate(self._recursive_split(content, self.separators), start=1):
                    out.append(_doc(text, week, f"{wk}_p{pn}_sub{sub}", pn))
        return out


def get_chunker(strategy: str = "exp2"):
    """Factory: 'exp1' page-level, 'exp2' sliding-window (best), 'exp3' structure-level."""
    s = strategy.lower()
    if s in ("exp1", "page", "page_level"):
        return PageLevelChunker()
    if s in ("exp2", "sliding", "fixed_overlap"):
        return SlidingWindowChunker()
    if s in ("exp3", "structure", "structure_level"):
        return StructureChunker()
    raise ValueError(f"Unknown strategy: {strategy}. Choose exp1 | exp2 | exp3.")
