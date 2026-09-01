"""
Counting tokens.

A "token" is a piece of a word. BGE can read 512 of them at a time, and 2 are
taken by markers the model adds itself, so 510 is the real ceiling for text.
Chunks are measured with this file so they never go over.

Only BGE is used now, so its tokenizer is the only one that counts and its
limit is the only one that binds.
"""

from typing import List, Optional, Tuple

from Share_components import configuration


MODEL_MAX_POSITIONS = 512      # BGE's limit
NUM_SPECIAL_TOKENS = 2         # the markers the model adds around every input
USABLE_TOKENS = MODEL_MAX_POSITIONS - NUM_SPECIAL_TOKENS   # 510

# Used by the statistics script to check chunks against the model.
EMBED_LIMITS = {
    "bge-small-en-v1.5": 512,
}

HF_NAMES = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}

# Caught here rather than three stages later, when the only symptom would be
# answers built from chunks whose endings were silently dropped.
if configuration.MAX_TOKENS > USABLE_TOKENS:
    raise ValueError(
        f"configuration.MAX_TOKENS is {configuration.MAX_TOKENS}, but the model can "
        f"only read {USABLE_TOKENS} tokens of content. Anything above that is "
        f"thrown away when the chunk is embedded. Lower MAX_TOKENS."
    )

_TOK = None


def usable_limit(model: str) -> int:
    """How much text a model can really read, once its markers are deducted."""
    return EMBED_LIMITS[model] - NUM_SPECIAL_TOKENS


def get_tokenizer():
    """Load the tokenizer the first time it is needed, then reuse it."""
    global _TOK
    if _TOK is not None:
        return _TOK

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install it first:  pip install transformers tokenizers"
        ) from exc

    try:
        _TOK = AutoTokenizer.from_pretrained(configuration.TOKENIZER, use_fast=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the tokenizer '{configuration.TOKENIZER}'. If you are "
            f"offline, download it once and point configuration.TOKENIZER at the folder."
        ) from exc

    # Measuring a long slide means encoding more than 512 tokens, which prints
    # a warning every time. It is harmless here, so silence it.
    try:
        import logging
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    except Exception:
        pass

    return _TOK


def init_tokenizer(name: Optional[str] = None):
    """Load the tokenizer up front. Used by 4-Corpus-statistics.py."""
    if name:
        configuration.TOKENIZER = name
    return get_tokenizer()


def tokenizer_name() -> str:
    return configuration.TOKENIZER


def tokenizer_provenance() -> dict:
    """A record of what did the counting, saved into the run manifest."""
    tok = get_tokenizer()
    return {
        "name_or_path": configuration.TOKENIZER,
        "class": type(tok).__name__,
        "vocab_size": getattr(tok, "vocab_size", None),
        "model_max_positions": MODEL_MAX_POSITIONS,
        "special_tokens_reserved": NUM_SPECIAL_TOKENS,
        "usable_tokens": USABLE_TOKENS,
        "counts_exclude_special_tokens": True,
    }


def count_tokens(text: str) -> int:
    """How many tokens this text is, not counting the model's own markers."""
    if not text:
        return 0
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def encode_with_offsets(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return the tokens plus where each one starts and ends in the original text.

    The sliding-window chunker uses these positions to cut the original string.
    Turning tokens back into text directly would lowercase everything and break
    the maths notation.
    """
    enc = get_tokenizer()(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    return enc["input_ids"], enc["offset_mapping"]


def split_text(text: str, max_tokens: int) -> List[str]:
    """Cut text into pieces that each fit in max_tokens.

    Tries to cut at a blank line first. If a piece is still too long, tries a
    single newline, then a full stop, then a space. This keeps sentences whole
    whenever it can.
    """
    return _split(text, max_tokens, ["\n\n", "\n", ". ", " ", ""])


def _split(text: str, max_tokens: int, separators: List[str]) -> List[str]:
    text = text.strip()
    if not text:
        return []

    # Already small enough, or we have run out of ways to cut it.
    if count_tokens(text) <= max_tokens or not separators:
        return [text]

    sep = separators[0]
    finer = separators[1:]

    pieces = text.split(sep) if sep != "" else list(text)
    chunks: List[str] = []
    current: List[str] = []

    for piece in pieces:
        candidate = sep.join(current + [piece]) if current else piece

        if count_tokens(candidate) <= max_tokens:
            current.append(piece)
        else:
            if current:
                chunks.append(sep.join(current).strip())
                current = []

            # This single piece is too big on its own, so cut it more finely.
            if count_tokens(piece) > max_tokens and finer:
                chunks.extend(_split(piece, max_tokens, finer))
            else:
                current.append(piece)

    if current:
        chunks.append(sep.join(current).strip())

    return [c for c in chunks if c]


def audit_chunks(chunks, label: str) -> dict:
    """Print how long the chunks are and whether any are too long."""
    counts = sorted(c.get("token_count", count_tokens(c["content"])) for c in chunks)
    if not counts:
        print(f"  [{label}] no chunks produced")
        return {"n": 0, "max": 0, "overflows": 0}

    stats = {
        "n": len(counts),
        "max": counts[-1],
        "mean": round(sum(counts) / len(counts), 1),
        "overflows": sum(1 for c in counts if c > USABLE_TOKENS),
    }

    print(f"  [{label}] {stats['n']} chunks, mean {stats['mean']} tokens, "
          f"longest {stats['max']} of {USABLE_TOKENS} allowed")

    if stats["overflows"]:
        print(f"  [{label}] WARNING: {stats['overflows']} chunk(s) are too long "
              f"and will be cut off when embedded")

    return stats
