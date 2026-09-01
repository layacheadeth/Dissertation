"""
Dropping chunks the reranker scored below zero.

The cross-encoder scores each chunk against the question, and a negative score
means it judged the pair a poor match. Under a token budget the list is cut by
length rather than by score, so a run can carry many negatively-scored chunks:
81% of exp1's chunks at budget 1500, for instance.

This filter removes them at scoring time, which answers "what would the metrics
look like if only positively-scored chunks counted?" without regenerating
anything.

    IMPORTANT. The model already saw the full context. Filtering here changes
    what is measured, not what was given to it, so the generation metrics
    (groundedness, context_utilisation) are then scored against a context the
    model never had. Retrieval metrics are unaffected by that objection: they
    ask whether the gold pages were retrieved and ranked well, which is a
    property of the retriever. Report filtered generation numbers as a
    diagnostic, not as the system's performance.

Off by default, for the same reason the relevance floor is: it changes the
numbers, so it is a condition to switch on and report, not a setting.
"""

# Only combo2 attaches reranker scores. A chunk with no score cannot be judged,
# so it is kept: dropping it would silently empty every non-combo2 run.
DEFAULT_THRESHOLD = 0.0


def keeps(chunk, threshold=DEFAULT_THRESHOLD):
    """True if this chunk survives the filter."""
    score = chunk.get("score")
    return True if score is None else float(score) > threshold


def filter_chunks(chunks, threshold=DEFAULT_THRESHOLD):
    """The chunks scoring above the threshold, in their original order."""
    return [c for c in chunks or [] if keeps(c, threshold)]


def make_filter(threshold=DEFAULT_THRESHOLD):
    """A one-argument callable for the evaluators' chunk_filter parameter.

    Returns None when filtering is off, which both evaluators read as "use the
    chunks as they are".
    """
    return lambda chunks: filter_chunks(chunks, threshold)


def describe(records, threshold=DEFAULT_THRESHOLD):
    """What the filter would do to one file, for reporting alongside the scores.

    emptied is the number of questions left with no chunks at all. Those are
    kept rather than dropped, so recall and nDCG score them as zero instead of
    the denominator quietly shrinking.
    """
    before = after = emptied = 0
    for record in records:
        chunks = record.get("chunks_used", [])
        kept = filter_chunks(chunks, threshold)
        before += len(chunks)
        after += len(kept)
        if chunks and not kept:
            emptied += 1
    return {
        "threshold": threshold,
        "chunks_before": before,
        "chunks_after": after,
        "chunks_dropped": before - after,
        "questions_emptied": emptied,
    }
