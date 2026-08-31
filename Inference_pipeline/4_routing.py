"""
Routing: the two decisions made around retrieval, not by it.

    meta router      answers "what can you do" without searching at all
    relevance gate   refuses after searching, when nothing scored high enough

Both produce a Plan, so Plan lives here too. Nothing in this module loads a
model or touches a store, so it imports cheaply and tests without fixtures.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.append(_p)

from Share_components import configuration as config_inference

NO_ANSWER = config_inference.NO_ANSWER


# ---------------------------------------------------------------------------
# Meta router
# ---------------------------------------------------------------------------
# These have no answer in the slides, so retrieval returns its nearest
# neighbours regardless and the model is left guessing between refusing and
# listing whatever came back. The patterns are anchored and narrow on purpose:
# a false positive refuses a real question, which is worse than letting an odd
# greeting through. Not matched, and must stay that way: "what does week 3
# cover", "how does BM25 work", "what do you know about cosine similarity".
META_PATTERNS = (
    # Anchored at both ends: a bare greeting is a greeting, but
    # "hello world example of tokenisation" is a course question.
    r"^\s*(hi|hello|hey|yo|good (morning|afternoon|evening))( there)?\b[\s,.!]*$",
    r"^\s*(thanks|thank you|cheers|ok|okay|nice|great)\s*[.!]*\s*$",
    r"^\s*(who|what) are you\b",
    r"^\s*what('?s| is) your (name|purpose|job|role)\b",
    r"\bwhat can you (do|teach|help)\b",
    r"\bare you (a |an )?(bot|ai|human|robot|chatgpt|llm|model)\b",
    r"^\s*how do you work\b",
    r"^\s*(help|\?+)\s*$",
)

_META_RE = tuple(re.compile(p, re.IGNORECASE) for p in META_PATTERNS)

META_ANSWER = getattr(
    config_inference, "META_ANSWER",
    "I answer questions about the COMP64702 lecture material. Ask me about "
    "anything covered in the course slides and I will answer from them.")


def is_meta_question(question):
    """True for questions about the system rather than about the course."""
    text = (question or "").strip()
    return any(pattern.search(text) for pattern in _META_RE)


# ---------------------------------------------------------------------------
# Relevance gate
# ---------------------------------------------------------------------------
# How relevant the best chunk must be before the model is allowed to answer.
# None turns the gate off, which is the default: it changes the answers, so it
# must be switched on deliberately and reported as its own condition.
#
# The number is a raw cross-encoder logit, unbounded and corpus-dependent, so
# there is no sensible universal value. Calibrate it before trusting it: run
# the benchmark once with --save-scores, then look at where the top scores of
# covered and uncovered questions separate.
RELEVANCE_FLOOR = getattr(config_inference, "RELEVANCE_FLOOR", None)

# The gate reads reranker scores, which only this combo produces.
GATE_COMBO = "combo2"


def gate_active(relevance_floor, active_combo):
    """The gate needs the reranker, so it only works under combo2."""
    return relevance_floor is not None and active_combo == GATE_COMBO


def check_floor_supported(relevance_floor, combo):
    """Fail when the pipeline is built, not silently at query time.

    A floor set under a mode with no reranker would look like it was applied
    and do nothing, and the results file would claim a gate that never ran.
    """
    if relevance_floor is not None and combo != GATE_COMBO:
        raise SystemExit(
            f"--relevance-floor needs the reranker, but combo is {combo!r}.\n"
            f"Use --combo {GATE_COMBO}, or drop the floor.")


def below_floor(scores, relevance_floor):
    """True when nothing was retrieved, or the best chunk misses the floor.

    The reranker read the question against each chunk, so a low best score is
    evidence about the corpus, not a guess made before looking at it.
    """
    return not scores or scores[0] < relevance_floor


# ---------------------------------------------------------------------------
# What the decisions produce
# ---------------------------------------------------------------------------
class Plan:
    """What routing and retrieval decided, before the model is asked.

    route is one of:
        meta       not a course question, answered without searching
        refused    searched, but nothing cleared the relevance floor
        answer     chunks found, the model still has to write the answer

    Keeping this separate from generation lets a caller print the chunks
    before the answer without searching twice, and lets the benchmark record
    why a question was refused.
    """

    __slots__ = ("route", "answer", "documents", "scores")

    def __init__(self, route, answer=None, documents=(), scores=None):
        self.route = route
        self.answer = answer
        self.documents = list(documents)
        self.scores = list(scores) if scores is not None else None

    @property
    def top_score(self):
        return self.scores[0] if self.scores else None

    def __repr__(self):
        return (f"Plan({self.route}, {len(self.documents)} chunks, "
                f"top_score={self.top_score})")


def meta_plan():
    """Answered without searching."""
    return Plan("meta", answer=META_ANSWER)


def refusal_plan(documents, scores):
    """A gate refusal. The chunks are carried so it can be inspected."""
    return Plan("refused", answer=NO_ANSWER, documents=documents, scores=scores)
