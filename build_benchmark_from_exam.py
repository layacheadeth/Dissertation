"""
============================================================================
 BENCHMARK BUILDER (FROM MOCK EXAM)  --  RAG-Socratic TA (COMP64702)
============================================================================

WHAT THIS IS
------------
Builds the evaluation benchmark from the REAL mock-exam questions (transcribed
in exam_source.py). The QA gold answers are the exam's verified correct answers.
For each question the script ALSO locates the source lecture PAGE, so every
entry has BOTH a week AND a page_number retrieval anchor.

WHY PAGE-LEVEL ANCHORING MATTERS
--------------------------------
The three chunking strategies split text differently, but every chunk traces
back to a source page. A "week only" anchor is far too loose (a week has ~50
slides), so retrieval scores would be inflated and could not separate the
strategies. Anchoring to (week, page_number) makes retrieval evaluation fair
and discriminative: a retrieval is a hit if it returns a chunk from that page.

HOW THE PAGES ARE FOUND
-----------------------
The exam questions are NOT copied from a single slide and their wording differs
from the slides, so keyword matching is unreliable. Instead, for each question
we give the STRONG LLM that question, its answer, and the numbered list of that
week's slides, and ask which slides contain the answer. Because the retriever
returns the TOP-5 chunks and a single chunk/page is a weak anchor, we ask for up
to the top-5 most relevant pages (ranked). A retrieval is a hit if it returns a
chunk from ANY of those pages. Every gold page also carries its ground-truth
text, so the entry is inspectable and supports faithfulness/context metrics.

THE SOCRATIC GOLD IS A HINT, NOT A POLISHED QUESTION
----------------------------------------------------
The model under test (Qwen-0.5B) cannot produce elaborate Socratic dialogue.
So the Socratic gold standard is a short HINT describing what a good response
should nudge the student toward WITHOUT revealing the answer. At evaluation
time you simply check whether the small model's response conveys that hint and
does not reveal the answer - if so, that already counts as success.

OUTPUT (two aligned files), each entry carrying week + the top-5 gold pages
(page_number = rank-1 page, also_pages = the rest) plus gold_pages with the
ground-truth text of every gold page:
  - benchmark_qa.json:       question + verified factual answer (all questions)
  - benchmark_socratic.json: question + Socratic hint (conceptual questions only)

SETUP
-----
Put your Nebius key in .env:   NEBIUS_API_KEY=your_key_here
Install:                       pip install openai python-dotenv
Set LECTURE_PATHS below to your extracted lecture .json files, then run:

    python build_benchmark_from_exam.py
============================================================================
"""

import glob
import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

from exam_source import EXAM


# ===========================================================================
#  CONFIGURATION  (hard-coded; no command-line arguments)
# ===========================================================================

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1"

# Extracted lecture pages. You can list exact .json FILES, per-week FOLDERS,
# or a mix - the loader accepts all three (files used directly, folders searched).
LECTURE_PATHS = [
    "Data/Data_week1/Week1_Intro_to_vector.json",
    "Data/Data_week2/Week2_Language_modelling.json",
    "Data/Data_week3/Week3_word2vec_RNN_Transformer.json",
    "Data/Data_week4/Week4_LLM-Data.json",
    "Data/Data_week5/Week5_TRIM_LLM-pretraining.json",
    "Data/Data_week6/Week6_TRIM_SFT_alignment.json",
    "Data/Data_week7/Week7-TRIM_Incontext-Evaluation.json",
    "Data/Data_week8/Week8_TRIM-Application-Multimodal.json",
    "Data/Data_week9/Week9_RAG_LLM_final.json",
    "Data/Data_week10/Week10_TM_last_lecture.json",
    "Data/Data_week11/Week11_TRIm_BERT.json",
    "Data/Data_week12/Week12_TM-lastlecture-1.json",
    "Data/Data_week13/Week13_revision_lecture_exam_focused.json",
]

QA_OUTPUT = "Data/Benchmark/benchmark_qa.json"
SOCRATIC_OUTPUT = "Data/Benchmark/benchmark_socratic.json"


# ===========================================================================
#  PART 1:  Reading the extracted lecture text  (reused loader)
# ===========================================================================

def read_pages_from_file(path):
    """
    Read one week's lecture file into a list of page dicts:
        {week_label, week_number, page_number, content}
    Skips near-empty pages (title/divider slides).
    A lecture file looks like {"week": "Week 1", "pages": [{page_number, content}]}.
    """
    data = json.load(open(path, encoding="utf-8"))
    week_label = data.get("week", "Unknown")
    m = re.search(r"(\d+)", week_label)
    week_number = int(m.group(1)) if m else 0

    pages = []
    for page in data.get("pages", []):
        content = (page.get("content") or "").strip()
        if len(content) < 40:
            continue
        pages.append({
            "week_label": week_label,
            "week_number": week_number,
            "page_number": page["page_number"],
            "content": content,
        })
    return pages


def _iter_json_paths(paths):
    """Expand each entry in paths: a file is used directly; a folder is searched
    recursively for .json files. So you can list folders OR exact files."""
    for item in paths:
        if os.path.isdir(item):
            for f in glob.glob(os.path.join(item, "**", "*.json"), recursive=True):
                if "__macosx" not in f.lower():
                    yield f
        elif os.path.isfile(item):
            yield item
        else:
            print(f"Warning: path not found, skipping: {item}")


def load_all_pages_by_week(paths):
    """
    Load every lecture file and group pages by week number:
        { 1: [page, ...], 2: [...], ... }

    Accepts files OR folders. Only genuine lecture files (a dict with 'week'
    and 'pages') are used; chunk files (JSON lists) are ignored automatically.
    """
    pages_by_week = {}
    for path in _iter_json_paths(paths):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not (isinstance(data, dict) and "pages" in data and "week" in data):
            continue                                    # skip chunk/other files
        for page in read_pages_from_file(path):
            pages_by_week.setdefault(page["week_number"], []).append(page)
    for wk in pages_by_week:
        pages_by_week[wk].sort(key=lambda p: p["page_number"])
    return pages_by_week


def check_corpus_loaded(pages_by_week):
    """
    Weeks are now DERIVED per question (not hand-assigned), so we only need to
    confirm that the lecture corpus loaded at all. If nothing loaded, stop with
    a clear message rather than producing None pages.
    """
    if not pages_by_week:
        sys.exit(
            "\nERROR: no lecture slides loaded. Check LECTURE_PATHS - the paths "
            "are probably wrong.\n(Tip: point LECTURE_PATHS at the per-week "
            "FOLDERS so a filename typo can't break loading.)"
        )


# ===========================================================================
#  PART 2:  LLM helpers  (page location + Socratic hint)
# ===========================================================================

def make_nebius_client():
    load_dotenv()
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        sys.exit("NEBIUS_API_KEY not found. Add it to your .env file.")
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=key)


def extract_json_object(text):
    """Parse a JSON object from an LLM reply, tolerating ```json fences and
    stray backslash escapes (common with maths notation like \\alpha)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model reply.")
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate))


WEEK_SUMMARY_SYSTEM = """You summarise the topics of one week's lecture for the
module COMP64702 "Transforming Text Into Meaning" (NLP / LLMs / RAG). Given the
concatenated slide text, reply with ONE short sentence listing the week's main
topics, using the module's terminology. Plain text only."""


def summarise_week(client, week_number, pages):
    """One-line topic summary for a week, used to pick the right week per question."""
    joined = "\n".join(p["content"] for p in pages)[:6000]
    reply = client.chat.completions.create(
        model=MODEL, temperature=0.2, max_tokens=80,
        messages=[{"role": "system", "content": WEEK_SUMMARY_SYSTEM},
                  {"role": "user", "content": f"Week {week_number} slides:\n{joined}"}],
    )
    return reply.choices[0].message.content.strip()


PICK_WEEK_SYSTEM = """You decide which lecture week teaches the answer to an exam
question. You are given the QUESTION, its correct ANSWER, and a numbered list of
weeks with a one-line summary of each. Choose the single week whose topics best
match the answer. Match on MEANING, not keywords.

Return ONLY: {"week_number": <int>, "confidence": "high"/"medium"/"low"}
No markdown, no commentary."""


def pick_week(client, item, week_summaries):
    """
    Ask the LLM which week's topics best match this question's answer.
    'week_summaries' is {week_number: "one-line summary"}.
    Returns (week_number, confidence). Guarantees a week that actually exists.
    """
    listing = "\n".join(f"Week {wk}: {summ}"
                        for wk, summ in sorted(week_summaries.items()))
    user = (f"QUESTION:\n{item['q']}\n\nCORRECT ANSWER:\n{item['answer']}\n\n"
            f"WEEKS:\n{listing}\n\nReturn only the JSON object.")
    reply = client.chat.completions.create(
        model=MODEL, temperature=0.1, max_tokens=60,
        messages=[{"role": "system", "content": PICK_WEEK_SYSTEM},
                  {"role": "user", "content": user}],
    )
    result = extract_json_object(reply.choices[0].message.content)
    week = result.get("week_number")
    conf = result.get("confidence", "medium")
    if week not in week_summaries:                      # guard against a bad week
        week = min(week_summaries)                      # harmless fallback
        conf = "low"
    return week, conf


PAGE_LOCATE_SYSTEM = """You locate where an exam answer is taught in lecture slides.
You are given an exam QUESTION, its correct ANSWER, and a numbered list of slides
from the relevant lecture week. Identify UP TO FIVE slides whose content most
directly supports the answer, ranked from most to least relevant (the pages a
student should read to learn it).

Match on MEANING, not exact words - the exam wording differs from the slides.
List only genuinely relevant slides - fewer than five is fine. Put the single
most relevant slide first.

Return ONLY a JSON object:
{"pages": [<int>, ...], "confidence": "high"/"medium"/"low"}
"pages" is ranked most-relevant-first, length 1 to 5. No markdown, no commentary."""


def locate_page(client, item, week_pages, top_k=5):
    """
    Ask the LLM for up to top_k slides (ranked) in the question's week that
    support the answer. Returns (pages, confidence) where pages is a ranked list
    of valid page numbers, most relevant first. Falls back to [] if the week has
    no loaded slides. This mirrors the retriever returning top-5 chunks: the gold
    set is the top-5 pages those chunks should come from.
    """
    if not week_pages:
        return [], "none"

    # Build a compact numbered slide listing (trim each slide to keep prompt small)
    listing = "\n".join(
        f"[Slide {p['page_number']}] {p['content'][:350]}"
        for p in week_pages
    )
    user = (
        f"QUESTION:\n{item['q']}\n\n"
        f"CORRECT ANSWER:\n{item['answer']}\n\n"
        f"SLIDES (Week {week_pages[0]['week_number']}):\n{listing}\n\n"
        f"Return only the JSON object."
    )
    reply = client.chat.completions.create(
        model=MODEL, temperature=0.1, max_tokens=120,
        messages=[{"role": "system", "content": PAGE_LOCATE_SYSTEM},
                  {"role": "user", "content": user}],
    )
    result = extract_json_object(reply.choices[0].message.content)
    raw = result.get("pages", []) or []
    conf = result.get("confidence", "medium")

    # keep only valid pages, preserve rank order, dedupe, cap at top_k
    valid = {p["page_number"] for p in week_pages}
    pages, seen = [], set()
    for pp in raw:
        if pp in valid and pp not in seen:
            pages.append(pp)
            seen.add(pp)
        if len(pages) == top_k:
            break
    if not pages:                                # fallback, flagged by low conf
        pages = [week_pages[0]["page_number"]]
        conf = "low"
    return pages, conf


SOCRATIC_HINT_SYSTEM = """ROLE
You write teaching HINTS for COMP64702 (NLP / LLMs / RAG).

You are given an exam QUESTION and its correct ANSWER. Do NOT change them.

TASK
Decide whether the question can be taught by hinting (conceptual "why/how/what"
questions can; pure calculation and bare recall cannot). If it can, write ONE
short HINT.

WHAT A HINT IS (important)
- The hint is the TARGET that a small tutor model's response will be checked
  against. It is NOT a polished Socratic question.
- It should state, in one sentence, the idea or direction a good tutoring
  response must nudge the student toward - WITHOUT revealing the answer.
- Keep it short and concrete: "Point the student toward X / get them to consider Y."

OUTPUT
Return ONLY a JSON object:
{"supports_socratic": true/false,
 "socratic_gold_standard": "<hint, or empty string>",
 "difficulty": "easy"/"medium"/"hard"}
No markdown, no commentary."""


def compose_query(item):
    """Use the worked example from the exam question when the item provides one
    (the question's given hypothesis); otherwise keep the question unchanged.

    The rule is data-driven: transcribe an `example` field into an EXAM item
    ONLY when the example genuinely fits that question. If it doesn't fit, leave
    the field out and the original question text is used as-is.
    """
    example = (item.get("example") or "").strip()
    q = item["q"].strip()
    return f"{q}\n\nExample:\n{example}" if example else q


def make_hint(client, item, question=None):
    """Ask the LLM for a Socratic HINT + difficulty for one exam question."""
    user = (f"QUESTION:\n{question or item['q']}\n\nCORRECT ANSWER:\n{item['answer']}\n\n"
            f"Topic: {item['topic']}. Return only the JSON object.")
    reply = client.chat.completions.create(
        model=MODEL, temperature=0.3, max_tokens=200,
        messages=[{"role": "system", "content": SOCRATIC_HINT_SYSTEM},
                  {"role": "user", "content": user}],
    )
    return extract_json_object(reply.choices[0].message.content)


# ===========================================================================
#  PART 3:  Build
# ===========================================================================

def build():
    client = make_nebius_client()
    pages_by_week = load_all_pages_by_week(LECTURE_PATHS)
    print(f"Loaded slides for weeks: {sorted(pages_by_week)}\n")
    check_corpus_loaded(pages_by_week)

    # Flat lookup so we can attach the grounded PAGE TEXT to each entry.
    # The content is already loaded, so this needs no extra API calls.
    page_index = {
        (p["week_number"], p["page_number"]): p["content"]
        for pages in pages_by_week.values() for p in pages
    }

    # Summarise each week ONCE, so we can pick the right week per question.
    print("Summarising each week's topics (once)...")
    week_summaries = {}
    for wk, pages in sorted(pages_by_week.items()):
        try:
            week_summaries[wk] = summarise_week(client, wk, pages)
        except Exception as error:
            print(f"  Week {wk}: summary failed ({error})")
            week_summaries[wk] = ""
    print("Done.\n")

    combined = []
    for i, item in enumerate(EXAM, 1):
        # 1) DERIVE the week from the data (not hand-assigned)
        try:
            week, week_conf = pick_week(client, item, week_summaries)
        except Exception as error:
            print(f"[{i}/{len(EXAM)}] pick-week ERROR: {error}")
            week, week_conf = min(pages_by_week), "low"

        week_pages = pages_by_week.get(week, [])

        # 2) locate up to the TOP-5 source pages within that week (ranked)
        try:
            pages, page_conf = locate_page(client, item, week_pages)
        except Exception as error:
            print(f"[{i}/{len(EXAM)}] page-locate ERROR: {error}")
            pages = [week_pages[0]["page_number"]] if week_pages else []
            page_conf = "low"
        page = pages[0] if pages else None       # primary anchor (most relevant)
        also = pages[1:]                         # the rest of the top-5

        # 3) compose the stored question (folds in the example only if present)
        q_text = compose_query(item)

        # 4) hint + difficulty (calculation questions get no Socratic hint)
        try:
            ann = make_hint(client, item, q_text)
        except Exception as error:
            print(f"[{i}/{len(EXAM)}] hint ERROR: {error} -- QA-only")
            ann = {"supports_socratic": False, "socratic_gold_standard": "",
                   "difficulty": "medium"}

        socratic = (ann.get("socratic_gold_standard") or "").strip()
        supports = (item["kind"] != "calculation"
                    and bool(ann.get("supports_socratic")) and bool(socratic))

        # overall confidence in the retrieval anchor = the weaker of the two steps
        rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
        anchor_conf = min([week_conf, page_conf], key=lambda c: rank.get(c, 0))

        # GROUNDED TEXT: the actual lecture page(s) the answer traces back to.
        # This is what makes each entry inspectable and lets faithfulness /
        # context metrics score against a real reference, not just a page number.
        # GROUNDED TEXT for the full TOP-5 gold page set (ranked, most relevant
        # first), each page carrying its own text. A retrieved chunk is a hit if
        # its page is any of these. gold_context stays as the single primary
        # page's text for quick inspection / faithfulness scoring.
        gold_pages = [
            {"rank": r, "page_number": pp, "content": page_index.get((week, pp), "")}
            for r, pp in enumerate(pages, 1)
        ]
        gold_context = gold_pages[0]["content"] if gold_pages else ""
        missing = [gp["page_number"] for gp in gold_pages if not gp["content"]]
        if missing:                                          # should not happen; flag if it does
            page_conf = "low"
            print(f"    (warning: no text for pages {missing}; anchor flagged low)")

        combined.append({
            "query_id": f"EXAM_{i:03d}",
            "week_number": week,                          # DERIVED retrieval anchor
            "page_number": page,                          # primary anchor (rank-1 page)
            "also_pages": also,                           # rest of the top-5 pages
            "week_confidence": week_conf,
            "page_confidence": page_conf,
            "anchor_confidence": anchor_conf,             # spot-check low ones
            "question_type": item["kind"],
            "difficulty": ann.get("difficulty", "medium"),
            "modes": ["qa", "socratic"] if supports else ["qa"],
            "student_query": q_text,
            "gold_context": gold_context,                 # grounded text (primary page)
            "gold_pages": gold_pages,                     # grounded text for ALL top-5 pages
            "qa_gold_standard": item["answer"],
            "socratic_gold_standard": socratic,
            "verified": False,
        })
        print(f"[{i}/{len(EXAM)}] {combined[-1]['query_id']} "
              f"week={week}({week_conf}) pages={pages}({page_conf}) "
              f"modes={combined[-1]['modes']}")

    # split into the two aligned files (both carry week + page)
    qa, soc = [], []
    for e in combined:
        shared = {
            "query_id": e["query_id"],
            "week_number": e["week_number"],
            "page_number": e["page_number"],
            "also_pages": e["also_pages"],
            "anchor_confidence": e["anchor_confidence"],
            "question_type": e["question_type"],
            "difficulty": e["difficulty"],
            "student_query": e["student_query"],
            "gold_context": e["gold_context"],            # primary page text (both files)
            "gold_pages": e["gold_pages"],                # top-5 pages + text (both files)
            "verified": False,
        }
        qa.append({**shared, "qa_gold_standard": e["qa_gold_standard"]})
        if "socratic" in e["modes"]:
            soc.append({**shared,
                        "socratic_gold_standard": e["socratic_gold_standard"]})

    os.makedirs(os.path.dirname(QA_OUTPUT) or ".", exist_ok=True)
    json.dump(qa, open(QA_OUTPUT, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump(soc, open(SOCRATIC_OUTPUT, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    low = [e["query_id"] for e in combined if e["anchor_confidence"] in ("low", "none")]
    print(f"\nDone.")
    print(f"  QA file:       {QA_OUTPUT}  ({len(qa)} questions)")
    print(f"  Socratic file: {SOCRATIC_OUTPUT}  ({len(soc)} questions)")
    if low:
        print(f"Low-confidence anchors to double-check: {low}")
    print("NEXT: spot-check the derived week/page for low-confidence items and "
          "skim the Socratic hints.")


if __name__ == "__main__":
    build()