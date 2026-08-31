"""
Running every stage of inference in one go.

The numbered scripts each demonstrate one part of the system and each writes
its own JSON record. Run separately they have to be invoked once per
strategy, once per model and once per question, and the stage numbering they
use predates the router: 4_routing.py has no runner of its own at all. This
script drives all five stages from one command and renumbers them so routing
is stage 1, which is where it actually happens:

    stage 1  routing              4_routing.py            (driven from here)
    stage 2  vector search        5-vector_search.py      run_stage()
    stage 3  hybrid + retrieval   6-ranking_n_retrieval.py run_stage()
    stage 4  LLM + prompt         7-llm_n_prompt.py       run_stage()
    stage 5  whole pipeline       8_RAG_pipeline.py       via run_inference_one_question

Nothing here reimplements a stage. Each stage is the same run_stage the
numbered script calls, imported and invoked, so the automated records and the
hand-run ones cannot disagree. The only thing this script changes is where the
records are written and what they call themselves: each module's STAGE and
STAGE_DIR are rebound to the new numbering before it is called, and put back
afterwards.

The stages are also chained, not merely run in sequence. Stage 3 records the
stage 2 file it followed, and stage 4 answers from exactly the chunks stage 3
chose rather than searching again, so a bad answer can be traced back to the
chunk that caused it.

Run it:
    python Inference_pipeline_automation.py
    python Inference_pipeline_automation.py --stages 1 2 3
    python Inference_pipeline_automation.py --strategies exp1 --llms 0.5b
    python Inference_pipeline_automation.py --questions "What is BM25?"
    python Inference_pipeline_automation.py --retrieval-only     # stages 1-3
    python Inference_pipeline_automation.py --overwrite

Retrieval runs equal-context by default: chunks are cut at a 500-token budget
in rank order rather than at a fixed top_n, which is the condition the
benchmark in Results_generation_bigfile/ was run under. --budget-tokens 0
switches back to fixed top-k.

Writes
    Data/Results_stage1/  routing         <slug>.json
    Data/Results_stage2/  vector search   <strategy>_<slug>.json
    Data/Results_stage3/  retrieval       <strategy>_<combo>_<slug>.json
    Data/Results_stage4/  generation      <strategy>_<model>_<slug>.json
    Data/Results_stage5/  pipeline        <strategy>_<model>_<slug>.json
    Data/Results_stage5/_automation_manifest.json

Stages 4 and 5 load a language model, so a full grid of three strategies by
three models by three questions is a long job. It is resumable: a cell whose
record already exists is skipped unless --overwrite is given, and a cell that
fails is recorded in the manifest and does not stop the rest.

Because it is a long job, it reports progress at the three timescales it
stalls on: a bar over the cells of each stage, a clock while a model's
weights load, and a bar over the tokens of each answer as it is written. The
token bar is the important one — a single answer on CPU is 30 to 100 seconds
during which nothing else moves. tqdm is optional; without it the bars become
no-ops and everything else behaves identically. --no-progress turns them off,
which is what you want when piping the log to a file.

Defaults come from Share_components/configuration.py.
"""

import argparse
import importlib
import importlib.util
import io
import json
import sys
import threading
import time
import traceback
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The root, for Share_components and Ingestion_pipeline; this folder, so the
# sibling modules import by bare name however this one was reached. Same two
# lines every module in this folder starts with, for the same reason.
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.append(_p)


# ---------------------------------------------------------------------------
# Loading the numbered modules
# ---------------------------------------------------------------------------
# The scripts import each other by bare name ("from vector_search import
# VectorSearch") but are filed with numeric prefixes ("5-vector_search.py"),
# which is not an importable module name: the hyphen alone makes it illegal.
# So each file is loaded from its path and registered in sys.modules under the
# bare name the others expect, in dependency order, before anything imports
# anything. Registering before executing matters — ranking_n_retrieval imports
# vector_search while it is still being executed itself.
#
# A file already sitting there unprefixed is used as-is. Both layouts work,
# which is the point: this script does not require the folder to be renamed.
MODULE_FILES = (
    # bare name              candidate filenames, in order of preference
    ("routing",              ("routing.py", "4_routing.py", "4-routing.py")),
    ("vector_search",        ("vector_search.py", "5-vector_search.py",
                              "5_vector_search.py")),
    ("ranking_n_retrieval",  ("ranking_n_retrieval.py", "6-ranking_n_retrieval.py",
                              "6_ranking_n_retrieval.py")),
    ("llm_n_prompt",         ("llm_n_prompt.py", "7-llm_n_prompt.py",
                              "7_llm_n_prompt.py")),
    ("RAG_pipeline",         ("RAG_pipeline.py", "rag_pipeline.py",
                              "8_RAG_pipeline.py", "8-RAG_pipeline.py")),
    ("run_inference_one_question", ("run_inference_one_question.py",)),
)


def _load_one(name, filenames):
    """Import one sibling by bare name, whatever its file is called."""
    if name in sys.modules:
        return sys.modules[name]

    # An unprefixed file is importable normally; let Python find it so that a
    # folder which has already been renamed behaves exactly as it would
    # without this script.
    try:
        return importlib.import_module(name)
    except ImportError:
        pass

    for filename in filenames:
        path = HERE / filename
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        # In sys.modules before execution: the module's own imports of its
        # siblings resolve against this table, and a circular import would
        # otherwise load a second copy.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
        return module

    raise SystemExit(
        f"Cannot find the module '{name}'. Looked for {', '.join(filenames)} "
        f"in {HERE}.\nRun this script from inside the Inference_pipeline "
        f"folder, with the numbered scripts alongside it.")


def load_modules():
    """All the stage modules, in dependency order. Returns them by bare name."""
    loaded = {}
    for name, filenames in MODULE_FILES:
        loaded[name] = _load_one(name, filenames)
    # RAG_pipeline is written with a capital in the imports and lower case in
    # __pycache__, so both spellings are made to point at the one module. Two
    # copies of the pipeline class would compare unequal and quietly load the
    # models twice.
    sys.modules.setdefault("rag_pipeline", loaded["RAG_pipeline"])
    return loaded


MODULES = load_modules()

routing = MODULES["routing"]
vector_search = MODULES["vector_search"]
ranking_n_retrieval = MODULES["ranking_n_retrieval"]
llm_n_prompt = MODULES["llm_n_prompt"]
RAG_pipeline = MODULES["RAG_pipeline"]
one_question = MODULES["run_inference_one_question"]

from Share_components import configuration as config_inference  # noqa: E402


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
# Three bars — over cells, over a model's loading time, over the tokens of an
# answer — all of which run_inference_bigfile.py needs too, so they live in
# pipeline_progress rather than in either entry point.
from pipeline_progress import (HAVE_TQDM, cell_bar, elapsed_bar,  # noqa: E402
                               free_model, install_generation_progress,
                               prints_below_bars, set_enabled)


# ---------------------------------------------------------------------------
# The renumbering
# ---------------------------------------------------------------------------
# The numbered scripts were written before the router had a stage of its own,
# so their internal numbering runs 1-4 with routing nowhere in it. This script
# reports 1-5 with routing first. Rather than edit five files, each module's
# STAGE and STAGE_DIR are rebound around the call: both are read at call time
# from the module globals, so rebinding them redirects the record without
# touching the code that writes it.
#
# Restored afterwards in a finally, so a module left imported in a notebook or
# a test still writes where its own docstring says it does.
DATA = config_inference.ROOT / "Data"

STAGE_LABELS = {
    1: "1-routing",
    2: "2-vector-search",
    3: "3-ranking-retrieval",
    4: "4-llm-prompt",
    5: "5-pipeline",
}

STAGE_DIRS = {number: DATA / f"Results_stage{number}" for number in STAGE_LABELS}


@contextmanager
def stage_renumbered(module, number):
    """Point one module's records at its new stage number, then put it back."""
    had_stage = hasattr(module, "STAGE")
    had_dir = hasattr(module, "STAGE_DIR")
    old_stage = getattr(module, "STAGE", None)
    old_dir = getattr(module, "STAGE_DIR", None)

    module.STAGE = STAGE_LABELS[number]
    module.STAGE_DIR = STAGE_DIRS[number]
    try:
        yield STAGE_DIRS[number]
    finally:
        if had_stage:
            module.STAGE = old_stage
        else:
            del module.STAGE
        if had_dir:
            module.STAGE_DIR = old_dir
        else:
            del module.STAGE_DIR


def slugify(question):
    """The filename slug, exactly as every stage computes it.

    Copied rather than imported because each stage inlines the same expression
    in its own writer. If they ever diverge, the file this script predicts and
    the file the stage writes must diverge with them, so the duplication is
    the honest version.
    """
    return "".join(c if c.isalnum() else "_" for c in question.lower()).strip("_")[:40]


def pipeline_model_tag(model_id):
    """The model label run_inference_one_question puts in its filename."""
    return (model_id or "nollm").split("/")[-1].replace(".", "").lower()


# ---------------------------------------------------------------------------
# What to run
# ---------------------------------------------------------------------------
# Three questions, chosen so the three routes the system can take are all
# exercised. One question would only ever show the third.
#
#   a covered question    retrieval finds it, the model answers from it
#   an uncovered one      retrieval returns its nearest neighbours anyway,
#                         and the answer should be the refusal sentence
#   a meta question       the router answers it without searching at all
#
# The last one is the reason routing is stage 1 rather than a footnote: it is
# the only stage whose behaviour it changes visibly.
DEFAULT_QUESTIONS = (
    "What is the Transformer?",
    "When is the exam?",
    "What can you do?",
)

DEFAULT_STRATEGIES = ("exp1", "exp2", "exp3")
DEFAULT_LLMS = ("360m", "0.5b", "1b")

# The demo runs equal-context, not fixed top-k. The retriever supports both —
# a budget makes it cut by total tokens and ignore top_n — and the benchmark
# in Results_generation_bigfile/ was run at 500 and 1500. 500 is the default
# here because it is the cheaper of the two and these records are an
# illustration rather than a result: the point is to show the mechanism the
# reported numbers were produced under, not to reproduce them.
#
# It also keeps the stage records honest about the system. Run at top_n=5
# they would show a cut the benchmark never used, and the settings block
# would be the only place saying so. Pass --budget-tokens 0 for top-k.
DEMO_BUDGET_TOKENS = 500


class Ledger:
    """What ran, what was skipped, what failed, and where each record went.

    Kept as one object so the manifest and the closing summary cannot report
    different things, and so a run that dies halfway still has a written
    account of everything before it.
    """

    def __init__(self):
        self.rows = []
        self.started = time.perf_counter()

    def add(self, stage, label, status, path=None, seconds=None, note=None):
        self.rows.append({
            "stage": stage,
            "label": label,
            "status": status,          # written | skipped | failed
            "path": str(path) if path else None,
            "seconds": round(seconds, 2) if seconds is not None else None,
            "note": note,
        })

    def counts(self, status):
        return [row for row in self.rows if row["status"] == status]

    def by_stage(self, stage):
        return [row for row in self.rows if row["stage"] == stage]


def announce(number, what):
    title = f"STAGE {number}  {what}"
    print(f"\n\n{'=' * 78}\n{title}\n{'=' * 78}")


def cell(ledger, stage, label, path, overwrite, work):
    """Run one cell of the grid: skip it, do it, or record why it failed.

    Every stage goes through here so all three outcomes are handled the same
    way. A failure in one cell must not lose the cells already written — a
    grid of 27 generation runs is an hour of work.
    """
    if path is not None and path.exists() and not overwrite:
        print(f"  already done: {label}  ({path.name})")
        ledger.add(stage, label, "skipped", path)
        return True

    started = time.perf_counter()
    try:
        work()
    except KeyboardInterrupt:
        # Not swallowed: Ctrl-C should stop the whole run, not skip one cell
        # and start the next long one.
        raise
    except (SystemExit, Exception) as e:
        # SystemExit is caught by name because the encoder self-check and the
        # model availability check both raise it, and a bare `except Exception`
        # would let them kill the run.
        seconds = time.perf_counter() - started
        message = f"{type(e).__name__}: {e}".splitlines()[0]
        print(f"  !! failed: {label}: {message}")
        traceback.print_exc(limit=2)
        ledger.add(stage, label, "failed", None, seconds, message)
        return False

    seconds = time.perf_counter() - started
    ledger.add(stage, label, "written", path, seconds)
    return True


# ---------------------------------------------------------------------------
# Stage 1: routing
# ---------------------------------------------------------------------------
# routing.py is the one module with no runner of its own, because it loads no
# model and touches no store: there was nothing to demonstrate interactively.
# It still makes two decisions that change the answer, so it gets a record
# here, in the same envelope as the rest.
def run_stage1(question, save=True):
    """Show what routing decides about a question, before anything is loaded."""
    print(f"\n[1] The question")
    print(f"  question    {question!r}")

    print(f"\n[2] Meta router")
    started = time.perf_counter()
    is_meta = routing.is_meta_question(question)
    seconds = time.perf_counter() - started

    # Which pattern fired, not just that one did. A question routed as meta by
    # the wrong pattern is a bug the boolean alone hides.
    matched = [pattern.pattern for pattern in routing._META_RE
               if pattern.search((question or "").strip())]
    print(f"  patterns    {len(routing._META_RE)} checked")
    print(f"  matched     {matched if matched else 'none'}")
    print(f"  meta        {is_meta}")

    print(f"\n[3] Relevance gate")
    floor = routing.RELEVANCE_FLOOR
    combo = config_inference.COMBO
    active = routing.gate_active(floor, combo)
    print(f"  floor       {floor}")
    print(f"  combo       {combo}  (the gate needs {routing.GATE_COMBO})")
    print(f"  active      {active}")
    if floor is not None and not active:
        print(f"  the floor is set but cannot apply under {combo}: no reranker "
              f"scores to compare it against")

    print(f"\n[4] The plan this produces")
    if is_meta:
        plan = routing.meta_plan()
        decided_by = "meta router"
    else:
        # Not a refusal and not an answer yet: routing alone cannot tell.
        # The gate only decides after retrieval has scored something, which
        # is stage 3's job, so the honest record here is "passed on".
        plan = routing.Plan("answer")
        decided_by = "passed to retrieval"
    print(f"  route       {plan.route}  ({decided_by})")
    print(f"  answer      {plan.answer!r}")
    print(f"  searched    {plan.route != 'meta'}")

    settings = {
        "route_meta": True,
        "meta_patterns": len(routing.META_PATTERNS),
        "relevance_floor": floor,
        "gate_combo": routing.GATE_COMBO,
        "combo": combo,
        "gate_active": active,
    }

    record = {
        "stage": STAGE_LABELS[1],
        "settings": settings,
        "settings_hash": _hash_settings(settings),
        "input": {"question": question},
        "output": {
            "route": plan.route,
            "is_meta": is_meta,
            "matched_patterns": matched,
            "answer": plan.answer,
            "searched": plan.route != "meta",
            "decided_by": decided_by,
            "gate_can_refuse_later": active,
        },
        "seconds": round(seconds, 6),
    }

    if not save:
        print(f"\n[5] Not saved (--no-save)")
        return None

    STAGE_DIRS[1].mkdir(parents=True, exist_ok=True)
    path = STAGE_DIRS[1] / f"{slugify(question)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"\n[5] Written to {path}")
    return path


def _hash_settings(settings):
    """The same short settings code the other stages write."""
    import hashlib
    parts = [f"{key}={settings[key]}" for key in sorted(settings)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Stage 5: the whole pipeline
# ---------------------------------------------------------------------------
# The only stage that runs the assembled system rather than a component, and
# so the only one where the router and the gate are visible in the answer
# rather than described. It reuses run_inference_one_question's own functions,
# so the record is the same one that file writes by hand.
def run_stage5(question, pipeline, search, retriever, save=True):
    result = one_question.answer_question(pipeline, search, retriever, question,
                                          show_scores=True)
    if not save:
        print("\n(not saved)")
        return None, result

    settings = one_question.stage_settings(None, search, retriever, pipeline)
    hashed = one_question.stage_hash(settings, pipeline)
    path = one_question.write_record(question, result, settings, hashed)
    print(f"\nWritten to {path}")
    return path, result


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run all five inference stages and save every record.")
    parser.add_argument("--stages", nargs="+", type=int,
                        choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5],
                        help="which stages to run (default: all five)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="stages 1 to 3 only: no language model is loaded, "
                             "which takes minutes instead of an hour")
    parser.add_argument("--questions", nargs="+", default=list(DEFAULT_QUESTIONS),
                        help="the questions to put through every stage")
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES),
                        help="chunking strategies to sweep (stages 2 to 5)")
    parser.add_argument("--llms", nargs="+", default=list(DEFAULT_LLMS),
                        help=f"models to sweep (stages 4 and 5); tags from "
                             f"{', '.join(llm_n_prompt.MODELS)} or full names")
    parser.add_argument("--embedder", default=config_inference.EMBEDDER)
    parser.add_argument("--combo", default=config_inference.COMBO,
                        choices=ranking_n_retrieval.COMBOS)
    parser.add_argument("--candidates", type=int, default=None,
                        help="chunks each search method puts forward "
                             "(default: 60 under a budget, otherwise "
                             f"{config_inference.CANDIDATES})")
    parser.add_argument("--top-n", type=int, default=config_inference.TOP_N)
    parser.add_argument("--budget-tokens", type=int, default=DEMO_BUDGET_TOKENS,
                        help=f"equal-context control: fill this many tokens of "
                             f"context in rank order instead of taking a fixed "
                             f"--top-n chunks (default {DEMO_BUDGET_TOKENS}; "
                             f"pass 0 for fixed top-k instead)")
    parser.add_argument("--max-tokens", type=int,
                        default=config_inference.MAX_NEW_TOKENS)
    parser.add_argument("--week", default=None,
                        help='restrict stage 2 to one week, e.g. "Week 3"')
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--relevance-floor", type=float,
                        default=routing.RELEVANCE_FLOOR,
                        help=f"stage 5 only, {routing.GATE_COMBO} only")
    parser.add_argument("--no-router", action="store_true",
                        help="stage 5: send meta questions through retrieval")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the encoder self-check in stage 5")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo cells whose record already exists")
    parser.add_argument("--no-progress", action="store_true",
                        help="no progress bars, plain scrolling output "
                             "(use when piping the log to a file)")
    parser.add_argument("--no-save", action="store_true",
                        help="print everything, write nothing")
    args = parser.parse_args()

    progress_on = set_enabled(not args.no_progress)

    stages = set(args.stages)
    if args.retrieval_only:
        stages &= {1, 2, 3}

    # Installed before any stage runs, because both stages that load a model
    # build it themselves and there is nothing to hook afterwards.
    if stages & {4, 5}:
        install_generation_progress(args.max_tokens)

    # Fails now, not forty minutes in, when stage 5 tries to apply a floor
    # under a combo that produces no scores for it to read.
    if 5 in stages:
        routing.check_floor_supported(args.relevance_floor, args.combo)

    # 0 means "no budget, use top_n". argparse cannot express that as None
    # without a second flag, and a budget of zero tokens is meaningless
    # anyway, so it is the off switch.
    if not args.budget_tokens:
        args.budget_tokens = None

    # A budget only holds context equal if the candidate pool is deep enough
    # to fill it: exp1's small chunks need about 20 to reach 500 tokens and
    # about 60 to reach 1500, while exp2's reach either easily. Left at the
    # configured 20, a budget run would cap exp1 below budget while exp2
    # filled, reintroducing the very imbalance the budget removes. So the
    # default follows the budget rather than the other way round; an explicit
    # --candidates still wins.
    budget_candidates = getattr(config_inference, "BUDGET_CANDIDATES", 60)
    if args.candidates is None:
        args.candidates = (budget_candidates if args.budget_tokens
                           else config_inference.CANDIDATES)

    # Both 6-ranking_n_retrieval.py and build_pipeline warn about an
    # underfilled budget, but the first does it in its __main__ block, which
    # calling run_stage directly walks straight past — so stage 3 would have
    # run underfilled and silently. Checked once here instead, before anything
    # loads. Only reachable now by asking for it explicitly.
    if args.budget_tokens and args.candidates < budget_candidates:
        print(f"\n  [warning] --budget-tokens {args.budget_tokens} with only "
              f"{args.candidates} candidates. Strategies with small chunks may "
              f"not reach the budget, and the comparison stops being "
              f"equal-context.\n            Suggested: --candidates "
              f"{budget_candidates}")
        print(f"            Check budget_underfilled in the stage 3 records "
              f"afterwards; anything above zero means it happened.")

    save = not args.no_save
    ledger = Ledger()

    print(f"{'=' * 78}")
    print(f"INFERENCE PIPELINE, STAGES {sorted(stages)}")
    print(f"{'=' * 78}")
    print(f"  questions   {len(args.questions)}: "
          f"{', '.join(repr(q) for q in args.questions)}")
    print(f"  strategies  {args.strategies}")
    print(f"  models      {args.llms if stages & {4, 5} else '(not needed)'}")
    cut = (f"budget_tokens={args.budget_tokens}" if args.budget_tokens
           else f"top_n={args.top_n}")
    print(f"  fixed       embedder={args.embedder} combo={args.combo} "
          f"candidates={args.candidates} {cut}")
    print(f"  condition   {'equal-context (budget)' if args.budget_tokens else 'equal-k (fixed top_n)'}")
    print(f"  router      {not args.no_router}   relevance_floor={args.relevance_floor}")
    print(f"  writing to  {DATA}/Results_stage{{{','.join(str(s) for s in sorted(stages))}}}/")
    if not save:
        print(f"  --no-save   nothing will be written")
    if not progress_on and not args.no_progress:
        print(f"  [note] tqdm is not installed, so there are no progress bars. "
              f"Install it with: pip install tqdm")

    # -- Stage 1 ------------------------------------------------------------
    # No model, no store, so it runs in milliseconds and is worth doing first:
    # it says which questions the router will divert before anything slow is
    # loaded on their behalf.
    if 1 in stages:
        # No bar: this stage loads nothing and finishes before a bar would
        # have drawn itself once.
        announce(1, "Routing — the two decisions made around retrieval")
        for question in args.questions:
            path = STAGE_DIRS[1] / f"{slugify(question)}.json" if save else None
            print(f"\n{'-' * 78}\nQ: {question}\n{'-' * 78}")
            cell(ledger, 1, question, path, args.overwrite,
                 lambda q=question: run_stage1(q, save=save))

    # -- Stage 2 ------------------------------------------------------------
    if 2 in stages:
        announce(2, "Vector search — the question as a vector, nearest chunks")
        total = len(args.strategies) * len(args.questions)
        with stage_renumbered(vector_search, 2), \
                cell_bar(total, "stage 2  vector search") as progress, \
                prints_below_bars():
            for strategy in args.strategies:
                for question in args.questions:
                    label = f"{strategy} | {question}"
                    progress.set_postfix_str(label)
                    path = (STAGE_DIRS[2] / f"{strategy}_{slugify(question)}.json"
                            if save else None)
                    print(f"\n{'-' * 78}\n{label}\n{'-' * 78}")
                    cell(ledger, 2, label, path, args.overwrite,
                         lambda s=strategy, q=question: vector_search.run_stage(
                             q, s, args.embedder, args.top_n, args.week, save=save))
                    progress.update(1)

    # -- Stage 3 ------------------------------------------------------------
    # Chained to stage 2 by file: the record says which stage 2 run it
    # followed, so the two can be read as one hand-off rather than two
    # unrelated searches that happened to use the same question.
    if 3 in stages:
        announce(3, "Hybrid search and retrieval — dense, sparse, merge, rerank, cut")
        total = len(args.strategies) * len(args.questions)
        with stage_renumbered(ranking_n_retrieval, 3), \
                cell_bar(total, "stage 3  retrieval") as progress, \
                prints_below_bars():
            for strategy in args.strategies:
                for question in args.questions:
                    label = f"{strategy} | {args.combo} | {question}"
                    progress.set_postfix_str(label)
                    slug = slugify(question)
                    path = (STAGE_DIRS[3] / f"{strategy}_{args.combo}_{slug}.json"
                            if save else None)
                    upstream = STAGE_DIRS[2] / f"{strategy}_{slug}.json"
                    from_stage2 = str(upstream) if upstream.exists() else None
                    print(f"\n{'-' * 78}\n{label}\n{'-' * 78}")
                    if from_stage2:
                        print(f"  following   {from_stage2}")
                    cell(ledger, 3, label, path, args.overwrite,
                         lambda s=strategy, q=question, u=from_stage2:
                             ranking_n_retrieval.run_stage(
                                 q, s, args.embedder, args.combo,
                                 args.candidates, args.top_n, args.budget_tokens,
                                 from_stage1=u, save=save))
                    progress.update(1)

    # -- Stage 4 ------------------------------------------------------------
    # Answers from the chunks stage 3 wrote, when there is a record to follow.
    # That is the whole point of chaining: a wrong answer here is either the
    # model's fault or retrieval's, and reading stage 3's record settles which
    # without running anything again.
    if 4 in stages:
        announce(4, "LLM and prompt — build the prompt, generate, classify")
        total = len(args.strategies) * len(args.llms) * len(args.questions)
        with stage_renumbered(llm_n_prompt, 4), \
                cell_bar(total, "stage 4  generation") as progress, \
                prints_below_bars():
            for strategy in args.strategies:
                for llm_name in args.llms:
                    for question in args.questions:
                        label = f"{strategy} | {llm_name} | {question}"
                        progress.set_postfix_str(label)
                        slug = slugify(question)
                        tag = llm_n_prompt.model_tag(llm_name)
                        path = (STAGE_DIRS[4] / f"{strategy}_{tag}_{slug}.json"
                                if save else None)
                        upstream = (STAGE_DIRS[3] /
                                    f"{strategy}_{args.combo}_{slug}.json")
                        from_stage3 = str(upstream) if upstream.exists() else None
                        print(f"\n{'-' * 78}\n{label}\n{'-' * 78}")
                        cell(ledger, 4, label, path, args.overwrite,
                             lambda s=strategy, m=llm_name, q=question,
                                    u=from_stage3: llm_n_prompt.run_stage(
                                 question=q,
                                 strategy=s,
                                 embedder=args.embedder,
                                 combo=args.combo,
                                 candidates=args.candidates,
                                 top_n=args.top_n,
                                 budget_tokens=args.budget_tokens,
                                 llm_name=m,
                                 max_tokens=args.max_tokens,
                                 greedy=True,
                                 from_stage2=u,
                                 save=save))
                        # run_stage built a model and dropped it; this is what
                        # actually returns the memory before the next one.
                        free_model()
                        progress.update(1)

    # -- Stage 5 ------------------------------------------------------------
    # The assembled system. Built once per strategy and model and reused for
    # every question: loading the models takes far longer than answering, so
    # rebuilding per question would be mostly waiting.
    if 5 in stages:
        announce(5, "The whole pipeline — router, search, gate, generation")
        total = len(args.strategies) * len(args.llms) * len(args.questions)
        with stage_renumbered(one_question, 5), \
                cell_bar(total, "stage 5  pipeline") as progress, \
                prints_below_bars():
            for strategy in args.strategies:
                for llm_name in args.llms:
                    model_id = llm_n_prompt.resolve_model(llm_name)
                    tag = pipeline_model_tag(model_id)
                    label = f"{strategy} | {llm_name}"
                    progress.set_postfix_str(label)

                    # Every question for this pair already answered: skip the
                    # build too, not just the answering. Loading a model to
                    # write nothing is the slowest possible no-op.
                    expected = [STAGE_DIRS[5] / f"{strategy}_{tag}_{slugify(q)}.json"
                                for q in args.questions]
                    if save and not args.overwrite and all(p.exists() for p in expected):
                        print(f"\n  already done: {label} (all "
                              f"{len(expected)} questions)")
                        for question, path in zip(args.questions, expected):
                            ledger.add(5, f"{label} | {question}", "skipped", path)
                        progress.update(len(expected))
                        continue

                    print(f"\n{'-' * 78}\n{label}: loading\n{'-' * 78}")
                    try:
                        # The embedder, the reranker and the generator all load
                        # here, none of them reporting anything, and on a first
                        # run they may be downloading. The clock is the only
                        # thing distinguishing that from a hang.
                        with elapsed_bar(f"building {label}"):
                            pipeline, search, retriever = RAG_pipeline.build_pipeline(
                                strategy=strategy,
                                combo=args.combo,
                                embedder=args.embedder,
                                llm_name=llm_name,
                                candidates=args.candidates,
                                top_n=args.top_n,
                                budget_tokens=args.budget_tokens,
                                device=args.device,
                                load_llm=True,
                                verify=not args.no_verify,
                                route_meta=not args.no_router,
                                relevance_floor=args.relevance_floor,
                            )
                    except KeyboardInterrupt:
                        raise
                    except (SystemExit, Exception) as e:
                        message = f"{type(e).__name__}: {e}".splitlines()[0]
                        print(f"  !! failed to build {label}: {message}")
                        for question in args.questions:
                            ledger.add(5, f"{label} | {question}", "failed",
                                       None, None, message)
                        progress.update(len(args.questions))
                        continue

                    for question, path in zip(args.questions, expected):
                        progress.set_postfix_str(f"{label} | {question}")
                        cell(ledger, 5, f"{label} | {question}",
                             path if save else None, args.overwrite,
                             lambda q=question: run_stage5(
                                 q, pipeline, search, retriever, save=save))
                        progress.update(1)

                    # Dropped before the next pair is built, so two models are
                    # never held in memory at once.
                    del pipeline, search, retriever
                    free_model()

    # -- What happened ------------------------------------------------------
    summarise(ledger, args, stages, save)


def summarise(ledger, args, stages, save):
    elapsed = time.perf_counter() - ledger.started
    print(f"\n\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"  {'stage':<24}{'written':<10}{'skipped':<10}{'failed':<10}")
    for number in sorted(stages):
        rows = ledger.by_stage(number)
        written = sum(1 for r in rows if r["status"] == "written")
        skipped = sum(1 for r in rows if r["status"] == "skipped")
        failed = sum(1 for r in rows if r["status"] == "failed")
        print(f"  {STAGE_LABELS[number]:<24}{written:<10}{skipped:<10}{failed:<10}")

    minutes, seconds = divmod(elapsed, 60)
    print(f"\n  {int(minutes)}m {seconds:04.1f}s in total")

    failures = ledger.counts("failed")
    if failures:
        print(f"\n  {len(failures)} cell(s) failed:")
        for row in failures:
            print(f"    stage {row['stage']}  {row['label']}: {row['note']}")
        print(f"  The rest were written. Fix these and re-run: finished cells "
              f"are skipped, so only the failures are redone.")

    if ledger.counts("skipped"):
        print(f"\n  {len(ledger.counts('skipped'))} cell(s) were already done. "
              f"Use --overwrite to redo them.")

    if not save:
        return

    # The manifest, so the results directory is readable without this log.
    # A directory whose provenance lives only in someone's terminal history is
    # how two prompt versions end up pooled in one comparison.
    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": {str(n): STAGE_LABELS[n] for n in sorted(stages)},
        "settings": {
            "questions": args.questions,
            "strategies": args.strategies,
            "llms": args.llms if stages & {4, 5} else [],
            "embedder": args.embedder,
            "combo": args.combo,
            "candidates": args.candidates,
            "top_n": args.top_n,
            "budget_tokens": args.budget_tokens,
            "max_new_tokens": args.max_tokens,
            "week": args.week,
            "route_meta": not args.no_router,
            "relevance_floor": args.relevance_floor,
            "prompt_format": llm_n_prompt.PromptTemplate.FORMAT_VERSION,
        },
        "seconds_total": round(elapsed, 1),
        "records": ledger.rows,
    }

    out_dir = STAGE_DIRS[max(stages)] if stages else DATA
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "_automation_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n  Manifest written to {path}")


if __name__ == "__main__":
    main()
