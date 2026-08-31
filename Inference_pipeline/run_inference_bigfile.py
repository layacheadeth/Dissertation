"""
Running the whole benchmark and saving the answers.

Run it:
    python run_inference_bigfile.py --benchmark questions.json
    python run_inference_bigfile.py --benchmark questions.json --grid
    python run_inference_bigfile.py --benchmark questions.json --strategy exp3 --llm 1b

--grid runs every chunking strategy against every model, one after another,
which is a long job. It saves progress as it goes and picks up where it left
off if it is interrupted, so you can stop it and restart it safely.

This file only handles the benchmark side: loading questions, timing, saving
progress, saving results. Building the system is RAG_pipeline.build_pipeline's
job and is imported, never copied. Nothing is imported from the interactive
tool any more: both files sit on top of RAG_pipeline instead.

It does not score anything. It writes the answers and the chunks they came
from; the scoring is a separate step.
"""

import argparse
import hashlib
import itertools
import json
import os
import time

# RAG_pipeline puts this folder on sys.path when it loads, which is what makes
# the two sibling imports below work. Import it first.
from RAG_pipeline import STRATEGIES, add_pipeline_flags, build_pipeline, get_device
from llm_n_prompt import MODELS, PromptTemplate, model_tag, resolve_model

# Three bars: over the grid, over the questions in a cell, over the tokens of
# an answer. Shared with Inference_pipeline_automation.py so the two report
# the same way, and a no-op when tqdm is not installed.
from pipeline_progress import (cell_bar, elapsed_bar, free_model,
                               install_generation_progress, make_bar,
                               prints_below_bars, set_enabled)

from Share_components import configuration as config_inference

# The question and its id can be under any of these names, depending on how the
# benchmark file was written.
QUESTION_FIELDS = ("student_query", "query", "question")
ID_FIELDS = ("query_id", "id", "qid")


def first_present(record, field_names):
    """The first of these fields that actually has a value."""
    for name in field_names:
        if record.get(name) not in (None, ""):
            return record[name]
    return None


def load_questions(path):
    """Read the benchmark file. Accepts a plain list or a wrapped one."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("queries", "benchmark", "questions", "results"):
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"{path}: could not find a list of questions")
    return data


def describe_chunk(doc, score=None):
    """One retrieved chunk, ready to save.

    The reranker score is saved when there is one. That is what makes the
    relevance floor tunable offline: with the scores in the file you can
    re-derive the refusal rate at any threshold from a single expensive run,
    instead of re-running the whole benchmark once per candidate value.
    """
    meta = dict(getattr(doc, "metadata", {}) or {})
    chunk_id = str(meta.get("chunk_id", "?"))
    row = {"chunk_id": chunk_id, "text": doc.page_content, "metadata": meta}
    if score is not None:
        row["score"] = round(float(score), 4)
    return row


# ---------------------------------------------------------------------------
# Recognising a saved run
# ---------------------------------------------------------------------------
def settings_hash(strategy, embedder, combo, model_id, candidates, top_n,
                  benchmark_path, limit=None, route_meta=True,
                  relevance_floor=None, budget_tokens=None):
    """A short code identifying the settings that produced a result file.

    Saved answers can only be reused if the system that produced them is the
    same one running now. Without this, editing the prompt and re-running would
    give you a results file quietly mixing two versions of the system, which is
    invisible in the file and only shows up when the numbers make no sense.

    It covers everything that changes an answer or the set of questions asked:
    the settings, the model, the prompt text, the benchmark file, and --limit.
    It deliberately leaves out the device, so moving a half-finished run from
    laptop to GPU resumes instead of starting over.
    """
    prompt = PromptTemplate()
    try:
        benchmark_stamp = str(int(os.path.getmtime(benchmark_path)))
    except OSError:
        benchmark_stamp = "?"

    parts = [strategy, embedder, combo, model_id, str(candidates), str(top_n),
             str(limit), os.path.basename(benchmark_path), benchmark_stamp,
             prompt.qa_system, prompt.qa_few_shot,
             str(route_meta), str(relevance_floor), str(budget_tokens)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def context_tokens(documents):
    """Total tokens of context handed to the model for one question.

    Measured with the BGE tokenizer, the same one that measured the chunks at
    ingestion, so budget numbers here and MAX_TOKENS there are the same unit.
    """
    try:
        from Share_components.chunking_tokenizer import count_tokens
        return sum(count_tokens(d.page_content) for d in documents)
    except Exception:
        return None


def progress_path(result_path):
    """Where partial progress is kept while a run is unfinished."""
    return str(result_path) + ".partial"


def load_progress(result_path, expected_hash):
    """Answers already saved for these settings, as {question_id: answer}.

    Looks at the partial file first, then the finished one. Anything produced
    by different settings is ignored rather than reused, because reusing it
    would corrupt the results.
    """
    for path in (progress_path(result_path), result_path):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"  {os.path.basename(path)} is unreadable, ignoring it")
            continue

        if data.get("settings", {}).get("hash") != expected_hash:
            print(f"  {os.path.basename(path)} used different settings, redoing it")
            continue

        answered = {r["question_id"]: r for r in data.get("results", [])}
        return answered, float(data.get("settings", {}).get("total_seconds", 0.0))

    return {}, 0.0


def save_progress(payload, result_path):
    """Write progress safely: to a temporary file, then rename.

    Renaming is instant, so being killed mid-write cannot leave a half-written
    file that the next run would try to resume from.
    """
    path = progress_path(result_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def clear_progress(result_path):
    for path in (progress_path(result_path), progress_path(result_path) + ".tmp"):
        if os.path.exists(path):
            os.remove(path)


def already_finished(result_path, expected_hash):
    """True only if a finished file exists AND used these exact settings."""
    if not os.path.exists(result_path):
        return False
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f).get("settings", {}).get("hash") == expected_hash
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Answering a whole benchmark
# ---------------------------------------------------------------------------
def run_benchmark(questions, result_path, args, strategy, llm_name,
                  run_hash, save_every=5):
    """Answer every question with one set of settings, saving as it goes.

    run_hash is passed in rather than worked out again: main() has already
    computed it to decide whether this run is needed at all, and two
    independent computations of the same code could quietly disagree after an
    edit, which would make finished runs look unfinished.
    """
    model_id = resolve_model(llm_name)
    device = args.device or get_device()

    print(f"\n--- {strategy} | {args.embedder} | {args.combo} | {llm_name} ---")

    answered, seconds_so_far = ({}, 0.0) if args.overwrite else \
        load_progress(result_path, run_hash)

    if args.limit:
        questions = questions[:args.limit]

    def question_id(record, position):
        found = first_present(record, ID_FIELDS)
        return str(found) if found is not None else str(position)

    remaining = sum(1 for i, q in enumerate(questions)
                    if question_id(q, i) not in answered)

    if answered:
        print(f"  resuming: {len(answered)} done, {remaining} to go")

    # Nothing left to do, so do not load the models at all. This is the whole
    # point of saving progress: reloading a model to answer zero questions
    # wastes minutes.
    if remaining == 0 and answered:
        pipeline = search = None
    else:
        # The embedder, the reranker and the generator all load here, none of
        # them reporting anything, and on a first run they may be downloading.
        # The clock is the only thing distinguishing that from a hang.
        with elapsed_bar(f"loading {strategy} | {llm_name}"):
            pipeline, search, _ = build_pipeline(
                strategy=strategy,
                combo=args.combo,
                embedder=args.embedder,
                llm_name=llm_name,
                candidates=args.candidates,
                top_n=args.top_n,
                budget_tokens=args.budget_tokens,
                device=device,
                verify=not args.no_verify,
                route_meta=not args.no_router,
                relevance_floor=args.relevance_floor,
            )

    results = []
    started = time.perf_counter()
    since_last_save = 0

    def payload():
        elapsed = seconds_so_far + (time.perf_counter() - started)
        return {
            "settings": {
                "hash": run_hash,
                "strategy": strategy,
                "embedder": args.embedder,
                "combo": args.combo,
                "llm": model_id,
                "device": device,
                "candidates": args.candidates,
                "top_n": args.top_n,
                "budget_tokens": args.budget_tokens,
                "context_tokens_mean": (
                    round(sum(r.get("context_tokens", 0) for r in results)
                          / max(len(results), 1), 1)),
                "route_meta": not args.no_router,
                "relevance_floor": args.relevance_floor,
                "collection": getattr(search, "collection_name", None),
                "routes": {name: sum(1 for r in results if r.get("route") == name)
                           for name in ("meta", "refused", "answer")},
                "n_questions": len(results),
                "total_seconds": round(elapsed, 1),
                "seconds_per_question": round(elapsed / max(len(results), 1), 2),
            },
            "results": results,
        }

    # Counted over what is left to do, not over the whole benchmark: on a
    # resumed run the questions already answered cost nothing, and counting
    # them would make the estimate say an hour when the work is ten minutes.
    question_bar = make_bar(remaining, f"  {strategy} | {llm_name}", unit="q")

    # closed in a finally: a cell that raises part-way through would
    # otherwise leave its bar stuck on screen at 83% while the grid
    # carries on underneath it.
    try:
        for position, record in enumerate(questions):
            question = str(first_present(record, QUESTION_FIELDS) or "").strip()
            if not question:
                continue

            qid = question_id(record, position)

            # Answered on an earlier attempt, so keep it, in benchmark order.
            if qid in answered:
                results.append(answered[qid])
                continue

            start = time.perf_counter()
            plan = pipeline.run(question)
            seconds = time.perf_counter() - start

            scores = plan.scores or [None] * len(plan.documents)
            results.append({
                "question_id": qid,
                "question": question,
                "answer": str(plan.answer),
                # How the answer was reached. A refusal the gate produced and a
                # refusal the model wrote are different results, and scoring them
                # as the same thing would hide which part is doing the work.
                "route": plan.route,
                "top_score": (round(plan.top_score, 4)
                              if plan.top_score is not None else None),
                "seconds": round(seconds, 3),
                # The number that shows the control did what it claims. Under a
                # budget these should be near-identical across strategies; under
                # fixed top_n they are not, which is the confound being removed.
                "context_tokens": context_tokens(plan.documents),
                "n_chunks": len(plan.documents),
                "chunks_used": [describe_chunk(d, s)
                                for d, s in zip(plan.documents, scores)],
            })
            print(f"  [{position+1}/{len(questions)}] {seconds:5.1f}s  {question[:55]}...")

            # The route is in the postfix because it is the one thing worth
            # watching live: a cell quietly refusing everything is a broken run,
            # and seeing it at question 5 saves waiting for question 60.
            question_bar.set_postfix_str(f"{plan.route} {question[:30]}")
            question_bar.update(1)

            since_last_save += 1
            if since_last_save >= save_every:
                save_progress(payload(), result_path)
                since_last_save = 0

    finally:
        question_bar.close()
    return payload()


def save_results(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    settings = payload["settings"]
    print(f"  saved {settings['n_questions']} answers to {path} "
          f"({settings['total_seconds']}s)")


def result_filename(strategy, embedder, combo, llm_name,
                    relevance_floor=None, route_meta=True, budget_tokens=None):
    """Every setting that changes the answers is in the name, so two runs
    cannot overwrite each other."""
    name = f"answers_{strategy}_{embedder}_{combo}_{model_tag(llm_name)}"
    if budget_tokens:
        # The equal-k and equal-budget conditions are the comparison. They must
        # sit side by side in the results directory, never overwrite one another.
        name += f"_budget{budget_tokens}"
    if relevance_floor is not None:
        name += f"_floor{relevance_floor:g}".replace(".", "p").replace("-", "m")
    if not route_meta:
        name += "_noroute"
    return name + ".json"


# The retrieval mode every benchmark run uses. Not a CLI choice: mixing
# retrieval modes within one results directory would make the cells
# incomparable in exactly the way mixing prompt versions did.
GRID_COMBO = "combo2"


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Answer a benchmark of questions.")
    parser.add_argument("--benchmark", required=True, help="the questions JSON file")
    add_pipeline_flags(parser)
    parser.add_argument("--out-dir", default=str(config_inference.RESULTS_DIR))
    parser.add_argument("--limit", type=int, default=None,
                        help="only answer the first N questions, for a quick test")
    parser.add_argument("--no-progress", action="store_true",
                        help="no progress bars, plain scrolling output "
                             "(use when piping the log to a file)")
    parser.add_argument("--overwrite", action="store_true",
                        help="ignore saved progress and redo everything")
    parser.add_argument("--grid", action="store_true",
                        help="run every strategy against every model")
    parser.add_argument("--grid-strategies", nargs="+", default=list(STRATEGIES),
                        choices=STRATEGIES)
    parser.add_argument("--grid-llms", nargs="+", default=list(MODELS),
                        help=f"tags from {sorted(MODELS)}")
    args = parser.parse_args()

    # The benchmark grid is combo2 only.
    #
    # combo2 is combo1 (dense + sparse, merged) plus the reranker, so it is a
    # strict superset and the one the project reports. Pinning it here means a
    # changed default in configuration.py cannot quietly produce combo1
    # answers that land in the same results directory and get pooled with
    # combo2 ones — the filename carries the combo, but nothing was checking
    # it matched. The interactive tools still accept every mode.
    if args.combo != GRID_COMBO:
        print(f"  [note] --combo {args.combo} ignored: the grid is {GRID_COMBO} only.")
    args.combo = GRID_COMBO

    progress_on = set_enabled(not args.no_progress)
    # Installed before the first cell, because build_pipeline builds the model
    # itself and there is nothing to hook afterwards.
    install_generation_progress(config_inference.MAX_NEW_TOKENS)

    questions = load_questions(args.benchmark)
    print(f"Loaded {len(questions)} questions from {args.benchmark}")
    if not progress_on and not args.no_progress:
        print("  [note] tqdm is not installed, so there are no progress bars. "
              "Install it with: pip install tqdm")

    # The experiment varies chunking strategy and generator, and holds
    # everything else fixed.
    #
    # Embedder is not swept: BGE is the only model the pipeline uses. In the
    # 24-cell v1 grid, swapping it changed retrieval by at most 0.003 mrr on
    # any strategy, while strategy changed it by 0.244 (exp1/exp3 0.586 ->
    # exp2 0.830), which is why the second model was dropped. It is still
    # recorded as a fixed setting rather than omitted silently.
    if args.grid:
        combinations = list(itertools.product(args.grid_strategies, args.grid_llms))
    else:
        combinations = [(args.strategy, args.llm)]

    # Printed so the log records what was held fixed, not just what varied.
    # A results directory whose provenance lives only in someone's memory is
    # how two prompt versions ended up pooled in one comparison.
    print(f"{len(combinations)} run(s) to do")
    print(f"  varying   : strategy {list(args.grid_strategies)} x "
          f"llm {list(args.grid_llms)}" if args.grid else "  single run")
    cut = (f"budget_tokens={args.budget_tokens}" if args.budget_tokens
           else f"top_n={args.top_n}")
    print(f"  fixed     : embedder={args.embedder} combo={args.combo} "
          f"candidates={args.candidates} {cut} "
          f"router={not args.no_router} relevance_floor={args.relevance_floor}")
    print(f"  condition : {'equal-context (budget)' if args.budget_tokens else 'equal-k (fixed top_n)'}")
    print(f"  prompt    : {PromptTemplate.FORMAT_VERSION}\n")

    skipped, failed = [], []

    # The bar that answers "how long until the whole grid is done". Its
    # estimate only means anything once a cell has finished — until then there
    # is no measured cell time to extrapolate from — and it assumes the cells
    # cost roughly the same, which the 1b model breaks. Treat the first
    # estimate as a guess and the one after cell three as real.
    grid_bar = make_bar(len(combinations), "grid", unit="run")

    # Everything the cells print goes through tqdm.write, so the log scrolls
    # above the bars instead of leaving a trail of duplicated ones.
    with prints_below_bars():
        for strategy, llm_name in combinations:
            path = os.path.join(args.out_dir,
                                result_filename(strategy, args.embedder,
                                                args.combo, llm_name,
                                                args.relevance_floor,
                                                not args.no_router,
                                                args.budget_tokens))
            run_hash = settings_hash(strategy, args.embedder, args.combo,
                                     resolve_model(llm_name), args.candidates,
                                     args.top_n, args.benchmark, args.limit,
                                     not args.no_router, args.relevance_floor,
                                     args.budget_tokens)

            label = f"{strategy} x {args.embedder} x {llm_name}"

            if not args.overwrite and already_finished(path, run_hash):
                print(f"  already done: {label}")
                skipped.append(label)
                grid_bar.update(1)
                continue

            if args.overwrite:
                clear_progress(path)

            try:
                payload = run_benchmark(questions, path, args, strategy, llm_name,
                                        run_hash)
            except KeyboardInterrupt:
                # Not caught quietly: Ctrl-C should stop everything, not skip one
                # run and start the next long one. Progress is already saved.
                print("\nStopped. Progress is saved, so re-running will continue.")
                raise
            except (SystemExit, Exception) as e:
                # SystemExit is listed too, because the self-check and the model
                # check raise it, and a plain `except Exception` would miss them.
                print(f"  !! failed: {label}: {type(e).__name__}: {e}")
                failed.append((label, str(e)))
                free_model()
                grid_bar.update(1)
                continue

            save_results(payload, path)
            clear_progress(path)

            # This cell's model is finished with. Dropping the last reference is
            # not the same as the memory coming back — cycles wait for the
            # collector, and torch keeps its own caches — and two models resident
            # at once is the difference between finishing and being killed on a
            # small machine.
            free_model()
            grid_bar.set_postfix_str(label)
            grid_bar.update(1)

        grid_bar.close()

    if skipped:
        print(f"\n{len(skipped)} run(s) were already done. "
              f"Use --overwrite to redo them.")
    if failed:
        print(f"\n{len(failed)} run(s) failed:")
        for label, message in failed:
            print(f"  {label}: {message.splitlines()[0]}")


if __name__ == "__main__":
    main()
