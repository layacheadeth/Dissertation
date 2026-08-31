"""
Progress reporting for the long runs.

Both long-running entry points need the same three bars, and neither can be
the place they live: run_inference_bigfile.py would have to import them from
Inference_pipeline_automation.py or the other way round, and whichever lost
would be importing a command-line tool for its internals. So they sit here,
imported by both, and this module has no command line and prints nothing on
import.

Three bars, at the three timescales these runs actually stall on:

    cells       one tick per unit of work finished, per stage or grid cell
    loading     elapsed time while a model's weights load, which is opaque
    tokens      one tick per generated token, which is the real progress

The token bar is the one that matters. A single answer on CPU is 30 to 100
seconds during which nothing else moves, and that silence is what makes a
working run look hung.

tqdm is optional. It is not in the pipeline's requirements and this module
must not be the reason the pipeline gains a dependency, so a missing tqdm
turns every bar into a no-op and changes nothing else.

    from pipeline_progress import (cell_bar, elapsed_bar, free_model,
                                   install_generation_progress,
                                   prints_below_bars, set_enabled)
"""

import io
import sys
import threading
from contextlib import contextmanager, redirect_stdout

# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
# A full grid is 27 generation runs and 27 pipeline runs, and the slow part is
# not the grid: it is one call to model.generate writing one answer, which on
# CPU is 30 to 100 seconds of complete silence. A bar over the cells alone
# would still sit still for a minute and a half at a time, which is the thing
# that makes a run look hung when it is working. So there are three bars, at
# the three timescales a run actually stalls on:
#
#   cells       one tick per record written, per stage
#   loading     elapsed time while a model's weights load, which is opaque
#   tokens      one tick per generated token, which is the real progress
#
# tqdm is optional. It is not in the pipeline's requirements and this script
# must not be the reason the pipeline gains a dependency, so a missing tqdm
# turns the bars into no-ops and changes nothing else.
try:
    from tqdm.auto import tqdm
    HAVE_TQDM = True
except ImportError:
    tqdm = None
    HAVE_TQDM = False

# The real stream, kept before anything is redirected. Bars and the lines the
# stages print both have to reach this rather than each other.
_REAL_STDOUT = sys.stdout

# Set from --no-progress in main(). Module-level because the generation hook
# is installed on a class and has no argument to read it from.
PROGRESS = HAVE_TQDM


class _NullBar:
    """What every bar helper returns when there are no bars.

    A shim rather than a None check at each call site: `with bar(...) as b`
    and `b.update()` then read the same whether tqdm is there or not.
    """

    def update(self, n=1):
        pass

    def set_postfix_str(self, text):
        pass

    def refresh(self):
        pass

    def close(self):
        pass


def make_bar(total, desc, unit="cell"):
    """A bar the caller closes itself.

    cell_bar is the version to prefer. This exists for loops already deep in
    an existing function, where wrapping them in a `with` would mean
    reindenting the whole body and burying the real change in whitespace.
    """
    if not PROGRESS or total <= 0:
        return _NullBar()
    return tqdm(total=total, desc=desc, unit=unit, file=_REAL_STDOUT,
                dynamic_ncols=True, leave=True)


@contextmanager
def cell_bar(total, desc):
    """One tick per cell of a stage's grid."""
    if not PROGRESS or total <= 0:
        yield _NullBar()
        return
    bar = tqdm(total=total, desc=desc, unit="cell", file=_REAL_STDOUT,
               dynamic_ncols=True, leave=True)
    try:
        yield bar
    finally:
        bar.close()


@contextmanager
def elapsed_bar(desc):
    """A ticking clock for a phase with no measurable progress.

    Loading weights reports nothing until it is done, and on a first run it
    may be downloading gigabytes. There is no percentage to show honestly, so
    this shows elapsed time and nothing else: enough to tell "slow" from
    "hung", which is the only question being asked.
    """
    if not PROGRESS:
        yield _NullBar()
        return
    bar = tqdm(bar_format="  {desc} {elapsed}", desc=desc, file=_REAL_STDOUT,
               dynamic_ncols=True, leave=False)
    stop = threading.Event()

    def tick():
        # Redrawn from a thread because nothing else will redraw it: the main
        # thread is inside from_pretrained and will not return for minutes.
        while not stop.wait(1.0):
            try:
                bar.refresh()
            except Exception:
                return

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield bar
    finally:
        stop.set()
        thread.join(timeout=2)
        bar.close()


class _TokenTicker:
    """A transformers streamer that counts tokens instead of printing them.

    generate() calls put() with the prompt first and then once per new token,
    so the first call is skipped and the rest are the answer being written.
    max_new_tokens is the total, and stop_strings or an EOS will usually end
    it early — a bar that finishes at 60% is correct here, not a bug.

    The bar is created on the first token rather than up front, so the pause
    before the first one is attributed to the prompt, not to a bar that
    appears and then sits at zero.
    """

    def __init__(self, total, desc="generating"):
        self.total = total
        self.desc = desc
        self.bar = None
        self.seen_prompt = False

    def put(self, value):
        if not self.seen_prompt:
            self.seen_prompt = True     # the prompt, not an answer token
            return
        if self.bar is None and PROGRESS:
            self.bar = tqdm(total=self.total, desc=f"  {self.desc}", unit="tok",
                            file=_REAL_STDOUT, dynamic_ncols=True,
                            leave=False)
        if self.bar is not None:
            self.bar.update(1)

    def end(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class _TqdmStream(io.TextIOBase):
    """Sends the stages' print() output through tqdm.write.

    The stages print a great deal, and printing to stdout underneath a live
    bar leaves a trail of duplicated bars up the terminal. tqdm.write clears
    the bar, writes the line and redraws, so the log stays readable and the
    bar stays at the bottom. Buffered to a whole line because print() arrives
    in fragments and each fragment would otherwise be redrawn separately.
    """

    def __init__(self):
        self.buffer_text = ""

    def write(self, text):
        self.buffer_text += text
        while "\n" in self.buffer_text:
            line, self.buffer_text = self.buffer_text.split("\n", 1)
            tqdm.write(line, file=_REAL_STDOUT)
        return len(text)

    def flush(self):
        if self.buffer_text:
            tqdm.write(self.buffer_text, file=_REAL_STDOUT)
            self.buffer_text = ""


@contextmanager
def prints_below_bars():
    """Keep the stages' output above the bars rather than through them."""
    if not PROGRESS:
        yield
        return
    stream = _TqdmStream()
    try:
        with redirect_stdout(stream):
            yield
    finally:
        stream.flush()


# ---------------------------------------------------------------------------
# Hooking generation
# ---------------------------------------------------------------------------
# Stage 4 builds its own LLM inside run_stage and stage 5 builds one inside
# build_pipeline, so neither hands one back to hook. The hook therefore goes
# on the class, once, and covers both. It wraps rather than edits: nothing in
# llm_n_prompt.py changes, and with --no-progress the wrapper is never
# installed at all.
def set_enabled(enabled):
    """Turn the bars on or off for this run. Off is still correct, just quiet."""
    global PROGRESS
    PROGRESS = bool(enabled) and HAVE_TQDM
    return PROGRESS


def install_generation_progress(default_max_tokens):
    """Put a loading clock and a token bar around every LLM this run builds.

    Imported lazily: this module is loaded by entry points that never build a
    language model, and importing llm_n_prompt at the top would make them pay
    for torch to draw a progress bar.
    """
    if not PROGRESS:
        return

    import llm_n_prompt

    LLM = llm_n_prompt.LLM
    if getattr(LLM, "_progress_installed", False):
        return

    original_init = LLM.__init__

    def __init__(self, model_name=None, device="cpu"):
        label = llm_n_prompt.resolve_model(model_name)
        with elapsed_bar(f"loading {label} on {device}"):
            original_init(self, model_name=model_name, device=device)
        _tick_tokens(self, default_max_tokens)

    LLM.__init__ = __init__
    LLM._progress_installed = True


def _tick_tokens(llm, default_max_tokens):
    """Inject a counting streamer into this model's generate().

    Wrapping model.generate rather than LLM.generate is deliberate: LLM.generate
    builds its keyword arguments internally and has no streamer parameter to
    pass one through, but it calls model.generate twice — once with
    stop_strings and once without, on older transformers — and wrapping the
    inner call covers both paths without duplicating that fallback here.

    The weakref matters more than it looks. Assigning a closure to
    model.generate that captured the *bound* method would make the model
    reference itself: model -> generate -> closure -> bound method -> model.
    Python collects cycles eventually, but not promptly, and stage 4 builds a
    new model for every one of its cells. Half a gigabyte apiece, held until
    the collector happens to run, is enough to get the process killed on a
    machine with a few gigabytes of RAM — silently, with no traceback, because
    the kernel does the killing. Holding the class's function and a weak
    reference to the instance keeps the model collectable the moment the cell
    that built it is done.
    """
    import weakref

    model = llm.model
    unbound = type(model).generate
    model_ref = weakref.ref(model)

    def generate(*args, **kwargs):
        target = model_ref()
        if target is None:                  # the model is gone; nothing to do
            raise RuntimeError("the model was freed before generate was called")
        if "streamer" in kwargs:            # someone else is already streaming
            return unbound(target, *args, **kwargs)
        total = kwargs.get("max_new_tokens") or default_max_tokens
        ticker = _TokenTicker(total)
        try:
            return unbound(target, *args, streamer=ticker, **kwargs)
        except TypeError:
            # A transformers old enough to reject streamer here would also
            # have rejected stop_strings above, so this is the same fallback
            # llm_n_prompt already makes: lose the bar, keep the answer.
            ticker.end()
            return unbound(target, *args, **kwargs)
        finally:
            ticker.end()

    model.generate = generate


def free_model():
    """Hand a finished cell's model back before the next one is built.

    Stage 4 builds an LLM inside run_stage for every cell and drops it on the
    way out, but dropping the last reference is not the same as the memory
    being returned: reference cycles wait for the collector, and torch keeps
    its own allocator caches. Two models resident at once is the difference
    between finishing and being killed on a small machine, so the collection
    is forced between cells rather than left to chance.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        # Freeing is an optimisation. Never let it be the thing that stops a run.
        pass


