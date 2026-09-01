"""
4-Corpus-statistics.py

Produce the corpus numbers cited in the dissertation, as files on disk rather
than as console output that scrolls away.

Reads what stages 1 and 2 already wrote to Data/All_extracted_text/Data_week*/:

    <stem>.json                  extraction output  (page counts, cleaned text)
    <stem>_exp1_chunks.json      page-level chunks
    <stem>_exp2_chunks.json      fixed-size overlapping chunks
    <stem>_exp3_chunks.json      section-aware chunks

and writes to Data/Analysis/:

    corpus_stats.json            everything, machine readable
    pages_by_week.csv            per-lecture page counts
    chunks_by_week.csv           per-lecture chunk counts per strategy
    chunk_length_stats.csv       length distribution per strategy
    tables.tex                   \\input-able booktabs tables
    corpus_stats.md              human-readable summary

Nothing is recomputed from the PDFs, so these figures describe exactly the
corpus that was ingested into ChromaDB.

Usage
    python 4-Corpus-statistics.py
    python 4-Corpus-statistics.py --data Data/All_extracted_text --out Data/Analysis
    python 4-Corpus-statistics.py --embed-tokenizers      # real model tokenizers
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------
# Token counting: the SAME WordPiece tokenizer the 2-x chunkers budget with
# ------------------------------------------------------------------
# Every token figure in this report is produced by the shared tokenizer in
# Share_components/chunking_tokenizer.py, which is the embedding model's own.
# Counting here in a different tokenizer from the one that enforced the budget
# would make the truncation audit unfalsifiable: the report could show zero
# overflows while the encoder truncated, or vice versa.
#
# There is no character-count fallback. If the tokenizer cannot be loaded the
# script stops, because ESTIMATED token figures have no place in a truncation
# audit that the write-up cites as a guarantee.
_HERE = Path(__file__).resolve().parent
for _base in (_HERE, *_HERE.parents):
    if (_base / "Share_components").is_dir():
        sys.path.insert(0, str(_base))
        break

from Share_components.chunking_tokenizer import (   # noqa: E402
    EMBED_LIMITS,
    HF_NAMES,
    NUM_SPECIAL_TOKENS,
    count_tokens,
    init_tokenizer,
    tokenizer_name,
    tokenizer_provenance,
    usable_limit,
)

# Limits expressed in CONTENT tokens, because count_tokens() excludes
# [CLS]/[SEP]. Comparing raw counts against 512 would under-report overflows by
# exactly two tokens per chunk.
USABLE_LIMITS = {model: usable_limit(model) for model in EMBED_LIMITS}

# The one embedding model the pipeline uses. Named once here so the overflow
# columns below cannot drift from it.
EMBED_MODEL = "bge-small-en-v1.5"
BGE_USABLE = USABLE_LIMITS[EMBED_MODEL]

STRATEGIES = {
    "exp1": {"suffix": "_exp1_chunks.json", "label": "Page-level"},
    "exp2": {"suffix": "_exp2_chunks.json", "label": "Fixed-size overlap"},
    "exp3": {"suffix": "_exp3_chunks.json", "label": "Section-aware"},
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def git_commit():
    """Record which revision of the code produced these numbers."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def week_sort_key(name):
    m = re.search(r"(\d+)", name or "")
    return int(m.group(1)) if m else 9999


def describe(values):
    """Summary statistics for a list of numbers."""
    if not values:
        return {"n": 0, "total": 0, "mean": 0, "median": 0, "sd": 0,
                "min": 0, "max": 0, "p95": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "total": sum(values),
        "mean": round(statistics.mean(values), 1),
        "median": statistics.median(values),
        "sd": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p95": ordered[max(0, int(0.95 * len(ordered)) - 1)],
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Stage 1: per-lecture extraction stats
# ------------------------------------------------------------------
def collect_lectures(data_dir):
    """Find every extraction JSON (not a *_chunks.json) under Data_week*/."""
    folders = sorted(glob.glob(os.path.join(data_dir, "Data_week*")),
                     key=lambda p: week_sort_key(os.path.basename(p)))
    if not folders:
        folders = [data_dir]

    lectures = []
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
            if "_chunks.json" in os.path.basename(path):
                continue
            try:
                data = load_json(path)
            except Exception as e:
                print(f"  ! could not read {path}: {e}", file=sys.stderr)
                continue
            if "pages" not in data:
                continue                       # not an extraction output

            raw = data.get("raw_total_pages", 0)
            dedup = data.get("deduplicated_total_pages", len(data.get("pages", [])))
            # pages_with_text was added later; fall back gracefully so old
            # extractions still report, just with the split unavailable.
            with_text = data.get("pages_with_text")
            has_split = with_text is not None
            if not has_split:
                with_text = None

            page_chars = [len(p.get("content", "")) for p in data.get("pages", [])]

            lectures.append({
                "folder": folder,
                "stem": Path(path).stem,
                "path": path,
                "sha256": sha256(path),
                "filename": data.get("filename", Path(path).name),
                "week": data.get("week", os.path.basename(folder)),
                "raw_total_pages": raw,
                "pages_with_text": with_text,
                "empty_pages_removed": data.get("empty_pages_removed"),
                "deduplicated_total_pages": dedup,
                "pages_removed_by_dedup": data.get("pages_removed_by_dedup"),
                "has_page_split": has_split,
                "total_chars": sum(page_chars),
                "total_tokens": sum(count_tokens(p.get("content", ""))
                                    for p in data.get("pages", [])),
            })
    return lectures


# ------------------------------------------------------------------
# Stage 2: per-lecture, per-strategy chunk stats
# ------------------------------------------------------------------
def collect_chunks(lecture, extra_tokenizers=None):
    """Load every strategy's chunk file for one lecture."""
    per_strategy = {}
    for key, cfg in STRATEGIES.items():
        path = os.path.join(lecture["folder"], lecture["stem"] + cfg["suffix"])
        if not os.path.exists(path):
            per_strategy[key] = None           # strategy not run for this lecture
            continue

        try:
            records = load_json(path)
        except Exception as e:
            print(f"  ! could not read {path}: {e}", file=sys.stderr)
            per_strategy[key] = None
            continue

        if not isinstance(records, list):
            records = [records]

        texts = [r.get("content", "") for r in records]
        chars = [len(t) for t in texts]
        tokens = [count_tokens(t) for t in texts]

        # Which source slides does this strategy's output actually cover?
        covered = set()
        for r in records:
            pages = r.get("page_number", [])
            if not isinstance(pages, list):
                pages = [pages]
            covered.update(p for p in pages if p is not None)

        ids = [r.get("chunk_id") for r in records]
        duplicate_ids = len(ids) - len(set(ids))

        entry = {
            "path": path,
            "sha256": sha256(path),
            "n_chunks": len(records),
            "duplicate_chunk_ids": duplicate_ids,
            "pages_covered": len(covered),
            "chars": describe(chars),
            "tokens": describe(tokens),
            # Compared against USABLE limits: count_tokens() excludes
            # [CLS]/[SEP], so the encoder's real headroom is limit - 2.
            "over_limit": {
                model: sum(1 for t in tokens if t > limit)
                for model, limit in USABLE_LIMITS.items()
            },
        }

        if extra_tokenizers:
            entry["model_tokens"] = {}
            for model, tok in extra_tokenizers.items():
                counts = [len(tok.encode(t, add_special_tokens=True)) for t in texts]
                entry["model_tokens"][model] = {
                    **describe(counts),
                    # add_special_tokens=True here, so compare against the
                    # full positional limit rather than the usable one.
                    "over_limit": sum(1 for c in counts if c > EMBED_LIMITS[model]),
                }

        per_strategy[key] = entry
    return per_strategy


def load_embed_tokenizers():
    """Optional: cross-check with the model's own tokenizer.

    The primary count is already BGE WordPiece, so this is a self-consistency
    check rather than a comparison: the counts should agree to within the two
    special tokens. A disagreement beyond that means the chunkers and this
    report have drifted apart -- most likely configuration.TOKENIZER was
    changed after stage 2 ran.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  ! transformers not installed; skipping --embed-tokenizers",
              file=sys.stderr)
        return None

    tokenizers = {}
    for model, hf_name in HF_NAMES.items():
        try:
            tokenizers[model] = AutoTokenizer.from_pretrained(
                hf_name, trust_remote_code=True
            )
            print(f"  loaded tokenizer: {hf_name}")
        except Exception as e:
            print(f"  ! could not load {hf_name}: {e}", file=sys.stderr)
    return tokenizers or None


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------
def aggregate(lectures, chunk_data, extra_tokenizers=None):
    totals = {
        "n_lectures": len(lectures),
        "raw_total_pages": sum(l["raw_total_pages"] for l in lectures),
        "deduplicated_total_pages": sum(l["deduplicated_total_pages"] for l in lectures),
        "total_chars": sum(l["total_chars"] for l in lectures),
        "total_tokens": sum(l["total_tokens"] for l in lectures),
    }

    # Only report the empty/dedup split if EVERY lecture recorded it, otherwise
    # a partial sum would be quietly wrong.
    if lectures and all(l["has_page_split"] for l in lectures):
        totals["pages_with_text"] = sum(l["pages_with_text"] for l in lectures)
        totals["empty_pages_removed"] = (
            totals["raw_total_pages"] - totals["pages_with_text"])
        totals["pages_removed_by_dedup"] = (
            totals["pages_with_text"] - totals["deduplicated_total_pages"])
        totals["split_available"] = True
    else:
        totals["split_available"] = False

    raw = totals["raw_total_pages"]
    totals["overall_reduction_pct"] = (
        round(100.0 * (raw - totals["deduplicated_total_pages"]) / raw, 1)
        if raw else 0.0
    )

    by_strategy = {}
    for key, cfg in STRATEGIES.items():
        all_chars, all_tokens = [], []
        n_chunks = 0
        n_lectures_present = 0
        dup_ids = 0
        over = defaultdict(int)
        model_counts = defaultdict(list)

        for stem, per in chunk_data.items():
            entry = per.get(key)
            if entry is None:
                continue
            n_lectures_present += 1
            n_chunks += entry["n_chunks"]
            dup_ids += entry["duplicate_chunk_ids"]
            for model, count in entry["over_limit"].items():
                over[model] += count
            all_chars.append(entry["chars"])
            all_tokens.append(entry["tokens"])
            if extra_tokenizers and "model_tokens" in entry:
                for model, stats in entry["model_tokens"].items():
                    model_counts[model].append(stats)

        # Re-derive corpus-wide mean from totals, not a mean of per-file means,
        # which would weight a 5-chunk lecture the same as a 90-chunk one.
        total_chars = sum(c["total"] for c in all_chars)
        total_tokens = sum(t["total"] for t in all_tokens)

        by_strategy[key] = {
            "label": cfg["label"],
            "n_lectures": n_lectures_present,
            "n_chunks": n_chunks,
            "duplicate_chunk_ids": dup_ids,
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "mean_chars": round(total_chars / n_chunks, 1) if n_chunks else 0,
            "mean_tokens": round(total_tokens / n_chunks, 1) if n_chunks else 0,
            "max_chars": max((c["max"] for c in all_chars), default=0),
            "max_tokens": max((t["max"] for t in all_tokens), default=0),
            "chunks_over_limit": dict(over),
            "chunks_per_page": (
                round(n_chunks / totals["deduplicated_total_pages"], 2)
                if totals["deduplicated_total_pages"] else 0
            ),
        }
        if model_counts:
            by_strategy[key]["model_tokens"] = {
                model: {
                    "max": max(s["max"] for s in per_file),
                    "over_limit": sum(s["over_limit"] for s in per_file),
                }
                for model, per_file in model_counts.items()
            }

    return totals, by_strategy


# ------------------------------------------------------------------
# Output writers
# ------------------------------------------------------------------
def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_pages_csv(out_dir, lectures):
    rows = sorted(lectures, key=lambda l: week_sort_key(l["week"]))
    write_csv(
        out_dir / "pages_by_week.csv", rows,
        ["week", "filename", "raw_total_pages", "pages_with_text",
         "empty_pages_removed", "deduplicated_total_pages",
         "pages_removed_by_dedup", "total_chars", "total_tokens", "sha256"],
    )


def write_chunks_csv(out_dir, lectures, chunk_data):
    rows = []
    for lec in sorted(lectures, key=lambda l: week_sort_key(l["week"])):
        row = {"week": lec["week"], "filename": lec["filename"],
               "deduplicated_total_pages": lec["deduplicated_total_pages"]}
        for key in STRATEGIES:
            entry = chunk_data[lec["stem"]].get(key)
            row[f"{key}_chunks"] = entry["n_chunks"] if entry else ""
            row[f"{key}_mean_tokens"] = entry["tokens"]["mean"] if entry else ""
        rows.append(row)

    fields = ["week", "filename", "deduplicated_total_pages"]
    for key in STRATEGIES:
        fields += [f"{key}_chunks", f"{key}_mean_tokens"]
    write_csv(out_dir / "chunks_by_week.csv", rows, fields)


def write_length_csv(out_dir, by_strategy):
    rows = []
    for key, s in by_strategy.items():
        rows.append({
            "strategy": key,
            "label": s["label"],
            "n_chunks": s["n_chunks"],
            "chunks_per_page": s["chunks_per_page"],
            "mean_chars": s["mean_chars"],
            "max_chars": s["max_chars"],
            "mean_tokens": s["mean_tokens"],
            "max_tokens": s["max_tokens"],
            f"over_{BGE_USABLE}_bge": s["chunks_over_limit"].get(EMBED_MODEL, 0),
        })
    write_csv(out_dir / "chunk_length_stats.csv", rows,
              list(rows[0].keys()) if rows else ["strategy"])


def write_tex(out_dir, totals, by_strategy, lectures):
    lines = []
    lines.append("% Generated by 4-Corpus-statistics.py — do not edit by hand.")
    lines.append(f"% Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")

    # --- Table 1: corpus reduction -------------------------------------
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Corpus size at each stage of extraction.}")
    lines.append(r"  \label{tab:corpus-reduction}")
    lines.append(r"  \begin{tabular}{lrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Stage & Pages & \% of raw \\")
    lines.append(r"    \midrule")
    raw = totals["raw_total_pages"]
    lines.append(f"    Raw slide pages & {raw} & 100.0 \\\\")
    if totals["split_available"]:
        wt = totals["pages_with_text"]
        lines.append(f"    With extractable text & {wt} & {100.0 * wt / raw:.1f} \\\\")
    dd = totals["deduplicated_total_pages"]
    lines.append(f"    After deduplication & {dd} & {100.0 * dd / raw:.1f} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table 2: chunk counts -----------------------------------------
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append("  \\caption{Chunk corpora produced by the four strategies "
                 f"from the same {dd} cleaned pages.}}")
    lines.append(r"  \label{tab:chunk-counts}")
    lines.append(r"  \begin{tabular}{lrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Strategy & Chunks & Per page & Mean tokens & Max tokens \\")
    lines.append(r"    \midrule")
    for key in sorted(by_strategy, key=lambda k: -by_strategy[k]["n_chunks"]):
        s = by_strategy[key]
        lines.append(f"    {s['label']} & {s['n_chunks']} & {s['chunks_per_page']:.2f} "
                     f"& {s['mean_tokens']:.0f} & {s['max_tokens']} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table 3: per-week pages ---------------------------------------
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Pages per lecture before and after deduplication.}")
    lines.append(r"  \label{tab:pages-by-week}")
    lines.append(r"  \begin{tabular}{lrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Week & Raw pages & Cleaned pages \\")
    lines.append(r"    \midrule")
    for lec in sorted(lectures, key=lambda l: week_sort_key(l["week"])):
        week = str(lec["week"]).replace("_", r"\_")
        lines.append(f"    {week} & {lec['raw_total_pages']} & "
                     f"{lec['deduplicated_total_pages']} \\\\")
    lines.append(r"    \midrule")
    lines.append(f"    Total & {raw} & {dd} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    (out_dir / "tables.tex").write_text("\n".join(lines), encoding="utf-8")


def write_markdown(out_dir, totals, by_strategy, provenance):
    dd = totals["deduplicated_total_pages"]
    raw = totals["raw_total_pages"]
    L = []
    L.append("# Corpus statistics\n")
    L.append(f"Generated: {provenance['generated_utc']}  ")
    L.append(f"Git commit: `{provenance['git_commit']}`  ")
    L.append(f"Token counter: {provenance['tokenizer']} "
             f"(WordPiece, counts exclude [CLS]/[SEP]; usable window "
             f"{provenance['tokenizer_details']['usable_tokens']})\n")

    L.append("## Pages\n")
    L.append(f"- Lectures: **{totals['n_lectures']}**")
    L.append(f"- Raw slide pages: **{raw}**")
    if totals["split_available"]:
        L.append(f"- Pages with extractable text: **{totals['pages_with_text']}** "
                 f"({totals['empty_pages_removed']} image-only or blank removed)")
        L.append(f"- After deduplication: **{dd}** "
                 f"({totals['pages_removed_by_dedup']} progressive reveals removed)")
    else:
        L.append(f"- After deduplication: **{dd}**")
        L.append("- NOTE: this extraction predates the `pages_with_text` field, so "
                 "the raw-to-cleaned gap mixes image-only slides with progressive "
                 "reveals. Re-run stage 1 to separate them.")
    L.append(f"- Overall reduction: **{totals['overall_reduction_pct']}%**\n")

    L.append("## Chunks\n")
    L.append(f"| Strategy | Chunks | Per page | Mean tokens | Max tokens | "
             f">{BGE_USABLE} (truncated) |")
    L.append("|---|---|---|---|---|---|")
    for key in sorted(by_strategy, key=lambda k: -by_strategy[k]["n_chunks"]):
        s = by_strategy[key]
        L.append(f"| {s['label']} | {s['n_chunks']} | {s['chunks_per_page']} | "
                 f"{s['mean_tokens']} | {s['max_tokens']} | "
                 f"{s['chunks_over_limit'].get(EMBED_MODEL, 0)} |")

    counts = [s["n_chunks"] for s in by_strategy.values() if s["n_chunks"]]
    if counts:
        L.append(f"\nSpread between largest and smallest corpus: "
                 f"**{max(counts) / min(counts):.1f}x** "
                 f"({max(counts)} vs {min(counts)} chunks).")

    dups = {k: s["duplicate_chunk_ids"] for k, s in by_strategy.items()
            if s["duplicate_chunk_ids"]}
    if dups:
        L.append(f"\n**Warning:** duplicate chunk_ids found: {dups}. "
                 "The ingest script deduplicates these, so the ChromaDB "
                 "collection count will differ from the JSON chunk count.")

    (out_dir / "corpus_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate verifiable corpus statistics from stage 1 and 2 output."
    )
    parser.add_argument("--data", default="Data/All_extracted_text",
                        help="directory containing Data_week*/ folders")
    parser.add_argument("--out", default="Data/Analysis",
                        help="where to write the statistics files")
    parser.add_argument("--embed-tokenizers", action="store_true",
                        help="cross-check with BGE's own tokenizer. It should "
                             "agree with the primary count to within 2 tokens; "
                             "a wider gap means stage 2 ran under a different "
                             "tokenizer than this report.")
    parser.add_argument("--tokenizer", default=None,
                        help="HuggingFace id or local path of the tokenizer "
                             "used for all token figures. MUST match the one "
                             "the 2-x chunkers ran with, or the truncation "
                             "audit describes a different corpus than the one "
                             "on disk. Default: the shared module's default.")
    args = parser.parse_args()

    data_dir = args.data
    out_dir = Path(args.out)

    if not os.path.isdir(data_dir):
        raise SystemExit(f"data directory does not exist: {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_dir} ...")
    lectures = collect_lectures(data_dir)
    if not lectures:
        raise SystemExit(
            f"No extraction JSON files found under {data_dir}/Data_week*/.\n"
            "Run the stage 1 and 2 pipeline first."
        )
    print(f"  found {len(lectures)} lecture(s)")

    init_tokenizer(args.tokenizer)
    print(f"  token counter: {tokenizer_name()}")

    extra_tokenizers = load_embed_tokenizers() if args.embed_tokenizers else None

    chunk_data = {}
    for lec in lectures:
        chunk_data[lec["stem"]] = collect_chunks(lec, extra_tokenizers)

    totals, by_strategy = aggregate(lectures, chunk_data, extra_tokenizers)

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "tokenizer": tokenizer_name(),
        "tokenizer_details": tokenizer_provenance(),
        "data_dir": os.path.abspath(data_dir),
        "python": sys.version.split()[0],
    }

    # ---- write everything -----------------------------------------
    with open(out_dir / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "provenance": provenance,
            "totals": totals,
            "by_strategy": by_strategy,
            "lectures": lectures,
            "chunks_by_lecture": chunk_data,
        }, f, indent=2, ensure_ascii=False)

    write_pages_csv(out_dir, lectures)
    write_chunks_csv(out_dir, lectures, chunk_data)
    write_length_csv(out_dir, by_strategy)
    write_tex(out_dir, totals, by_strategy, lectures)
    write_markdown(out_dir, totals, by_strategy, provenance)

    # ---- console summary: the citable figures ----------------------
    raw = totals["raw_total_pages"]
    dd = totals["deduplicated_total_pages"]
    print(f"\n{'=' * 60}")
    print("CORPUS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Lectures                 {totals['n_lectures']}")
    print(f"  Raw slide pages          {raw}")
    if totals["split_available"]:
        print(f"  Pages with text          {totals['pages_with_text']} "
              f"(-{totals['empty_pages_removed']} image-only/blank)")
        print(f"  After deduplication      {dd} "
              f"(-{totals['pages_removed_by_dedup']} progressive reveals)")
    else:
        print(f"  After deduplication      {dd}")
        print("  ! pages_with_text missing — cannot separate blank-slide removal")
        print("    from deduplication. Re-run stage 1 with the updated extractor.")
    print(f"  Overall reduction        {totals['overall_reduction_pct']}%")
    print()
    print(f"  {'Strategy':<22}{'Chunks':>8}{'/page':>8}{'Mean tok':>10}{'Max tok':>9}")
    for key in sorted(by_strategy, key=lambda k: -by_strategy[k]["n_chunks"]):
        s = by_strategy[key]
        print(f"  {s['label']:<22}{s['n_chunks']:>8}{s['chunks_per_page']:>8.2f}"
              f"{s['mean_tokens']:>10.0f}{s['max_tokens']:>9}")

    for key, s in by_strategy.items():
        over = s["chunks_over_limit"].get(EMBED_MODEL, 0)
        if over:
            pct = 100.0 * over / s["n_chunks"]
            print(f"\n  ! {s['label']}: {over} chunks ({pct:.1f}%) exceed BGE's "
                  f"{BGE_USABLE}-token usable window "
                  f"and are truncated at ingestion")

    print(f"\nWritten to {out_dir}/")
    for name in ("corpus_stats.json", "corpus_stats.md", "tables.tex",
                 "pages_by_week.csv", "chunks_by_week.csv",
                 "chunk_length_stats.csv"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
