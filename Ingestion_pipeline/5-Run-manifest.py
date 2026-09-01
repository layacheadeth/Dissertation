"""
5-Run-manifest.py

Capture everything needed to reproduce a pipeline run that the corpus
statistics do not already cover.

4-Corpus-statistics.py describes the DATA (how many pages, how many chunks,
how long they are). This script describes the RUN: which code, which package
versions, which parameters, which source PDFs, and what ended up in ChromaDB.
Together they are the evidence base for a methodology chapter.

Writes to Data/Analysis/:

    run_manifest.json     everything, machine readable
    run_manifest.md       human-readable summary

Usage
    python 5-Run-manifest.py
    python 5-Run-manifest.py --chroma-path Data/Database/chroma_db
    python 5-Run-manifest.py --params Data/Analysis/chunker_params.json
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# The tokenizer that budgeted the chunkers is part of the run's identity: the
# same parameters under a different vocabulary produce a different corpus.
_HERE = Path(__file__).resolve().parent
for _base in (_HERE, *_HERE.parents):
    if (_base / "Share_components").is_dir():
        sys.path.insert(0, str(_base))
        break

try:
    from Share_components.chunking_tokenizer import tokenizer_provenance
except Exception as _tok_exc:                      # noqa: BLE001
    _TOK_EXC = _tok_exc

    def tokenizer_provenance():
        return {"error": f"tokenizer unavailable: {_TOK_EXC}"}

# Packages whose versions materially affect the results. transformers and
# tokenizers are here because they supply the WordPiece vocabulary that BUDGETS
# the chunkers, not just the one that embeds: a vocabulary change would move
# every chunk boundary in the corpus. sentence-transformers runs BGE, which
# embeds every chunk and every question.
TRACKED_PACKAGES = [
    "torch",
    "transformers",
    "sentence-transformers",
    "tokenizers",
    "huggingface-hub",
    "chromadb",
    "langchain-core",
    "pypdf",
    "numpy",
]


# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
def package_versions():
    """Record installed versions of everything that can change the output."""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:                        # pragma: no cover
        return {"error": "importlib.metadata unavailable"}

    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
        except Exception as e:                 # noqa: BLE001
            versions[pkg] = f"error: {e}"
    return versions


def environment():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "packages": package_versions(),
        "cuda": cuda_info(),
    }


def cuda_info():
    """GPU presence changes runtime, and occasionally numerics."""
    try:
        import torch
    except ImportError:
        return {"torch": "not installed"}
    try:
        info = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
        }
        if info["available"]:
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
        return info
    except Exception as e:                     # noqa: BLE001
        return {"error": str(e)}


# ------------------------------------------------------------------
# Code version
# ------------------------------------------------------------------
def git_state(repo_dir):
    """Which revision of the code produced this run.

    A dissertation claim of reproducibility is weak without this, so when git
    is unavailable the manifest says so explicitly rather than leaving a
    silently empty field.
    """
    def run(args):
        out = subprocess.run(["git", "-C", str(repo_dir)] + args,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None

    try:
        commit = run(["rev-parse", "HEAD"])
    except Exception as e:                     # noqa: BLE001
        return {"available": False,
                "reason": f"git not usable: {type(e).__name__}",
                "advice": "commit the pipeline to a git repo so runs are traceable"}

    if not commit:
        return {"available": False,
                "reason": f"{repo_dir} is not a git repository (or git is missing)",
                "advice": "run 'git init && git add -A && git commit' in the "
                          "pipeline directory so each run records a code version"}

    status = run(["status", "--porcelain"])
    return {
        "available": True,
        "commit": commit,
        "short_commit": commit[:8],
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "uncommitted_files": len(status.splitlines()) if status else 0,
    }


# ------------------------------------------------------------------
# Source PDFs
# ------------------------------------------------------------------
def sha256(path, limit_bytes=None):
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
            read += len(block)
            if limit_bytes and read >= limit_bytes:
                break
    return h.hexdigest()


def source_pdfs(lecture_dir):
    """Hash the input PDFs.

    The extracted JSON is hashed by 4-Corpus-statistics.py, but that only
    proves the chunks match the extraction — not that the extraction came from
    the slide set as released. This closes that link.
    """
    paths = sorted(glob.glob(os.path.join(lecture_dir, "*.pdf")))
    records = []
    for path in paths:
        try:
            records.append({
                "filename": os.path.basename(path),
                "bytes": os.path.getsize(path),
                "sha256": sha256(path),
            })
        except Exception as e:                 # noqa: BLE001
            records.append({"filename": os.path.basename(path),
                            "error": str(e)})
    return {"directory": os.path.abspath(lecture_dir), "count": len(records),
            "files": records}


# ------------------------------------------------------------------
# Chunker parameters
# ------------------------------------------------------------------
def observed_chunk_params(data_dir):
    """Recover the parameters actually baked into the chunk files.

    Configured values are recorded by the pipeline, but these are read back
    out of the emitted records, so they cannot drift from what was really run.
    """
    observed = {}

    # exp2 stores its window settings on every record.
    sizes, overlaps = Counter(), Counter()
    for path in glob.glob(os.path.join(data_dir, "**", "*_exp2_chunks.json"),
                          recursive=True):
        try:
            for rec in json.load(open(path, encoding="utf-8")):
                if "chunk_size_tokens" in rec:
                    sizes[rec["chunk_size_tokens"]] += 1
                if "overlap_tokens" in rec:
                    overlaps[rec["overlap_tokens"]] += 1
        except Exception:                      # noqa: BLE001
            continue
    if sizes:
        observed["exp2"] = {
            "chunk_size_tokens": dict(sizes),
            "overlap_tokens": dict(overlaps),
            "consistent": len(sizes) == 1 and len(overlaps) == 1,
        }

    # exp1 records whether the oversized-page guard fired.
    split_pages, total = 0, 0
    for path in glob.glob(os.path.join(data_dir, "**", "*_exp1_chunks.json"),
                          recursive=True):
        try:
            for rec in json.load(open(path, encoding="utf-8")):
                total += 1
                if rec.get("was_split"):
                    split_pages += 1
        except Exception:                      # noqa: BLE001
            continue
    if total:
        observed["exp1"] = {
            "chunks": total,
            "chunks_from_split_pages": split_pages,
            "guard_fired": split_pages > 0,
        }

    # exp3 reports how many sections it detected.
    sections, parts = set(), 0
    for path in glob.glob(os.path.join(data_dir, "**", "*_exp3_chunks.json"),
                          recursive=True):
        try:
            for rec in json.load(open(path, encoding="utf-8")):
                parts += 1
                sections.add((os.path.basename(path), rec.get("section_index")))
        except Exception:                      # noqa: BLE001
            continue
    if parts:
        observed["exp3"] = {"chunks": parts, "sections_detected": len(sections)}

    return observed


def configured_chunk_params(params_path):
    """Load the parameter block the pipeline wrote when it ran the chunkers."""
    if not params_path or not os.path.exists(params_path):
        return {
            "available": False,
            "reason": f"no parameter file at {params_path}",
            "advice": "run the pipeline via Ingestion-pipeline-automation-allweek.py, "
                      "which writes chunker_params.json",
        }
    try:
        with open(params_path, encoding="utf-8") as f:
            return {"available": True, "source": os.path.abspath(params_path),
                    **json.load(f)}
    except Exception as e:                     # noqa: BLE001
        return {"available": False, "reason": str(e)}


# ------------------------------------------------------------------
# ChromaDB state
# ------------------------------------------------------------------
def chroma_state(chroma_path):
    """Snapshot what is actually in the vector store.

    Records the distance metric explicitly: Chroma's default is L2, and since
    the pipeline writes L2-normalised vectors, L2 and cosine give identical
    RANKINGS. The methodology should state which was configured rather than
    assume.
    """
    if not os.path.isdir(chroma_path):
        return {"available": False, "reason": f"no database at {chroma_path}"}

    try:
        import chromadb
    except ImportError:
        return {"available": False, "reason": "chromadb not installed"}

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collections = []
        for col in client.list_collections():
            name = col.name if hasattr(col, "name") else str(col)
            entry = {"name": name}
            try:
                handle = client.get_collection(name)
                entry["count"] = handle.count()
                entry["metadata"] = handle.metadata
                # Dimensionality is not exposed directly; read one vector.
                sample = handle.get(limit=1, include=["embeddings"])
                vecs = sample.get("embeddings") or []
                if len(vecs) and vecs[0] is not None:
                    entry["dimensions"] = len(vecs[0])
            except Exception as e:             # noqa: BLE001
                entry["error"] = str(e)
            collections.append(entry)

        collections.sort(key=lambda c: c["name"])
        return {
            "available": True,
            "path": os.path.abspath(chroma_path),
            "n_collections": len(collections),
            "total_vectors": sum(c.get("count", 0) for c in collections),
            "collections": collections,
        }
    except Exception as e:                     # noqa: BLE001
        return {"available": False, "reason": str(e)}


# ------------------------------------------------------------------
# Markdown report
# ------------------------------------------------------------------
def write_markdown(out_dir, manifest):
    L = ["# Run manifest\n"]
    L.append(f"Generated: {manifest['generated_utc']}\n")

    git = manifest["code"]["git"]
    L.append("## Code version\n")
    if git.get("available"):
        dirty = " (UNCOMMITTED CHANGES PRESENT)" if git.get("dirty") else ""
        L.append(f"- Commit: `{git['short_commit']}` on `{git.get('branch')}`{dirty}")
        if git.get("dirty"):
            L.append(f"- {git['uncommitted_files']} file(s) modified since the commit — "
                     "this run is NOT reproducible from the commit alone")
    else:
        L.append(f"- **Not available**: {git.get('reason')}")
        L.append(f"- Fix: {git.get('advice')}")
    L.append("")

    env = manifest["environment"]
    L.append("## Environment\n")
    L.append(f"- Python {env['python']} on {env['platform']}")
    cuda = env.get("cuda", {})
    if cuda.get("available"):
        L.append(f"- GPU: {cuda.get('device_name')} (CUDA {cuda.get('torch_cuda_version')})")
    else:
        L.append("- GPU: none detected (CPU inference)")
    L.append("")
    L.append("| Package | Version |")
    L.append("|---|---|")
    for pkg, ver in env["packages"].items():
        L.append(f"| {pkg} | {ver} |")
    L.append("")

    pdfs = manifest["source_pdfs"]
    L.append("## Source documents\n")
    L.append(f"- {pdfs['count']} PDF(s) in `{pdfs['directory']}`")
    L.append("- SHA-256 of each is recorded in run_manifest.json\n")

    cfg = manifest["chunker_parameters"]["configured"]
    L.append("## Chunker parameters\n")
    if cfg.get("available"):
        L.append("| Strategy | Parameters |")
        L.append("|---|---|")
        for strat, params in (cfg.get("strategies") or {}).items():
            rendered = ", ".join(f"{k}={v}" for k, v in params.items()) or "defaults"
            L.append(f"| {strat} | {rendered} |")
    else:
        L.append(f"- **Not recorded**: {cfg.get('reason')}")
        L.append(f"- Fix: {cfg.get('advice')}")
    L.append("")

    obs = manifest["chunker_parameters"]["observed"]
    if obs.get("exp2"):
        e = obs["exp2"]
        flag = "" if e.get("consistent") else "  **INCONSISTENT ACROSS FILES**"
        L.append(f"Read back from the chunk files: exp2 window "
                 f"{list(e['chunk_size_tokens'])}, overlap "
                 f"{list(e['overlap_tokens'])}.{flag}")
    if obs.get("exp1"):
        e = obs["exp1"]
        L.append(f"exp1 oversized-page guard: "
                 f"{'fired' if e['guard_fired'] else 'did not fire'} "
                 f"({e['chunks_from_split_pages']} of {e['chunks']} chunks came "
                 f"from split pages).")
    if obs.get("exp3"):
        e = obs["exp3"]
        L.append(f"exp3 detected {e['sections_detected']} section(s), "
                 f"emitted as {e['chunks']} chunk(s).")
    L.append("")

    ch = manifest["vector_store"]
    L.append("## Vector store\n")
    if ch.get("available"):
        L.append(f"- {ch['n_collections']} collection(s), "
                 f"{ch['total_vectors']} vectors total, at `{ch['path']}`\n")
        L.append("| Collection | Vectors | Dims |")
        L.append("|---|---|---|")
        for c in ch["collections"]:
            L.append(f"| {c['name']} | {c.get('count', '?')} | {c.get('dimensions', '?')} |")
    else:
        L.append(f"- **Not available**: {ch.get('reason')}")
    L.append("")

    (out_dir / "run_manifest.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Capture code version, environment, parameters and vector "
                    "store state for one pipeline run."
    )
    parser.add_argument("--data", default="Data/All_extracted_text")
    parser.add_argument("--lectures", default="Data/All_lectures")
    parser.add_argument("--chroma-path", default="Data/Database/chroma_db")
    parser.add_argument("--out", default="Data/Analysis")
    parser.add_argument("--params", default="Data/Analysis/chunker_params.json",
                        help="parameter file written by the automation script")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent),
                        help="directory to read git state from")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building run manifest...")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {"git": git_state(args.repo), "repo_dir": os.path.abspath(args.repo)},
        "environment": environment(),
        "source_pdfs": source_pdfs(args.lectures),
        "chunker_parameters": {
            "configured": configured_chunk_params(args.params),
            "observed": observed_chunk_params(args.data),
            "tokenizer": tokenizer_provenance(),
        },
        "vector_store": chroma_state(args.chroma_path),
    }

    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_markdown(out_dir, manifest)

    # ---- console summary, flagging anything that weakens reproducibility ----
    print(f"\n{'=' * 60}")
    print("RUN MANIFEST")
    print(f"{'=' * 60}")

    git = manifest["code"]["git"]
    if git.get("available"):
        state = "dirty" if git.get("dirty") else "clean"
        print(f"  Code            {git['short_commit']} ({state})")
        if git.get("dirty"):
            print(f"  ! {git['uncommitted_files']} uncommitted file(s): this run "
                  f"cannot be reproduced from the commit alone")
    else:
        print(f"  ! Code version NOT recorded: {git.get('reason')}")
        print(f"    {git.get('advice')}")

    env = manifest["environment"]
    print(f"  Python          {env['python']}")
    for pkg in ("transformers", "sentence-transformers", "chromadb"):
        print(f"  {pkg:<15} {env['packages'].get(pkg)}")

    print(f"  Source PDFs     {manifest['source_pdfs']['count']} hashed")

    cfg = manifest["chunker_parameters"]["configured"]
    if not cfg.get("available"):
        print(f"  ! Chunker parameters NOT recorded: {cfg.get('reason')}")

    obs = manifest["chunker_parameters"]["observed"].get("exp2")
    if obs and not obs.get("consistent"):
        print("  ! exp2 chunk files disagree on window size — some were generated "
              "with different settings. Re-run stage 2 for all weeks.")

    ch = manifest["vector_store"]
    if ch.get("available"):
        print(f"  Collections     {ch['n_collections']} "
              f"({ch['total_vectors']} vectors)")
    else:
        print(f"  ! Vector store not read: {ch.get('reason')}")

    print(f"\nWritten to {out_dir}/")
    print("  run_manifest.json")
    print("  run_manifest.md")


if __name__ == "__main__":
    main()
