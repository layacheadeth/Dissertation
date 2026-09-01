"""
Runs the whole pipeline, start to finish.

Run it:
    python Ingestion-pipeline-automation-allweek.py

It does the same thing as running these one after another:

    1-Extract_clean_format_text.py        PDFs        -> clean text
    2-1-PageLevel-chunking.py             clean text  -> exp1 chunks
    2-2-FixedSizeOverlapping-chunking.py  clean text  -> exp2 chunks
    2-3-SectionAware-chunking.py          clean text  -> exp3 chunks
    4-Corpus-statistics.py                chunks      -> tables for the report
    3-1-Ingest-to-ChromaDB-bge-Embed.py   chunks      -> database (BGE)
    5-Run-manifest.py                     everything  -> a record of this run

If you only want part of it, run the individual script instead. To change a
number or a folder, edit Share_components/configuration.py.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))

from Share_components import configuration
from Share_components.chunking_strategies import STRATEGY_NAMES, chunk_all
from Ingestion_pipeline.ingestion_strategy import ingest_all

sys.path.append(str(HERE))
from importlib import import_module

config_ingestion = configuration

# The file name starts with a number, so it cannot be imported normally.
extract_all = import_module("1-Extract_clean_format_text").extract_all


def save_settings() -> None:
    """Write the settings to a file so 5-Run-manifest.py can record them."""
    config_ingestion.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    settings = {
        "strategies": {
            "exp1": {"max_tokens": config_ingestion.MAX_TOKENS,
                     "min_slide_chars": config_ingestion.MIN_SLIDE_CHARS},
            "exp2": {"max_tokens": config_ingestion.MAX_TOKENS,
                     "overlap_tokens": config_ingestion.OVERLAP_TOKENS},
            "exp3": {"max_tokens": config_ingestion.MAX_TOKENS,
                     "min_slide_chars": config_ingestion.MIN_SLIDE_CHARS},
        },
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "token_unit": "WordPiece (the embedding model's own tokenizer)",
        "tokenizer": config_ingestion.TOKENIZER,
    }
    path = config_ingestion.ANALYSIS_DIR / "chunker_params.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def run_script(name: str) -> None:
    """Run one of the numbered scripts as a separate program.

    Started from the project root, because 4- and 5- look for folders like
    Data/Analysis relative to wherever they are run from.
    """
    subprocess.run([sys.executable, str(HERE / name)],
                   cwd=str(config_ingestion.ROOT), check=True)


def banner(text: str) -> None:
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")


if __name__ == "__main__":
    save_settings()

    banner("Step 1: reading the PDFs")
    extract_all()

    for strategy in STRATEGY_NAMES:
        banner(f"Step 2: chunking, {strategy}")
        chunk_all(strategy)

    banner("Step 3: corpus statistics")
    run_script("4-Corpus-statistics.py")

    banner("Step 4: loading the database with BGE")
    ingest_all()

    banner("Step 5: recording what this run did")
    run_script("5-Run-manifest.py")

    print(f"\nFinished. Database is in {config_ingestion.CHROMA_DIR}")
    print(f"Report tables are in {config_ingestion.ANALYSIS_DIR}")
