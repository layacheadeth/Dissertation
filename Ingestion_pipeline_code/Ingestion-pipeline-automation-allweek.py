import argparse
import glob
import os
import re
import subprocess
from pathlib import Path


DATA_DIR = "Data/All_extracted_text"
CHROMA_PATH = "Data/Database/chroma_db"


def ingest_to_chromadb(chroma_path=CHROMA_PATH, data_dir=DATA_DIR, reset=True):
    """Stage 3: load every week's chunk files into ChromaDB.

    Runs ONCE after all lectures are chunked, not per lecture: 3-1 globs
    Data/Data_week*/ and ingests every week in one pass, so calling it inside
    the loop would re-ingest the whole corpus for each PDF.

    reset=True deletes each collection first. That is the right default here
    because this script has just regenerated every chunk file: without it,
    upsert only overwrites matching ids, so chunks that no longer exist under
    a changed id would linger in the collection forever.
    """
    print(f"\n=== Stage 3: Ingesting all weeks into ChromaDB ===")
    cmd = [
        "python3",
        "Ingestion_pipeline_code/3-1-Ingest-to-ChromaDB.py",
        "--data", data_dir,
        "--chroma-path", chroma_path,
    ]
    if reset:
        cmd.append("--reset")
    subprocess.run(cmd, check=True)


def process_lectures(chroma_path=CHROMA_PATH, data_dir=DATA_DIR, run_ingest=True, reset=True):
    # Find all PDF files in the lectures directory
    pdf_files = sorted(glob.glob("Data/All_lectures/*.pdf"))

    if not pdf_files:
        print("No PDF files found in Data/All_lectures/")
        return

    print(f"Found {len(pdf_files)} lectures to process...\n")

    processed = 0

    for pdf in pdf_files:
        pdf_path = Path(pdf)
        # Extract week number from filename (e.g., 'Week1' -> '1')
        match = re.search(r"Week(\d+)", pdf_path.name, re.IGNORECASE)

        if match:
            week_num = match.group(1)
            output_dir = f"Data/All_extracted_text/Data_week{week_num}"
            os.makedirs(output_dir, exist_ok=True)

            # File paths for all pipeline outputs
            extracted_json = Path(output_dir) / f"{pdf_path.stem}.json"
            exp1_out = Path(output_dir) / f"{pdf_path.stem}_exp1_chunks.json"
            exp2_out = Path(output_dir) / f"{pdf_path.stem}_exp2_chunks.json"
            exp3_out = Path(output_dir) / f"{pdf_path.stem}_exp3_chunks.json"
            exp4_out = Path(output_dir) / f"{pdf_path.stem}_exp4_chunks.json"
            exp5_out = Path(output_dir) / f"{pdf_path.stem}_exp5_chunks.json"


            print(f"=== Processing Week {week_num}: {pdf_path.name} ===")

            # Step 1: Extract, Clean, and Deduplicate Text
            print("  [1/6] Extracting raw text...")
            subprocess.run(
                [
                    "python3",
                    "Ingestion_pipeline_code/1-Extract_clean_format_text.py",
                    "--input",
                    str(pdf_path),
                    "--output-dir",
                    output_dir,
                ],
                check=True,
            )

            # Step 2: Page-Level Structural Chunking (Exp 1)
            print("  [2/6] Running Exp 1 (Page-Level)...")
            subprocess.run(
                [
                    "python3",
                    "Ingestion_pipeline_code/2-1-PageLevel-chunking.py",
                    "--input",
                    str(extracted_json),
                    "--output",
                    str(exp1_out),
                ],
                check=True,
            )

            # Step 3: Fixed-Size Overlapping Window Chunking (Exp 2)
            print("  [3/6] Running Exp 2 (Sliding Window)...")
            subprocess.run(
                [
                    "python3",
                    "Ingestion_pipeline_code/2-2-FixedSizeOverlapping-chunking.py",
                    "--input",
                    str(extracted_json),
                    "--output",
                    str(exp2_out),
                ],
                check=True,
            )

            # Step 4: Hybrid Recursive Structural Chunking (Exp 3)
            print("  [4/6] Running Exp 3 (Structure chunking)...")
            subprocess.run(
                [
                    "python3",
                    "Ingestion_pipeline_code/2-3-StructureLevel-chunking.py",
                    "--input",
                    str(extracted_json),
                    "--output",
                    str(exp3_out),
                ],
                check=True,
            )

            # Step 5: SematicAware Chunking (Exp 4)
            print("  [5/6] Running Exp 4 (SemanticAware chunking)...")
            subprocess.run(
                [
                    "python3",
                    "Ingestion_pipeline_code/2-4-SemanticAware-chunking.py",
                    "--input",
                    str(extracted_json),
                    "--output",
                    str(exp4_out),
                ],
                check=True,
            )

            # Step 6: SectionAware Chunking (Exp 5)
            print("  [6/6] Running Exp 5 (SectionAware chunking)...")
            subprocess.run(
                [
                    "python3",
                    "Ingestion_pipeline_code/2-5-SectionAware-chunking.py",
                    "--input",
                    str(extracted_json),
                    "--output",
                    str(exp5_out),
                ],
                check=True,
            )
                 

            processed += 1
            print(f"Finished processing Week {week_num}.\n")

        else:
            print(f"⚠️ Could not extract week number from {pdf}, skipping...")

    if processed == 0:
        print("No lectures were processed — skipping ChromaDB ingestion.")
        return

    print(f"All {processed} lectures processed through stages 1 and 2-1..2-5.")

    if not run_ingest:
        print("\nSkipping ChromaDB ingestion (--skip-ingest).")
        return

    # Stage 3 runs once, after every week has been chunked.
    ingest_to_chromadb(chroma_path, data_dir, reset=reset)

    print(f"\nPipeline complete: {processed} lectures chunked and ingested "
          f"into {chroma_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full ingestion pipeline: extract -> chunk -> ChromaDB."
    )
    parser.add_argument("--data", default=DATA_DIR,
                        help="directory containing Data_week*/ folders")
    parser.add_argument("--chroma-path", default=CHROMA_PATH,
                        help="where to write the ChromaDB store")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="run stages 1 and 2 only, leaving ChromaDB untouched")
    parser.add_argument("--ingest-only", action="store_true",
                        help="skip extraction and chunking; only run stage 3 on "
                             "the chunk files already on disk")
    parser.add_argument("--no-reset", action="store_true",
                        help="upsert into existing collections instead of "
                             "rebuilding them (rarely what you want)")
    args = parser.parse_args()

    if args.ingest_only:
        ingest_to_chromadb(args.chroma_path, args.data, reset=not args.no_reset)
    else:
        process_lectures(
            chroma_path=args.chroma_path,
            data_dir=args.data,
            run_ingest=not args.skip_ingest,
            reset=not args.no_reset,
        )