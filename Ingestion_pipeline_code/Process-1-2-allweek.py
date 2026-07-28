import glob
import os
import re
import subprocess
from pathlib import Path


def process_lectures():
    # Find all PDF files in the lectures directory
    pdf_files = sorted(glob.glob("Data/All_lectures/*.pdf"))

    if not pdf_files:
        print("No PDF files found in Data/All_lectures/")
        return

    print(f"Found {len(pdf_files)} lectures to process...\n")

    for pdf in pdf_files:
        pdf_path = Path(pdf)
        # Extract week number from filename (e.g., 'Week1' -> '1')
        match = re.search(r"Week(\d+)", pdf_path.name, re.IGNORECASE)

        if match:
            week_num = match.group(1)
            output_dir = f"Data/Data_week{week_num}"
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

            # Step 6: SectionAware Chunking (Exp 3)
            print("  [6/6] Running Exp 5 (SemanticAware chunking)...")
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
                 

            print(f"Finished processing Week {week_num}.\n")

        else:
            print(f"⚠️ Could not extract week number from {pdf}, skipping...")

    print("All lectures processed successfully through stages 1, 2-1, 2-2, 2-3, 2-4 and 2-5!")


if __name__ == "__main__":
    process_lectures()