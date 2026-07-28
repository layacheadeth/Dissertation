import argparse
import json
import os
from typing import List, Dict, Any

def run_experiment_1(input_json_path: str, min_char_threshold: int = 25) -> List[Dict[str, Any]]:
    """
    Experiment 1: Page-Level Structural Chunking
    Keeps each deduplicated slide page as a single chunk.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filename = data.get("filename", "")
    week = data.get("week", "")
    pages = data.get("pages", [])

    chunks = []
    
    for page in pages:
        content = page.get("content", "").strip()
        page_num = page.get("page_number")

        # Filter low-information / transition slides
        if len(content) < min_char_threshold:
            continue

        chunk_record = {
            "experiment_id": "exp1_page_level",
            "chunk_id": f"{week.replace(' ', '')}_p{page_num}",
            "filename": filename,
            "week": week,
            "page_number": page_num,
            "char_count": len(content),
            "word_count": len(content.split()),
            "content": content
        }
        chunks.append(chunk_record)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 1: Page-Level Structural Chunking on slide JSON data."
    )
    
    # Required arguments
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        type=str, 
        help="Path to the input JSON file (e.g., Data/Data_week1/Week1_Intro_to_vector.json)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output", 
        required=False, 
        type=str, 
        default=None, 
        help="Path to save the output JSON file (optional). If omitted, prints summary to console."
    )
    
    parser.add_argument(
        "-t", "--threshold", 
        required=False, 
        type=int, 
        default=25, 
        help="Minimum character length threshold to keep a page (default: 25)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    # Run the chunking experiment
    results = run_experiment_1(args.input, min_char_threshold=args.threshold)
    print(f"[Exp 1] Successfully generated {len(results)} page-level chunks.")

    # Save to output file if specified
    if args.output:
        # Ensure target directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Exp 1] Saved results to: {args.output}")
    else:
        print("\n--- Sample Output Chunk ---")
        if results:
            print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()