import argparse
import json
import os
from typing import List, Dict, Any

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def token_length(text: str) -> int:
        return len(tokenizer.encode(text))
    def tokenize(text: str) -> List[int]:
        return tokenizer.encode(text)
    def decode(tokens: List[int]) -> str:
        return tokenizer.decode(tokens)
except ImportError:
    # Fallback to character approximation (~4 chars/token) if tiktoken not installed
    def token_length(text: str) -> int:
        return len(text) // 4
    def tokenize(text: str) -> List[str]:
        return text.split()
    def decode(tokens: List[str]) -> str:
        return " ".join(tokens)


def run_experiment_2(
    input_json_path: str, 
    chunk_size: int = 300, 
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Experiment 2: Fixed-Size Overlapping Sliding Window Chunking
    Flattens text and applies fixed-token windowing ignoring page boundaries.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filename = data.get("filename", "")
    week = data.get("week", "")
    pages = data.get("pages", [])

    # Concatenate all page contents with explicit source tracking tags
    flattened_text = ""
    for page in pages:
        content = page.get("content", "").strip()
        if content:
            flattened_text += f"\n[Slide {page.get('page_number')}]\n" + content

    tokens = tokenize(flattened_text)
    chunks = []
    chunk_idx = 1
    step = chunk_size - overlap

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = decode(chunk_tokens).strip()

        if not chunk_text:
            continue

        chunk_record = {
            "experiment_id": "exp2_sliding_window",
            "chunk_id": f"{week.replace(' ', '')}_sw_{chunk_idx}",
            "filename": filename,
            "week": week,
            "chunk_size_tokens": chunk_size,
            "overlap_tokens": overlap,
            "actual_token_count": len(chunk_tokens),
            "content": chunk_text
        }
        chunks.append(chunk_record)
        chunk_idx += 1

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 2: Fixed-Size Overlapping Sliding Window Chunking on slide JSON data."
    )
    
    # Required CLI arguments
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        type=str, 
        help="Path to input JSON file"
    )
    
    # Optional CLI arguments
    parser.add_argument(
        "-o", "--output", 
        required=False, 
        type=str, 
        default=None, 
        help="Path to save output JSON file"
    )
    
    parser.add_argument(
        "-s", "--size", 
        required=False, 
        type=int, 
        default=300, 
        help="Target chunk size in tokens (default: 300)"
    )
    
    parser.add_argument(
        "-v", "--overlap", 
        required=False, 
        type=int, 
        default=50, 
        help="Overlap size in tokens (default: 50)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    # Run the fixed-size sliding window chunking
    results = run_experiment_2(args.input, chunk_size=args.size, overlap=args.overlap)
    print(f"[Exp 2] Generated {len(results)} fixed-window chunks.")

    # Save to file if output argument is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Exp 2] Saved results to: {args.output}")
    else:
        print("\n--- Sample Output Chunk ---")
        if results:
            print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()