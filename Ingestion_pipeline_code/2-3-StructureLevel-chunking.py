import argparse
import json
import os
from typing import List, Dict, Any

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
except ImportError:
    # Fallback to character approximation (~4 chars per token) if tiktoken isn't installed
    def count_tokens(text: str) -> int:
        return len(text) // 4


def recursive_split_text(
    text: str, 
    max_tokens: int = 100, 
    separators: List[str] = None
) -> List[str]:
    """
    Recursively splits text using hierarchical separators (\n\n -> \n -> . -> space)
    until each segment fits within max_tokens.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    text = text.strip()
    if not text:
        return []

    # Base case: if text is within limit or no separators left, return as is
    if count_tokens(text) <= max_tokens or not separators:
        return [text]

    sep = separators[0]
    next_separators = separators[1:]

    splits = text.split(sep) if sep != "" else list(text)
    chunks = []
    current_chunk = []

    for split in splits:
        # Build test string using current accumulated pieces plus next candidate
        if current_chunk:
            candidate = sep.join(current_chunk + [split])
        else:
            candidate = split

        if count_tokens(candidate) <= max_tokens:
            current_chunk.append(split)
        else:
            # Current chunk has reached maximum safe capacity
            if current_chunk:
                chunks.append(sep.join(current_chunk).strip())
                current_chunk = []
            
            # If a single split segment itself is oversized, recursively break it down
            if count_tokens(split) > max_tokens and next_separators:
                sub_chunks = recursive_split_text(split, max_tokens, next_separators)
                chunks.extend(sub_chunks)
            else:
                current_chunk.append(split)

    if current_chunk:
        chunks.append(sep.join(current_chunk).strip())

    return [c for c in chunks if c]


def run_experiment_3(
    input_json_path: str, 
    max_tokens: int = 100
) -> List[Dict[str, Any]]:
    """
    Experiment 3: Hybrid Recursive Structural Chunking
    Respects page boundaries while recursively splitting large page text 
    down to token limits using natural text separators.
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

        if not content:
            continue

        # Perform hybrid recursive splitting on page content
        page_chunks = recursive_split_text(content, max_tokens=max_tokens)

        for sub_idx, chunk_text in enumerate(page_chunks, start=1):
            chunk_record = {
                "experiment_id": "exp3_hybrid_recursive",
                "chunk_id": f"{week.replace(' ', '')}_p{page_num}_sub{sub_idx}",
                "filename": filename,
                "week": week,
                "page_number": page_num,
                "sub_chunk_index": sub_idx,
                "token_count": count_tokens(chunk_text),
                "content": chunk_text
            }
            chunks.append(chunk_record)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 3: Hybrid Recursive Structural Chunking on slide JSON data."
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        type=str, 
        help="Path to input JSON file"
    )
    
    parser.add_argument(
        "-o", "--output", 
        required=False, 
        type=str, 
        default=None, 
        help="Path to save output JSON file"
    )
    
    parser.add_argument(
        "-m", "--max-tokens", 
        required=False, 
        type=int, 
        default=150, 
        help="Maximum allowed tokens per chunk (default: 250)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    results = run_experiment_3(args.input, max_tokens=args.max_tokens)
    print(f"[Exp 3] Generated {len(results)} hybrid recursive chunks.")

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Exp 3] Saved results to: {args.output}")
    else:
        print("\n--- Sample Output Chunk ---")
        if results:
            print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()