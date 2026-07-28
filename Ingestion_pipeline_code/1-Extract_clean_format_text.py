#!/usr/bin/env bash
#!/usr/bin/env python3
"""
PDF Text Extractor, Cleaner & Deduplicator
------------------------------------------
Extracts text from a single PDF document, cleans noise (bullet points, 
slide numbers), deduplicates consecutive build-up slides (retaining only the final, complete slide),
and exports structured JSON.

Usage:
    python Extract_clean_format_text.py --input Data/All_lectures/Week1_Intro_to_vector.pdf --output-dir Data/Data_week1
"""

import argparse
import json
import logging
import re
from pathlib import Path
from pypdf import PdfReader

# Suppress non-fatal pypdf warnings regarding broken XRef tables
logging.getLogger("pypdf").setLevel(logging.ERROR)


def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from PDF slides/documents.
    """
    if not text:
        return ""

    # Remove special bullet symbols or non-standard bullet characters
    text = re.sub(r'[\u25a0\u25a1\u25aa\u25ab\u2022\u2023\u2043\u2219]', '', text)

    # Normalize Windows line endings to Unix line endings
    text = text.replace('\r\n', '\n')

    # Filter out lines that consist solely of page/slide numbers (1-3 digits)
    lines = text.split('\n')
    filtered_lines = [
        line for line in lines 
        if not re.match(r'^\s*\d{1,3}\s*$', line)
    ]
    text = '\n'.join(filtered_lines)

    # Replace multiple horizontal spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse 3 or more consecutive newlines into double newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_week_info(filename: str) -> str:
    """
    Extracts the week string (e.g., 'Week 1') from a file name.
    """
    match = re.search(r'week[_\s\-]*(\d+)', filename, re.IGNORECASE)
    if match:
        week_num = match.group(1)
        return f"Week {week_num}"
    return "Unknown"


def normalize_for_comparison(text: str) -> str:
    """
    Collapses all whitespace and lowercases text for flexible subset comparison.
    This prevents minor line-wrap or spacing variations in PDF extraction 
    from breaking progressive slide detection.
    """
    return " ".join(text.lower().split())


def deduplicate_pages(pages: list[dict]) -> list[dict]:
    """
    Deduplicates consecutive slides that build up text incrementally.
    
    If slide N's content is completely contained within slide N+1 (progressive reveal),
    slide N is discarded so that only the final, complete slide (N+1) is retained.
    """
    if not pages:
        return []

    deduplicated = []
    num_pages = len(pages)
    
    for i in range(num_pages - 1):
        current_page = pages[i]
        next_page = pages[i + 1]
        
        current_content = current_page["content"]
        next_content = next_page["content"]

        # Normalize texts to catch subset relationships regardless of spacing/formatting changes
        norm_current = normalize_for_comparison(current_content)
        norm_next = normalize_for_comparison(next_content)

        # Case 1: Exact duplicate consecutive pages
        if norm_current == norm_next:
            continue

        # Case 2: Progressive animation slide (Current slide is a subset of the next slide)
        # Skip the current page because the next page contains all of its content plus additional reveals
        if norm_current and norm_current in norm_next and len(norm_current) < len(norm_next):
            continue

        # If it's a unique or final state slide, keep it
        deduplicated.append(current_page)

    # Always retain the very last page of the document
    deduplicated.append(pages[-1])

    return deduplicated


def extract_pdf_data(pdf_path: Path) -> dict:
    """
    Extracts, cleans, and deduplicates text page by page from a PDF file.
    """
    reader = PdfReader(str(pdf_path))
    total_raw_pages = len(reader.pages)
    raw_extracted_pages = []

    for index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_content = clean_text(raw_text)

        # Only add pages that actually contain readable text
        if cleaned_content:
            raw_extracted_pages.append({
                "page_number": index,
                "content": cleaned_content
            })

    # Deduplicate consecutive incremental slides
    clean_pages = deduplicate_pages(raw_extracted_pages)

    # Extract week metadata
    week_attribute = extract_week_info(pdf_path.name)

    return {
        "filename": pdf_path.name,
        "week": week_attribute,
        "raw_total_pages": total_raw_pages,
        "deduplicated_total_pages": len(clean_pages),
        "pages": clean_pages
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract, clean, and deduplicate text from a single PDF and export as JSON."
    )
    
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True, 
        help="Path to the target input PDF file."
    )
    
    parser.add_argument(
        "-o", "--output-dir", 
        type=str, 
        default=".", 
        help="Directory path where output JSON file will be saved. (Default: current directory)"
    )

    args = parser.parse_args()

    input_file = Path(args.input).resolve()
    output_directory = Path(args.output_dir).resolve()

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        return
        
    if not input_file.suffix.lower() == ".pdf":
        print(f"Error: Input file '{input_file}' is not a PDF file.")
        return

    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / f"{input_file.stem}.json"

    print(f"Processing: {input_file.name} ...")

    try:
        data = extract_pdf_data(input_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Success!")
        print(f"  - Original pages: {data['raw_total_pages']}")
        print(f"  - Deduplicated pages: {data['deduplicated_total_pages']}")
        print(f"  - Output written to: {output_file}")

    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")


if __name__ == "__main__":
    main()