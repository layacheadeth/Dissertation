"""
Step 1: read the lecture PDFs and save clean text.

Run it:
    python 1-Extract_clean_format_text.py

Reads  every PDF in Data/All_lectures/
Writes Data/All_extracted_text/Data_week1/Week1_Intro.json  (and so on)

It removes bullet symbols and page numbers, and drops the half-finished copies
of animated slides so only the complete version is kept.
"""

import json
import logging
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pypdf import PdfReader

from Share_components import configuration

config_ingestion = configuration

# pypdf complains loudly about PDFs it then reads perfectly well.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def clean_text(text: str) -> str:
    """Remove bullet symbols, lone page numbers and messy spacing."""
    if not text:
        return ""

    text = re.sub(r"[\u25a0\u25a1\u25aa\u25ab\u2022\u2023\u2043\u2219]", "", text)
    text = text.replace("\r\n", "\n")

    # Drop any line that is only a number, which is a slide number.
    lines = [l for l in text.split("\n") if not re.match(r"^\s*\d{1,3}\s*$", l)]
    text = "\n".join(lines)

    text = re.sub(r"[ \t]+", " ", text)       # runs of spaces become one space
    text = re.sub(r"\n{3,}", "\n\n", text)    # runs of blank lines become one
    return text.strip()


def find_week(filename: str) -> str:
    """'Week1_Intro.pdf' becomes 'Week 1'."""
    match = re.search(r"week[_\s\-]*(\d+)", filename, re.IGNORECASE)
    return f"Week {match.group(1)}" if match else "Unknown"


def simplify(text: str) -> str:
    """Lowercase and squash spaces, so wrapping differences do not matter."""
    return " ".join(text.lower().split())


def remove_animation_duplicates(pages: list) -> list:
    """Drop a slide if the next slide contains everything it says.

    An animated slide comes out of the PDF as several pages that each add a
    line to the one before, so only the last one is worth keeping.
    """
    if not pages:
        return []

    kept = []
    for i in range(len(pages) - 1):
        this_slide = simplify(pages[i]["content"])
        next_slide = simplify(pages[i + 1]["content"])

        if this_slide == next_slide:
            continue
        if this_slide and this_slide in next_slide and len(this_slide) < len(next_slide):
            continue

        kept.append(pages[i])

    kept.append(pages[-1])      # the last slide is always a finished one
    return kept


def extract_one_pdf(pdf_path: Path) -> dict:
    """Read one PDF and return its cleaned pages plus some counts."""
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    pages_with_text = []
    for number, page in enumerate(reader.pages, start=1):
        content = clean_text(page.extract_text() or "")
        if content:
            pages_with_text.append({"page_number": number, "content": content})

    final_pages = remove_animation_duplicates(pages_with_text)

    # Three counts, kept apart so you can say where each lost page went:
    # picture-only slides are one thing, animation duplicates are another.
    return {
        "filename": pdf_path.name,
        "week": find_week(pdf_path.name),
        "raw_total_pages": total_pages,
        "pages_with_text": len(pages_with_text),
        "empty_pages_removed": total_pages - len(pages_with_text),
        "deduplicated_total_pages": len(final_pages),
        "pages_removed_by_dedup": len(pages_with_text) - len(final_pages),
        "pages": final_pages,
    }


def extract_all() -> None:
    """Read every lecture PDF and save one JSON file each."""
    pdf_files = sorted(config_ingestion.LECTURES_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {config_ingestion.LECTURES_DIR}")
        return

    print(f"Found {len(pdf_files)} lecture(s)")

    for pdf_path in pdf_files:
        match = re.search(r"week(\d+)", pdf_path.name, re.IGNORECASE)
        if not match:
            print(f"  ! {pdf_path.name} has no week number in its name, skipping")
            continue

        output_dir = config_ingestion.EXTRACTED_DIR / f"Data_week{match.group(1)}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pdf_path.stem}.json"

        data = extract_one_pdf(pdf_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"\n{pdf_path.name}")
        print(f"  {data['raw_total_pages']} pages in the PDF")
        print(f"  {data['empty_pages_removed']} picture-only or blank, removed")
        print(f"  {data['pages_removed_by_dedup']} animation duplicates, removed")
        print(f"  {data['deduplicated_total_pages']} pages kept -> {output_path.name}")


if __name__ == "__main__":
    extract_all()
