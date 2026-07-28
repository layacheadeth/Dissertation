import argparse
import json
import os
import re
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Token counting (same convention as experiments 2, 3 and 4)
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------
_CONT_MARKERS = re.compile(
    r"\s*[\(\[]?\s*(cont(inued|\.|d)?|ctd\.?|\d+\s*/\s*\d+|part\s+\d+)\s*[\)\]]?\s*$",
    re.IGNORECASE,
)

_NUMBERED = re.compile(r"^\s*((\d+(\.\d+)*)|([IVXLC]+\.)|([A-Z]\.))\s+\S")

_SECTION_KEYWORDS = re.compile(
    r"^\s*(agenda|outline|overview|introduction|intro|background|motivation|"
    r"summary|recap|conclusion|references|appendix|q\s*&\s*a|questions|"
    r"objectives|learning outcomes|chapter|section|part|topic|lab|exercise|"
    r"example[s]?|case study)\b",
    re.IGNORECASE,
)


def normalise_title(title: str) -> str:
    """Strip continuation markers and trailing punctuation for comparison."""
    t = _CONT_MARKERS.sub("", title.strip())
    t = re.sub(r"[\s:;.\-–—]+$", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def extract_heading(content: str, max_heading_words: int = 12) -> Optional[str]:
    """
    Treat the first non-empty line of a slide as a candidate heading if it looks
    like one: short, not sentence-terminated, and either numbered, title/upper
    cased, or matching a structural keyword.
    """
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        return None

    first = lines[0]
    words = first.split()

    if len(words) > max_heading_words:
        return None
    if first.endswith((".", "?", "!")) and not _NUMBERED.match(first):
        return None

    looks_numbered = bool(_NUMBERED.match(first))
    looks_keyword = bool(_SECTION_KEYWORDS.match(first))
    alpha = [c for c in first if c.isalpha()]
    looks_upper = bool(alpha) and sum(c.isupper() for c in alpha) / len(alpha) > 0.6
    capitalised_words = [w for w in words if w[:1].isupper()]
    looks_titlecase = bool(words) and len(capitalised_words) / len(words) >= 0.5

    if looks_numbered or looks_keyword or looks_upper or looks_titlecase:
        return first
    return None


# ---------------------------------------------------------------------------
# Recursive splitter (reused from Experiment 3 so oversized sections stay usable)
# ---------------------------------------------------------------------------
def recursive_split_text(
    text: str,
    max_tokens: int = 300,
    separators: List[str] = None
) -> List[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    text = text.strip()
    if not text:
        return []

    if count_tokens(text) <= max_tokens or not separators:
        return [text]

    sep = separators[0]
    next_separators = separators[1:]

    splits = text.split(sep) if sep != "" else list(text)
    chunks = []
    current_chunk = []

    for split in splits:
        candidate = sep.join(current_chunk + [split]) if current_chunk else split

        if count_tokens(candidate) <= max_tokens:
            current_chunk.append(split)
        else:
            if current_chunk:
                chunks.append(sep.join(current_chunk).strip())
                current_chunk = []

            if count_tokens(split) > max_tokens and next_separators:
                chunks.extend(recursive_split_text(split, max_tokens, next_separators))
            else:
                current_chunk.append(split)

    if current_chunk:
        chunks.append(sep.join(current_chunk).strip())

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Experiment 5
# ---------------------------------------------------------------------------
def build_sections(
    pages: List[Dict[str, Any]],
    min_char_threshold: int = 25,
) -> List[Dict[str, Any]]:
    """
    Group consecutive slides into sections. A new section starts when a slide
    presents a heading that differs from the current section heading. Slides
    with no heading, or with a repeated/continuation heading, are absorbed into
    the section already in progress.
    """
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for page in pages:
        content = page.get("content", "").strip()
        page_num = page.get("page_number")

        if len(content) < min_char_threshold:
            continue

        heading = extract_heading(content)
        norm = normalise_title(heading) if heading else None

        starts_new = (
            current is None
            or (norm is not None and norm != current["norm_title"])
        )

        if starts_new:
            current = {
                "title": heading if heading else f"Untitled section (slide {page_num})",
                "norm_title": norm if norm else f"__untitled_{page_num}",
                "pages": [],
                "body": [],
            }
            sections.append(current)

        current["pages"].append(page_num)
        current["body"].append(content)

    return sections


def run_experiment_5(
    input_json_path: str,
    max_tokens: int = 300,
    min_char_threshold: int = 25,
    prepend_title: bool = True,
) -> List[Dict[str, Any]]:
    """
    Experiment 5: Section-Aware Chunking
    Detects slide headings, merges consecutive slides that belong to the same
    section (including "cont." continuation slides), and emits one chunk per
    section. Sections exceeding the token budget are recursively split, with the
    section title prepended to every part so retrieval context is preserved.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = data.get("filename", "")
    week = data.get("week", "")
    pages = data.get("pages", [])

    sections = build_sections(pages, min_char_threshold=min_char_threshold)

    chunks: List[Dict[str, Any]] = []

    for sec_idx, section in enumerate(sections, start=1):
        body = "\n\n".join(section["body"]).strip()
        if not body:
            continue

        parts = recursive_split_text(body, max_tokens=max_tokens)
        page_list = [p for p in section["pages"] if p is not None]

        for part_idx, part in enumerate(parts, start=1):
            if prepend_title and not part.lstrip().startswith(section["title"]):
                content = f"{section['title']}\n\n{part}"
            else:
                content = part

            chunk_record = {
                "experiment_id": "exp5_section_aware",
                "chunk_id": f"{week.replace(' ', '')}_sec{sec_idx}_part{part_idx}",
                "filename": filename,
                "week": week,
                "section_index": sec_idx,
                "section_title": section["title"],
                "part_index": part_idx,
                "part_count": len(parts),
                "page_start": page_list[0] if page_list else None,
                "page_end": page_list[-1] if page_list else None,
                "source_pages": page_list,
                "slide_count": len(section["pages"]),
                "token_count": count_tokens(content),
                "char_count": len(content),
                "content": content,
            }
            chunks.append(chunk_record)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 5: Section-Aware Chunking on slide JSON data."
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
        default=300,
        help="Maximum tokens per chunk before a section is split (default: 300)"
    )

    parser.add_argument(
        "-t", "--threshold",
        required=False,
        type=int,
        default=25,
        help="Minimum character length for a slide to be considered (default: 25)"
    )

    parser.add_argument(
        "--no-title-prefix",
        action="store_true",
        help="Do not prepend the section title to each chunk"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    results = run_experiment_5(
        args.input,
        max_tokens=args.max_tokens,
        min_char_threshold=args.threshold,
        prepend_title=not args.no_title_prefix,
    )
    section_count = len({c["section_index"] for c in results})
    print(f"[Exp 5] Generated {len(results)} section-aware chunks across {section_count} sections.")

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Exp 5] Saved results to: {args.output}")
    else:
        print("\n--- Sample Output Chunk ---")
        if results:
            print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
