"""
The three chunking strategies.

Each one takes a lecture (already extracted by stage 1) and returns a list of
chunks. The 2-x scripts just call chunk_all() with a strategy name.

    exp1  one chunk per slide
    exp2  a fixed-size window sliding over the whole lecture
    exp3  slides sharing a heading are merged, then cut if too long
"""

import json
import re
from typing import Any, Dict, List, Optional

from Share_components import configuration
from Share_components.chunking_tokenizer import (
    audit_chunks,
    count_tokens,
    encode_with_offsets,
    split_text,
)


STRATEGY_NAMES = ["exp1", "exp2", "exp3"]

LABELS = {
    "exp1": "page-level",
    "exp2": "fixed-size overlapping",
    "exp3": "section-aware",
}


def _week_prefix(week: str) -> str:
    """'Week 1' becomes 'Week1', used to build chunk ids."""
    return (week or "").replace(" ", "")


# ---------------------------------------------------------------------------
# exp1: one chunk per slide
# ---------------------------------------------------------------------------
def chunk_page_level(lecture: Dict[str, Any]) -> List[Dict]:
    """One chunk per slide.

    A slide too long for the model is cut into parts, otherwise the model would
    read the start of it and silently ignore the rest.
    """
    filename = lecture.get("filename", "")
    week = lecture.get("week", "")
    chunks = []

    for page in lecture.get("pages", []):
        content = page.get("content", "").strip()
        page_num = page.get("page_number")

        if len(content) < configuration.MIN_SLIDE_CHARS:
            continue

        if count_tokens(content) > configuration.MAX_TOKENS:
            parts = split_text(content, configuration.MAX_TOKENS)
        else:
            parts = [content]

        base_id = f"{_week_prefix(week)}_p{page_num}"

        for part_index, part in enumerate(parts, start=1):
            chunks.append({
                "experiment_id": "exp1_page_level",
                # Slides that were not cut keep their plain id.
                "chunk_id": base_id if len(parts) == 1 else f"{base_id}_part{part_index}",
                "filename": filename,
                "week": week,
                "page_number": [page_num] if page_num is not None else [],
                "char_count": len(part),
                "word_count": len(part.split()),
                "token_count": count_tokens(part),
                "part_index": part_index,
                "part_count": len(parts),
                "was_split": len(parts) > 1,
                "content": part,
            })

    return chunks


# ---------------------------------------------------------------------------
# exp2: a fixed-size window sliding over the whole lecture
# ---------------------------------------------------------------------------
def chunk_fixed_window(lecture: Dict[str, Any]) -> List[Dict]:
    """Slide a fixed window across the lecture, ignoring slide boundaries.

    Neighbouring chunks share OVERLAP_TOKENS of text so a sentence split across
    the join still appears whole in one of them.
    """
    filename = lecture.get("filename", "")
    week = lecture.get("week", "")

    # Turn the whole lecture into one long list of tokens, remembering for each
    # token which slide it came from and where it sits in that slide's text.
    slide_texts: List[str] = []
    slide_numbers: List[Any] = []
    token_slide: List[int] = []
    token_position: List[tuple] = []

    for page in lecture.get("pages", []):
        content = page.get("content", "").strip()
        if not content:
            continue

        _ids, positions = encode_with_offsets(content)
        if not positions:
            continue

        slide_index = len(slide_texts)
        slide_texts.append(content)
        slide_numbers.append(page.get("page_number"))
        token_slide.extend([slide_index] * len(positions))
        token_position.extend([(int(a), int(b)) for a, b in positions])

    total = len(token_position)
    step = configuration.MAX_TOKENS - configuration.OVERLAP_TOKENS
    chunks = []
    number = 1

    for start in range(0, total, step):
        end = min(start + configuration.MAX_TOKENS, total)
        if end <= start:
            break

        # Walk the window and cut the original slide text wherever the window
        # crosses from one slide to the next.
        pieces: List[str] = []
        slides_used: List[Any] = []

        slide = token_slide[start]
        text_from, text_to = token_position[start]

        for i in range(start + 1, end):
            if token_slide[i] == slide:
                text_to = token_position[i][1]
            else:
                pieces.append(slide_texts[slide][text_from:text_to])
                slides_used.append(slide_numbers[slide])
                slide = token_slide[i]
                text_from, text_to = token_position[i]

        pieces.append(slide_texts[slide][text_from:text_to])
        slides_used.append(slide_numbers[slide])

        text = "\n".join(p.strip() for p in pieces if p.strip()).strip()
        if not text:
            continue

        chunks.append({
            "experiment_id": "exp2_sliding_window",
            "chunk_id": f"{_week_prefix(week)}_sw_{number}",
            "filename": filename,
            "week": week,
            "page_number": sorted({s for s in slides_used if s is not None}),
            "chunk_size_tokens": configuration.MAX_TOKENS,
            "overlap_tokens": configuration.OVERLAP_TOKENS,
            "window_token_count": end - start,
            "actual_token_count": count_tokens(text),
            "content": text,
        })
        number += 1

        if end >= total:
            break

    return chunks


# ---------------------------------------------------------------------------
# exp3: slides sharing a heading are merged into one section
# ---------------------------------------------------------------------------
# Matches "(cont.)", "2/3", "part 2" at the end of a heading.
_CONTINUED = re.compile(
    r"\s*[\(\[]?\s*(cont(inued|\.|d)?|ctd\.?|\d+\s*/\s*\d+|part\s+\d+)\s*[\)\]]?\s*$",
    re.IGNORECASE,
)

# Matches "1.", "2.3", "IV.", "A." at the start of a line.
_NUMBERED = re.compile(r"^\s*((\d+(\.\d+)*)|([IVXLC]+\.)|([A-Z]\.))\s+\S")

# Words that almost always start a new section.
_SECTION_WORDS = re.compile(
    r"^\s*(agenda|outline|overview|introduction|intro|background|motivation|"
    r"summary|recap|conclusion|references|appendix|q\s*&\s*a|questions|"
    r"objectives|learning outcomes|chapter|section|part|topic|lab|exercise|"
    r"example[s]?|case study)\b",
    re.IGNORECASE,
)


def normalise_title(title: str) -> str:
    """Strip '(cont.)' and trailing punctuation so two headings compare equal."""
    text = _CONTINUED.sub("", title.strip())
    text = re.sub(r"[\s:;.\-–—]+$", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_heading(content: str) -> Optional[str]:
    """Return the slide's first line if it looks like a heading, else None.

    A heading is short, does not end like a sentence, and is either numbered,
    a known section word, mostly capitals, or in Title Case.
    """
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        return None

    first = lines[0]
    words = first.split()

    if len(words) > 12:
        return None
    if first.endswith((".", "?", "!")) and not _NUMBERED.match(first):
        return None

    letters = [c for c in first if c.isalpha()]
    mostly_capitals = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.6
    capitalised = [w for w in words if w[:1].isupper()]
    title_case = bool(words) and len(capitalised) / len(words) >= 0.5

    if (_NUMBERED.match(first) or _SECTION_WORDS.match(first)
            or mostly_capitals or title_case):
        return first
    return None


def build_sections(pages: List[Dict]) -> List[Dict]:
    """Group slides under the same heading into one section.

    A slide with no heading, or with the same heading as the slide before it,
    joins the section already open.
    """
    sections: List[Dict] = []
    current: Optional[Dict] = None

    for page in pages:
        content = page.get("content", "").strip()
        page_num = page.get("page_number")

        if len(content) < configuration.MIN_SLIDE_CHARS:
            continue

        heading = extract_heading(content)
        simplified = normalise_title(heading) if heading else None

        starts_new_section = (
            current is None
            or (simplified is not None and simplified != current["simplified_title"])
        )

        if starts_new_section:
            current = {
                "title": heading or f"Untitled section (slide {page_num})",
                "simplified_title": simplified or f"__untitled_{page_num}",
                "pages": [],
                "body": [],
            }
            sections.append(current)

        current["pages"].append(page_num)
        current["body"].append(content)

    return sections


def chunk_section_aware(lecture: Dict[str, Any]) -> List[Dict]:
    """One chunk per section, cut into parts if the section is too long.

    The heading is put at the top of every part so a chunk taken from the
    middle of a section still says what it is about. Its tokens are subtracted
    from the budget first, so adding it back cannot push a part over the limit.
    """
    filename = lecture.get("filename", "")
    week = lecture.get("week", "")
    chunks = []

    for section_number, section in enumerate(build_sections(lecture.get("pages", [])), start=1):
        body = "\n\n".join(section["body"]).strip()
        if not body:
            continue

        title = section["title"]
        budget = configuration.MAX_TOKENS - count_tokens(f"{title}\n\n")
        use_title = budget >= 1

        # A heading long enough to fill the whole budget is dropped instead,
        # so there is still room for actual content.
        if not use_title:
            budget = configuration.MAX_TOKENS

        parts = split_text(body, budget)
        slides = sorted({p for p in section["pages"] if p is not None})

        for part_index, part in enumerate(parts, start=1):
            if use_title and not part.lstrip().startswith(title):
                content = f"{title}\n\n{part}"
            else:
                content = part

            chunks.append({
                "experiment_id": "exp3_section_aware",
                "chunk_id": f"{_week_prefix(week)}_sec{section_number}_part{part_index}",
                "filename": filename,
                "week": week,
                "page_number": slides,
                "section_index": section_number,
                "section_title": title,
                "part_index": part_index,
                "part_count": len(parts),
                "slide_count": len(section["pages"]),
                "token_count": count_tokens(content),
                "char_count": len(content),
                "content": content,
            })

    return chunks


STRATEGIES = {
    "exp1": chunk_page_level,
    "exp2": chunk_fixed_window,
    "exp3": chunk_section_aware,
}


# ---------------------------------------------------------------------------
# What the 2-x scripts call
# ---------------------------------------------------------------------------
def chunk_all(strategy: str) -> None:
    """Chunk every extracted lecture with one strategy and save the results.

    Reads  Data/All_extracted_text/Data_week*/<lecture>.json
    Writes Data/All_extracted_text/Data_week*/<lecture>_<strategy>_chunks.json
    """
    chunker = STRATEGIES[strategy]
    lecture_files = sorted(configuration.EXTRACTED_DIR.glob("Data_week*/*.json"))
    lecture_files = [p for p in lecture_files if "_chunks" not in p.name]

    if not lecture_files:
        print(f"No extracted lectures found in {configuration.EXTRACTED_DIR}")
        print("Run 1-Extract_clean_format_text.py first.")
        return

    print(f"Chunking {len(lecture_files)} lecture(s), {LABELS[strategy]} ({strategy})")

    total = 0
    for path in lecture_files:
        with open(path, "r", encoding="utf-8") as f:
            lecture = json.load(f)

        chunks = chunker(lecture)
        out_path = path.with_name(f"{path.stem}_{strategy}_chunks.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        print(f"\n{path.name}")
        audit_chunks(chunks, strategy)
        print(f"  saved to {out_path.name}")
        total += len(chunks)

    print(f"\nDone. {total} chunks in total.")
