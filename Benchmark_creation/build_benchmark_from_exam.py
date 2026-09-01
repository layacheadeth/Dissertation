import glob
import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from Evaluation.exam_source import EXAM

LECTURE_PATHS = [
    "Data/All_extracted_text/Data_week1/Week1_Intro_to_vector.json",
    "Data/All_extracted_text/Data_week2/Week2_Language_modelling.json",
    "Data/All_extracted_text/Data_week3/Week3_word2vec_RNN_Transformer.json",
    "Data/All_extracted_text/Data_week4/Week4_LLM-Data.json",
    "Data/All_extracted_text/Data_week5/Week5_TRIM_LLM-pretraining.json",
    "Data/All_extracted_text/Data_week6/Week6_TRIM_SFT_alignment.json",
    "Data/All_extracted_text/Data_week7/Week7-TRIM_Incontext-Evaluation.json",
    "Data/All_extracted_text/Data_week8/Week8_TRIM-Application-Multimodal.json",
    "Data/All_extracted_text/Data_week9/Week9_RAG_LLM_final.json",
    "Data/All_extracted_text/Data_week10/Week10_TM_last_lecture.json",
    "Data/All_extracted_text/Data_week11/Week11_TRIM_BERT.json",
    "Data/All_extracted_text/Data_week12/Week12_TM-lastlecture-1.json",
    "Data/All_extracted_text/Data_week13/Week13_revision_lecture_exam_focused.json",
]

# --- CONFIGURATION ---
load_dotenv()
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

QA_OUTPUT = "Data/Benchmark/benchmark_qa.json"
SOCRATIC_OUTPUT = "Data/Benchmark/benchmark_socratic.json"

MAX_GOLD_PAGES = 5


# --- 1. DATA LOADER & CROSS-WEEK INDEX ---
def _resolve_paths(paths):
    resolved_files = []
    for path in paths:
        if os.path.isfile(path):
            resolved_files.append(path)
        elif os.path.isdir(path):
            resolved_files.extend(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
        else:
            print(f"Warning: path not found, skipping: {path}")
    return resolved_files


def load_lecture_data(paths):
    """Returns {(week, primary_page): {"merged_pages": [...], "content": str}} and a TOC string."""
    page_index = {}
    toc_entries = []

    for file_path in _resolve_paths(paths):
        if "__macosx" in file_path.lower() or "_chunks.json" in file_path.lower():
            continue

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not (isinstance(data, dict) and "pages" in data and "week" in data):
            continue

        match = re.search(r"(\d+)", data.get("week", "0"))
        week_num = int(match.group(1)) if match else 0

        for page in data.get("pages", []):
            content = (page.get("content") or "").strip()
            if len(content) < 40:
                continue

            raw_page = page.get("page_number")
            if isinstance(raw_page, list):
                pages_list = raw_page or [1]
            else:
                pages_list = [raw_page if raw_page else 1]
            primary_page = pages_list[0]

            page_index[(week_num, primary_page)] = {
                "merged_pages": pages_list,
                "content": content,
            }

            snippet = content[:120].replace("\n", " ")
            toc_entries.append(f"W{week_num} P{primary_page}: {snippet}...")

    return page_index, "\n".join(toc_entries)


# --- 2. SYSTEM PROMPT ---
SYSTEM_PROMPT = f"""You are a curriculum assistant for an NLP course.
Given an EXAM QUESTION, its CORRECT ANSWER, and a LECTURE SLIDE INDEX, return a JSON object.

Return strictly JSON with these keys:

1. "gold_pages": between 1 and {MAX_GOLD_PAGES} objects, ranked most relevant first. Each object:
     {{"week": <int>, "page": <int>, "role": "primary" | "supporting", "why": "<max 10 words: what this slide supplies>"}}
   INCLUSION RULE: include a page ONLY if a student could not fully answer the
   question without it. Most questions are covered by 1 or 2 slides. Returning
   {MAX_GOLD_PAGES} pages is unusual and is correct only when the question genuinely
   spans several lectures. Prefer fewer, precise pages over more. Do NOT pad the list
   with loosely related slides. Exactly one page must have role "primary" (rank 1).
   If no slide in the index covers the question, return an empty list.

2. "keywords": 3 to 7 key domain terms or technical concepts critical to answering this question.

3. "socratic_hint": ONE short teaching hint guiding a student toward the solution
   WITHOUT revealing the final answer.
   - Conceptual questions: nudge toward the underlying mechanism or definition.
   - Calculation questions: point to the required formula, key variables, or steps.

4. "difficulty": "easy" | "medium" | "hard"

Return strictly valid JSON. No markdown fences, no extra commentary."""


def query_llm(item, question_text, toc_text):
    user_prompt = (
        f"QUESTION:\n{question_text}\n\n"
        f"CORRECT ANSWER:\n{item['answer']}\n\n"
        f"LECTURE INDEX (ALL WEEKS):\n{toc_text}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


# --- 3. RESOLVE MODEL ANCHORS AGAINST THE REAL INDEX ---
def resolve_gold_pages(raw_gold_pages, page_index):
    """Keeps only anchors that exist in the index. Returns (gold_pages, dropped)."""
    gold_pages, dropped, seen = [], [], set()

    for gp in raw_gold_pages[:MAX_GOLD_PAGES]:
        try:
            week = int(gp.get("week"))
            page = int(gp.get("page"))
        except (TypeError, ValueError):
            dropped.append(gp)
            continue

        if (week, page) in seen:
            continue
        seen.add((week, page))

        data = page_index.get((week, page))
        if data is None:
            # Hallucinated anchor: never emit it with empty content.
            dropped.append({"week": week, "page": page})
            continue

        entry = {
            "rank": len(gold_pages) + 1,
            "week": week,
            "page": page,
            "role": gp.get("role", "primary" if not gold_pages else "supporting"),
            "why": (gp.get("why") or "").strip(),
            "content": data["content"],
        }
        # Only carry a page list when the slide really spans more than one page.
        if len(data["merged_pages"]) > 1:
            entry["merged_pages"] = data["merged_pages"]

        gold_pages.append(entry)

    return gold_pages, dropped


# --- 4. BUILD BENCHMARK ---
def build():
    print("Loading slides across all weeks and building global index...")
    page_index, toc_text = load_lecture_data(LECTURE_PATHS)
    print(f"Loaded {len(page_index)} slides. Processing {len(EXAM)} exam questions...\n")

    qa_benchmark, socratic_benchmark = [], []
    length_counts = {}

    for i, item in enumerate(EXAM, 1):
        query_id = f"EXAM_{i:03d}"
        try:
            question_text = item.get("q") or item.get(f"q{i}") or list(item.values())[0]

            res = query_llm(item, question_text, toc_text)
            gold_pages, dropped = resolve_gold_pages(res.get("gold_pages", []), page_index)

            review_reasons = []
            if not gold_pages:
                # No fabricated W1P1 fallback: an unanchored item is flagged, not invented.
                review_reasons.append("no_gold_page_resolved")
            if dropped:
                review_reasons.append(f"unresolved_anchors:{len(dropped)}")

            entry = {
                "query_id": query_id,
                "student_query": question_text,
                "question_type": item.get("kind", "conceptual"),
                "difficulty": res.get("difficulty", "medium"),
                "keywords": res.get("keywords", []),
                "gold_pages": gold_pages,
                "qa_gold_standard": item["answer"],
            }
            if review_reasons:
                entry["needs_review"] = review_reasons

            qa_benchmark.append(entry)

            hint = (res.get("socratic_hint") or "").strip()
            if hint:
                socratic_benchmark.append({**entry, "socratic_gold_standard": hint})

            length_counts[len(gold_pages)] = length_counts.get(len(gold_pages), 0) + 1

            summary = ", ".join(f"W{p['week']}P{p['page']}" for p in gold_pages) or "NONE"
            flag = "  [REVIEW]" if review_reasons else ""
            print(f"[{i:02d}/{len(EXAM):02d}] {query_id} -> [{summary}] | {item.get('kind')}{flag}")

        except Exception as e:
            print(f"[{i:02d}/{len(EXAM):02d}] Error processing {query_id}: {e}")

    os.makedirs(os.path.dirname(QA_OUTPUT) or ".", exist_ok=True)
    json.dump(qa_benchmark, open(QA_OUTPUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(socratic_benchmark, open(SOCRATIC_OUTPUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\nSuccessfully built benchmark dataset!")
    print(f"  QA File:        {QA_OUTPUT} ({len(qa_benchmark)} items)")
    print(f"  Socratic File:  {SOCRATIC_OUTPUT} ({len(socratic_benchmark)} items)")

    print("\nGold pages per question (check this for padding):")
    for n in sorted(length_counts):
        print(f"  {n} page(s): {length_counts[n]} question(s)")
    flagged = sum(1 for e in qa_benchmark if "needs_review" in e)
    if flagged:
        print(f"\n{flagged} item(s) flagged with needs_review.")


if __name__ == "__main__":
    build()