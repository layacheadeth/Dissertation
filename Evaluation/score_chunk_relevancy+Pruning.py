"""
Grade each gold chunk 0-3 by contribution to the gold answer, then prune.

    python score_chunk_utility.py              # score only
    python score_chunk_utility.py --prune 2    # score, then prune (keep >= 2)

Schema expected per gold_page: rank, week, page, role, why, content.
Title-only slides are graded 0 without an LLM call. The judge sees the
neighbouring slides as context, since slides are short and sequentially
dependent (p54 defines what p55 uses).
"""

import argparse, glob, json, os, re, statistics, sys
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

# --- CONFIGURATION -----------------------------------------------------

QA_INPUT = "Data/Benchmark/benchmark_qa.json"
QA_SCORED = "Data/Benchmark/benchmark_qa_scored.json"
QA_PRUNED = "Data/Benchmark/benchmark_qa_pruned.json"

LECTURE_PATHS = [
    "Data/Data_week1/Week1_Intro_to_vector.json",
    "Data/Data_week2/Week2_Language_modelling.json",
    "Data/Data_week3/Week3_word2vec_RNN_Transformer.json",
    "Data/Data_week4/Week4_LLM-Data.json",
    "Data/Data_week5/Week5_TRIM_LLM-pretraining.json",
    "Data/Data_week6/Week6_TRIM_SFT_alignment.json",
    "Data/Data_week7/Week7-TRIM_Incontext-Evaluation.json",
    "Data/Data_week8/Week8_TRIM-Application-Multimodal.json",
    "Data/Data_week9/Week9_RAG_LLM_final.json",
    "Data/Data_week10/Week10_TM_last_lecture.json",
    "Data/Data_week11/Week11_TRIM_BERT.json",
    "Data/Data_week12/Week12_TM-lastlecture-1.json",
    "Data/Data_week13/Week13_revision_lecture_exam_focused.json",
]

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
BASE_URL = "https://api.studio.nebius.com/v1"
API_KEY_ENV = "NEBIUS_API_KEY"

N_SAMPLES = 3          # grades per chunk, median taken
SAMPLE_TEMP = 0.3      # must be > 0 when N_SAMPLES > 1
MAX_CHARS = 4000
NEIGHBOURS = 1         # slides of context either side; 0 disables
TITLE_MAX_CHARS = 150  # below this, with no math, counts as a title slide
LIMIT = 35           # e.g. 5 to test

load_dotenv()

MATH = re.compile(r"[=∑√·⊤|²]|\d\s*[,.]\s*\d|\[[\d,\s.]+\]")

JUDGE_PROMPT = """You grade how much a lecture slide contributes to answering an exam question.

Judge ONLY the TARGET SLIDE. Neighbouring slides are shown for context because
lectures split one idea across consecutive slides; do not grade them.
Topical similarity is NOT contribution.

3 = contains a fact, formula, or definition the answer directly depends on
2 = contributes partially: one of several needed pieces, or setup the answer
    relies on (e.g. defines a vocabulary or example used in the calculation)
1 = same topic, supplies nothing the answer uses
0 = off-topic, or a section header / navigational slide

Return strictly this JSON, no fences:
{"grade": <0|1|2|3>, "justification": "<one short sentence>"}"""


# --- slide index (for neighbour context) --------------------------------

def load_slides():
    """{week: {page: content}} across all lecture files."""
    index = {}
    for path in LECTURE_PATHS:
        files = [path] if os.path.isfile(path) else glob.glob(
            os.path.join(path, "**", "*.json"), recursive=True)
        for fp in files:
            if "__macosx" in fp.lower():
                continue
            try:
                data = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            if not (isinstance(data, dict) and "pages" in data and "week" in data):
                continue
            m = re.search(r"(\d+)", str(data["week"]))
            wk = int(m.group(1)) if m else 0
            for pg in data["pages"]:
                num = pg.get("page_number")
                num = num[0] if isinstance(num, list) else num
                index.setdefault(wk, {})[num or 1] = (pg.get("content") or "").strip()
    return index


def neighbours(index, week, page, n=NEIGHBOURS):
    """Content of the n slides before and after, in deck order."""
    if not n or week not in index:
        return "", ""
    pages = sorted(index[week])
    if page not in pages:
        return "", ""
    i = pages.index(page)
    before = "\n\n".join(index[week][p] for p in pages[max(0, i - n):i])
    after = "\n\n".join(index[week][p] for p in pages[i + 1:i + 1 + n])
    return before, after


def is_title_slide(text):
    """Short, no math: a section header or image slide."""
    t = (text or "").strip()
    return len(t) < TITLE_MAX_CHARS and not MATH.search(t)


# --- grading -----------------------------------------------------------

def grade_chunk(client, index, question, answer, gp):
    text = (gp.get("content") or "").strip()
    if not text:
        return 0, [], "empty chunk", True
    if is_title_slide(text):
        return 0, [], "title / navigational slide (no LLM call)", True

    before, after = neighbours(index, gp["week"], gp["page"])

    ctx = ""
    if before:
        ctx += f"\n\nPRECEDING SLIDE (context only):\n{before[:MAX_CHARS]}"
    if after:
        ctx += f"\n\nFOLLOWING SLIDE (context only):\n{after[:MAX_CHARS]}"

    prompt = (f"QUESTION:\n{question}\n\nCORRECT ANSWER:\n{answer}\n\n"
              f"TARGET SLIDE:\n{text[:MAX_CHARS]}{ctx}")

    grades, why = [], ""
    for _ in range(N_SAMPLES):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=SAMPLE_TEMP,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": JUDGE_PROMPT},
                          {"role": "user", "content": prompt}])
            p = json.loads(r.choices[0].message.content)
            grades.append(max(0, min(3, int(p.get("grade", 0)))))
            why = why or p.get("justification", "")
        except Exception as e:
            print(f"    call failed: {e}", file=sys.stderr)

    if not grades:
        return None, [], "all calls failed", False
    return round(statistics.median(grades)), grades, why, False


def score(client, index, benchmark):
    dist, by_rank, by_role, unstable, skipped = Counter(), {}, {}, [], 0

    for i, item in enumerate(benchmark, 1):
        q, a = item["student_query"], item["qa_gold_standard"]
        for gp in item.get("gold_pages", []):
            g, samples, why, structural = grade_chunk(client, index, q, a, gp)
            gp["utility"], gp["utility_samples"], gp["utility_reason"] = g, samples, why
            skipped += structural
            if g is None:
                continue
            dist[g] += 1
            by_rank.setdefault(gp["rank"], []).append(g)
            by_role.setdefault(gp.get("role", "?"), []).append(g)
            if len(samples) > 1 and max(samples) - min(samples) >= 2:
                unstable.append((item["query_id"], gp["rank"], samples))

        line = "  ".join(
            f"r{gp['rank']}(W{gp['week']}P{gp['page']})={gp['utility']}"
            for gp in item["gold_pages"])
        useful = sum(1 for gp in item["gold_pages"] if (gp["utility"] or 0) >= 2)
        print(f"[{i:03d}/{len(benchmark)}] {item['query_id']}  {line}  "
              f"| useful: {useful}/{len(item['gold_pages'])}")

    return dist, by_rank, by_role, unstable, skipped


def report(dist, by_rank, by_role, unstable, skipped, benchmark):
    total = sum(dist.values()) or 1
    print("\n" + "=" * 55)
    print("GRADE DISTRIBUTION")
    for g in (3, 2, 1, 0):
        print(f"  {g}: {dist[g]:4d} ({dist[g]/total:5.1%}) {'#' * int(35*dist[g]/total)}")

    print("\nMEAN GRADE BY RANK  (should decrease; if flat, rank is noise)")
    for r in sorted(by_rank):
        v = by_rank[r]
        print(f"  rank {r}: {statistics.mean(v):.2f}  (n={len(v)})")

    print("\nMEAN GRADE BY ROLE  (primary should outscore supporting)")
    for role in sorted(by_role):
        v = by_role[role]
        print(f"  {role:<12} {statistics.mean(v):.2f}  (n={len(v)})")

    print("\nUSEFUL CHUNKS PER QUESTION (>= 2)")
    for k, n in sorted(Counter(
            sum(1 for gp in it.get("gold_pages", []) if (gp.get("utility") or 0) >= 2)
            for it in benchmark).items()):
        print(f"  {k}: {n} question(s)")

    print(f"\nTitle slides graded 0 without an LLM call: {skipped}")

    if unstable:
        print(f"\nUNSTABLE ({len(unstable)}) - review by hand:")
        for qid, r, s in unstable[:15]:
            print(f"  {qid} rank {r}: {s}")


def prune(benchmark, threshold):
    rescued = []
    for item in benchmark:
        pages = item.get("gold_pages", [])
        kept = [gp for gp in pages if (gp.get("utility") or 0) >= threshold]
        if not kept and pages:
            kept = [max(pages, key=lambda gp: gp.get("utility") or 0)]
            rescued.append(item["query_id"])
        for r, gp in enumerate(kept, 1):
            gp["rank"] = r
            gp["role"] = "primary" if r == 1 else "supporting"
        item["gold_pages"] = kept
        item["n_gold"] = len(kept)
        item["prune_threshold"] = threshold
    return rescued


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", type=int, choices=[1, 2, 3], metavar="N",
                    help="keep chunks with utility >= N")
    args = ap.parse_args()

    if N_SAMPLES > 1 and SAMPLE_TEMP == 0:
        sys.exit("SAMPLE_TEMP 0 with N_SAMPLES > 1 gives identical draws.")

    key = os.environ.get(API_KEY_ENV)
    if not key:
        sys.exit(f"No API key in ${API_KEY_ENV}")
    client = OpenAI(base_url=BASE_URL, api_key=key)

    index = load_slides()
    print(f"Slide index: {sum(len(v) for v in index.values())} pages "
          f"across {len(index)} weeks")

    benchmark = json.load(open(QA_INPUT, encoding="utf-8"))
    if LIMIT:
        benchmark = benchmark[:LIMIT]
    n = sum(len(it.get("gold_pages", [])) for it in benchmark)
    print(f"{len(benchmark)} questions, {n} chunks, up to {n*N_SAMPLES} calls\n")

    dist, by_rank, by_role, unstable, skipped = score(client, index, benchmark)

    os.makedirs(os.path.dirname(QA_SCORED) or ".", exist_ok=True)
    json.dump(benchmark, open(QA_SCORED, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    report(dist, by_rank, by_role, unstable, skipped, benchmark)
    print(f"\nScored -> {QA_SCORED}")

    if args.prune:
        rescued = prune(benchmark, args.prune)
        json.dump(benchmark, open(QA_PRUNED, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        sizes = dict(sorted(Counter(it["n_gold"] for it in benchmark).items()))
        print(f"\nPruned at >= {args.prune} -> {QA_PRUNED}\n  sizes: {sizes}")
        if rescued:
            print(f"  {len(rescued)} question(s) had nothing above threshold; "
                  f"kept best chunk, verify: {rescued[:10]}")


if __name__ == "__main__":
    main()