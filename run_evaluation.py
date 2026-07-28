import sys
import io
from datetime import datetime

from evaluation import evaluate_rag_pipeline

OUTPUT_PATH    = "inputs_and_outputs/output_qa.json"
BENCHMARK_PATH = "data/benchmark_qa.json"
REPORT_PATH    = "evaluation_report.txt"


class _Tee:
    """Write everything to the real stdout AND to an in-memory buffer, so the
    user still sees live progress while we capture the full report for a file.
    Delegates any other stream attributes (isatty, encoding, fileno, ...) to the
    first stream so libraries that introspect sys.stdout keep working."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        first = self.streams[0]
        return first.isatty() if hasattr(first, "isatty") else False
    def __getattr__(self, name):
        # Fallback for any attribute we didn't explicitly define.
        return getattr(self.streams[0], name)


def main():
    buffer = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, buffer)

    header = (
        "=" * 65 + "\n"
        "  RAG EVALUATION REPORT\n"
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Output    : {OUTPUT_PATH}\n"
        f"  Benchmark : {BENCHMARK_PATH}\n"
        + "=" * 65
    )
    print(header)

    try:
        df = evaluate_rag_pipeline(OUTPUT_PATH, BENCHMARK_PATH)

        # Append the full per-query breakdown so the report has ALL results,
        # not just the aggregate means printed by the evaluator.
        if df is not None and not df.empty:
            print("\n" + "=" * 65)
            print("  PER-QUERY RESULTS")
            print("=" * 65)
            with_full = df.copy()
            print(with_full.to_string(index=False))

            print("\n" + "=" * 65)
            print("  COLUMN AVERAGES")
            print("=" * 65)
            numeric = df.select_dtypes(include="number")
            print(numeric.mean().to_string())
    finally:
        # Restore stdout and flush the captured report to disk.
        sys.stdout = real_stdout
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
        print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
