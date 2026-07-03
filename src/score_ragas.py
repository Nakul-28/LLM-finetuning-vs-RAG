"""Score RAG outputs with RAGAS-style summary tables and manual-review exports."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import EVAL_REPORTS_DIR, ensure_dir, read_jsonl


def summarize_rows(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    numeric_fields = ["exact_match", "answer_f1", "context_precision", "faithfulness", "latency_s", "input_tokens_est", "output_tokens_est"]
    summary: dict[str, float] = {}
    for field in numeric_fields:
        values = [float(row.get(field, 0.0)) for row in rows if row.get(field) is not None]
        if values:
            summary[field] = sum(values) / len(values)
    return summary


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    summary = summarize_rows(rows)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in sorted(summary.items()):
            writer.writerow({"metric": key, "value": round(value, 6)})


def write_manual_sample(path: Path, rows: list[dict], sample_size: int) -> None:
    ensure_dir(path.parent)
    sample = rows if len(rows) <= sample_size else random.sample(rows, sample_size)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "dataset", "question", "reference", "prediction", "exact_match", "answer_f1", "hallucination_flag", "fluency", "correctness", "completeness", "helpfulness"])
        writer.writeheader()
        for row in sample:
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "dataset": row.get("dataset", ""),
                    "question": row.get("question", ""),
                    "reference": row.get("reference", ""),
                    "prediction": row.get("prediction", ""),
                    "exact_match": row.get("exact_match", 0.0),
                    "answer_f1": row.get("answer_f1", 0.0),
                    "hallucination_flag": 0 if float(row.get("exact_match", 0.0)) > 0 else 1,
                    "fluency": "",
                    "correctness": "",
                    "completeness": "",
                    "helpfulness": "",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=EVAL_REPORTS_DIR)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise FileNotFoundError(f"No RAG outputs found in {args.input}")

    summary_path = ensure_dir(args.output_dir) / f"ragas_{args.dataset}.csv"
    manual_path = ensure_dir(args.output_dir) / f"manual_{args.dataset}.csv"
    write_summary_csv(summary_path, rows)
    write_manual_sample(manual_path, rows, args.sample_size)
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote manual review sample to {manual_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())