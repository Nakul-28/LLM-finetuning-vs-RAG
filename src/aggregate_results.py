"""Aggregate results from baseline, fine-tuned, and RAG runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import EVAL_REPORTS_DIR, RESULTS_DIR, ensure_dir, read_jsonl


def aggregate(results_dir: Path, output_path: Path) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            dataset = str(row.get("dataset", path.stem))
            method = str(row.get("method", path.stem.split("_", 1)[0]))
            grouped[(dataset, method)].append(row)

    summary_rows: list[dict] = []
    for (dataset, method), rows in sorted(grouped.items()):
        numeric_fields = ["exact_match", "answer_f1", "context_precision", "faithfulness", "latency_s", "input_tokens_est", "output_tokens_est"]
        summary = {"dataset": dataset, "method": method, "count": len(rows)}
        for field in numeric_fields:
            values = [float(row.get(field, 0.0)) for row in rows if row.get(field) is not None]
            summary[field] = round(sum(values) / len(values), 6) if values else 0.0
        summary_rows.append(summary)

    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["dataset", "method", "count"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=EVAL_REPORTS_DIR / "summary_table.csv")
    args = parser.parse_args()

    rows = aggregate(args.results_dir, args.output)
    print(f"Wrote {len(rows)} summary rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())