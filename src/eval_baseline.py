"""Evaluate the base model without fine-tuning or retrieved context."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from src.common import BASE_MODEL, ROOT_DIR, RESULTS_DIR, build_prompt, ensure_dir, estimate_generation_tokens, exact_match, extract_answer, extract_question, read_jsonl, run_ollama, token_f1, word_count, write_jsonl


def evaluate(dataset_path: Path, output_path: Path, *, model: str = BASE_MODEL, limit: int | None = None) -> list[dict]:
    records = read_jsonl(dataset_path)
    if limit is not None:
        records = records[:limit]

    outputs: list[dict] = []
    for record in tqdm(records, desc="Evaluating baseline"):
        question = extract_question(record)
        prompt = build_prompt(question, "")
        start = perf_counter()
        prediction, raw = run_ollama(prompt, model=model)
        latency_s = perf_counter() - start
        reference = extract_answer(record)
        outputs.append(
            {
                "id": record.get("id"),
                "dataset": record.get("dataset"),
                "method": "baseline",
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "latency_s": round(latency_s, 4),
                "input_tokens_est": word_count(prompt),
                "output_tokens_est": estimate_generation_tokens(prediction),
                "exact_match": exact_match(prediction, reference),
                "answer_f1": token_f1(prediction, reference),
                "raw_response": raw,
            }
        )

    write_jsonl(output_path, outputs)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    dataset_path = ROOT_DIR / "data" / "processed" / args.dataset / f"{args.split}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset split: {dataset_path}")
    output_path = ensure_dir(args.output_dir) / f"baseline_{args.dataset}.jsonl"
    evaluate(dataset_path, output_path, model=args.model, limit=args.limit)
    print(f"Wrote baseline results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())