"""Prepare the SQuAD dataset for baseline, fine-tuning, and RAG evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import PROCESSED_DIR, RAW_DIR, RANDOM_SEED, ensure_dir, normalize_text, split_records, write_jsonl


def load_squad_file(file_path: Path, split_name: str) -> list[dict]:
    import json

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []
    for item in data.get("data", []):
        for paragraph in item.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "")
                qa_id = qa.get("id", "")
                is_impossible = qa.get("is_impossible", False)

                if is_impossible:
                    answer = "no answer"
                else:
                    answers = qa.get("answers", [])
                    if answers:
                        answer = answers[0].get("text", "")
                    else:
                        answer = "no answer"

                records.append(
                    {
                        "id": qa_id or f"squad-{split_name}-{len(records)}",
                        "dataset": "squad",
                        "split": split_name,
                        "question": normalize_text(question),
                        "context": normalize_text(context),
                        "answer": normalize_text(answer),
                        "source": "squad",
                    }
                )
    return records


def prepare(output_dir: Path | None = None) -> dict[str, Path]:
    output_root = ensure_dir(output_dir or (PROCESSED_DIR / "squad"))
    raw_squad_dir = RAW_DIR / "squad"
    train_file = raw_squad_dir / "train-v2.0.json"
    dev_file = raw_squad_dir / "dev-v2.0.json"

    if not train_file.exists() or not dev_file.exists():
        raise FileNotFoundError(f"Please place train-v2.0.json and dev-v2.0.json in {raw_squad_dir}")

    print(f"Loading {train_file}...")
    train_rows = load_squad_file(train_file, "train")

    print(f"Loading {dev_file}...")
    test_rows = load_squad_file(dev_file, "test")

    train_split, val_split, _ = split_records(train_rows, seed=RANDOM_SEED, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0)

    paths = {
        "train": output_root / "train.jsonl",
        "val": output_root / "val.jsonl",
        "test": output_root / "test.jsonl",
    }
    write_jsonl(paths["train"], train_split)
    write_jsonl(paths["val"], val_split)
    write_jsonl(paths["test"], test_rows)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    paths = prepare(args.output_dir)
    for split_name, path in paths.items():
        print(f"Wrote {split_name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())