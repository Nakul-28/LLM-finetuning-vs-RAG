"""Prepare the RecipeNLG dataset for the comparison study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import PROCESSED_DIR, RAW_DIR, ensure_dir, normalize_text, split_records, truncate_words, write_jsonl
from src.config import MAX_PROMPT_WORDS, RANDOM_SEED, RECIPENLG_FILENAME


def parse_multi_value(value: str | None) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return "; ".join(normalize_text(item) for item in parsed if normalize_text(item))
        except json.JSONDecodeError:
            pass
    return text.replace("|", "; ")


def pick(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            value = normalize_text(lowered[candidate.lower()])
            if value:
                return value
    return ""


def build_records(rows: list[dict[str, str]], split_name: str) -> list[dict]:
    records: list[dict] = []
    for index, row in enumerate(rows):
        title = pick(row, ("title", "name", "recipe_title"))
        ingredients = parse_multi_value(pick(row, ("ingredients", "ingredient", "ingredients_list")))
        directions = parse_multi_value(pick(row, ("directions", "instruction", "instructions", "method")))

        prompt = ingredients or title
        answer = normalize_text(" ".join(part for part in (title, directions) if part))
        if not answer:
            answer = truncate_words(prompt, MAX_PROMPT_WORDS)

        records.append(
            {
                "id": f"recipenlg-{split_name}-{index}",
                "dataset": "recipenlg",
                "split": split_name,
                "question": truncate_words(prompt, MAX_PROMPT_WORDS),
                "context": ingredients,
                "answer": answer,
                "source": "recipenlg",
                "title": title,
            }
        )
    return records


def load_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def prepare(input_path: Path | None = None, output_dir: Path | None = None) -> dict[str, Path]:
    input_file = input_path or (RAW_DIR / "recipenlg" / RECIPENLG_FILENAME)
    if not input_file.exists():
        raise FileNotFoundError(
            f"RecipeNLG input not found: {input_file}. Place the CSV there or pass --input_path."
        )

    rows = load_rows(input_file)
    records = build_records(rows, "all")
    train_split, val_split, test_split = split_records(records, seed=RANDOM_SEED, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    output_root = ensure_dir(output_dir or (PROCESSED_DIR / "recipenlg"))
    paths = {
        "train": output_root / "train.jsonl",
        "val": output_root / "val.jsonl",
        "test": output_root / "test.jsonl",
    }
    write_jsonl(paths["train"], train_split)
    write_jsonl(paths["val"], val_split)
    write_jsonl(paths["test"], test_split)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    paths = prepare(args.input_path, args.output_dir)
    for split_name, path in paths.items():
        print(f"Wrote {split_name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())