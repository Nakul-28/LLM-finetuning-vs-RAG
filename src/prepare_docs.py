"""Prepare a documentation corpus for the RAG experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import CHROMA_DIR, DOC_EXTENSIONS, PROCESSED_DIR, RAW_DIR, chunk_text, ensure_dir, normalize_text, split_records, truncate_words, write_jsonl
from src.config import MAX_PROMPT_WORDS, RANDOM_SEED


def collect_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in DOC_EXTENSIONS]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def build_records(files: list[Path]) -> list[dict]:
    records: list[dict] = []
    for file_index, path in enumerate(files):
        text = normalize_text(read_text(path))
        if not text:
            continue
        for chunk_index, chunk in enumerate(chunk_text(text)):
            records.append(
                {
                    "id": f"docs-{file_index}-{chunk_index}",
                    "dataset": "docs",
                    "split": "all",
                    "question": f"What does the passage from {path.name} explain?",
                    "context": chunk,
                    "answer": truncate_words(chunk, MAX_PROMPT_WORDS),
                    "source": str(path),
                }
            )
    return records


def prepare(input_dir: Path | None = None, output_dir: Path | None = None) -> dict[str, Path]:
    input_root = input_dir or (RAW_DIR / "docs")
    files = collect_files(input_root)
    if not files:
        raise FileNotFoundError(f"No documentation files found under {input_root}")

    records = build_records(files)
    train_split, val_split, test_split = split_records(records, seed=RANDOM_SEED, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    output_root = ensure_dir(output_dir or (PROCESSED_DIR / "docs"))
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
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    paths = prepare(args.input_dir, args.output_dir)
    for split_name, path in paths.items():
        print(f"Wrote {split_name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())