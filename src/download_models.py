"""Download or warm up models used by the study."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import try_run
from src.config import EMBED_MODEL, RERANKER_MODEL


def pull_ollama_models(models: Iterable[str]) -> None:
    for model in models:
        print(f"Pulling Ollama model: {model}")
        try_run(["ollama", "pull", model])


def warmup_huggingface_models(reranker_model: str) -> None:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install 'sentence-transformers' to warm up the reranker model.") from exc

    print(f"Loading reranker model: {reranker_model}")
    CrossEncoder(reranker_model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", action="store_true", help="Pull the Ollama base model(s).")
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    parser.add_argument("--warmup_hf", action="store_true", help="Warm up the embedding and reranker models.")
    parser.add_argument("--embed_model", default=EMBED_MODEL)
    parser.add_argument("--reranker_model", default=RERANKER_MODEL)
    args = parser.parse_args()

    if args.pull:
        pull_ollama_models([args.model])
        pull_ollama_models([args.embed_model])

    if args.warmup_hf:
        warmup_huggingface_models(args.reranker_model)

    if not args.pull and not args.warmup_hf:
        print("Nothing to do. Pass --pull and/or --warmup_hf.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())