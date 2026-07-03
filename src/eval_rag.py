"""Evaluate the RAG pipeline on the frozen test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import sys

import chromadb

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from src.common import CHROMA_DIR, BASE_MODEL, ROOT_DIR, RESULTS_DIR, ensure_dir, estimate_generation_tokens, exact_match, extract_answer, extract_question, read_jsonl, token_f1, word_count, write_jsonl, context_support_score
from src.config import CHUNK_SIZE, EMBED_MODEL, RERANKER_MODEL, TOP_K
from src.rag_query import answer_question, load_rag_stack


def evaluate(dataset: str, dataset_path: Path, collection_name: str, output_path: Path, *, model: str = BASE_MODEL, top_k: int = TOP_K, limit: int | None = None, max_tokens: int = 512) -> list[dict]:
    records = read_jsonl(dataset_path)
    if limit is not None:
        records = records[:limit]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(collection_name)
    embedder, reranker = load_rag_stack(collection, embed_model=EMBED_MODEL, reranker_model=RERANKER_MODEL)

    outputs: list[dict] = []
    for record in tqdm(records, desc=f"Evaluating RAG on {dataset}"):
        question = extract_question(record)
        start = perf_counter()
        result = answer_question(collection, embedder, question, top_k=top_k, reranker=reranker, model=model, max_tokens=max_tokens)
        latency_s = perf_counter() - start
        prediction = result["answer"]
        reference = extract_answer(record)
        context = result["context"]
        outputs.append(
            {
                "id": record.get("id"),
                "dataset": dataset,
                "method": "rag",
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "context": context,
                "retrieved": result["retrieved"],
                "latency_s": round(latency_s, 4),
                "input_tokens_est": word_count(result["prompt"]),
                "output_tokens_est": estimate_generation_tokens(prediction),
                "exact_match": exact_match(prediction, reference),
                "answer_f1": token_f1(prediction, reference),
                "context_precision": context_support_score(prediction, context),
                "faithfulness": context_support_score(prediction, context),
            }
        )

    write_jsonl(output_path, outputs)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--collection_name", default=None)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    dataset_path = ROOT_DIR / "data" / "processed" / args.dataset / f"{args.split}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset split: {dataset_path}")
    collection_name = args.collection_name or f"{args.dataset}_{args.chunk_size}"
    output_path = ensure_dir(args.output_dir) / f"rag_{args.dataset}.jsonl"
    evaluate(args.dataset, dataset_path, collection_name, output_path, model=args.model, top_k=args.top_k, limit=args.limit, max_tokens=args.max_tokens)
    print(f"Wrote RAG results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())