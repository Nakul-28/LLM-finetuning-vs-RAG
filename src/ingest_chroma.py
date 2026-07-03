"""Build the ChromaDB collection used for RAG experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import chromadb

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import CHROMA_DIR, PROCESSED_DIR, ROOT_DIR, chunk_text, ensure_dir, extract_context, extract_question, extract_answer, read_jsonl
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, RANDOM_SEED


def _load_embedder(model_name: str):
    if ":" in model_name or "/" not in model_name:
        from src.common import OllamaEmbedder
        return OllamaEmbedder(model_name)

    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, trust_remote_code=True)


def _document_text(record: dict) -> str:
    for candidate in (extract_context(record), extract_answer(record), extract_question(record)):
        if candidate:
            return candidate
    return ""


def ingest(dataset: str, *, collection_name: str, splits: tuple[str, ...], chunk_size: int, chunk_overlap: int, limit: int | None = None, embed_model: str = EMBED_MODEL) -> int:
    source_rows: list[dict] = []
    for split in splits:
        path = ROOT_DIR / "data" / "processed" / dataset / f"{split}.jsonl"
        source_rows.extend(read_jsonl(path))

    if limit is not None:
        source_rows = source_rows[:limit]

    if not source_rows:
        raise FileNotFoundError(f"No processed rows found for dataset '{dataset}'.")

    embedder = _load_embedder(embed_model)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    seen_chunks = set()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for row_index, record in enumerate(source_rows):
        text = _document_text(record)
        if not text:
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap) or [text]
        for chunk_index, chunk in enumerate(chunks):
            if chunk in seen_chunks:
                continue
            seen_chunks.add(chunk)
            ids.append(f"{dataset}-{record.get('id', row_index)}-{chunk_index}")
            documents.append(chunk)
            metadatas.append(
                {
                    "dataset": dataset,
                    "record_id": record.get("id", row_index),
                    "split": record.get("split", "unknown"),
                    "chunk_index": chunk_index,
                    "source": record.get("source", ""),
                }
            )

    print(f"Processed {len(source_rows)} rows. Generated {len(ids)} unique context chunks.")
    embeddings = embedder.encode(documents, normalize_embeddings=True, show_progress_bar=True).tolist()
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return len(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--collection_name", default=None)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embed_model", default=EMBED_MODEL)
    args = parser.parse_args()

    collection_name = args.collection_name or f"{args.dataset}_{args.chunk_size}"
    count = ingest(
        args.dataset,
        collection_name=collection_name,
        splits=tuple(args.splits),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        limit=args.limit,
        embed_model=args.embed_model,
    )
    print(f"Ingested {count} chunks into collection '{collection_name}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())