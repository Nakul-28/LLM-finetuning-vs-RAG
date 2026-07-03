"""Retrieve context and generate an answer with the base model."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import BASE_MODEL, build_prompt, extract_question, run_ollama


def _load_embedder(model_name: str):
    if ":" in model_name or "/" not in model_name:
        from src.common import OllamaEmbedder
        return OllamaEmbedder(model_name)

    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, trust_remote_code=True)


def _load_reranker(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    return CrossEncoder(model_name)


def retrieve_documents(collection, embedder, question: str, *, top_k: int = 5, reranker=None) -> list[dict[str, Any]]:
    question_embedding = embedder.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=max(top_k * 3, top_k), include=["documents", "metadatas", "distances"])
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    ranked: list[dict[str, Any]] = []
    if reranker is not None and documents:
        pairs = [[question, document] for document in documents]
        scores = reranker.predict(pairs)
        scored = sorted(zip(scores, documents, metadatas), key=lambda item: item[0], reverse=True)
        for score, document, metadata in scored[:top_k]:
            ranked.append({"score": float(score), "document": document, "metadata": metadata})
        return ranked

    distances = results.get("distances", [[]])[0]
    scored = sorted(zip(distances, documents, metadatas), key=lambda item: item[0])
    for distance, document, metadata in scored[:top_k]:
        ranked.append({"score": float(distance), "document": document, "metadata": metadata})
    return ranked


def answer_question(collection, embedder, question: str, *, top_k: int = 5, reranker=None, model: str = BASE_MODEL, max_tokens: int = 512) -> dict[str, Any]:
    retrieved = retrieve_documents(collection, embedder, question, top_k=top_k, reranker=reranker)
    context = "\n\n".join(item["document"] for item in retrieved)
    prompt = build_prompt(question, context)
    answer, raw = run_ollama(prompt, model=model, max_tokens=max_tokens)
    return {
        "question": question,
        "context": context,
        "answer": answer,
        "prompt": prompt,
        "raw_response": raw,
        "retrieved": retrieved,
    }


def load_rag_stack(collection, *, embed_model: str, reranker_model: str | None = None):
    embedder = _load_embedder(embed_model)
    reranker = _load_reranker(reranker_model) if reranker_model else None
    return embedder, reranker
