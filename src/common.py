from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .config import BASE_MODEL, CHUNK_OVERLAP, CHUNK_SIZE, MAX_TOKENS, PROMPT_TEMPLATE, RANDOM_SEED, TEMPERATURE


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
BASE_MODELS_DIR = MODELS_DIR / "base"
FINETUNED_MODELS_DIR = MODELS_DIR / "finetuned"
RESULTS_DIR = ROOT_DIR / "results"
EVAL_REPORTS_DIR = ROOT_DIR / "eval_reports"
CHROMA_DIR = ROOT_DIR / "chroma_db"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def truncate_words(text: str, max_words: int) -> str:
    words = normalize_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def split_records(
    records: list[dict[str, Any]],
    *,
    seed: int = RANDOM_SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items = list(records)
    random.Random(seed).shuffle(items)
    total = len(items)
    if total == 0:
        return [], [], []
    if total == 1:
        return items, [], []
    if total == 2:
        return [items[0]], [], [items[1]]

    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio))) if total >= 4 else 0
    if train_count + val_count >= total:
        val_count = max(0, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count + val_count + test_count > total:
            train_count = max(1, total - val_count - test_count)

    train_end = train_count
    val_end = train_end + val_count
    return items[:train_end], items[train_end:val_end], items[val_end:]


def chunk_text(text: str, *, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = normalize_text(text).split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def word_count(text: str) -> int:
    return len(normalize_text(text).split())


def extract_question(record: dict[str, Any]) -> str:
    for key in ("question", "prompt", "instruction", "title"):
        value = normalize_text(record.get(key))
        if value:
            return value
    return ""


def extract_context(record: dict[str, Any]) -> str:
    for key in ("context", "text", "document", "content", "passage", "body", "ingredients"):
        value = record.get(key)
        if isinstance(value, list):
            value = "\n".join(normalize_text(item) for item in value if normalize_text(item))
        value = normalize_text(value)
        if value:
            return value
    return ""


def extract_answer(record: dict[str, Any]) -> str:
    for key in ("answer", "output", "response", "target", "completion"):
        value = normalize_text(record.get(key))
        if value:
            return value

    answers = record.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text") or []
        if texts:
            return normalize_text(texts[0])
    if isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, dict):
            return normalize_text(first.get("text") or first.get("answer"))
        return normalize_text(first)
    return ""


def build_prompt(question: str, context: str = "") -> str:
    return PROMPT_TEMPLATE.format(question=normalize_text(question), context=normalize_text(context))


def estimate_generation_tokens(text: str) -> int:
    return max(1, word_count(text))


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text).lower()).strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_for_match(prediction) == normalize_for_match(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_for_match(prediction).split()
    ref_tokens = normalize_for_match(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def context_support_score(prediction: str, context: str) -> float:
    prediction_tokens = set(normalize_for_match(prediction).split())
    if not prediction_tokens:
        return 0.0
    context_tokens = set(normalize_for_match(context).split())
    return len(prediction_tokens & context_tokens) / len(prediction_tokens)


def run_ollama(prompt: str, *, model: str = BASE_MODEL, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> tuple[str, dict[str, Any]]:
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install the 'ollama' Python package and run 'ollama serve' before evaluating.") from exc

    options = {"temperature": temperature, "num_predict": max_tokens}
    try:
        response = ollama.generate(model=model, prompt=prompt, options=options)
    except AttributeError:
        client = ollama.Client()
        response = client.generate(model=model, prompt=prompt, options=options)

    if isinstance(response, dict):
        text = response.get("response")
        if text is None:
            message = response.get("message") or {}
            text = message.get("content", "") if isinstance(message, dict) else ""
        return normalize_text(text), response

    text = getattr(response, "response", "") or ""
    return normalize_text(text), {"response": text}


def try_run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")


class OllamaEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts: list[str] | str, normalize_embeddings: bool = True, show_progress_bar: bool = False) -> Any:
        import numpy as np
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("Install the 'ollama' Python package and run 'ollama serve' to use Ollama embeddings.") from exc

        if isinstance(texts, str):
            texts = [texts]

        if show_progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc=f"Ollama embedding ({self.model_name})")
            except ImportError:
                iterator = texts
        else:
            iterator = texts

        embeddings = []
        for text in iterator:
            current_text = text
            while True:
                try:
                    try:
                        response = ollama.embeddings(model=self.model_name, prompt=current_text)
                    except AttributeError:
                        client = ollama.Client()
                        response = client.embeddings(model=self.model_name, prompt=current_text)
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    if "context length" in err_msg or "too long" in err_msg or "length exceeds" in err_msg or "status code: 500" in err_msg:
                        words = current_text.split()
                        if len(words) <= 1:
                            raise e
                        new_len = int(len(words) * 0.8)
                        truncated_text = " ".join(words[:new_len])
                        if len(truncated_text) >= len(current_text):
                            raise e
                        print(f"Warning: Truncated embedding input from {len(words)} to {new_len} words due to context length limit.")
                        current_text = truncated_text
                    else:
                        raise e
            
            embedding = response["embedding"]
            if normalize_embeddings:
                arr = np.array(embedding)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                embedding = arr.tolist()
            embeddings.append(embedding)
            
        return np.array(embeddings)
