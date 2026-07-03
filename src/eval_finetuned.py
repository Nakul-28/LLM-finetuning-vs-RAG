"""Evaluate a fine-tuned checkpoint on the frozen test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import sys

import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from src.common import BASE_MODEL, ROOT_DIR, RESULTS_DIR, build_prompt, ensure_dir, estimate_generation_tokens, exact_match, extract_answer, extract_question, read_jsonl, token_f1, word_count, write_jsonl


def _load_finetuned_model(model_dir: Path, base_model: str):
    import importlib

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if base_model == "qwen2.5:3b-instruct":
        base_model = "Qwen/Qwen2.5-3B-Instruct"
    elif base_model == "qwen2.5:7b-instruct":
        base_model = "Qwen/Qwen2.5-7B-Instruct"

    peft_module = importlib.import_module("peft")
    PeftModel = getattr(peft_module, "PeftModel")

    tokenizer_source = model_dir if (model_dir / "tokenizer.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    quantize = False
    try:
        import bitsandbytes  # noqa: F401
        quantize = torch.cuda.is_available()
    except ImportError:
        pass

    from transformers import BitsAndBytesConfig

    model_kwargs = {"device_map": "auto", "offload_folder": "offload"}
    if quantize:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    if (model_dir / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    model.eval()
    return tokenizer, model


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    decoded = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return decoded.strip()


def evaluate(dataset_path: Path, model_dir: Path, output_path: Path, *, base_model: str = BASE_MODEL, limit: int | None = None, max_new_tokens: int = 256) -> list[dict]:
    records = read_jsonl(dataset_path)
    if limit is not None:
        records = records[:limit]

    tokenizer, model = _load_finetuned_model(model_dir, base_model)
    outputs: list[dict] = []
    for record in tqdm(records, desc="Evaluating fine-tuned"):
        question = extract_question(record)
        prompt = build_prompt(question, "")
        start = perf_counter()
        prediction = _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        latency_s = perf_counter() - start
        reference = extract_answer(record)
        outputs.append(
            {
                "id": record.get("id"),
                "dataset": record.get("dataset"),
                "method": "finetuned",
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "latency_s": round(latency_s, 4),
                "input_tokens_est": word_count(prompt),
                "output_tokens_est": estimate_generation_tokens(prediction),
                "exact_match": exact_match(prediction, reference),
                "answer_f1": token_f1(prediction, reference),
            }
        )

    write_jsonl(output_path, outputs)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    dataset_path = ROOT_DIR / "data" / "processed" / args.dataset / f"{args.split}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset split: {dataset_path}")
    output_path = ensure_dir(args.output_dir) / f"finetuned_{args.dataset}.jsonl"
    evaluate(dataset_path, args.model_dir, output_path, base_model=args.base_model, limit=args.limit, max_new_tokens=args.max_new_tokens)
    print(f"Wrote fine-tuned results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())