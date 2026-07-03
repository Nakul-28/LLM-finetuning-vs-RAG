"""Run QLoRA or LoRA fine-tuning for a processed dataset split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common import build_prompt, ensure_dir, extract_answer, extract_context, extract_question, normalize_text, read_jsonl


def _has_bnb_support() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _target_modules(model) -> list[str]:
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return [candidate for candidate in candidates if candidate in names]


def _build_training_text(record: dict[str, str]) -> str:
    prompt = build_prompt(extract_question(record), extract_context(record))
    answer = normalize_text(extract_answer(record))
    return f"{prompt} {answer}".strip()


def _load_tokenizer_and_model(base_model: str, quantize: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if base_model == "qwen2.5:3b-instruct":
        base_model = "Qwen/Qwen2.5-3B-Instruct"
    elif base_model == "qwen2.5:7b-instruct":
        base_model = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs: dict = {"device_map": "auto"}
    if quantize:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    return tokenizer, model


def train(
    dataset_path: Path,
    output_dir: Path,
    *,
    base_model: str,
    epochs: float,
    max_length: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int,
    grad_accumulation: int,
) -> Path:
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install 'transformers', 'datasets', and 'peft' before fine-tuning.") from exc

    records = read_jsonl(dataset_path)
    if not records:
        raise ValueError(f"No training rows found in {dataset_path}")

    quantize = _has_bnb_support()
    tokenizer, model = _load_tokenizer_and_model(base_model, quantize)
    if quantize:
        model = prepare_model_for_kbit_training(model)

    target_modules = _target_modules(model)
    if not target_modules:
        raise RuntimeError("Could not detect LoRA target modules in the loaded model.")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    dataset = Dataset.from_list([
        {"text": _build_training_text(record)}
        for record in records
    ])

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokenized = tokenizer(batch["text"], truncation=True, max_length=max_length)
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ensure_dir(output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if quantize else "adamw_torch",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    import inspect
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized,
        "data_collator": collator,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_model", default="qwen2.5:3b-instruct")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation", type=int, default=8)
    args = parser.parse_args()
    output_dir = train(
        args.dataset,
        args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
    )
    print(f"Saved fine-tuned adapter to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())