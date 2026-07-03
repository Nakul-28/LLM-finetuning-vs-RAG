# LLM Fine-Tuning vs RAG

This repository is the working area for a comparative study of fine-tuning versus retrieval-augmented generation (RAG) for domain-specific LLM tasks.

The project is based on the two workspace documents at the repository root:

- [implementation-plan.md](../implementation-plan.md)
- [GUIDE.md](../GUIDE.md)

## What this repo contains

The guide assumes a small Python project with a fixed configuration module, dataset preparation scripts, evaluation scripts, and a local RAG index. The repository is scaffolded to support that workflow:

- `src/` for implementation code
- `src/common.py` for shared path, split, metric, and generation helpers
- `data/` for raw and processed datasets
- `models/` for base and fine-tuned checkpoints
- `chroma_db/` for persistent vector storage
- `results/` and `eval_reports/` for evaluation outputs

## Recommended workflow

1. Run `start.bat` from the repository root to bootstrap local sample inputs, install dependencies, and execute the full pipeline.
2. If you want to work step by step instead, create and activate a Python virtual environment and install `requirements.txt` manually.
3. Use the individual scripts in `src/` only when you need to rerun a single stage.

## Project layout

```text
LLM-finetuning-vs-RAG/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
	├── __init__.py
	├── config.py
	├── download_models.py
	├── prepare_squad.py
	├── prepare_recipenlg.py
	├── prepare_docs.py
	├── eval_baseline.py
	├── finetune_qlora.py
	├── eval_finetuned.py
	├── ingest_chroma.py
	├── rag_query.py
	├── eval_rag.py
	├── score_ragas.py
	└── aggregate_results.py
```

## Notes

- The initial goal is to validate one dataset end-to-end before expanding to the full comparison matrix.
- The constants in `src/config.py` are intended to be the single source of truth for controlled variables.
- Large generated artifacts such as models, processed datasets, and vector stores should stay out of version control.