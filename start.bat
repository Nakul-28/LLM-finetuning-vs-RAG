@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist ".venv\Scripts\activate.bat" (
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :error

python -m pip install --upgrade pip
if errorlevel 1 goto :error

python -m pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist "data\raw\recipenlg" mkdir "data\raw\recipenlg"
if not exist "data\raw\docs" mkdir "data\raw\docs"

if not exist "data\raw\recipenlg\recipes.csv" (
    > "data\raw\recipenlg\recipes.csv" (
        echo title,ingredients,directions
        echo Simple Tomato Pasta,tomato^|pasta^|olive oil,"Boil pasta and toss with tomato sauce."
        echo Lemon Rice,rice^|lemon^|salt,"Cook rice and finish with lemon juice and salt."
        echo Veggie Soup,carrot^|celery^|onion^|broth,"Simmer vegetables in broth until tender."
    )
)

if not exist "data\raw\docs\sample.md" (
    > "data\raw\docs\sample.md" (
        echo # Sample Documentation
        echo.
        echo This local corpus exists so the pipeline can run end to end even before external documents are added.
        echo.
        echo It explains a simple retrieval example, a support process, and a few system notes.
    )
)

if not exist "data\raw\docs\notes.md" (
    > "data\raw\docs\notes.md" (
        echo # Project Notes
        echo.
        echo Fine-tuning is expected to help on generation-heavy tasks, while RAG should help on document lookup and volatile facts.
    )
)

echo === Step 1: Download or warm up models ===
python src\download_models.py --pull --warmup_hf
if errorlevel 1 goto :error

echo === Step 2: Prepare datasets ===
python src\prepare_squad.py
if errorlevel 1 goto :error

python src\prepare_recipenlg.py --input_path data\raw\recipenlg\recipes.csv
if errorlevel 1 goto :error

python src\prepare_docs.py --input_dir data\raw\docs
if errorlevel 1 goto :error

echo === Step 3: Baseline evaluation ===
python src\eval_baseline.py --dataset squad
if errorlevel 1 goto :error

python src\eval_baseline.py --dataset recipenlg
if errorlevel 1 goto :error

python src\eval_baseline.py --dataset docs
if errorlevel 1 goto :error

echo === Step 4: Fine-tuning track ===
python src\finetune_qlora.py --dataset data\processed\recipenlg\train.jsonl --output_dir models\finetuned\recipenlg --epochs 1 --max_length 256 --batch_size 1 --grad_accumulation 1
if errorlevel 1 goto :error

python src\eval_finetuned.py --dataset recipenlg --model_dir models\finetuned\recipenlg --max_new_tokens 128
if errorlevel 1 goto :error

echo === Step 5: RAG track ===
python src\ingest_chroma.py --dataset squad --splits train val --collection_name squad_512
if errorlevel 1 goto :error

python src\ingest_chroma.py --dataset recipenlg --splits train val --collection_name recipenlg_512
if errorlevel 1 goto :error

python src\ingest_chroma.py --dataset docs --splits train val --collection_name docs_512
if errorlevel 1 goto :error

python src\eval_rag.py --dataset squad --collection_name squad_512
if errorlevel 1 goto :error

python src\eval_rag.py --dataset recipenlg --collection_name recipenlg_512
if errorlevel 1 goto :error

python src\eval_rag.py --dataset docs --collection_name docs_512
if errorlevel 1 goto :error

echo === Step 6: Score and aggregate ===
python src\score_ragas.py --input results\rag_squad.jsonl --dataset squad
if errorlevel 1 goto :error

python src\score_ragas.py --input results\rag_recipenlg.jsonl --dataset recipenlg
if errorlevel 1 goto :error

python src\score_ragas.py --input results\rag_docs.jsonl --dataset docs
if errorlevel 1 goto :error

python src\aggregate_results.py
if errorlevel 1 goto :error

echo.
echo Pipeline complete.
exit /b 0

:error
echo.
echo Pipeline failed. Review the message above for the first failing step.
exit /b 1