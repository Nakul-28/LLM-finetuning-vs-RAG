BASE_MODEL = "qwen2.5:3b-instruct"
TEMPERATURE = 0.2
MAX_TOKENS = 512
PROMPT_TEMPLATE = (
    "You are a helpful assistant.\n\nContext:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)
EMBED_MODEL = "nomic-embed-text-v2-moe:latest"
RERANKER_MODEL = "BAAI/bge-reranker-base"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DOC_EXTENSIONS = (".md", ".markdown", ".txt", ".rst", ".mdx")
RECIPENLG_FILENAME = "recipes.csv"
MAX_PROMPT_WORDS = 120