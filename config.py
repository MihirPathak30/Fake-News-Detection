from pathlib import Path

APP_TITLE = "AI Powered Fake News Detection System"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
REPORT_DIR = BASE_DIR / "report"
LOG_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

TRUE_DATASET_NAME = "True.csv"
FAKE_DATASET_NAME = "Fake.csv"
DATASET_SEARCH_DIRS = (
    BASE_DIR,
    BASE_DIR / "datasets",
    BASE_DIR / "data",
    BASE_DIR.parent,
)

ALLOWED_UPLOAD_EXTENSIONS = {".txt"}
MAX_CONTENT_LENGTH = 1 * 1024 * 1024
TFIDF_MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42


def ensure_runtime_directories() -> None:
    for folder in (REPORT_DIR, LOG_DIR, STATIC_DIR, TEMPLATE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def find_dataset_file(filename: str) -> Path | None:
    for directory in DATASET_SEARCH_DIRS:
        candidate = directory / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None
