from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    APP_TITLE,
    LOG_DIR,
    MAX_CONTENT_LENGTH,
    ensure_runtime_directories,
)
from predict import FakeNewsPredictor, PredictorError, load_training_metrics


ensure_runtime_directories()
LOG_FILE = LOG_DIR / "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = "fake-news-detection-secret-key"
predictor = FakeNewsPredictor()


def get_training_metrics() -> dict:
    return load_training_metrics()


def read_uploaded_text(file_storage) -> str:
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        raise ValueError("Please select a TXT file to upload.")

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a .txt file only.")

    payload = file_storage.read()
    if not payload:
        raise ValueError("The uploaded file is empty. Please upload a TXT file with news content.")

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("The TXT file could not be decoded. Please save it as UTF-8 or ANSI text.")


def extract_news_text() -> str:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        return str(payload.get("text", "")).strip()

    uploaded_file = request.files.get("news_file")
    typed_text = request.form.get("news_text", "").strip()

    if uploaded_file and uploaded_file.filename:
        return read_uploaded_text(uploaded_file).strip()
    return typed_text


def build_page_context(active_page: str, result: dict | None = None, error: str | None = None, input_text: str = "") -> dict:
    metrics = get_training_metrics()
    return {
        "app_title": APP_TITLE,
        "active_page": active_page,
        "metrics": metrics,
        "result": result,
        "error_message": error,
        "input_text": input_text,
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", **build_page_context("home"))


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if request.method == "GET":
        return render_template("index.html", **build_page_context("prediction"))

    try:
        news_text = extract_news_text()
        result = predictor.predict(news_text).to_dict()
        return render_template("index.html", **build_page_context("prediction", result=result, input_text=news_text))
    except (ValueError, PredictorError) as exc:
        logger.warning("Prediction warning: %s", exc)
        return render_template("index.html", **build_page_context("prediction", error=str(exc), input_text=request.form.get("news_text", ""))), 400
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("Unexpected server error during prediction: %s", exc)
        return render_template("index.html", **build_page_context("prediction", error="Something went wrong while processing your request. Please try again.")), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        news_text = extract_news_text()
        result = predictor.predict(news_text).to_dict()
        return jsonify({"success": True, "result": result})
    except (ValueError, PredictorError) as exc:
        logger.warning("API prediction warning: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("Unexpected API error during prediction: %s", exc)
        return jsonify({"success": False, "message": "Internal server error. Please try again later."}), 500


@app.route("/about", methods=["GET"])
def about_page():
    return render_template("about.html", **build_page_context("about"))


@app.route("/readme", methods=["GET"])
def readme_page():
    readme_path = Path(__file__).resolve().parent / "README.md"
    try:
        readme_text = readme_path.read_text(encoding="utf-8")
    except OSError:
        readme_text = "README.md could not be loaded."
    context = build_page_context("readme")
    context["readme_text"] = readme_text
    return render_template("readme.html", **context)


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", **build_page_context("home", error="The page you requested was not found."), status_code=404), 404


@app.errorhandler(413)
def too_large(_error):
    return render_template("error.html", **build_page_context("prediction", error="The uploaded file is too large. Please keep TXT uploads under 1 MB."), status_code=413), 413


@app.errorhandler(500)
def internal_error(_error):
    return render_template("error.html", **build_page_context("home", error="An unexpected server error occurred. Please restart the app and try again."), status_code=500), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
