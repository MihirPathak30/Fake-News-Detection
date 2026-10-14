from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from config import (
    FAKE_DATASET_NAME,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TRUE_DATASET_NAME,
    VECTORIZER_PATH,
    ensure_runtime_directories,
    find_dataset_file,
)
from preprocessing import batch_clean, combine_fields


class TrainingError(Exception):
    """Raised when the training pipeline cannot proceed."""


def load_dataset(file_path: Path, label: int) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        dataframe = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")

    expected_columns = {"title", "text"}
    if not expected_columns.issubset(dataframe.columns):
        raise TrainingError(f"{file_path.name} must contain title and text columns.")

    dataframe = dataframe.fillna("")
    dataframe["content"] = dataframe.apply(lambda row: combine_fields(row["title"], row["text"]), axis=1)
    dataframe["label"] = label
    return dataframe[["content", "label"]]


def write_reports(report_dir: Path, metrics: dict, report_text: str, cm_values: list[list[int]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    confusion_df = pd.DataFrame(cm_values, index=["Actual Fake", "Actual Real"], columns=["Pred Fake", "Pred Real"])
    confusion_df.to_csv(report_dir / "confusion_matrix.csv", encoding="utf-8")

    summary_lines = [
        f"Generated: {metrics['generated_at_utc']}",
        f"Samples used: {metrics['samples_used']}",
        f"Vocabulary size: {metrics['vocabulary_size']}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1 Score: {metrics['f1_score']:.4f}",
        f"scikit-learn version: {metrics['sklearn_version']}",
    ]
    (report_dir / "training_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> int:
    ensure_runtime_directories()

    true_path = find_dataset_file(TRUE_DATASET_NAME)
    fake_path = find_dataset_file(FAKE_DATASET_NAME)
    if not true_path or not fake_path:
        raise TrainingError(
            "Datasets not found. Place True.csv and Fake.csv in the project root, data/, or datasets/ folder, then run setup_project.bat again."
        )

    true_df = load_dataset(true_path, label=1)
    fake_df = load_dataset(fake_path, label=0)
    dataset = pd.concat([true_df, fake_df], ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    dataset["clean_text"] = batch_clean(dataset["content"])
    dataset = dataset[dataset["clean_text"].str.strip() != ""].reset_index(drop=True)

    if dataset.empty:
        raise TrainingError("Training data became empty after preprocessing. Please verify the CSV content.")

    x_train, x_test, y_train, y_test = train_test_split(
        dataset["clean_text"],
        dataset["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataset["label"],
    )

    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="liblinear")
    model.fit(x_train_features, y_train)

    predictions = model.predict(x_test_features)
    report_text = classification_report(y_test, predictions, target_names=["Fake", "Real"], zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    metrics = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "samples_used": int(len(dataset)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "sklearn_version": sklearn.__version__,
        "model": "Logistic Regression",
        "vectorizer": f"TF-IDF (max_features={TFIDF_MAX_FEATURES})",
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    write_reports(REPORT_DIR, metrics, report_text, cm.tolist())

    print("Training completed successfully.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Saved model to: {MODEL_PATH.name}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
