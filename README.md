# AI Powered Fake News Detection System

A production-quality Flask machine learning project that trains its own fake-news classifier from **True.csv** and **Fake.csv** using **Python 3.12**, **scikit-learn**, **pandas**, **nltk**, and **joblib**.

## Installation

1. Extract the `FakeNewsDetection` project folder.
2. Place `True.csv` and `Fake.csv` in the project root folder.
   - Supported fallback locations: `data/` or `datasets/`
3. Double-click `setup_project.bat`
4. Wait until the script finishes and shows **SUCCESS**.

## Running

1. Double-click `run_project.bat`
2. The script validates the trained files.
3. If `model.pkl` or `vectorizer.pkl` is missing, the project automatically retrains.
4. Flask starts locally.
5. Your browser opens automatically at `http://127.0.0.1:5000`

## Features

- Fresh model training from the provided CSV files
- Text preprocessing with lowercasing, URL removal, HTML removal, punctuation cleanup, digit removal, stopword removal, and lemmatization
- TF-IDF vectorization with `max_features=5000`
- Logistic Regression classification model
- 80/20 train-test split with `random_state=42`
- Automatic generation of:
  - `model.pkl`
  - `vectorizer.pkl`
  - classification report
  - accuracy / precision / recall / F1 summary
  - confusion matrix CSV
- Prediction from pasted text or uploaded TXT files
- Friendly error handling and logging
- Premium dark glassmorphism UI

## Project Structure

```text
FakeNewsDetection/
│
├── app.py
├── config.py
├── train.py
├── predict.py
├── preprocessing.py
├── requirements.txt
├── setup_project.bat
├── run_project.bat
├── model.pkl                # generated after setup
├── vectorizer.pkl           # generated after setup
│
├── templates/
├── static/
├── report/                  # generated training reports
├── screenshots/
└── logs/
```

## Requirements

- Windows
- Python 3.12 installed and available in PATH
- Internet connection during the first setup for dependency installation and NLTK downloads
- `True.csv` and `Fake.csv` available in the project root, `data/`, or `datasets/`

## Troubleshooting

### Python 3.12 not found
Install Python 3.12 and make sure it is added to PATH.

### Setup fails during package installation
Check internet access, then run `setup_project.bat` again.

### Dataset not found
Place `True.csv` and `Fake.csv` beside `setup_project.bat`, or inside `data/` / `datasets/`.

### Browser does not open automatically
Open this URL manually:

```text
http://127.0.0.1:5000
```

### Model mismatch concern
`model.pkl` and `vectorizer.pkl` are always generated locally by `train.py` using the exact `scikit-learn` version installed from `requirements.txt`.

## Notes

- The distributed ZIP intentionally excludes `venv/`, CSV datasets, cached files, generated reports, and trained artifacts.
- Re-running setup skips training when both `model.pkl` and `vectorizer.pkl` already exist.
- Logs are written to `logs/app.log`.
