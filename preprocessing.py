import html
import re
from typing import Iterable

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
SPACE_PATTERN = re.compile(r"\s+")
LEMMATIZER = WordNetLemmatizer()


try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    STOP_WORDS = set()


def _safe_lemma(token: str) -> str:
    try:
        return LEMMATIZER.lemmatize(token)
    except LookupError:
        return token


def clean_text(text: object) -> str:
    if text is None:
        return ""

    normalized = html.unescape(str(text)).lower().strip()
    if not normalized:
        return ""

    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = HTML_PATTERN.sub(" ", normalized)
    normalized = NON_ALPHA_PATTERN.sub(" ", normalized)
    normalized = SPACE_PATTERN.sub(" ", normalized).strip()

    tokens: list[str] = []
    for token in normalized.split():
        if token in STOP_WORDS:
            continue
        lemma = _safe_lemma(token)
        if len(lemma) > 1:
            tokens.append(lemma)

    return " ".join(tokens)


def combine_fields(title: object, body: object) -> str:
    title_text = "" if title is None else str(title).strip()
    body_text = "" if body is None else str(body).strip()
    return f"{title_text} {body_text}".strip()


def batch_clean(texts: Iterable[object]) -> list[str]:
    return [clean_text(text) for text in texts]
