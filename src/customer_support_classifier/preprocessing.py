"""Text preprocessing utilities."""

from __future__ import annotations

import re
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are not already present."""
    resources = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for resource_name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource_name)


class LemmatizingTokenizer:
    """Callable tokenizer that normalises, optionally removes stopwords, and lemmatises tokens."""

    def __init__(self, remove_stopwords: bool) -> None:
        ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english")) if remove_stopwords else None

    def __call__(self, text: str) -> List[str]:
        tokens = re.findall(TOKEN_PATTERN, text.lower())
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
        if self.stopwords:
            return [token for token in lemmas if token not in self.stopwords]
        return lemmas


def create_tfidf_vectorizer(config: dict) -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer based on configuration parameters.

    Parameters
    ----------
    config:
        Preprocessing configuration dictionary.

    Returns
    -------
    TfidfVectorizer
        Configured TF-IDF vectorizer.
    """
    tokenizer = None
    if config.get("lemmatize", True):
        tokenizer = LemmatizingTokenizer(remove_stopwords=config.get("remove_stopwords", True))

    ngram_range = config.get("ngram_range", [1, 1])
    if isinstance(ngram_range, Iterable):
        ngram_range = tuple(ngram_range)  # type: ignore[assignment]

    return TfidfVectorizer(
        lowercase=config.get("lowercase", True),
        analyzer="word",
        tokenizer=tokenizer,
        token_pattern=None if tokenizer else TOKEN_PATTERN,
        stop_words=None if tokenizer else ("english" if config.get("remove_stopwords", True) else None),
        max_features=config.get("max_features"),
        ngram_range=ngram_range,
    )

