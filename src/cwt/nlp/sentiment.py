"""Post-level sentiment.

Why not just VADER or FinBERT?
* VADER is a general-purpose lexicon; crypto slang ("rekt", "hodl", "wagmi", "moon", rocket emoji)
  is mostly out of vocabulary. CryptoBERT's authors report VADER at ~50% 3-class accuracy on
  crypto posts (Kulakowski & Frasincar, IEEE Intelligent Systems 38(4), 2023).
* CryptoBERT itself is better but costs ~20-40 ms/post on CPU; scoring millions of posts every
  refresh is not practical in a laptop-reproducible pipeline.

So we *distil the same supervision signal* that CryptoBERT was fine-tuned on: StockTwits crypto
posts whose bullish/bearish tag was set by the author at posting time (exogenous labels, no
annotator bias). A word+emoji TF-IDF logistic model learns crypto vocabulary, runs at
~50k posts/s, and is benchmarked out-of-sample against VADER in the report.

Score = P(bullish) - P(bearish) in [-1, 1]. We also keep P(bull), P(bear) separately because
the literature (e.g. Guba/Eastmoney studies in China) finds negative tone carries more
information than positive tone - an asymmetry a single net score would hide.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
NUM = re.compile(r"(?<![\w$])\d[\d,\.]*k?")
WS = re.compile(r"\s+")
# words, $cashtags, #tags and single emoji/pictographs as tokens
TOKEN = r"(?u)[$#]?\b\w[\w']+\b|[\U0001F300-\U0001FAFF\u2600-\u27BF]"


def normalize(text: str) -> str:
    t = str(text).lower()
    t = URL.sub(" url ", t)
    t = MENTION.sub(" @user ", t)
    t = NUM.sub(" num ", t)
    t = t.replace("&amp;", "&")
    return WS.sub(" ", t).strip()


class CryptoSentiment:
    """TF-IDF(1-2 gram words + emoji) -> multinomial logistic regression (bear/neutral/bull)."""

    def __init__(self):
        self.pipe = None

    def fit(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalize, token_pattern=TOKEN,
                                      ngram_range=(1, 2), min_df=5, max_features=300_000,
                                      sublinear_tf=True, dtype=np.float32)),
            ("lr", LogisticRegression(C=2.0, max_iter=600, class_weight="balanced")),
        ])
        self.pipe.fit(texts, labels)
        return self

    def proba(self, texts) -> np.ndarray:
        """Returns columns [p_bear, p_neutral, p_bull]."""
        return self.pipe.predict_proba(texts)

    def score(self, texts) -> pd.DataFrame:
        p = self.proba(list(texts))
        return pd.DataFrame({"p_bear": p[:, 0], "p_bull": p[:, 2],
                             "sent": p[:, 2] - p[:, 0]})

    def save(self, path: Path):
        joblib.dump(self.pipe, path)

    @classmethod
    def load(cls, path: Path):
        m = cls()
        m.pipe = joblib.load(path)
        return m


def vader_scores(texts) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    return np.array([a.polarity_scores(str(t))["compound"] for t in texts])
