"""Train the crypto sentiment classifier on author-labelled StockTwits crypto posts
(ElKulako/stocktwits-crypto, HF) and benchmark it out-of-sample against VADER.

Split is *chronological-agnostic random* because the file carries no timestamps; we report
it as a text-classification benchmark only, never as evidence of price predictability.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt.config import MODELS, RAW, REPORTS  # noqa: E402
from cwt.nlp.sentiment import CryptoSentiment, vader_scores  # noqa: E402

URL = "https://huggingface.co/datasets/ElKulako/stocktwits-crypto/resolve/main/st-data-full.xlsx"


def load() -> pd.DataFrame:
    pq = RAW / "stocktwits.parquet"
    if not pq.exists():
        import urllib.request
        xl = RAW / "st-data-full.xlsx"
        if not xl.exists():
            urllib.request.urlretrieve(URL, xl)
        df = pd.read_excel(xl)
        df["text"] = df["text"].astype(str)
        df.to_parquet(pq, index=False)
    return pd.read_parquet(pq)


def main():
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    df = load().dropna()
    df = df[df.text.str.split().str.len() >= 3]
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(df))
    cut = int(0.85 * len(df))
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]

    t0 = time.time()
    m = CryptoSentiment().fit(tr.text.tolist(), tr.label.values)
    fit_s = time.time() - t0
    m.save(MODELS / "cryptolex.joblib")  # save first: benchmarking must not risk the artefact
    t0 = time.time()
    p = m.proba(te.text.tolist())
    rate = len(te) / (time.time() - t0)
    pred = p.argmax(1)

    # VADER on a 20k subsample (it is slow); map compound to 3 classes with standard +-0.05
    sub = te.sample(20000, random_state=7)
    v = vader_scores(sub.text)
    vpred = np.where(v > 0.05, 2, np.where(v < -0.05, 0, 1))
    ps = m.proba(sub.text.tolist())
    mp, sp = ps.argmax(1), ps[:, 2] - ps[:, 0]

    # Directional agreement on polar posts only (bull vs bear), the part that matters for trading
    polar = sub.label.values != 1
    res = {
        "n_train": int(len(tr)), "n_test": int(len(te)), "fit_seconds": round(fit_s, 1),
        "posts_per_second": int(rate),
        "cryptolex_acc_3class": round(accuracy_score(te.label, pred), 4),
        "cryptolex_macroF1_3class": round(f1_score(te.label, pred, average="macro"), 4),
        "cryptolex_acc_3class_sub": round(accuracy_score(sub.label, mp), 4),
        "vader_acc_3class_sub": round(accuracy_score(sub.label, vpred), 4),
        "cryptolex_macroF1_sub": round(f1_score(sub.label, mp, average="macro"), 4),
        "vader_macroF1_sub": round(f1_score(sub.label, vpred, average="macro"), 4),
        # threshold-free: does the continuous score rank bullish above bearish posts?
        "cryptolex_bull_vs_bear_auc": round(roc_auc_score(sub.label.values[polar] == 2, sp[polar]), 4),
        "vader_bull_vs_bear_auc": round(roc_auc_score(sub.label.values[polar] == 2, v[polar]), 4),
    }
    print(json.dumps(res, indent=2))
    (REPORTS / "sentiment_benchmark.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
