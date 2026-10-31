"""Purged, expanding-window walk-forward validation (Lopez de Prado 2018, ch. 7 & 12).

* Folds are defined on ELIGIBLE rows (covered hours), in time order.
* Each fold: fit on all eligible rows strictly before the test block, minus a purge window
  (training rows whose 6h label horizon would overlap the test start are dropped), then predict the
  next block. Models are refit every `step` eligible rows (~ weekly at full coverage).
* Hyper-parameters are fixed a priori in config.py - nothing is tuned on test folds, so every test
  prediction is genuinely out-of-sample. The number of variants we compared is recorded and fed into
  the Deflated Sharpe Ratio.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from ..config import EXP


def folds(df: pd.DataFrame, min_train: int = 1200, step: int = 250):
    el = df.index[df["eligible"]]
    purge = pd.Timedelta(hours=EXP.purge_hours)
    start = min_train
    while start < len(el):
        test_idx = el[start:start + step]
        t0 = test_idx[0]
        train_idx = el[:start]
        train_idx = train_idx[train_idx < t0 - purge]
        yield train_idx, test_idx
        start += step


def run(df: pd.DataFrame, make_model: Callable, min_train=1200, step=250) -> pd.DataFrame:
    preds = []
    for k, (tr, te) in enumerate(folds(df, min_train, step)):
        m = make_model().fit(df.loc[tr])
        p = m.predict(df.loc[te])
        p["fold"] = k
        preds.append(p)
    return pd.concat(preds)
