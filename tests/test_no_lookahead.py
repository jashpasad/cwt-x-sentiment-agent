"""Leakage tests. If any future information leaks into features, perturbing the future must change
past features - so we perturb prices/posts AFTER a cut-off and assert features BEFORE (and AT) the
cut-off row are bit-identical."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt.backtest.metrics import max_drawdown, sharpe  # noqa: E402
from cwt.features import PRICE_FEATS, SENT_FEATS, build  # noqa: E402


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    c = 30000 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    prices = pd.DataFrame({"open": c, "high": c * 1.002, "low": c * 0.998, "close": c}, index=idx)
    sent = pd.DataFrame({
        "n_posts": rng.integers(40, 400, n).astype(float), "n_human": 100.0, "bot_share": 0.3,
        "sent_mean": rng.normal(0.2, 0.1, n), "sent_std": 0.5, "bull_mean": 0.5, "bear_mean": 0.2,
        "sent_w": rng.normal(0.2, 0.1, n), "h_sent_mean": rng.normal(0.2, 0.1, n), "h_sent_std": 0.5,
        "h_bull_mean": 0.5, "h_bear_mean": rng.uniform(0.1, 0.3, n), "h_sent_w": 0.2,
        "h_bullishness": rng.normal(0.5, 0.2, n), "bullishness": 0.5, "authors": 50.0}, index=idx)
    return prices, sent


def test_features_do_not_see_the_future():
    prices, sent = _toy()
    cut = 400
    f1 = build(prices, sent)
    p2, s2 = prices.copy(), sent.copy()
    p2.iloc[cut:, :] *= 1.5          # shock bar `cut` and everything after
    s2.iloc[cut:, :] = s2.iloc[cut:, :] * -3
    f2 = build(p2, s2)
    cols = PRICE_FEATS + SENT_FEATS
    # row `cut` (decision at the OPEN of bar `cut`) must not know bar `cut`'s data
    pd.testing.assert_frame_equal(f1[cols].iloc[:cut + 1], f2[cols].iloc[:cut + 1])


def test_target_alignment():
    prices, sent = _toy()
    f = build(prices, sent)
    c = np.log(prices["close"])
    n = 300
    assert np.isclose(f["y_0"].iloc[n], c.iloc[n] - c.iloc[n - 1])        # n+0: current bar close
    assert np.isclose(f["y_5"].iloc[n], c.iloc[n + 5] - c.iloc[n - 1])    # n+5: 6h ahead
    assert np.isclose(f["p_ref"].iloc[n], prices["close"].iloc[n - 1])


def test_metrics():
    r = pd.Series([0.01, -0.02, 0.01, 0.01])
    assert max_drawdown(r) < 0
    assert np.isfinite(sharpe(r))
