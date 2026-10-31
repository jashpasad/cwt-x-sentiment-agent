"""Feature engineering from X sentiment and price ONLY (brief constraint; exchange trading volume is
deliberately excluded because it is neither sentiment nor price).

TIMING CONTRACT (the single most common source of fake alpha in sentiment papers is a
one-bar look-ahead, so it is spelled out and unit-tested in tests/test_no_lookahead.py):

  * Bars are UTC hours labelled by OPEN time. close[t] is the traded price at t+1h.
  * Decision time for row n is the top of hour t_n (the open of bar n).
  * Information set at t_n: posts with created_utc < t_n and closes of bars <= n-1.
    => every feature is computed on bar-indexed series and then shifted by ONE bar.
  * Target for horizon h in {0..5}:  y_h = log(close[n+h] / close[n-1])
    so "n+0" = price at the end of the current hour (1h ahead), "n+5" = 6h ahead.
    n+0 is therefore a genuine forecast, not a nowcast of an already-known price.
  * Targets are vol-normalised: z_h = y_h / (sigma_n * sqrt(h+1)), sigma_n = trailing 72h hourly
    realised vol known at t_n. This makes 2021 bull and 2022 bear regimes comparable; forecasts
    are mapped back to price as  P_hat = close[n-1] * exp(z_hat * sigma_n * sqrt(h+1)).

COVERAGE: the historical X corpus has collection gaps. Sentiment features are computed only over
covered hours (NaN otherwise, which LightGBM handles natively) and post *volume* is expressed
relative to the trailing covered-hour baseline, so collection-effort swings are not mistaken for
attention shocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATS = ["r_1", "r_3", "r_6", "r_12", "r_24", "r_72", "rv_6", "rv_24", "rv_72", "rv_ratio",
               "park_24", "dd_72", "du_72", "hod_sin", "hod_cos", "dow"]
SENT_FEATS = ["att_rel1", "att_rel6", "h_sent_mean", "h_sent_w", "h_bullishness", "h_bear_mean",
              "h_bull_mean", "sent_mean", "sent_std", "h_sent_std", "bot_share", "sent_ema6",
              "sent_mom", "sent_z72", "sent_chg6", "sent_ret_div", "sent_x_vol", "bear_x_ret",
              "hours_since_post"]
TIME_FEATS = ["hod_sin", "hod_cos", "dow"]
HORIZONS = (0, 1, 2, 3, 4, 5)


def _ema_nan(x: pd.Series, span: int) -> pd.Series:
    """EMA over observed points only, then forward-filled at most `span` bars (stale after that)."""
    e = x.dropna().ewm(span=span, min_periods=1).mean().reindex(x.index)
    return e.ffill(limit=span)


def build(prices: pd.DataFrame, sent: pd.DataFrame, min_posts: int = 50) -> pd.DataFrame:
    idx = prices.index
    c, hi, lo = np.log(prices["close"]), np.log(prices["high"]), np.log(prices["low"])
    r = c.diff()
    f = pd.DataFrame(index=idx)
    # ---- price block (bar-indexed; shifted below) ----
    f["r_1"] = r
    for k in (3, 6, 12, 24, 72):
        f[f"r_{k}"] = c - c.shift(k)
    for k in (6, 24, 72):
        f[f"rv_{k}"] = np.sqrt((r ** 2).rolling(k, min_periods=k // 2).mean())
    f["rv_ratio"] = f["rv_6"] / f["rv_72"]
    f["park_24"] = np.sqrt(((hi - lo) ** 2).rolling(24, min_periods=12).mean() / (4 * np.log(2)))
    f["dd_72"] = c - c.rolling(72, min_periods=24).max()
    f["du_72"] = c - c.rolling(72, min_periods=24).min()

    # ---- sentiment block ----
    s = sent.reindex(idx)
    n = s["n_posts"]
    covered = n.fillna(0) >= min_posts
    logn = np.log1p(n.where(covered))
    base = logn.rolling(72, min_periods=6).mean()           # covered-hour baseline (NaNs skipped)
    f["att_rel1"] = logn - base
    f["att_rel6"] = logn.rolling(6, min_periods=3).mean() - base
    for col in ["h_sent_mean", "h_sent_w", "h_bullishness", "h_bear_mean", "h_bull_mean",
                "sent_mean", "sent_std", "h_sent_std", "bot_share"]:
        f[col] = s[col].where(covered)
    hs = f["h_sent_mean"]
    f["sent_ema6"] = _ema_nan(hs, 6)
    f["sent_mom"] = f["sent_ema6"] - _ema_nan(hs, 24)
    mu, sd = hs.rolling(72, min_periods=12).mean(), hs.rolling(72, min_periods=12).std()
    f["sent_z72"] = (hs - mu) / sd
    f["sent_chg6"] = f["sent_ema6"] - f["sent_ema6"].shift(6)
    zr = f["r_6"] / (f["rv_72"] * np.sqrt(6))
    zs = f["sent_chg6"] / f["sent_chg6"].rolling(168, min_periods=24).std()
    f["sent_ret_div"] = zs - zr                               # crowd mood vs what price did
    f["sent_x_vol"] = f["sent_z72"] * f["rv_ratio"]
    f["bear_x_ret"] = f["h_bear_mean"] * zr
    last_cov = pd.Series(np.where(covered, np.arange(len(idx)), np.nan), index=idx).ffill()
    f["hours_since_post"] = np.arange(len(idx)) - last_cov
    f["covered"] = covered.astype(float)

    # ---- shift EVERYTHING by one bar: features for row n use bars <= n-1 ----
    f = f.shift(1)
    # calendar features describe the decision hour itself (known in advance) -> not shifted
    f["hod_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hod_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["dow"] = idx.dayofweek

    # ---- targets ----
    sigma = f["rv_72"]
    f["sigma"] = sigma
    f["p_ref"] = prices["close"].shift(1)                     # last known price at t_n
    for h in HORIZONS:
        y = c.shift(-h) - c.shift(1)
        f[f"y_{h}"] = y
        f[f"z_{h}"] = y / (sigma * np.sqrt(h + 1))
    f["eligible"] = (f["covered"] == 1) & f["rv_72"].notna()
    return f
