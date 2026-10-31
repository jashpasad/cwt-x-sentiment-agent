"""Turn forecasts into an hourly long/short BTC position and compute net returns.

pos_n = clip(signal_n / scale_n, -1, 1) * min(1, sigma_target / sigma_n)
  * signal: zhat_0 (1h-ahead) or the horizon-averaged zhat (smoother, lower turnover)
  * scale_n: causal expanding std of past signals (no look-ahead; warm-up uses a fixed prior)
  * vol targeting: size down when trailing realised vol is high (Moreira & Muir 2017 logic)
  * outside covered hours the strategy is FLAT (it only trades when it has X data)
Return_n = pos_n * y_0[n] - cost * |pos_n - pos_{n-1}|   (y_0 = log return over bar n)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EXP
from ..features import HORIZONS

SIGMA_TARGET_H = EXP.target_vol_annual / np.sqrt(24 * 365)


def positions(pred: pd.DataFrame, feats: pd.DataFrame, signal: str = "h0") -> pd.Series:
    if signal == "h0":
        s = pred["zhat_0"]
    else:  # horizon ensemble
        s = pred[[f"zhat_{h}" for h in HORIZONS]].mean(axis=1)
    scale = s.expanding(min_periods=100).std().shift(1).fillna(0.15).clip(lower=0.02)
    raw = (s / scale).clip(-1, 1)
    vol_adj = np.minimum(1.0, SIGMA_TARGET_H / feats.loc[s.index, "sigma"])
    return raw * vol_adj


def backtest(pos: pd.Series, feats: pd.DataFrame, cost_bps: float = EXP.cost_bps) -> pd.DataFrame:
    span = feats.loc[pos.index.min():pos.index.max()]
    p = pos.reindex(span.index).fillna(0.0)
    gross = p * span["y_0"].fillna(0)
    cost = cost_bps / 1e4 * p.diff().abs().fillna(p.abs())
    return pd.DataFrame({"pos": p, "gross": gross, "net": gross - cost, "mkt": span["y_0"].fillna(0)})


def positions_low_turnover(pred: pd.DataFrame, feats: pd.DataFrame, every: int = 6, band: float = 0.25) -> pd.Series:
    """Decide only every `every` hours on the n+5 forecast (non-overlapping 6h bets), hold for 6h,
    and skip re-sizing unless the target moves by more than `band` (no-trade band)."""
    s = pred["zhat_5"]
    scale = s.expanding(min_periods=100).std().shift(1).fillna(0.15).clip(lower=0.02)
    target = (s / scale).clip(-1, 1) * np.minimum(1.0, SIGMA_TARGET_H / feats.loc[s.index, "sigma"])
    target = target[target.index.hour % every == 0]
    full_idx = pd.date_range(target.index.min(), target.index.max() + pd.Timedelta(hours=every - 1),
                             freq="h", tz="UTC")
    pos, cur, out = target.reindex(full_idx), 0.0, []
    for t, v in pos.items():
        if np.isfinite(v) and abs(v - cur) > band:
            cur = v
        elif not np.isfinite(v) and t.hour % every == 0:
            cur = 0.0      # no X data at a decision time -> flat
        out.append(cur)
    return pd.Series(out, index=full_idx)


def vol_managed_long(lrv_pred: pd.Series, feats: pd.DataFrame, cap: float = 1.5) -> pd.Series:
    """Long-only BTC scaled by inverse predicted variance (Moreira & Muir 2017 'volatility-managed
    portfolios'): w = c / sigma_hat^2, normalised to mean ~1 using only past forecasts."""
    var_h = np.exp(lrv_pred) / 6.0
    inv = 1.0 / var_h
    norm = inv.expanding(min_periods=100).mean().shift(1)
    return (inv / norm).clip(0, cap).fillna(0.0)
