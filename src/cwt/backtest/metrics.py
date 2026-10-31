"""Forecast-quality and trading metrics.

Trading: Sharpe, Sortino, Max Drawdown, Calmar, hit rate, turnover, plus
  * Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012) - accounts for skew/kurtosis and length
  * Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) - additionally deflates for the number of
    strategy variants tried (selection bias / "backtest overfitting")
  * stationary block-bootstrap 90% CI for the Sharpe ratio
Forecast: RMSE, directional accuracy, out-of-sample R^2 vs random walk (Campbell & Thompson 2008),
Spearman IC, conformal coverage/width, Diebold-Mariano test (HAC, Harvey-Leybourne-Newbold correction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

HOURS_PER_YEAR = 24 * 365


def sharpe(r, ann=HOURS_PER_YEAR):
    r = pd.Series(r).dropna()
    return np.nan if r.std() == 0 else r.mean() / r.std() * np.sqrt(ann)


def sortino(r, ann=HOURS_PER_YEAR):
    r = pd.Series(r).dropna()
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return np.nan if dd == 0 else r.mean() / dd * np.sqrt(ann)


def max_drawdown(r):
    eq = np.exp(pd.Series(r).fillna(0).cumsum())
    return float((eq / eq.cummax() - 1).min())


def psr(r, sr_star=0.0):
    """P(true per-period SR > sr_star) given the sample (non-annualised)."""
    r = pd.Series(r).dropna()
    n, sr = len(r), r.mean() / r.std()
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / den))


def dsr(r, trial_srs):
    """Deflated SR: PSR against the expected max SR of N independent trials."""
    trial_srs = np.asarray([s for s in trial_srs if np.isfinite(s)])
    N = max(len(trial_srs), 2)
    v = np.var(trial_srs, ddof=1) if len(trial_srs) > 1 else 0.0
    emc = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - emc) * stats.norm.ppf(1 - 1 / N) + emc * stats.norm.ppf(1 - 1 / (N * np.e)))
    return psr(r, sr0)


def block_bootstrap_sharpe(r, block=24, n_boot=1000, seed=7):
    """Stationary (Politis-Romano) block bootstrap, vectorised."""
    r = pd.Series(r).dropna().values
    n = len(r)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        new_block = rng.random(n) < 1.0 / block
        new_block[0] = True
        bid = np.cumsum(new_block) - 1
        first = np.flatnonzero(new_block)
        starts = rng.integers(0, n, len(first))
        idx = (starts[bid] + np.arange(n) - first[bid]) % n
        x = r[idx]
        sd = x.std()
        out[b] = x.mean() / sd * np.sqrt(HOURS_PER_YEAR) if sd > 0 else 0.0
    return np.percentile(out, [5, 95])


def trading_summary(r, pos=None, trial_srs=None) -> dict:
    r = pd.Series(np.asarray(r, dtype=float)).fillna(0)
    active = (pd.Series(np.asarray(pos, dtype=float)).abs() > 1e-9) if pos is not None else r != 0
    mdd = max_drawdown(r)
    ann_ret = r.mean() * HOURS_PER_YEAR
    lo, hi = block_bootstrap_sharpe(r)
    d = {
        "sharpe": sharpe(r), "sharpe_ci90": (round(lo, 2), round(hi, 2)), "sortino": sortino(r),
        "max_drawdown": mdd, "calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "ann_return_log": ann_ret, "ann_vol": r.std() * np.sqrt(HOURS_PER_YEAR),
        "hit_rate": float((r[active] > 0).mean()) if active.any() else np.nan,
        "psr_vs_0": psr(r[r != 0]) if (r != 0).sum() > 30 else np.nan,
        "hours": int(len(r)), "active_hours": int(active.sum()),
    }
    if pos is not None:
        d["turnover_per_day"] = float(pd.Series(np.asarray(pos, dtype=float)).diff().abs().sum() / max(len(r) / 24, 1))
    if trial_srs is not None:
        d["dsr"] = dsr(r[r != 0], trial_srs)
    return d


def dm_test(e1, e2, h=1):
    """Diebold-Mariano on squared errors; H1: model 1 more accurate (negative stat). HLN-corrected."""
    d = pd.Series(e1 ** 2 - e2 ** 2).dropna().values
    n = len(d)
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = dbar / np.sqrt(var) if var > 0 else np.nan
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = stats.t.cdf(stat, df=n - 1)  # one-sided
    return float(stat), float(p)


def forecast_summary(df: pd.DataFrame, pred: pd.DataFrame, h: int) -> dict:
    j = pred.join(df[[f"y_{h}", f"z_{h}", "p_ref"]], how="inner").dropna(subset=[f"y_{h}"])
    y, rh = j[f"y_{h}"], j[f"rhat_{h}"]
    price = j["p_ref"] * np.exp(y)
    nz = y != 0
    return {
        "n": int(len(j)),
        "rmse_ret_bp": float(np.sqrt(((y - rh) ** 2).mean()) * 1e4),
        "r2_oos_vs_rw": float(1 - ((y - rh) ** 2).sum() / (y ** 2).sum()),
        "dir_acc": float((np.sign(rh[nz]) == np.sign(y[nz])).mean()),
        "ic_spearman": float(stats.spearmanr(rh, y).statistic) if rh.std() > 0 else np.nan,
        "mape_price_pct": float((np.abs(j[f"phat_{h}"] - price) / price).mean() * 100),
        "pi80_coverage": float(((price >= j[f"plo_{h}"]) & (price <= j[f"phi_{h}"])).mean()),
        "pi80_width_pct": float(((j[f"phi_{h}"] - j[f"plo_{h}"]) / j["p_ref"]).mean() * 100),
    }


def dm_loss(l1, l2, h=1):
    """Diebold-Mariano on an arbitrary loss differential (e.g. QLIKE); negative stat => model 1 better."""
    d = pd.Series(np.asarray(l1) - np.asarray(l2)).dropna().values
    n = len(d)
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(stats.t.cdf(stat, df=n - 1))
