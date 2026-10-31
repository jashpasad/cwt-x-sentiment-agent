"""Multi-horizon forecaster: one gradient-boosted model per horizon on vol-normalised targets,
with split-conformal prediction intervals.

Why GBM and not an LSTM/Transformer? ~3.5k covered hourly samples is far too few for deep sequence
models to beat a regularised tree ensemble out of sample; the literature's large LSTM "accuracy"
numbers are mostly in-sample or price-level R^2 (trivially high because P_t ~ P_{t-1}). We
report return-space metrics against a random walk instead.

Why conformal? A point forecast of BTC 6h ahead is almost always "about the current price"; the
decision-relevant output is a *calibrated range*. Split conformal (Vovk et al.; Romano et al. 2019)
gives finite-sample coverage guarantees under exchangeability; we check empirical coverage per
horizon in the evaluation report rather than assuming it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EXP
from ..features import HORIZONS


class Forecaster:
    def __init__(self, feats: list[str], kind: str = "lgbm", params: dict | None = None):
        self.feats, self.kind = feats, kind
        self.params = params or EXP.lgb_params
        self.models, self.q = {}, {}

    def _new(self):
        if self.kind == "lgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(random_state=EXP.seed, **self.params)
        if self.kind == "ridge":
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=50.0))
        raise ValueError(self.kind)

    def fit(self, df: pd.DataFrame, calib_frac: float = EXP.calib_frac, alpha: float = EXP.conformal_alpha):
        cut = int(len(df) * (1 - calib_frac))
        tr, ca = df.iloc[:cut], df.iloc[cut:]
        for h in HORIZONS:
            t = tr.dropna(subset=[f"z_{h}"])
            y = t[f"z_{h}"].clip(-6, 6)                      # winsorise flash-crash prints
            m = self._new().fit(t[self.feats], y)
            c = ca.dropna(subset=[f"z_{h}"])
            res = np.abs(c[f"z_{h}"].values - m.predict(c[self.feats]))
            k = int(np.ceil((len(res) + 1) * (1 - alpha)))
            self.q[h] = np.sort(res)[min(k, len(res)) - 1] if len(res) else np.nan
            self.models[h] = m
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for h in HORIZONS:
            z = self.models[h].predict(df[self.feats])
            s = df["sigma"].values * np.sqrt(h + 1)
            out[f"zhat_{h}"] = z
            out[f"rhat_{h}"] = z * s
            out[f"phat_{h}"] = df["p_ref"].values * np.exp(z * s)
            out[f"plo_{h}"] = df["p_ref"].values * np.exp((z - self.q[h]) * s)
            out[f"phi_{h}"] = df["p_ref"].values * np.exp((z + self.q[h]) * s)
        return out

    def importance(self) -> pd.DataFrame:
        if self.kind != "lgbm":
            return pd.DataFrame()
        return pd.DataFrame({h: pd.Series(m.booster_.feature_importance("gain"), index=self.feats)
                             for h, m in self.models.items()})


class RandomWalk:
    """Martingale benchmark: best forecast of price is the last price."""

    def fit(self, df, **kw):
        self.q = {}
        cut = int(len(df) * (1 - EXP.calib_frac))
        ca = df.iloc[cut:]
        for h in HORIZONS:
            res = np.abs(ca[f"z_{h}"].dropna().values)
            k = int(np.ceil((len(res) + 1) * (1 - EXP.conformal_alpha)))
            self.q[h] = np.sort(res)[min(k, len(res)) - 1]
        return self

    def predict(self, df):
        out = pd.DataFrame(index=df.index)
        for h in HORIZONS:
            s = df["sigma"].values * np.sqrt(h + 1)
            out[f"zhat_{h}"] = 0.0
            out[f"rhat_{h}"] = 0.0
            out[f"phat_{h}"] = df["p_ref"].values
            out[f"plo_{h}"] = df["p_ref"].values * np.exp(-self.q[h] * s)
            out[f"phi_{h}"] = df["p_ref"].values * np.exp(self.q[h] * s)
        return out
