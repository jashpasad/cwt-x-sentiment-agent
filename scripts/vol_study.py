"""Hypothesis test suggested by the literature (Suardi et al. 2022; Akbar et al. 2025):
X attention / disagreement carries information about future VOLATILITY rather than direction.

Target: log realised variance over the forecast window n..n+5 (6 hourly squared log returns).
Models (same purged walk-forward folds as the price study):
  har       HAR-RV benchmark (Corsi 2009): OLS on log rv_6, rv_24, rv_72  - the standard vol benchmark
  price_gbm LightGBM on price features
  full_gbm  LightGBM on price + X sentiment features
  har_x     HAR + X attention/disagreement terms (linear, interpretable)
Metrics: MSE of log-RV, QLIKE (robust loss for variance forecasts, Patton 2011), DM tests.
Then: vol-adaptive conformal intervals (width scaled by predicted vol) vs the constant-width version,
checked for conditional coverage in calm vs turbulent hours.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt.backtest.metrics import dm_loss, dm_test  # noqa: E402
from cwt.config import EXP, PROC, REPORTS  # noqa: E402
from cwt.features import PRICE_FEATS, SENT_FEATS  # noqa: E402
from cwt.validation.walkforward import folds  # noqa: E402

X_TERMS = ["att_rel1", "att_rel6", "sent_std", "h_sent_std", "bot_share", "h_bear_mean", "sent_z72"]


def add_target(f: pd.DataFrame) -> pd.DataFrame:
    y0 = f["y_0"]  # log return of bar n
    rv = sum(y0.shift(-k) ** 2 for k in range(6))
    f = f.copy()
    f["lrv_fwd"] = np.log(rv + 1e-10)
    for k in (6, 24, 72):
        f[f"lrv_{k}"] = np.log(f[f"rv_{k}"] ** 2 + 1e-12)
    return f


def fit_predict(kind, tr, te):
    y = tr["lrv_fwd"]
    if kind in ("har", "har_x"):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        cols = ["lrv_6", "lrv_24", "lrv_72"] + (X_TERMS if kind == "har_x" else []) + ["hod_sin", "hod_cos"]
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)).fit(tr[cols], y)
        return m.predict(te[cols])
    import lightgbm as lgb
    cols = PRICE_FEATS + ["lrv_6", "lrv_24", "lrv_72"] + (SENT_FEATS if kind == "full_gbm" else [])
    p = dict(EXP.lgb_params, objective="l2")
    p.pop("alpha", None)
    m = lgb.LGBMRegressor(random_state=EXP.seed, **p).fit(tr[cols], y)
    return m.predict(te[cols])


def qlike(lrv_true, lrv_pred):
    s2, h = np.exp(lrv_true), np.exp(lrv_pred)
    return s2 / h - np.log(s2 / h) - 1


def main():
    f = add_target(pd.read_parquet(PROC / "features_hist.parquet"))
    f = f[f["lrv_fwd"].notna() | ~f["eligible"]]
    preds = {k: [] for k in ["har", "har_x", "price_gbm", "full_gbm"]}
    for tr, te in folds(f):
        trd, ted = f.loc[tr].dropna(subset=["lrv_fwd"]), f.loc[te].dropna(subset=["lrv_fwd"])
        for k in preds:
            preds[k].append(pd.Series(fit_predict(k, trd, ted), index=ted.index))
    P = pd.DataFrame({k: pd.concat(v) for k, v in preds.items()})
    y = f.loc[P.index, "lrv_fwd"]
    res = {}
    for k in P:
        e = y - P[k]
        res[k] = {"mse_logrv": float((e ** 2).mean()), "qlike": float(qlike(y, P[k]).mean()),
                  "r2_vs_har": float(1 - (e ** 2).sum() / ((y - P["har"]) ** 2).sum()),
                  "corr": float(np.corrcoef(y, P[k])[0, 1])}
    pairs = [("har_x", "har"), ("full_gbm", "price_gbm"), ("full_gbm", "har"), ("price_gbm", "har")]
    dm = {f"{a}_vs_{b}": {"mse": dm_test(y - P[a], y - P[b], h=6),
                          "qlike": dm_loss(qlike(y, P[a]), qlike(y, P[b]), h=6)} for a, b in pairs}

    # vol-adaptive conformal: width proportional to predicted sigma (from full_gbm), calibrated causally
    sig_hat = np.exp(P["full_gbm"] / 2) / np.sqrt(6)            # hourly-equivalent sigma forecast
    j = f.loc[P.index]
    z_adapt = (j["y_5"] / (sig_hat * np.sqrt(6))).abs()
    z_const = (j["y_5"] / (j["sigma"] * np.sqrt(6))).abs()
    q_adapt = z_adapt.expanding(200).quantile(0.8).shift(1)
    q_const = z_const.expanding(200).quantile(0.8).shift(1)
    cov_a = (z_adapt <= q_adapt)
    cov_c = (z_const <= q_const)
    turb = j["rv_72"] > j["rv_72"].median()
    width_a = (q_adapt * sig_hat * np.sqrt(6) * 2)
    width_c = (q_const * j["sigma"] * np.sqrt(6) * 2)
    cond = {
        "adaptive": {"coverage_all": float(cov_a[q_adapt.notna()].mean()),
                     "coverage_calm": float(cov_a[q_adapt.notna() & ~turb].mean()),
                     "coverage_turbulent": float(cov_a[q_adapt.notna() & turb].mean()),
                     "mean_width_pct": float(width_a.mean() * 100)},
        "constant": {"coverage_all": float(cov_c[q_const.notna()].mean()),
                     "coverage_calm": float(cov_c[q_const.notna() & ~turb].mean()),
                     "coverage_turbulent": float(cov_c[q_const.notna() & turb].mean()),
                     "mean_width_pct": float(width_c.mean() * 100)},
    }
    P.to_parquet(PROC / "vol_preds.parquet")

    # Vol-managed long-only BTC during covered hours: does the X-informed vol forecast size risk better?
    from cwt.backtest.metrics import trading_summary
    from cwt.backtest.strategy import vol_managed_long
    prior = json.loads((REPORTS / "results.json").read_text())
    r1 = j["y_0"]                                   # return of bar n, decided at its open
    vm = {"long_unmanaged": pd.Series(1.0, index=P.index)}
    for k in ["har", "har_x", "full_gbm"]:
        vm[f"vol_managed_{k}"] = vol_managed_long(P[k], j)
    trial_srs = []
    rets = {}
    for k, w in vm.items():
        cost = EXP.cost_bps / 1e4 * w.diff().abs().fillna(w.abs())
        rets[k] = w * r1 - cost
        trial_srs.append(rets[k].mean() / rets[k].std())
    vm_res = {k: trading_summary(v, vm[k], trial_srs) for k, v in rets.items()}
    out = {"n_test": int(len(P)), "models": res, "diebold_mariano": dm, "interval_n5": cond,
           "vol_managed_long": vm_res}
    (REPORTS / "vol_study.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
