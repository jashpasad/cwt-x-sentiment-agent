"""End-to-end historical experiment.

1. Load hourly prices + hourly X sentiment into DuckDB, read back the joined view.
2. Build features (sentiment & price only) with the timing contract in features.py.
3. Purged expanding walk-forward for every variant below (fixed hyper-parameters, no test tuning).
4. Forecast metrics per horizon n+0..n+5, Diebold-Mariano tests, trading backtests, DSR.

Variants (the ablation design is the scientific core: does X sentiment add information?):
  rw            random walk (martingale)                 - the bar every forecaster must clear
  price_lgbm    price features only
  sent_lgbm     sentiment (+ calendar) features only
  full_lgbm     price + human-filtered sentiment         - the proposed model
  full_allposts price + sentiment WITHOUT the bot filter - tests the bot-filter hypothesis
  full_ridge    linear model on the same features        - tests whether non-linearity matters
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt import db  # noqa: E402
from cwt.backtest import metrics as M  # noqa: E402
from cwt.backtest.strategy import backtest, positions, positions_low_turnover  # noqa: E402
from cwt.config import EXP, PROC, RAW, REPORTS  # noqa: E402
from cwt.features import HORIZONS, PRICE_FEATS, SENT_FEATS, TIME_FEATS, build  # noqa: E402
from cwt.models.forecaster import Forecaster, RandomWalk  # noqa: E402
from cwt.validation import walkforward as WF  # noqa: E402

START, END = "2021-01-20", "2023-01-31"


def load_inputs():
    fp = RAW / "btc_1h.parquet"
    if not fp.exists():  # first run: pull the keyless Binance archive
        from cwt.ingest.prices import binance_hourly
        binance_hourly("2021-01-01", "2023-03-31").to_parquet(fp)
    prices = pd.read_parquet(fp).loc[START:END]
    sent = pd.read_parquet(PROC / "sent_hourly_hist.parquet")
    sent.index = pd.to_datetime(sent.index, utc=True)
    con = db.connect()
    db.upsert(con, "prices_1h", prices.rename_axis("time"))
    db.upsert(con, "sent_hourly", sent.assign(source="kaggle_hist").rename_axis("time"))
    mi = db.load_model_input(con, START, END)
    con.close()
    return mi[["open", "high", "low", "close"]], mi.drop(columns=["open", "high", "low", "close"])


def allposts_feats(f: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    """Same features but built from ALL posts (no bot filter)."""
    s2 = sent.copy()
    for a, b in [("h_sent_mean", "sent_mean"), ("h_sent_w", "sent_w"), ("h_bullishness", "bullishness"),
                 ("h_bear_mean", "bear_mean"), ("h_bull_mean", "bull_mean"), ("h_sent_std", "sent_std")]:
        s2[a] = s2[b]
    return s2


def main():
    prices, sent = load_inputs()
    f = build(prices, sent, min_posts=50)
    f_all = build(prices, allposts_feats(None, sent), min_posts=50)
    print("eligible hours:", int(f.eligible.sum()), "of", len(f))
    f.to_parquet(PROC / "features_hist.parquet")

    full = PRICE_FEATS + SENT_FEATS
    variants = {
        "rw": (f, lambda: RandomWalk()),
        "price_lgbm": (f, lambda: Forecaster(PRICE_FEATS)),
        "sent_lgbm": (f, lambda: Forecaster(SENT_FEATS + TIME_FEATS)),
        "full_lgbm": (f, lambda: Forecaster(full)),
        "full_allposts": (f_all, lambda: Forecaster(full)),
        "full_ridge": (f, lambda: Forecaster(full, kind="ridge")),
    }
    preds, fc = {}, {}
    for name, (frame, mk) in variants.items():
        fp = PROC / f"pred_{name}.parquet"
        if fp.exists() and "--reuse" in sys.argv:
            p = pd.read_parquet(fp)
        else:
            p = WF.run(frame, mk)
            p.to_parquet(fp)
        preds[name] = p
        fc[name] = {h: M.forecast_summary(frame, p, h) for h in HORIZONS}
        print(name, {h: round(fc[name][h]["dir_acc"], 3) for h in HORIZONS}, flush=True)

    # Diebold-Mariano: proposed vs each benchmark, per horizon
    dm = {}
    for bench in ["rw", "price_lgbm", "full_allposts", "full_ridge"]:
        dm[bench] = {}
        for h in HORIZONS:
            j = preds["full_lgbm"][[f"rhat_{h}"]].join(preds[bench][[f"rhat_{h}"]], rsuffix="_b") \
                .join(f[f"y_{h}"]).dropna()
            dm[bench][h] = M.dm_test(j[f"y_{h}"] - j[f"rhat_{h}"], j[f"y_{h}"] - j[f"rhat_{h}_b"], h=h + 1)

    # Trading: every (model x signal) combination is a "trial" for the DSR
    bt, trial_sr = {}, []
    for name in ["price_lgbm", "sent_lgbm", "full_lgbm", "full_allposts", "full_ridge"]:
        for sig in ["h0", "ens"]:
            pos = positions(preds[name], f, sig)
            r = backtest(pos, f)
            bt[f"{name}|{sig}"] = r
            trial_sr.append(r["net"].mean() / r["net"].std())
    for name in ["price_lgbm", "full_lgbm"]:   # low-turnover redesign, added after round 1: extra trials
        r = backtest(positions_low_turnover(preds[name], f), f)
        bt[f"{name}|6h_band"] = r
        trial_sr.append(r["net"].mean() / r["net"].std())
    trading = {k: M.trading_summary(v["net"], v["pos"], trial_sr) for k, v in bt.items()}
    ref = bt["full_lgbm|ens"]
    covered_long = ref["mkt"] * (f["eligible"].reindex(ref.index).fillna(False))
    trading["buy_hold_covered_hours"] = M.trading_summary(covered_long, (covered_long != 0).astype(float))
    trading["buy_hold_continuous"] = M.trading_summary(ref["mkt"], np.ones(len(ref)))
    for c in [0.0, 10.0]:
        r = backtest(positions(preds["full_lgbm"], f, "ens"), f, cost_bps=c)
        trading[f"full_lgbm|ens@{int(c)}bps"] = M.trading_summary(r["net"], r["pos"])
    pd.concat({k: v for k, v in bt.items()}, axis=1).to_parquet(PROC / "backtests.parquet")

    imp = Forecaster(full).fit(f[f.eligible]).importance()
    imp.to_csv(REPORTS / "feature_importance_gain.csv")

    out = {"eligible_hours": int(f.eligible.sum()),
           "test_hours": int(len(preds["full_lgbm"])),
           "test_start": str(preds["full_lgbm"].index.min()), "test_end": str(preds["full_lgbm"].index.max()),
           "n_trials_for_dsr": len(trial_sr), "forecast": fc, "diebold_mariano": dm, "trading": trading,
           "config": {k: (v if not isinstance(v, dict) else v) for k, v in EXP.__dict__.items()}}
    (REPORTS / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: {m: round(v[m], 3) if isinstance(v[m], float) else v[m]
                          for m in ["sharpe", "sortino", "max_drawdown", "dsr"] if m in v}
                      for k, v in trading.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
