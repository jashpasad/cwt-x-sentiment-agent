"""Generate every chart in outputs/charts from the artefacts written by run_backtest.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt.config import CHARTS, PROC, RAW, REPORTS  # noqa: E402
from cwt.viz import charts as C  # noqa: E402


def main():
    prices = pd.read_parquet(RAW / "btc_1h.parquet").loc["2021-01-20":"2023-01-31"]
    sent = pd.read_parquet(PROC / "sent_hourly_hist.parquet")
    sent.index = pd.to_datetime(sent.index, utc=True)
    f = pd.read_parquet(PROC / "features_hist.parquet")
    pred = pd.read_parquet(PROC / "pred_full_lgbm.parquet")
    bt = pd.read_parquet(PROC / "backtests.parquet")
    res = json.loads((REPORTS / "results.json").read_text())
    imp = pd.read_csv(REPORTS / "feature_importance_gain.csv", index_col=0)
    out = [
        C.sentiment_vs_price_daily(prices, sent, CHARTS / "01_sentiment_vs_price.png"),
        C.longest_segment(prices, f, CHARTS / "02_hourly_zoom_longest_window.png"),
        C.lead_lag(f, CHARTS / "03_lead_lag.png"),
        C.quintile_matrix(f, CHARTS / "04_visual_matrix_quintile_by_horizon.png"),
        C.rolling_corr(f, CHARTS / "05_rolling_correlation.png"),
        C.forecast_fan(prices, f, pred, CHARTS / "06_forecast_fan.png"),
        C.equity(bt, CHARTS / "07_equity_drawdown.png",
                 ["full_lgbm|ens", "full_lgbm|6h_band", "price_lgbm|6h_band", "sent_lgbm|ens", "full_allposts|ens", "market"]),
        C.horizon_metrics(res, CHARTS / "08_metrics_by_horizon.png"),
        C.importance(imp, CHARTS / "09_feature_importance.png"),
        C.dashboard(prices, f, pred, CHARTS / "dashboard.html"),
    ]
    print("\n".join(map(str, out)))


if __name__ == "__main__":
    main()
