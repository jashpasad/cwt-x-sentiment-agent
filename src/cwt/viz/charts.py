"""Visual Matrix Output Generator: how X sentiment moves against price through time.

Static PNGs (report-ready) + one interactive HTML dashboard (plotly, zoomable, range slider).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..features import HORIZONS  # noqa: E402

INK, ACC, NEG, POS, MUTED = "#1f2937", "#2563eb", "#dc2626", "#16a34a", "#9ca3af"
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def sentiment_vs_price_daily(prices, sent, path):
    d = sent[sent.n_posts >= 50].copy()
    daily = d.groupby(d.index.floor("D")).apply(
        lambda g: pd.Series({"sent": np.average(g.h_sent_mean.fillna(g.sent_mean), weights=g.n_posts),
                             "n": g.n_posts.sum(), "bot": np.average(g.bot_share, weights=g.n_posts)}))
    px = prices["close"].resample("D").last().loc[daily.index.min():daily.index.max()]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(px.index, px.values, color=INK, lw=1.1, label="BTC/USDT close (daily)")
    ax.set_yscale("log")
    ax.set_ylabel("BTC price (log scale)")
    ax2 = ax.twinx()
    sc = ax2.scatter(daily.index, daily.sent, c=daily.sent, cmap="RdYlGn", vmin=0.1, vmax=0.5,
                     s=6 + 60 * daily.n / daily.n.max(), alpha=.85, edgecolor="none")
    ax2.plot(daily.sent.rolling(7, min_periods=2).mean(), color=ACC, lw=1, alpha=.7, label="7-obs mean")
    ax2.set_ylabel("Human X sentiment (P(bull) - P(bear))")
    ax2.spines["right"].set_visible(True)
    ax.set_title("X sentiment vs BTC price, Feb 2021 - Jan 2023 (dot size = posts that day; gaps = no collection)")
    fig.colorbar(sc, ax=ax2, pad=0.08, fraction=0.02)
    return _save(fig, path)


def longest_segment(prices, f, path):
    cov = f["covered"].fillna(0) > 0
    grp = (cov != cov.shift()).cumsum()
    seg = cov[cov].groupby(grp[cov]).apply(lambda s: (s.index.min(), s.index.max(), len(s)))
    a, b, _ = max(seg, key=lambda x: x[2])
    w = f.loc[a - pd.Timedelta(hours=12):b]
    p = prices.loc[w.index, "close"]
    fig, axs = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4, 1]})
    axs[0].plot(p, color=INK, lw=1.2)
    axs[0].set_ylabel("BTC (USD)")
    axs[0].set_title(f"Longest continuous X coverage window: {a:%d %b %Y} - {b:%d %b %Y} (hourly)")
    s = w["h_sent_mean"]
    axs[1].plot(s, color=MUTED, lw=.6, label="hourly human sentiment")
    axs[1].plot(w["sent_ema6"], color=ACC, lw=1.4, label="EMA-6h")
    axs[1].fill_between(w.index, w["h_bear_mean"], color=NEG, alpha=.15, label="P(bear) mean")
    axs[1].legend(loc="upper left", ncol=3, frameon=False)
    axs[1].set_ylabel("sentiment")
    colors = np.where(w["att_rel1"] > 0, ACC, MUTED)
    axs[2].bar(w.index, w["att_rel1"], width=1 / 24, color=colors)
    axs[2].set_ylabel("abnormal\nattention")
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    return _save(fig, path)


def lead_lag(f, path, max_lag=24):
    """corr(signal_t, r_{t+k}). k>0: signal LEADS price; k<0: signal FOLLOWS price."""
    raw = f[["h_sent_mean", "att_rel1", "sent_std", "y_0"]].copy()
    # undo the one-bar feature shift so signal and return refer to the same bar index
    sig = raw[["h_sent_mean", "att_rel1", "sent_std"]].shift(-1)
    r = raw["y_0"]
    cov = f["covered"].shift(-1).fillna(0) > 0
    fig, ax = plt.subplots(figsize=(9, 3.8))
    lags = range(-max_lag, max_lag + 1)
    for col, colr, lab in [("h_sent_mean", ACC, "sentiment level"), ("att_rel1", NEG, "abnormal attention"),
                           ("sent_std", POS, "disagreement (dispersion)")]:
        s = sig[col].where(cov)
        cc = [s.corr(r.shift(-k)) for k in lags]
        ax.plot(list(lags), cc, marker="o", ms=2.5, color=colr, lw=1.2, label=lab)
    n = int(cov.sum())
    ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color=MUTED, alpha=.25, label="95% band (iid)")
    ax.axvline(0, color=INK, lw=.6)
    ax.set_xlabel("lag k (hours).  k > 0: X signal at t vs return at t+k (signal leads).  k < 0: signal follows price")
    ax.set_ylabel("correlation")
    ax.set_title("Lead-lag: does X sentiment lead BTC returns, or chase them?")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, path)


def quintile_matrix(f, path):
    """Visual matrix: signal quintile (rows) x horizon n+0..n+5 (cols) -> mean forward return / |return|."""
    e = f[f["eligible"]].copy()
    wk = e.index.tz_localize(None).to_period("W")
    e["bear_wk"] = e["h_bear_mean"] - e.groupby(wk)["h_bear_mean"].transform("mean")
    for h in HORIZONS:
        e[f"yw_{h}"] = e[f"y_{h}"] - e.groupby(wk)[f"y_{h}"].transform("mean")
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.7))
    specs = [("sent_z72", "y", "Sentiment surprise (z vs 72h)", "mean fwd return (bp)", "RdYlGn"),
             ("h_bear_mean", "y", "Bearish share P(bear) - RAW", "mean fwd return (bp)", "RdYlGn"),
             ("bear_wk", "yw", "Bearish share, WEEK-demeaned\n(raw pattern = regime; flips contrarian)", "bp", "RdYlGn"),
             ("att_rel1", "abs", "Abnormal attention", "mean |fwd return| (bp)", "Purples")]
    for ax, (col, kind, title, cbl, cmap) in zip(axs, specs):
        q = pd.qcut(e[col], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
        def tgt(h):
            return e[f"y_{h}"] if kind == "y" else (e[f"yw_{h}"] if kind == "yw" else e[f"y_{h}"].abs())
        mat = pd.DataFrame({f"n+{h}": tgt(h).groupby(q, observed=True).mean() * 1e4 for h in HORIZONS})
        if kind in ("y", "yw"):
            v = np.nanmax(np.abs(mat.values))
            im = ax.imshow(mat.values, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        else:
            im = ax.imshow(mat.values, cmap=cmap, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.values[i, j]:.0f}", ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(6), mat.columns)
        ax.set_yticks(range(5), mat.index)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046, label=cbl)
    fig.suptitle("Visual matrix: X-signal quintile vs forward BTC move at each horizon (descriptive, all covered hours 2021-23; not walk-forward)",
                 fontweight="bold", fontsize=10)
    return _save(fig, path)


def rolling_corr(f, path):
    e = f[f["eligible"]][["sent_chg6", "r_6", "h_sent_mean", "y_0"]].dropna()
    roll = e["h_sent_mean"].rolling(168, min_periods=60).corr(e["y_0"])
    contemp = e["sent_chg6"].rolling(168, min_periods=60).corr(e["r_6"])
    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.plot(contemp.values, color=MUTED, lw=1, label="contemporaneous: 6h sentiment change vs 6h return")
    ax.plot(roll.values, color=ACC, lw=1.2, label="predictive: sentiment at t vs next-hour return")
    ax.axhline(0, color=INK, lw=.6)
    ticks = np.linspace(0, len(e) - 1, 9).astype(int)
    ax.set_xticks(ticks, [e.index[i].strftime("%b %y") for i in ticks])
    ax.set_xlabel("covered hours in time order (gaps compressed)")
    ax.set_ylabel("rolling corr (168 covered h)")
    ax.set_title("Sentiment tracks price contemporaneously far more than it predicts it")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def forecast_fan(prices, f, pred, path, n_paths=1):
    last = pred.index[-300]
    w = prices.loc[last - pd.Timedelta(hours=36): last + pd.Timedelta(hours=8), "close"]
    row = pred.loc[last]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(w.index, w.values, color=INK, lw=1.3, label="actual")
    ts = [last + pd.Timedelta(hours=h + 1) for h in HORIZONS]
    lo = [row[f"plo_{h}"] for h in HORIZONS]
    hi = [row[f"phi_{h}"] for h in HORIZONS]
    mid = [row[f"phat_{h}"] for h in HORIZONS]
    t0 = [last]
    p0 = [f.loc[last, "p_ref"]]
    ax.fill_between(t0 + ts, p0 + lo, p0 + hi, color=ACC, alpha=.18, label="80% conformal interval")
    ax.plot(t0 + ts, p0 + mid, color=ACC, lw=1.6, marker="o", ms=3, label="forecast n+0 .. n+5")
    ax.axvline(last, color=MUTED, ls="--", lw=.8)
    ax.set_title(f"Out-of-sample multi-horizon forecast issued {last:%Y-%m-%d %H:%M} UTC")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def equity(bt: pd.DataFrame, path, keys):
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    pal = [ACC, NEG, POS, "#a855f7", "#f59e0b", INK]
    for k, c in zip(keys, pal):
        if k == "market":
            r = bt[(keys[0], "mkt")]
            lab = "BTC buy & hold (continuous)"
        else:
            r = bt[(k, "net")]
            lab = k + " (net 5bp)"
        eq = np.exp(r.cumsum())
        axs[0].plot(eq, color=c if k != "market" else MUTED, lw=1.2, label=lab)
        if k != "market":
            axs[1].plot(eq / eq.cummax() - 1, color=c, lw=1)
    axs[0].set_yscale("log")
    axs[0].legend(frameon=False, fontsize=8, ncol=2)
    axs[0].set_title("Walk-forward equity (strategies trade only in hours with X coverage; flat otherwise)")
    axs[1].set_ylabel("drawdown")
    return _save(fig, path)


def horizon_metrics(results: dict, path):
    fc = results["forecast"]
    names = ["rw", "price_lgbm", "sent_lgbm", "full_lgbm", "full_allposts", "full_ridge"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    for nm, c in zip(names, [MUTED, INK, NEG, ACC, POS, "#a855f7"]):
        for ax, m in zip(axs, ["dir_acc", "r2_oos_vs_rw", "pi80_coverage"]):
            ax.plot([f"n+{h}" for h in HORIZONS], [fc[nm][str(h)][m] if str(h) in fc[nm] else fc[nm][h][m]
                                                    for h in HORIZONS], marker="o", ms=3, color=c, label=nm)
    axs[0].axhline(.5, color=INK, lw=.5)
    axs[0].set_title("Directional accuracy")
    axs[1].axhline(0, color=INK, lw=.5)
    axs[1].set_title("Out-of-sample R² vs random walk")
    axs[2].axhline(.8, color=INK, lw=.5, ls="--")
    axs[2].set_title("80% interval empirical coverage")
    axs[0].legend(frameon=False, fontsize=7)
    return _save(fig, path)


def importance(imp: pd.DataFrame, path, top=18):
    s = (imp / imp.sum()).mean(axis=1).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 5))
    sent = {"att_rel1", "att_rel6", "h_sent_mean", "h_sent_w", "h_bullishness", "h_bear_mean", "h_bull_mean",
            "sent_mean", "sent_std", "h_sent_std", "bot_share", "sent_ema6", "sent_mom", "sent_z72",
            "sent_chg6", "sent_ret_div", "sent_x_vol", "bear_x_ret", "hours_since_post"}
    ax.barh(s.index, s.values, color=[ACC if i in sent else INK for i in s.index])
    ax.set_title("Feature importance (gain share, avg over horizons)\nblue = X-sentiment features, black = price")
    return _save(fig, path)


def dashboard(prices, f, pred, path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    e = f[f["covered"].fillna(0) > 0]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.5, .3, .2], vertical_spacing=.03)
    px = prices["close"].loc[e.index.min():e.index.max()]
    fig.add_trace(go.Scattergl(x=px.index, y=px, name="BTC close", line=dict(color=INK, width=1)), 1, 1)
    fig.add_trace(go.Scattergl(x=pred.index, y=pred["phat_0"], name="forecast n+0 (OOS)", mode="markers",
                               marker=dict(size=2, color=ACC)), 1, 1)
    fig.add_trace(go.Scattergl(x=e.index, y=e["h_sent_mean"], name="human sentiment", mode="markers",
                               marker=dict(size=2, color=e["h_sent_mean"], colorscale="RdYlGn")), 2, 1)
    fig.add_trace(go.Scattergl(x=e.index, y=e["sent_ema6"], name="sentiment EMA-6h", line=dict(color=ACC)), 2, 1)
    fig.add_trace(go.Bar(x=e.index, y=e["att_rel1"], name="abnormal attention", marker_color=NEG), 3, 1)
    fig.update_layout(height=760, title="CWT X-Sentiment Agent - interactive sentiment vs price explorer (2021-23)",
                      template="plotly_white", legend=dict(orientation="h"))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=.04), row=3, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.write_html(path, include_plotlyjs=True)  # self-contained: works offline and under strict CSPs
    return path


def live_180d(px: pd.DataFrame, sent_live: pd.DataFrame, path, fng: pd.DataFrame | None = None):
    """Last 180 days: BTC price vs X sentiment (daily, from sparse Apify samples) and X attention
    (estimated posts/hour from capped Latest samples)."""
    d = sent_live.copy()
    daily = d.groupby(d.index.floor("D")).agg(sent=("h_sent_mean", "mean"), bear=("h_bear_mean", "mean"),
                                              rate=("n_posts", "median"), k=("n_human", "sum"))
    pxd = px["close"].resample("D").last().loc[daily.index.min():]
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1.4, 1]})
    axs[0].plot(pxd, color=INK, lw=1.3)
    axs[0].set_ylabel("BTC (USD)")
    axs[0].set_title("Live window: BTC vs X sentiment, last 180 days (Apify: 1 sample/day + hourly for the last 7 days)")
    axs[1].scatter(daily.index, daily.sent, s=10, c=daily.sent, cmap="RdYlGn", vmin=-.1, vmax=.4)
    axs[1].plot(daily.sent.rolling(7, min_periods=3).mean(), color=ACC, lw=1.5, label="7-day mean, human net sentiment")
    axs[1].plot(daily.bear.rolling(7, min_periods=3).mean(), color=NEG, lw=1.1, label="7-day mean P(bear)")
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    axs[2].bar(daily.index, daily.rate, color=MUTED, width=.9)
    axs[2].plot(daily.rate.rolling(7, min_periods=3).median(), color=NEG, lw=1.2)
    axs[2].set_ylabel("apparent\nposts/hour")
    axs[2].set_yscale("log")
    axs[2].set_title("Sampled posting rate - falls with age because X's search index thins older posts; "
                     "only comparable within the recent dense window", fontsize=8, fontweight="normal")
    if fng is not None:
        ax3 = axs[1].twinx()
        ax3.plot(fng.set_index("date")["value"].loc[daily.index.min():], color="#f59e0b", lw=1, alpha=.8)
        ax3.set_ylabel("Fear & Greed", color="#f59e0b")
        ax3.spines["right"].set_visible(True)
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return _save(fig, path)
