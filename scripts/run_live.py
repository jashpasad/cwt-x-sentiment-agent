"""LIVE agent run (last 180 days).

  Apify X posts (budgeted sampling, scripts/scrape_x.py)  -> crypto sentiment + bot filter
  -> hourly sentiment with attention = estimated posts/hour -> DuckDB (joined with Binance prices)
  -> (a) live out-of-sample check on the dense last-7-day window, models trained ONLY on 2021-23
  -> (b) forecast n+0..n+5 for the next hour with 80% conformal ranges
  -> Tavily news (+ Exa if configured) -> 1-page outlook + live chart

If no APIFY_TOKEN is set it degrades to proxy mode (StockTwits + Fear & Greed, price-only forecast).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt import db  # noqa: E402
from cwt.aggregate import finalize, hourly_sums, score_posts  # noqa: E402
from cwt.backtest import metrics as M  # noqa: E402
from cwt.config import APIFY_TOKEN, CHARTS, MODELS, PROC, RAW, REPORTS, ROOT, TAVILY_API_KEY  # noqa: E402
from cwt.features import HORIZONS, PRICE_FEATS, SENT_FEATS, build  # noqa: E402
from cwt.ingest import prices as P  # noqa: E402
from cwt.ingest import proxies  # noqa: E402
from cwt.models.forecaster import Forecaster, RandomWalk  # noqa: E402
from cwt.nlp.sentiment import CryptoSentiment  # noqa: E402
from cwt.report import outlook  # noqa: E402
from cwt.viz import charts as C  # noqa: E402


def next_row(px):
    nxt = px.index[-1] + pd.Timedelta(hours=1)
    return pd.concat([px, pd.DataFrame(index=[nxt], columns=px.columns, dtype=float)])


def live_sentiment(model):
    from cwt.ingest.apify_x import load_cache, load_legacy
    posts, rates = load_cache()
    posts = pd.concat([posts, load_legacy()]).drop_duplicates("post_id").reset_index(drop=True)
    posts.to_parquet(RAW / "apify_posts.parquet")
    rates.to_parquet(RAW / "apify_rates.parquet")
    p = score_posts(posts, model)          # posts bucketed by their TRUE created hour
    s = finalize(hourly_sums(p))
    r = rates.assign(hour=pd.to_datetime(rates.sample_end, utc=True) - pd.Timedelta(hours=1)).set_index("hour")
    s["n_sampled"] = s["n_posts"]
    s["n_posts"] = r["rate_per_h"].reindex(s.index)          # attention = estimated posts/hour
    return p, s


def main():
    model = CryptoSentiment.load(MODELS / "cryptolex.joblib")
    now = pd.Timestamp.now(tz="UTC").floor("h")
    start = (now - pd.Timedelta(days=185)).strftime("%Y-%m-%d")
    px_live = P.hourly_prices(start, now.strftime("%Y-%m-%d %H:%M"))
    px_hist = pd.read_parquet(RAW / "btc_1h.parquet").loc["2021-01-20":"2023-01-31"]
    sent_hist = pd.read_parquet(PROC / "sent_hourly_hist.parquet")
    sent_hist.index = pd.to_datetime(sent_hist.index, utc=True)
    con = db.connect()
    db.upsert(con, "prices_1h", px_live.rename_axis("time"))
    f_hist = build(px_hist, sent_hist)
    fng = proxies.fear_greed(200)
    fng.to_csv(REPORTS / "fear_greed_200d.csv", index=False)
    live = bool(APIFY_TOKEN) and (RAW / "apify").exists()
    state = {"live_mode": live}

    if live:
        posts, sent_live = live_sentiment(model)
        db.upsert(con, "sent_hourly", sent_live.assign(source="apify_live").rename_axis("time"))
        sent_live.to_parquet(PROC / "sent_hourly_live.parquet")
        pxn = next_row(px_live)
        # min_posts gate now applies to estimated posts/hour; require >= 10 actually-sampled posts too
        f_live = build(pxn, sent_live.reindex(pxn.index), min_posts=50)
        ok = sent_live["n_sampled"].reindex(pxn.index).shift(1).fillna(0) >= 10
        # only the dense recent window is age-homogeneous (X's search index thins older posts, which
        # biases sampled posting rates downward with age), so only it supplies hourly model rows
        dense = pxn.index >= now - pd.Timedelta(days=7)
        f_live["eligible"] = f_live["eligible"] & ok & dense
        C.live_180d(px_live, sent_live, CHARTS / "10_live_180d_sentiment_vs_price.png", fng)

        # (a) live out-of-sample check: models fitted on 2021-23 ONLY, evaluated on dense recent hours
        dense_start = now - pd.Timedelta(days=7)
        te = f_live[(f_live.index >= dense_start) & f_live.eligible & f_live["y_5"].notna()]
        tr = f_hist[f_hist.eligible]
        chk = {}
        for name, mk in {"rw": RandomWalk, "price_lgbm": lambda: Forecaster(PRICE_FEATS),
                         "full_lgbm": lambda: Forecaster(PRICE_FEATS + SENT_FEATS)}.items():
            pr = mk().fit(tr).predict(te)
            chk[name] = {h: M.forecast_summary(te, pr, h) for h in (0, 5)}
        state["live_oos_check"] = {"hours": int(len(te)), "from": str(te.index.min()), "to": str(te.index.max()),
                                   "metrics": chk}
        from cwt.ingest.apify_x import usage_usd
        state["apify"] = {"posts": int(len(posts)), "samples": int(sent_live.shape[0]),
                          "from": str(posts.created_utc.min()), "to": str(posts.created_utc.max()),
                          "bot_share": float(posts.bot.mean()), "cost_usd": usage_usd()}
        hp = posts[~posts.bot]
        last24 = hp[hp.created_utc > now - pd.Timedelta(hours=24)]
        last7 = hp[hp.created_utc > now - pd.Timedelta(days=7)]
        base = hp[hp.created_utc <= now - pd.Timedelta(days=7)]
        rate = sent_live["n_posts"]
        soc = {"net_sent": float(last24.sent.mean()), "net_sent_base": float(base.sent.mean()),
               "fng": int(fng.value.iloc[-1]),
               "text": (f"X via Apify: {len(posts):,} posts sampled across {sent_live.shape[0]} hourly windows "
                        f"(180 days); {posts.bot.mean():.0%} flagged automation-like and excluded below. Human net "
                        f"sentiment: last 24h {last24.sent.mean():+.2f}, last 7d {last7.sent.mean():+.2f}, vs "
                        f"{base.sent.mean():+.2f} over the prior ~173 days. Bearish share P(bear): last 24h "
                        f"{last24.p_bear.mean():.2f} vs {base.p_bear.mean():.2f} baseline. Attention: "
                        f"~{rate.iloc[-24:].median():,.0f} posts/hour over the last day vs "
                        f"~{rate.loc[now - pd.Timedelta(days=7):].median():,.0f} 7-day median (rate is only "
                        f"comparable within the recent window). Crypto Fear & Greed {fng.value.iloc[-1]} "
                        f"({fng.value_classification.iloc[-1]}), 30-day mean {fng.value.iloc[-30:].mean():.0f}."),
               "headline": ""}
        train = pd.concat([tr, f_live[f_live.eligible & f_live["y_5"].notna()]])
        feats = PRICE_FEATS + SENT_FEATS
    else:
        f_live = build(next_row(px_live), pd.DataFrame(index=next_row(px_live).index,
                                                       columns=sent_hist.columns, dtype=float))
        train = pd.concat([f_hist.dropna(subset=["rv_72"]), f_live.dropna(subset=["rv_72", "y_5"])])
        feats = PRICE_FEATS
        st = proxies.stocktwits()
        st = pd.concat([st.reset_index(drop=True), model.score(st.text.tolist())], axis=1)
        soc = {"net_sent": float(st.sent.mean()), "net_sent_base": float(sent_hist.h_sent_mean.mean()),
               "fng": int(fng.value.iloc[-1]), "headline": "Proxy mode (no Apify token).",
               "text": f"StockTwits proxy, {len(st)} posts, net {st.sent.mean():+.2f}; F&G {fng.value.iloc[-1]}."}

    fc_model = Forecaster(feats).fit(train)
    row = f_live.iloc[[-1]]
    pr = fc_model.predict(row).iloc[0]
    t0 = row.index[0]
    fc = pd.DataFrame({"phat": [pr[f"phat_{h}"] for h in HORIZONS], "plo": [pr[f"plo_{h}"] for h in HORIZONS],
                       "phi": [pr[f"phi_{h}"] for h in HORIZONS], "ret": [pr[f"rhat_{h}"] for h in HORIZONS]},
                      index=[t0 + pd.Timedelta(hours=h + 1) for h in HORIZONS])
    fc.to_csv(REPORTS / "live_forecast.csv")

    if TAVILY_API_KEY:
        from cwt.ingest.news import exa_news, score_news, tavily_news
        nd = pd.concat([tavily_news(days=7, max_results=8), exa_news(days=7)])
        nd = score_news(nd.dropna(subset=["title"]), model)
        nd.to_csv(REPORTS / "news_scored.csv", index=False)
        nd = nd.sort_values(["published", "score"], ascending=False).drop_duplicates("title")
        state["news_sent_mean"] = float(nd.sent.mean())
        news = [{"date": f"{r.published:%d %b}" if pd.notna(r.published) else "", "title": r.title,
                 "source": r.url.split("/")[2]} for r in nd.head(8).itertuples()]
    else:
        news = json.loads((ROOT / "docs" / "news_snapshot.json").read_text())

    ps = outlook.price_state(px_live)
    notes = ROOT / "docs" / "analyst_notes.md"
    prov = ("Data: Binance BTCUSDT 1h archive + Coinbase; X via Apify actor kaitoeasyapi/twitter-x-data-tweet-scraper-"
            "pay-per-result-cheapest (CJdippxWmn9uRfooo); news via Tavily. Forecast: LightGBM on price + X features, "
            "trained on 2021-23 and the last 180 days." if live else "Proxy mode.")
    md = outlook.render(ps, soc, news, fc, prov, notes.read_text() if notes.exists() else "")
    (REPORTS / "sentiment_outlook.md").write_text(md)
    state.update({"price_state": {k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in ps.items()},
                  "social": soc, "forecast": fc.reset_index().astype(str).to_dict("records")})
    (REPORTS / "live_state.json").write_text(json.dumps(state, indent=2, default=str))
    print(md)
    print(json.dumps(state.get("live_oos_check", {}), indent=1, default=str))
    con.close()


if __name__ == "__main__":
    main()
