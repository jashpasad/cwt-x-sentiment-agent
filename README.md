# CWT X-Sentiment Price Prediction Agent (BTC)

An end-to-end, reproducible pipeline that scrapes X (Twitter) posts via **Apify**, pulls recent news via **Tavily**
(and optionally **Exa**), joins them with hourly prices in **DuckDB**, and forecasts **BTC at n+0 … n+5 hours** from
**X sentiment and price only**. It is validated with a purged walk-forward against honest benchmarks.

**Deliverables:** [`outputs/reports/EVALUATION_REPORT.pdf`](outputs/reports/EVALUATION_REPORT.pdf) (also `.md`) ·
[`outputs/reports/sentiment_outlook.md`](outputs/reports/sentiment_outlook.md) (1-page current outlook) ·
[`outputs/charts/`](outputs/charts) (visual matrix + interactive `dashboard.html`) ·
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) (literature, international practice, design).

## Headline findings

| | Result |
|---|---|
| Data integrity | The suggested "Bitcoin Tweets 2025–2026" Kaggle set is the 2021–23 corpus re-dated by **+1,826 days** (10,593/10,774 rows identical). Rejected. See `scripts/audit_provenance.py`. |
| Sentiment model | Crypto-domain classifier trained on 510k author-labelled StockTwits posts: **bull-vs-bear AUC 0.862 vs VADER 0.575**, ~20k posts/s on CPU. |
| Direction (n+0…n+5) | **No model beats a random walk** out of sample (OOS R² < 0; directional accuracy 48–52%, 2,807 test hours). Adding X sentiment to price gives no significant gain (Diebold–Mariano p > 0.18). |
| Trading | All long/short variants lose after 5 bp costs. Best Sharpe ≈ −0.7 (6-hour, no-trade-band variant), Deflated Sharpe ≈ 0.06. BTC buy-and-hold over the same window: Sharpe −0.91, max drawdown −77%. |
| **Volatility** | **X attention and disagreement improve volatility forecasts.** HAR-X beats HAR on QLIKE, **DM t = −3.54, p = 0.0002**. Vol-adaptive intervals fix calm-hour under-coverage (76% → 83%, target 80%). |
| Lead–lag | Sentiment correlates 0.22 with the *previous* hour's return and ~0.04 with the *next* hour's: **X chases price**. |

The value of X data here is **risk/volatility information, not direction**. That matches the most rigorous studies
(Guégan & Renault 2021; Suardi et al. 2022; Akbar et al. 2025) and contradicts the in-sample "90% accuracy" claims
common online.

## Architecture

```
Apify (X posts) ─┐                                       ┌─ walk-forward eval ─ metrics.py (Sharpe, Sortino, MDD, DSR, DM)
Kaggle 2021-23 ──┼─ nlp/sentiment.py ─ nlp/bots.py ─ aggregate.py ─┐            │
Binance archive ─┼──────────────────────────────── db.py (DuckDB view) ─ features.py ─ models/forecaster.py (LGBM × 6 horizons + conformal)
Tavily / Exa ────┘ (news → outlook only)                                          └─ viz/charts.py · report/outlook.py
```

| Module | Purpose |
|---|---|
| `ingest/apify_x.py` | Budget-aware X scraping. Hourly windows; **posting rate recovered from capped "Latest" samples** ((K−1)/span); cost guard. |
| `ingest/news.py` | Tavily news (topic=news) + optional Exa; scored with the same sentiment model |
| `ingest/historical.py` | Robust streaming loader for the malformed Kaggle CSV |
| `nlp/sentiment.py` | TF-IDF (words + emoji) logistic model distilling StockTwits author labels (the supervision CryptoBERT used) |
| `nlp/bots.py` | Automation filter (client, templated text, account age, hashtag stuffing, ticker bots) |
| `features.py` | Timing contract (no look-ahead, unit-tested), vol-normalised targets, sentiment features |
| `models/forecaster.py` | One LightGBM per horizon + split-conformal 80% intervals; random-walk benchmark |
| `validation/walkforward.py` | Purged expanding walk-forward |
| `backtest/` | Strategies (hourly, low-turnover, vol-managed) and metrics incl. PSR/DSR, block-bootstrap CI, DM tests |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # add APIFY_TOKEN and TAVILY_API_KEY (EXA_API_KEY optional)
make audit                      # provenance check of the suggested dataset
make data sentiment hourly      # historical corpus → scored hourly sentiment (~15 min, 1 CPU)
make backtest                   # walk-forward, all variants  → outputs/reports/results.json
python scripts/vol_study.py     # volatility hypothesis       → outputs/reports/vol_study.json
python scripts/scrape_x.py      # Apify, last 180 days (≈ US$3.8 cap on the free plan)
make live                       # DB join, live OOS check, n+0..n+5 forecast, 1-page outlook
make charts report              # charts + EVALUATION_REPORT.{md,pdf}
make test                       # look-ahead / alignment tests
```

## Apify and Tavily (for reproduction)

| Service | Identifier | Notes |
|---|---|---|
| Apify actor (used) | `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`, id `CJdippxWmn9uRfooo` | $0.25 / 1k tweets on the free tier; API allowed on the free plan |
| Apify actor (supported) | `apidojo/tweet-scraper`, id `61RPP7dywgiy0JPD0` | On the free plan: 10 items/run, 5 runs/month, verified 31 Oct 2026. Use `actor="apidojo"` on a paid plan |
| Tavily | `POST https://api.tavily.com/search`, `topic="news"`, 7-day window | |
| Exa (optional) | `POST https://api.exa.ai/search`, `category="news"` | |

Tokens are read from `.env` (git-ignored) and are supplied separately in the submission email.

## Honest limitations

The historical X corpus is sampled and gappy. The live sample is budget-limited (~20 posts per sampled hour).
The test period is mostly the 2022 bear market. Two strategy variants were added after round 1; they are disclosed
and included in the Deflated Sharpe. See §9 of the report.
