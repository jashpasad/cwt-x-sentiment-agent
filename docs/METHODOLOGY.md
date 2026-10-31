# Methodology, literature and design rationale

## 1. What the literature actually says (and what we took from it)

| # | Study | Setting | Finding | Design consequence here |
|---|---|---|---|---|
| 1 | Bollen, Mao & Zeng (2011), *J. Comp. Sci.* | Twitter mood vs DJIA (US) | Mood dimensions Granger-cause index moves at days' lag | Origin of the field; daily lag claims later failed to replicate out of sample, so we validate strictly OOS |
| 2 | Kraaijeveld & De Smedt (2020), *J. Int. Fin. Markets* (UK/Edinburgh) | 9 coins, crypto lexicon, Granger | Sentiment has predictive power for BTC; bullishness ratio strongest; bots present | Antweiler–Frank bullishness index; explicit bot filter |
| 3 | Guégan & Renault (2021), *Finance Research Letters* (France) | ~1M StockTwits msgs, intraday BTC | Significant only up to ~15-minute horizons; vanishes at lower frequency, concentrated in the 2017 bubble | Hourly (not daily) bars; expect a weak, regime-dependent signal; don't over-promise |
| 4 | Suardi, Rasel & Liu (2022), *Int. Rev. Econ. & Finance* (Australia) | VADER sentiment dispersion & attention | Dispersion predicts volatility; attention predicts volume, not returns | Disagreement (sent_std) feature; volatility-aware sizing; conformal ranges instead of point bets |
| 5 | Maître, Pugachyov & Weigert (2025), CFR Cologne WP 25-02 (Germany) | Cross-section of cryptos 2018-22 | *Abnormal* attention (log posts − 30-day mean) predicts next-day returns; driven by investor ticker-tweets, not official accounts | `att_rel` = log volume vs trailing baseline; human-filtered signal |
| 6 | Akbar et al. (2025), *Studies in Econ. & Finance* (Emerald) | 16M BTC tweets, human vs automation split | Polarity has no return predictability; attention drives volatility; automation-like accounts dominate and dilute polarity | Two parallel signal sets (all vs human) and an ablation that tests the bot filter |
| 7 | Kulakowski & Frasincar (2023), *IEEE Intelligent Systems* 38(4) (NL) — CryptoBERT | 3.2M crypto posts; StockTwits author labels | Domain model ≫ VADER (≈70% vs ≈50% 3-class) | We distil the same author-label supervision into a fast TF-IDF model (AUC 0.86 vs VADER 0.58 on bull-vs-bear) |
| 8 | Ciganovic & D'Amario (2024), LASSO-VAR (Italy) | 10 coins, Twitter+Reddit+Google Trends | Social sentiment does **not** Granger-cause returns; no forecast-accuracy gain for BTC | Must benchmark against a random walk and a price-only model and report honestly |
| 9 | Huang et al. (2022), *Emerging Mkts Fin. & Trade* (China, Guba/Eastmoney) | Retail forum sentiment | Negative sentiment matters; positive does not (asymmetry) | Bearish share P(bear) kept as a separate feature, not netted away |
| 10 | Lv et al. (2022), *Applied Econ. Letters* (China, Eastmoney) | VAR on forum sentiment | Negative indicators explain returns better; sentiment also *reacts* to returns | Lead-lag chart tests direction of causality explicitly |
| 11 | Trushkovskyi (2025), *J. Advanced Research in Law & Economics* (Ukraine) | Twitter+Reddit daily index 2020-25 | Lagged sentiment +0.24% next-day; ML beats OLS | Non-linear GBM vs linear Ridge ablation |
| 12 | Bailey & López de Prado (2012, 2014); López de Prado (2018) | Methodology | PSR/DSR; purged walk-forward; backtest overfitting | Purged expanding walk-forward, fixed hyper-params, DSR over all variants tried |

**Synthesis.** The more rigorous the evaluation, the smaller the directional edge from social *polarity*; the
robust findings are about **attention, disagreement and volatility**, about **negative-tone asymmetry**, and
about **bots diluting signal**. So the system is designed to (a) forecast a calibrated *range* for n+0..n+5,
(b) let attention/disagreement modulate risk, and (c) prove—or disprove—incremental value of X data with
ablations against price-only and random-walk benchmarks.

## 2. How this is done in practice abroad

* **US** — commercial social/news sentiment feeds sold to funds: LSEG MarketPsych (sentiment, fear, disagreement
  indices for equities and crypto), Bloomberg's social-sentiment fields, Santiment and LunarCrush for crypto (social
  volume, "social dominance", weighted sentiment). Common thread: they ship *volume/attention and dispersion*
  alongside polarity — mirrored in our feature set.
* **UK / Euro area** — central banks use text sentiment as a *nowcasting and risk* input (BoE and ECB research on
  news-based sentiment), i.e. for conditional risk rather than point return prediction — our conformal-range framing.
* **China** — Guba/Eastmoney forum indices (e.g. GubaSenti) are the most studied retail-sentiment measures; the
  consistent finding of negative-tone asymmetry and short-term reversal informs our separate bear-share feature.
* **Russia / CIS** — retail crypto discussion has migrated to Telegram channels; the CryptoBERT post-training
  corpus includes Telegram. Our ingestion layer is source-agnostic (shared schema), so Telegram or StockTwits can be
  added as extra `origin`s without touching the model code.

## 3. Timing contract (no look-ahead)

Hourly UTC bars labelled by open time. Decision for row *n* is taken at the open of bar *n* using posts created
before it and closes up to bar *n−1*. Target `y_h = log(close[n+h] / close[n−1])`, so **n+0 is the close of the
current hour (1 h ahead)** and **n+5 is 6 h ahead** — n+0 is a real forecast, not a copy of a known price.
`tests/test_no_lookahead.py` shocks all data from bar *n* onward and asserts features up to row *n* are unchanged.

## 4. Data quality decisions

1. **Provenance audit** — the "Bitcoin Tweets 2025–2026" Kaggle set is the 2021–23 corpus shifted by exactly 1,826 days
   (10,593/10,774 sampled rows identical; quoted BTC prices median $45.4k vs real $68.2k in the stamped month and
   $47.2k five years earlier). Rejected. Script: `scripts/audit_provenance.py`.
2. **Malformed CSV** — unbalanced quotes crash pandas' C parser; streamed with `csv` and kept only 13-field rows
   (4,487 of 4.69M dropped).
3. **Collection gaps** — 220 well-covered days over 24 months; longest unbroken run 15 days. Sentiment features are
   computed over covered hours only; volume is measured relative to a trailing covered-hour baseline so collection
   effort is not mistaken for attention. A row is modelled only if the previous hour has ≥ 50 posts.
4. **Bots** — 47% of posts flagged as automation-like (non-human client, templated text, account < 30 days,
   hashtag stuffing, price-ticker templates).
5. **Live sampling design under a US$5 budget** — the brief's reference actor (`apidojo/tweet-scraper`) is capped at
   10 items/run and 5 runs/month on Apify's free plan (verified), so the pipeline uses
   `kaitoeasyapi/...-cheapest` (US$0.25/1k on the free tier). Each call samples the newest 20 posts of one 1-hour
   window; the posting rate is recovered as (K−1)/span. Plan: every hour of the last 7 days + one rotating-hour
   sample per day for the other 173 days. A single 180-day "Top" query would over-sample viral hours and turn post
   volume into a popularity artefact.
6. **Two live-data pitfalls found and fixed.** (a) Operator precedence: `bitcoin OR $BTC since_time:… until_time:…`
   binds the time filter only to the last OR term; the first pass therefore returned un-windowed posts. Fixed by
   parenthesising the OR group and verified on a recent and a 4-month-old window. The first-pass posts are kept
   only for the daily series (correct timestamps, not hour-targeted). (b) X's search index thins with age: the same
   query yields ~1,500 posts/h today but ~40/h for a July window. Sampled rates are therefore compared only within
   the recent dense window; older samples feed sentiment only.

## 5. Model

One LightGBM regressor per horizon on **volatility-normalised** targets `z_h = y_h / (σ_72h·√(h+1))`
(comparable across the 2021 bull and 2022 bear regimes), Huber loss, strong regularisation
(15 leaves, ≥200 samples/leaf, L2 = 5), winsorised targets. Price forecast `P̂ = P_{n−1}·exp(ẑ·σ·√(h+1))`.
**Split-conformal** 80% intervals from the last 20% of each training window. Why not LSTM/Transformer: ~3.5k
covered samples; deep sequence models need far more data to beat a random walk OOS.

**Why not reinforcement learning for the policy?** With ~3.5k non-contiguous hourly samples and a non-stationary
reward, an RL agent would mostly learn the training path (high variance, no replay of genuinely new states). A
forecast → volatility-targeted position rule is the transparent, lower-variance choice; it is a drop-in seam
(`backtest/strategy.positions`) where an RL policy can be swapped once months of continuous live X data accumulate.

## 6. Validation

Purged expanding walk-forward over eligible hours: first fit on 1,200 hours, refit every 250 hours, 6-hour purge
before each test block. Hyper-parameters fixed a priori (config.py); nothing tuned on test folds. Benchmarks on the
**same test hours**: random walk, price-only, sentiment-only, full (human-filtered), full without bot filter, linear
Ridge. Statistics: OOS R² vs random walk, directional accuracy, Spearman IC, conformal coverage,
Diebold–Mariano (HAC, HLN-corrected), Sharpe/Sortino/MDD/Calmar, block-bootstrap Sharpe CI, PSR and DSR across all
model × signal variants.
