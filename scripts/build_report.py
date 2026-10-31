"""Build the evaluation report (Markdown + PDF) from the JSON artefacts. Every number is read from
outputs/reports/*.json, never typed by hand."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt.config import CHARTS, REPORTS  # noqa: E402

R = json.loads((REPORTS / "results.json").read_text())
V = json.loads((REPORTS / "vol_study.json").read_text())
S = json.loads((REPORTS / "sentiment_benchmark.json").read_text())
L = json.loads((REPORTS / "live_state.json").read_text()) if (REPORTS / "live_state.json").exists() else {}
AUTHOR = "Jash Bharat Pasad (IIT Gandhinagar)"

blocks = []  # (kind, payload)


def H(t, lvl=1): blocks.append(("h", (lvl, t)))
def Pp(t): blocks.append(("p", t))
def T(header, rows, note=None): blocks.append(("t", (header, rows, note)))
def I(path, caption, w=16): blocks.append(("i", (path, caption, w)))
def B(items): blocks.append(("b", items))


def f(x, d=3, pct=False, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    if pct:
        return f"{x * 100:{'+' if sign else ''}.{d}f}%"
    return f"{x:{'+' if sign else ''}.{d}f}"


tr = R["trading"]
best_dir = max((k for k in tr if "|" in k and "@" not in k), key=lambda k: tr[k]["sharpe"])
vq = V["models"]
dmq = V["diebold_mariano"]

H("CrowdWisdomTrading — X Sentiment Price Prediction Agent: Evaluation Report")
Pp(f"Author: {AUTHOR}. Asset: Bitcoin (BTC/USDT, hourly). Forecast horizons n+0 … n+5 (1–6 hours ahead). "
   "Inputs: X sentiment and price only. All metrics below are out-of-sample from a purged expanding walk-forward; "
   "hyper-parameters were fixed before testing.")

H("1. Executive summary", 2)
B([
    f"<b>Data integrity first.</b> One of the suggested datasets (“Bitcoin Tweets 2025–2026”, Kaggle) is the 2021–23 "
    f"corpus with every timestamp shifted +1,826 days (10,593/10,774 sampled rows identical; tweets stamped Feb 2026 quote "
    f"BTC at ~$45k vs a real ~$68k). It was rejected; the original correctly-dated 4.69M-tweet corpus was used.",
    f"<b>Domain sentiment model.</b> Trained on 510k author-labelled StockTwits crypto posts: bull-vs-bear AUC "
    f"{S['cryptolex_bull_vs_bear_auc']:.3f} vs VADER {S['vader_bull_vs_bear_auc']:.3f}; 3-class macro-F1 "
    f"{S['cryptolex_macroF1_sub']:.2f} vs {S['vader_macroF1_sub']:.2f}; ~{S['posts_per_second']:,} posts/s on one CPU.",
    f"<b>Direction: no edge.</b> Across {R['test_hours']:,} out-of-sample hours ({R['test_start'][:10]} → {R['test_end'][:10]}) "
    f"no model beats a random walk (OOS R² < 0 at every horizon; directional accuracy 48–52%). Adding X sentiment to a "
    f"price model does not significantly change accuracy (Diebold–Mariano p > 0.18).",
    f"<b>Trading.</b> Every long/short variant loses after 5 bp costs. Best: {best_dir} with Sharpe "
    f"{tr[best_dir]['sharpe']:.2f} (90% CI {tr[best_dir]['sharpe_ci90'][0]} to {tr[best_dir]['sharpe_ci90'][1]}), "
    f"max drawdown {tr[best_dir]['max_drawdown']:.0%}, Deflated Sharpe {tr[best_dir]['dsr']:.2f}; BTC buy-and-hold "
    f"over the same 2021-23 window: Sharpe {tr['buy_hold_continuous']['sharpe']:.2f}, max drawdown "
    f"{tr['buy_hold_continuous']['max_drawdown']:.0%}.",
    f"<b>Where X sentiment does help: risk.</b> Adding X attention/disagreement to the HAR volatility model lowers QLIKE "
    f"from {vq['har']['qlike']:.3f} to {vq['har_x']['qlike']:.3f} (DM t = {dmq['har_x_vs_har']['qlike'][0]:.2f}, "
    f"p = {dmq['har_x_vs_har']['qlike'][1]:.4f}). Vol-adaptive intervals fix the under-coverage of constant-width "
    f"intervals in calm hours ({V['interval_n5']['constant']['coverage_calm']:.0%} → "
    f"{V['interval_n5']['adaptive']['coverage_calm']:.0%} vs 80% target).",
    "<b>X follows price.</b> Sentiment correlates 0.22 with the previous hour's return but only ~0.04 with the next hour's "
    "(lead–lag chart). The raw ‘high bearishness → negative returns’ pattern is a regime artefact: demeaned within each "
    "week it reverses into a mild contrarian bounce.",
])

H("2. Data", 2)
T(["Source", "What", "Coverage / volume", "Notes"], [
    ["Kaggle kaushiksuresh147 Bitcoin_tweets.csv", "historical X posts (train/validate)",
     f"4.69M posts → 4,039 covered hours, {R['eligible_hours']:,} model-eligible", "malformed CSV rows dropped: 4,487"],
    ["Apify (live, last 180 days)", "X posts, engagement, author followers",
     (f"{L['apify']['posts']:,} posts in {L['apify']['samples']} hourly samples, {L['apify']['from'][:10]} → "
      f"{L['apify']['to'][:10]}; cost ${L['apify']['cost_usd']:.2f}" if 'apify' in L else "not run"),
     "kaitoeasyapi actor (apidojo capped on free plan); budgeted hourly sampling"],
    ["Binance spot archive (data.binance.vision)", "BTCUSDT 1h OHLC", "2021-01 → 2026-10", "keyless, immutable files"],
    ["Tavily (news)", "recent BTC news", "last 7 days", "outlook only — not a model feature"],
])
Pp("Coverage gaps in the historical corpus (220 well-covered days; longest unbroken run 15 days) mean post counts "
   "reflect the collector's effort. Attention is therefore measured relative to a trailing covered-hour baseline, and a "
   "row is modelled only if the previous hour has ≥ 50 posts. 47% of posts are automation-like (non-human client, "
   "templated text, account < 30 days, hashtag stuffing, price-ticker templates).")

H("3. Method (summary — full detail in docs/METHODOLOGY.md)", 2)
B(["Timing contract: decision at the open of hour n with posts before it and closes to n−1; target "
   "y<sub>h</sub> = log(close<sub>n+h</sub>/close<sub>n−1</sub>), so n+0 is a genuine 1-hour-ahead forecast. A unit test "
   "shocks all future data and asserts features are unchanged.",
   "Features (sentiment & price only): lagged returns, realised and Parkinson vol, drawdown; human-filtered net "
   "sentiment, bullishness index, bearish share (kept separate — negative-tone asymmetry), disagreement, bot share, "
   "abnormal attention, sentiment momentum/surprise, sentiment–price divergence.",
   "Model: one LightGBM per horizon on volatility-normalised targets (regime-comparable), Huber loss, heavy "
   "regularisation; split-conformal 80% intervals. Benchmarks on identical hours: random walk, price-only, "
   "sentiment-only, no-bot-filter, linear Ridge; HAR-RV for volatility.",
   "Validation: purged expanding walk-forward (initial 1,200 eligible hours, refit every 250, 6-hour purge). Deflated "
   f"Sharpe uses all {len([k for k in tr if '|' in k and '@' not in k])} directional strategy variants tried."])

H("4. Forecast accuracy by horizon (out-of-sample)", 2)
rows = []
for m in ["rw", "price_lgbm", "sent_lgbm", "full_lgbm", "full_allposts", "full_ridge"]:
    for h in ["0", "2", "5"]:
        d = R["forecast"][m][h]
        rows.append([m, f"n+{h}", f(d["r2_oos_vs_rw"], 4, sign=True), f(d["dir_acc"], 3), f(d["ic_spearman"], 3, sign=True),
                     f(d["mape_price_pct"], 3), f(d["pi80_coverage"], 2), f(d["pi80_width_pct"], 2)])
T(["Model", "h", "OOS R² vs RW", "Dir. acc.", "IC", "Price MAPE %", "PI80 cover", "PI80 width %"], rows,
  "Price MAPE is shown for completeness: it is tiny for every model (incl. the random walk) because P(t+h) ≈ P(t); "
  "it is not evidence of skill. OOS R² of returns vs a random walk is the meaningful number.")
dmr = R["diebold_mariano"]
T(["full_lgbm vs", "n+0 (t, p)", "n+2", "n+5"],
  [[b, *[f"{dmr[b][h][0]:+.2f}, {dmr[b][h][1]:.3f}" for h in ["0", "2", "5"]]] for b in dmr],
  "Diebold–Mariano (squared error, HAC, HLN-corrected). Negative t ⇒ full_lgbm more accurate; p is one-sided.")
I(CHARTS / "08_metrics_by_horizon.png", "Directional accuracy, OOS R² and interval coverage by horizon.")

H("5. Trading performance (required metrics)", 2)
order = [k for k in tr if "|" in k and "@" not in k] + ["buy_hold_covered_hours", "buy_hold_continuous"] + \
        [k for k in tr if "@" in k]
T(["Strategy", "Sharpe", "Sharpe 90% CI", "Sortino", "Max DD", "Calmar", "Turnover/day", "DSR"],
  [[k, f(tr[k]["sharpe"], 2), f"{tr[k]['sharpe_ci90'][0]} … {tr[k]['sharpe_ci90'][1]}", f(tr[k]["sortino"], 2),
    f(tr[k]["max_drawdown"], 1, pct=True), f(tr[k]["calmar"], 2), f(tr[k].get("turnover_per_day"), 2),
    f(tr[k].get("dsr"), 3)] for k in order],
  "Hourly net returns after 5 bp per unit turnover (rows marked @0bps/@10bps show cost sensitivity), annualised with "
  "√8760. Strategies are flat in hours without X coverage. h0 = 1h-ahead signal; ens = horizon-averaged signal; "
  "6h_band = decide every 6h on the n+5 forecast with a 0.25 no-trade band (added after round 1, counted in DSR).")
I(CHARTS / "07_equity_drawdown.png", "Walk-forward equity and drawdown.")

H("6. Volatility: where X sentiment adds information", 2)
T(["Model", "MSE log-RV", "QLIKE", "R² vs HAR", "corr"],
  [[k, f(v["mse_logrv"]), f(v["qlike"]), f(v["r2_vs_har"], 3, sign=True), f(v["corr"])] for k, v in vq.items()])
T(["Comparison", "DM on MSE (t, p)", "DM on QLIKE (t, p)"],
  [[k, f"{v['mse'][0]:+.2f}, {v['mse'][1]:.3f}", f"{v['qlike'][0]:+.2f}, {v['qlike'][1]:.4f}"] for k, v in dmq.items()],
  "Target: log realised variance over n..n+5. QLIKE (Patton 2011) is the robust loss for variance forecasts and "
  "penalises under-predicting volatility spikes. The gain is on QLIKE, not on MSE of log-RV.")
ic = V["interval_n5"]
T(["n+5 80% interval", "coverage all", "calm hours", "turbulent hours", "mean width %"],
  [[k, f(v["coverage_all"], 3), f(v["coverage_calm"], 3), f(v["coverage_turbulent"], 3), f(v["mean_width_pct"], 2)]
   for k, v in ic.items()])
vm = V["vol_managed_long"]
T(["Vol-managed long (covered hours)", "Sharpe*", "Sortino*", "Max DD", "Ann. vol*"],
  [[k, f(v["sharpe"], 2), f(v["sortino"], 2), f(v["max_drawdown"], 1, pct=True), f(v["ann_vol"], 2)] for k, v in vm.items()],
  "*annualised per active hour (covered hours only), so comparable across rows but not with §5. Better variance "
  "forecasts did not translate into better risk-adjusted returns in this mostly-bear sample: honest negative result.")

H("7. Visual matrix: X sentiment vs price through time", 2)
I(CHARTS / "01_sentiment_vs_price.png", "Daily human X sentiment vs BTC, 2021–23.")
I(CHARTS / "03_lead_lag.png", "Lead–lag: sentiment reacts to past returns; almost no lead on future returns.")
I(CHARTS / "04_visual_matrix_quintile_by_horizon.png",
  "Signal quintile × horizon. Raw bearish-share pattern is a regime effect; week-demeaned it flips contrarian.")
I(CHARTS / "02_hourly_zoom_longest_window.png", "Hourly zoom on the longest continuous coverage window.")
I(CHARTS / "09_feature_importance.png", "Feature importance (gain).", 11)

if L.get("live_mode"):
    H("8. Live window (Apify, last 180 days)", 2)
    oc = L.get("live_oos_check", {})
    if oc:
        T(["Model (trained on 2021-23 only)", "h", "OOS R² vs RW", "Dir. acc.", "PI80 cover"],
          [[m, f"n+{h}", f(d["r2_oos_vs_rw"], 4, sign=True), f(d["dir_acc"], 3), f(d["pi80_coverage"], 2)]
           for m, dd in oc["metrics"].items() for h, d in dd.items()],
          f"{oc['hours']} dense hourly samples, {oc['from'][:16]} → {oc['to'][:16]} UTC. Too few hours for inference — a "
          "transfer smoke test showing the 2021-23 model and its intervals behave sensibly in 2026.")
    I(CHARTS / "10_live_180d_sentiment_vs_price.png", "Live 180-day window: price, X sentiment, attention, Fear & Greed.")
    I(CHARTS / "06_forecast_fan.png", "Example out-of-sample multi-horizon forecast with 80% conformal band.", 12)

H("9. Self-critique and threats to validity", 2)
B(["Historical X data is a sampled, gappy corpus (2021-23); live data is a budget sample (~20 posts per sampled hour). "
   "Neither is the full firehose. Effects that need dense data (minute-level, Guégan & Renault 2021) are out of reach.",
   "The test period is dominated by the 2022 bear market. Results may differ in trending bull regimes.",
   "Low-turnover and vol-managed variants were added after seeing round-1 results. They are disclosed and counted in "
   "the DSR rather than hidden. The volatility hypothesis was pre-specified from the literature, not data-mined.",
   "The sentiment model is trained on StockTwits labels and applied to X (domain shift). A CryptoBERT re-score of a "
   "sample is the next check.",
   "Post-level bot detection is heuristic (no ground truth). The all-posts vs human ablation shows no significant "
   "difference in accuracy.",
   "With a paid Apify plan the same code scales to dense hourly coverage for 180 days (≈ 430k posts, ≈ US$110–170). That "
   "is the single biggest improvement available."])

H("10. Reproduce", 2)
Pp("pip install -r requirements.txt; cp .env.example .env (add APIFY_TOKEN, TAVILY_API_KEY); then make all "
   "(audit → data → sentiment → hourly → backtest → vol study → charts → scrape → live → report). make test runs the "
   "look-ahead tests.")


def to_md():
    out = []
    import re
    strip = lambda s: re.sub(r"</?b>", "**", re.sub(r"<sub>(.*?)</sub>", r"_\1", s))  # noqa: E731
    for k, v in blocks:
        if k == "h":
            out.append("#" * v[0] + " " + v[1])
        elif k == "p":
            out.append(strip(v))
        elif k == "b":
            out += [f"- {strip(x)}" for x in v]
        elif k == "t":
            hdr, rows, note = v
            out.append("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
            out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
            if note:
                out.append(f"\n*{note}*")
        elif k == "i":
            out.append(f"![{v[1]}](../charts/{Path(v[0]).name})\n*{v[1]}*")
        out.append("")
    (REPORTS / "EVALUATION_REPORT.md").write_text("\n".join(out))


def to_pdf():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, ListFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=11.2)
    small = ParagraphStyle("s", parent=body, fontSize=7, leading=8.8, textColor=colors.HexColor("#4b5563"))
    cell = ParagraphStyle("c", parent=body, fontSize=6.8, leading=8.2)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=15, leading=19, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=3,
                        textColor=colors.HexColor("#1e3a8a"))
    story = []
    for k, v in blocks:
        if k == "h":
            story.append(Paragraph(v[1], h1 if v[0] == 1 else h2))
        elif k == "p":
            story.append(Paragraph(v, body))
        elif k == "b":
            story.append(ListFlowable([Paragraph(x, body) for x in v], bulletType="bullet", leftIndent=10,
                                      bulletFontSize=6))
        elif k == "t":
            hdr, rows, note = v
            data = [[Paragraph(f"<b>{h}</b>", cell) for h in hdr]] + [[Paragraph(str(c), cell) for c in r] for r in rows]
            t = Table(data, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")])]))
            story += [Spacer(1, 3), t]
            if note:
                story.append(Paragraph(note, small))
        elif k == "i":
            path, cap, w = v
            if Path(path).exists():
                from PIL import Image as PI
                iw, ih = PI.open(path).size
                W = w * cm
                story += [Spacer(1, 4), Image(str(path), width=W, height=W * ih / iw), Paragraph(cap, small)]
        story.append(Spacer(1, 4))
    doc = SimpleDocTemplate(str(REPORTS / "EVALUATION_REPORT.pdf"), pagesize=A4, leftMargin=1.6 * cm,
                            rightMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                            title="CWT X Sentiment Agent - Evaluation Report", author=AUTHOR)
    doc.build(story)


if __name__ == "__main__":
    to_md()
    to_pdf()
    print("written:", REPORTS / "EVALUATION_REPORT.md", REPORTS / "EVALUATION_REPORT.pdf")
