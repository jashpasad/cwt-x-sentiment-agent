"""One-page current sentiment outlook (Markdown), generated from live data.

Structure: verdict line -> price state -> social sentiment state -> news drivers -> model forecast
table (n+0..n+5 with 80% conformal ranges) -> what would change the view -> data provenance.
The verdict is rule-based and auditable (no LLM in the loop), so the same inputs always produce the
same page.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_state(px: pd.DataFrame) -> dict:
    c = px["close"]
    last = c.iloc[-1]
    r = np.log(c).diff()
    ath180 = c.iloc[-180 * 24:].max()

    def ret(h):
        return float(last / c.iloc[-1 - h] - 1)
    return {"last": float(last), "time": c.index[-1], "r24h": ret(24), "r7d": ret(24 * 7),
            "r30d": ret(24 * 30), "r180d": ret(min(len(c) - 1, 24 * 180)),
            "dd_from_180d_high": float(last / ath180 - 1), "high_180d": float(ath180),
            "low_180d": float(c.iloc[-180 * 24:].min()),
            "vol24h_ann": float(r.iloc[-24:].std() * np.sqrt(8760)),
            "vol30d_ann": float(r.iloc[-720:].std() * np.sqrt(8760)),
            "above_30d_mean": bool(last > c.iloc[-720:].mean())}


def verdict(ps: dict, soc: dict) -> str:
    score = 0
    score += 1 if ps["r7d"] > 0 else -1
    score += 1 if ps["above_30d_mean"] else -1
    score += 1 if soc.get("net_sent", 0) > soc.get("net_sent_base", 0) else -1
    score += -1 if soc.get("fng", 50) < 35 else (1 if soc.get("fng", 50) > 60 else 0)
    if score <= -2:
        return "CAUTIOUS / BEARISH-LEANING"
    if score >= 2:
        return "CONSTRUCTIVE / BULLISH-LEANING"
    return "NEUTRAL / RANGE-BOUND"


def render(ps, soc, news, fc: pd.DataFrame, provenance: str, analyst_notes: str = "") -> str:
    v = verdict(ps, soc)
    rows = "\n".join(
        f"| n+{h} | {t:%d %b %H:%M} | ${r.phat:,.0f} | ${r.plo:,.0f} – ${r.phi:,.0f} | {r.ret*1e4:+.0f} bp |"
        for h, (t, r) in enumerate(fc.iterrows()))
    nl = "\n".join(f"- **{n['date']}** — {n['title']} ({n['source']})" for n in news[:6])
    return f"""# BTC Sentiment Outlook — {ps['time']:%d %b %Y %H:%M} UTC

**Verdict: {v}.** {soc.get('headline', '')}

**Price state.** BTC ${ps['last']:,.0f}: {ps['r24h']:+.1%} 24h, {ps['r7d']:+.1%} 7d, {ps['r30d']:+.1%} 30d, \
{ps['r180d']:+.1%} over 180d. {ps['dd_from_180d_high']:.1%} below the 180-day high (${ps['high_180d']:,.0f}); \
180-day low ${ps['low_180d']:,.0f}. Realised vol {ps['vol24h_ann']:.0%} (24h) vs {ps['vol30d_ann']:.0%} (30d), annualised.

**Social sentiment.** {soc.get('text', '')}

**News drivers (last 7 days).**
{nl}

**Model forecast** (issued at the top of the next hour; 80% split-conformal ranges; n+0 = end of current hour):

| Horizon | Time (UTC) | Point | 80% range | Expected move |
|---|---|---|---|---|
{rows}

{analyst_notes}

<sub>{provenance}</sub>
"""
