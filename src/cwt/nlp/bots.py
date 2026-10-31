"""Automation-likeness score per post.

Motivation: in Bitcoin X data, automation-like accounts dominate volume and dilute polarity signals
(Akbar et al., Studies in Economics and Finance, 2025: 16M BTC tweets; Kraaijeveld & De Smedt 2020
also flag bot pump-and-dump). We therefore build *two* parallel signals - all posts vs human-like
posts - and let the walk-forward decide which carries information.

Rules are transparent heuristics (no labelled bot data exists for this corpus), each cheap to compute
on millions of posts:
  1. posting client not in an allow-list of human apps (dlvr.it, IFTTT, custom price tickers ...)
  2. templated text: the number-stripped text appears >= 5 times in the same UTC day
  3. account younger than 30 days at posting time
  4. hashtag stuffing: >= 6 hashtags
  5. price-ticker template ("price update", "1 BTC =", "24H Low")
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

HUMAN_CLIENTS = {
    "twitter for iphone", "twitter for android", "twitter web app", "twitter for ipad",
    "twitter web client", "tweetdeck", "tweetbot for ios", "tweetbot for mac", "twitter for mac",
    "tweetdeck web app", "x for iphone", "x for android", "x web app", "", "nan", "none",
}
TICKER = re.compile(r"price update|1\s*₿\s*=|1 ?btc ?=|24h (?:low|high)|#btc price|current price", re.I)
STRIP = re.compile(r"https?://\S+|[\d,\.]+|@\w+|\s+")


def bot_flags(df: pd.DataFrame) -> pd.Series:
    src = df["source"].astype(str).str.strip().str.lower()
    client = ~src.isin(HUMAN_CLIENTS)
    if "origin" in df and (df["origin"] == "apify").all():
        client = pd.Series(False, index=df.index)  # Apify output rarely carries a client name
    key = df["text"].astype(str).str.lower().str.replace(STRIP, "", regex=True).str[:80]
    day = df["created_utc"].dt.floor("D")
    templ = key.groupby([day, key]).transform("size") >= 5
    age_days = (df["created_utc"] - df["author_created"]).dt.days
    young = age_days.lt(30).fillna(False)
    stuffing = df["text"].astype(str).str.count("#") >= 6
    ticker = df["text"].astype(str).str.contains(TICKER)
    return (client | templ | young | stuffing | ticker).astype(bool)


def influence_weight(df: pd.DataFrame) -> np.ndarray:
    """Reach weight: log followers, plus log engagement when available (Apify path).
    Capped so a single mega-account cannot dominate an hour."""
    f = np.log1p(pd.to_numeric(df["author_followers"], errors="coerce").fillna(0).clip(0, 5e7))
    eng = (pd.to_numeric(df.get("likes"), errors="coerce").fillna(0)
           + 2 * pd.to_numeric(df.get("reposts"), errors="coerce").fillna(0))
    w = 1.0 + f + np.log1p(eng)
    return np.minimum(w.values, np.nanpercentile(w.values, 99) if len(w) else 1)
