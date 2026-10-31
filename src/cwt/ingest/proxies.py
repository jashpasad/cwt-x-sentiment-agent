"""Keyless live social-sentiment proxies used ONLY when no Apify token is configured, so the agent can
still produce a current outlook. They are clearly labelled as proxies in every output and are never
mixed into model training.

* StockTwits public stream for BTC.X: latest messages with author-set Bullish/Bearish tags. This is
  the same platform/label type our sentiment model was trained on, so it doubles as a live
  calibration check of the classifier (model score vs author tag).
* Alternative.me Crypto Fear & Greed Index (daily composite incl. social/search components).
"""
from __future__ import annotations

import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (research; CWT sentiment agent)"}


def stocktwits(symbol: str = "BTC.X", pages: int = 10) -> pd.DataFrame:
    rows, max_id = [], None
    for _ in range(pages):
        params = {"max": max_id} if max_id else {}
        r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json",
                         params=params, headers=UA, timeout=30)
        if r.status_code != 200:
            break
        js = r.json()
        for m in js.get("messages", []):
            ent = (m.get("entities") or {}).get("sentiment") or {}
            u = m.get("user") or {}
            rows.append({"id": m["id"], "created_utc": pd.to_datetime(m["created_at"], utc=True),
                         "text": m.get("body", ""), "tag": ent.get("basic"),
                         "author": u.get("username"), "author_followers": u.get("followers")})
        max_id = js.get("cursor", {}).get("max")
        if not js.get("cursor", {}).get("more"):
            break
        time.sleep(0.6)
    return pd.DataFrame(rows).drop_duplicates("id")


def fear_greed(days: int = 60) -> pd.DataFrame:
    js = requests.get(f"https://api.alternative.me/fng/?limit={days}", timeout=30).json()
    df = pd.DataFrame(js["data"])
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["value"] = df["value"].astype(int)
    return df[["date", "value", "value_classification"]].sort_values("date")
