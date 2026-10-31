"""Hourly OHLC for the asset.

Primary: Binance public bulk archive (data.binance.vision) - keyless, immutable monthly/daily files,
so a reviewer re-downloads byte-identical history. Fallback for the most recent hours: Coinbase
Exchange public candles API. (api.binance.com is geo-blocked in some regions - HTTP 451 - which is
why the archive, not the REST API, is primary.)

Timestamps: Binance spot archives switched from milliseconds to microseconds in 2025; handled.
All bars are UTC, labelled by their OPEN time; close of bar t is the price at t+1h.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from ..config import ASSET, RAW

BASE = "https://data.binance.vision/data/spot"


def _read_zip(content: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(df.iloc[0, 0]).isdigit():  # some files carry a header row
        df = df.iloc[1:]
    df = df.iloc[:, :5].set_axis(["ts", "open", "high", "low", "close"], axis=1).astype(float)
    ts = df["ts"].astype("int64")
    ts = ts.where(ts < 1e14, ts // 1000)
    df["time"] = pd.to_datetime(ts, unit="ms", utc=True)
    return df.drop(columns="ts")


def binance_hourly(start: str, end: str, symbol: str | None = None) -> pd.DataFrame:
    symbol = symbol or ASSET.price_symbol
    cache = RAW / "klines"
    cache.mkdir(exist_ok=True)
    frames = []
    start_p, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    last_full_month = (pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M") - 1)
    for p in pd.period_range(start_p.to_period("M"), end_ts.to_period("M"), freq="M"):
        if p <= last_full_month:
            urls = [f"{BASE}/monthly/klines/{symbol}/1h/{symbol}-1h-{p}.zip"]
        else:
            days = pd.date_range(p.start_time, min(p.end_time, pd.Timestamp.now().normalize()), freq="D")
            urls = [f"{BASE}/daily/klines/{symbol}/1h/{symbol}-1h-{d:%Y-%m-%d}.zip" for d in days]
        for u in urls:
            fp = cache / u.rsplit("/", 1)[1]
            if not fp.exists():
                r = requests.get(u, timeout=60)
                if r.status_code != 200:
                    continue
                fp.write_bytes(r.content)
            frames.append(_read_zip(fp.read_bytes()))
    df = pd.concat(frames).drop_duplicates("time").set_index("time").sort_index()
    return df.loc[start:end]


def coinbase_recent(hours: int = 300) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").floor("h")
    out = []
    while hours > 0:
        n = min(hours, 300)
        st = end - pd.Timedelta(hours=n)
        r = requests.get(f"https://api.exchange.coinbase.com/products/{ASSET.coinbase_product}/candles",
                         params={"granularity": 3600, "start": st.isoformat(), "end": end.isoformat()},
                         timeout=30).json()
        out += r
        end, hours = st, hours - n
    df = pd.DataFrame(out, columns=["ts", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.drop_duplicates("time").set_index("time").sort_index()[["open", "high", "low", "close"]]


def hourly_prices(start: str, end: str) -> pd.DataFrame:
    df = binance_hourly(start, end)
    try:
        cb = coinbase_recent(72)
        df = pd.concat([df, cb[cb.index > df.index.max()]])
    except Exception:
        pass
    df = df[df.index < pd.Timestamp.now(tz="UTC").floor("h")]   # drop the still-forming bar
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    return df.reindex(full).ffill()   # exchange outages: carry last price (flagged by zero return)
