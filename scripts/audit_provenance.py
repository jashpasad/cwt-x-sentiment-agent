"""Data-provenance audit: is the "Bitcoin Tweets Dataset (2025-2026)" really 2025-26 data?

Two independent tests, using only HTTP range requests (no full download):
 1. Row identity: compare the first rows of the "2025-26" file with the original 2021-23 corpus.
 2. In-text price check: tweets quoting a "$BTC price" are compared with the actual BTC price on the
    stamped date vs the date shifted back 5 years.
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd
import requests

NEW = ("https://www.kaggle.com/api/v1/datasets/download/pokeash/bitcoin-tweets-dataset-20252026/"
       "bitcoin_tweets_latest.csv")
ORIG = "https://www.kaggle.com/api/v1/datasets/download/kaushiksuresh147/bitcoin-tweets/Bitcoin_tweets.csv"


def head_new(nbytes=4_000_000) -> pd.DataFrame:
    b = requests.get(NEW, headers={"Range": f"bytes=0-{nbytes}"}, timeout=120).content
    txt = b.decode("utf-8", "replace").rsplit("\n", 1)[0]
    return pd.read_csv(io.StringIO(txt), engine="python", on_bad_lines="skip")


def head_orig(n=20000) -> pd.DataFrame:
    # the original is served as a zip; deflate is a stream, so a prefix decodes the first rows
    b = requests.get(ORIG, headers={"Range": "bytes=0-6000000"}, timeout=120).content
    import zlib
    off = 30 + int.from_bytes(b[26:28], "little") + int.from_bytes(b[28:30], "little")
    raw = zlib.decompressobj(-15).decompress(b[off:])
    txt = raw.decode("utf-8", "replace").rsplit("\n", 1)[0]
    return pd.read_csv(io.StringIO(txt), engine="python", on_bad_lines="skip").head(n)


def main():
    new, orig = head_new(), head_orig()
    m = new.merge(orig, on=["user_name", "text"], suffixes=("_new", "_orig"))
    m["shift_days"] = (pd.to_datetime(m.date_new, errors="coerce") -
                       pd.to_datetime(m.date_orig, errors="coerce")).dt.days
    print(f"rows compared: {len(new):,}; identical (user,text) pairs: {len(m):,}")
    print("timestamp shift (days) distribution:\n", m.shift_days.value_counts().head())

    # in-text price check
    pat = re.compile(r"(?:1\s*₿\s*=\s*\$|\$BTC[^\d$]{0,20}\$?)(\d{4,6}(?:\.\d+)?)")
    q = new.assign(px=new.text.astype(str).str.extract(pat)[0].astype(float)).dropna(subset=["px"])
    q = q[(q.px > 1000) & (q.px < 200000)]
    q["stamped"] = pd.to_datetime(q.date, errors="coerce")
    print(f"tweets quoting a BTC price: {len(q)}; median quoted price ${q.px.median():,.0f}; "
          f"stamped dates {q.stamped.min()} .. {q.stamped.max()}")
    def month_median(ts):
        u = (f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/"
             f"BTCUSDT-1h-{ts:%Y-%m}.zip")
        z = zipfile.ZipFile(io.BytesIO(requests.get(u, timeout=60).content))
        return pd.read_csv(z.open(z.namelist()[0]), header=None)[4].median()
    stamped = q.stamped.median()
    real_now = month_median(stamped)
    real_shift = month_median(stamped - pd.Timedelta(days=1826))
    print(f"actual BTC median price, stamped month {stamped:%Y-%m}: ${real_now:,.0f}; "
          f"same month minus 5y: ${real_shift:,.0f}")
    verdict = abs(q.px.median() - real_shift) < abs(q.px.median() - real_now)
    print("Verdict:", "RE-DATED copy of the 2021-23 corpus - do NOT use as 2025-26 data" if verdict
          else "dates consistent with quoted prices")


if __name__ == "__main__":
    main()
