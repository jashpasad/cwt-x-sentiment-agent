"""Historical X corpus used to train/validate the model (the brief: "you have historical tweets
to learn / teach the model in").

Source: Kaggle `kaushiksuresh147/bitcoin-tweets` (CC0), file Bitcoin_tweets.csv, 4.69M #Bitcoin/#BTC
tweets, Feb-2021 .. Jan-2023, collected via the X API (Tweepy) in bursts.

PROVENANCE WARNING (found during this project, see scripts/audit_provenance.py):
the list's "Bitcoin Tweets Dataset (2025-2026)" (Kaggle `pokeash/...`) is the SAME corpus with
every timestamp shifted by +5 years (row-for-row identical text; tweets stamped 2026 quote BTC at
~$16.8k and ETH at ~$1.26k, i.e. late-2022 prices). Joining it to 2025-26 prices would train a
model on pure noise while looking like recent data. We therefore use the original, correctly dated
file.

The CSV has malformed rows (unbalanced quotes), which crash pandas' C parser; we stream it with the
stdlib csv module and keep only well-formed 13-field rows (4,487 of ~4.69M dropped).
"""
from __future__ import annotations

import csv
import glob
import io
import sys
import zipfile

import pandas as pd

from ..config import RAW

KAGGLE_URL = ("https://www.kaggle.com/api/v1/datasets/download/kaushiksuresh147/"
              "bitcoin-tweets/Bitcoin_tweets.csv")
KEEP = ["date", "text", "user_followers", "user_friends", "user_favourites", "user_verified",
        "user_created", "is_retweet", "source", "user_name"]


def download(dst=None):
    import requests
    dst = dst or RAW / "btc_tweets_orig.zip"
    if not dst.exists():
        with requests.get(KAGGLE_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
    return dst


def extract(zip_path=None, out_dir=None, chunk=400_000):
    csv.field_size_limit(sys.maxsize)
    zip_path = zip_path or download()
    out_dir = out_dir or RAW / "parts"
    out_dir.mkdir(exist_ok=True)
    z = zipfile.ZipFile(zip_path)
    f = io.TextIOWrapper(z.open("Bitcoin_tweets.csv"), encoding="utf-8", errors="replace", newline="")
    r = csv.reader(f)
    hdr = next(r)
    idx = {h: i for i, h in enumerate(hdr)}
    buf, part, bad = [], 0, 0
    for row in r:
        if len(row) != len(hdr):
            bad += 1
            continue
        buf.append([row[idx[k]] for k in KEEP])
        if len(buf) >= chunk:
            pd.DataFrame(buf, columns=KEEP).to_parquet(out_dir / f"p{part:03d}.parquet", index=False)
            buf, part = [], part + 1
    if buf:
        pd.DataFrame(buf, columns=KEEP).to_parquet(out_dir / f"p{part:03d}.parquet", index=False)
    return bad


def to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map the Kaggle columns onto the shared post schema used by the Apify ingester."""
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    out = pd.DataFrame({
        "created_utc": pd.to_datetime(df["date"], errors="coerce", format="mixed", utc=True),
        "text": df["text"].astype(str),
        "author": df["user_name"],
        "author_followers": num(df["user_followers"]),
        "author_following": num(df["user_friends"]),
        "author_created": pd.to_datetime(df["user_created"], errors="coerce", format="mixed", utc=True),
        "author_verified": df["user_verified"].astype(str).str.lower().eq("true"),
        "source": df["source"],
        "is_repost": df["is_retweet"].astype(str).str.lower().eq("true"),
        # engagement counts are not in this corpus; the Apify path fills them
        "likes": float("nan"), "reposts": float("nan"), "views": float("nan"),
        "origin": "kaggle_kaushiksuresh147",
    })
    return out.dropna(subset=["created_utc"])


def iter_parts(parts_dir=None):
    for fp in sorted(glob.glob(str((parts_dir or RAW / "parts") / "*.parquet"))):
        yield to_schema(pd.read_parquet(fp))
