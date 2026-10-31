"""X (Twitter) ingestion via Apify, budget-aware.

Actors
* DEFAULT  `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest` (id CJdippxWmn9uRfooo):
  US$0.25 / 1k tweets on Apify's FREE tier and callable via API on the free plan. Each call has a
  minimum charge that the actor pads with "mock" items (filtered out below).
* `apidojo/tweet-scraper` (Tweet Scraper V2, id 61RPP7dywgiy0JPD0): the brief's reference actor, but on
  the free plan it is capped at 10 items/run and 5 runs/month (verified 31 Oct 2026), so it is only
  usable on a paid plan. Supported via `actor="apidojo"`.

Sampling design (why it is not "one big 180-day query")
X search with sort=Latest returns the newest K posts *before* the window end. We issue one call per
sample time with a 1-hour window and K=20:
  * the K posts give a sentiment sample for that hour;
  * the time span they cover gives the TRUE posting rate, rate ~= (K-1)/span (estimate_rate) - a real
    attention measure from a tiny, cheap sample. (29 Oct 14:00-15:00 UTC: the newest 25 "bitcoin"
    posts spanned ~1 minute, i.e. ~1,500 posts/hour.)
Plan under the US$5 free-plan budget:
  * DENSE: every hour of the last `dense_days` days  -> hourly features + a live out-of-sample check
  * SPARSE: `per_day` samples/day over 180 days at rotating hours (no fixed time-of-day bias)
A cost guard reads the account's live usage and stops before `budget_usd`.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from ..config import APIFY_TOKEN, RAW

KAITO = "kaitoeasyapi~twitter-x-data-tweet-scraper-pay-per-result-cheapest"
APIDOJO = "apidojo~tweet-scraper"
QUERY = "(bitcoin OR $BTC OR #BTC OR #Bitcoin)"   # parenthesised: X applies OR before time operators
API = "https://api.apify.com/v2"

SCHEMA = ["post_id", "created_utc", "text", "author", "author_followers", "author_following",
          "author_created", "author_verified", "likes", "reposts", "replies", "quotes", "views",
          "lang", "source", "is_repost", "query", "origin", "sample_end"]


def _tok():
    if not APIFY_TOKEN:
        raise RuntimeError("APIFY_TOKEN missing. Put it in .env (see .env.example).")
    return APIFY_TOKEN


def usage_usd() -> float:
    js = requests.get(f"{API}/users/me/limits", params={"token": _tok()}, timeout=30).json()
    return float(js["data"]["current"]["monthlyUsageUsd"])


def build_input(end: datetime, k: int = 20, actor: str = "kaito", query: str = QUERY) -> dict:
    t1 = int(end.timestamp())
    t0 = t1 - 3600
    if actor == "kaito":
        return {"twitterContent": query, "lang": "en", "since_time": str(t0), "until_time": str(t1),
                "maxItems": k, "queryType": "Latest"}
    return {"searchTerms": [f"{query} since_time:{t0} until_time:{t1} lang:en -filter:replies"],
            "sort": "Latest", "maxItems": k}


def normalize_item(it: dict, sample_end) -> dict | None:
    if not it.get("createdAt") or str(it.get("text", "")).startswith("From KaitoEasyAPI"):
        return None   # mock / padding item
    a = it.get("author") or {}
    return {
        "post_id": str(it.get("id")),
        "created_utc": pd.to_datetime(it.get("createdAt"), utc=True, errors="coerce", format="mixed"),
        "text": it.get("fullText") or it.get("text") or "",
        "author": a.get("userName"), "author_followers": a.get("followers"),
        "author_following": a.get("following"),
        "author_created": pd.to_datetime(a.get("createdAt"), utc=True, errors="coerce", format="mixed"),
        "author_verified": bool(a.get("isBlueVerified") or a.get("isVerified")),
        "likes": it.get("likeCount"), "reposts": it.get("retweetCount"), "replies": it.get("replyCount"),
        "quotes": it.get("quoteCount"), "views": it.get("viewCount"), "lang": it.get("lang"),
        "source": it.get("source"), "is_repost": bool(it.get("retweeted_tweet") or it.get("isRetweet")),
        "query": QUERY, "origin": "apify", "sample_end": sample_end,
    }


def estimate_rate(times: pd.Series, k_requested: int, window_end) -> float:
    """Posts/hour implied by a capped Latest sample ending at window_end."""
    n = len(times)
    if n < 2 or n < k_requested * 0.9:      # window exhausted -> we saw everything
        return float(n)
    span_h = max((pd.Timestamp(window_end) - times.min()).total_seconds() / 3600, 1 / 3600)
    return (n - 1) / span_h


def sample_plan(now: datetime, days: int = 180, dense_days: int = 10, per_day: int = 2) -> list:
    now = now.replace(minute=0, second=0, microsecond=0)
    pts = {now - timedelta(hours=h) for h in range(0, dense_days * 24)}
    for d in range(dense_days, days):
        base = (now - timedelta(days=d)).replace(hour=0)
        for j in range(per_day):
            hour = (d * 7 + j * (24 // per_day)) % 24          # rotating hours
            pts.add(base + timedelta(hours=hour + 1))
    return sorted(p for p in pts if p <= now)


def _call(end: datetime, k: int, actor: str) -> list:
    cache = RAW / "apify"
    cache.mkdir(parents=True, exist_ok=True)
    fp = cache / f"{actor}_{end:%Y%m%dT%H}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    act = KAITO if actor == "kaito" else APIDOJO
    for attempt in range(3):
        try:
            r = requests.post(f"{API}/acts/{act}/run-sync-get-dataset-items",
                              params={"token": _tok(), "timeout": 240},
                              json=build_input(end, k, actor), timeout=300)
            if r.status_code in (200, 201):
                items = r.json()
                fp.write_text(json.dumps(items, default=str))
                return items
        except Exception as e:  # noqa: BLE001
            print("[apify]", end, e, flush=True)
        time.sleep(5 * (attempt + 1))
    return []


def load_cache(actor: str = "kaito", k: int = 20):
    rows, rates = [], []
    for fp in sorted((RAW / "apify").glob(f"{actor}_*.json")):
        end = pd.Timestamp(datetime.strptime(fp.stem.split("_")[1], "%Y%m%dT%H"), tz="UTC")
        items = [x for x in (normalize_item(i, end) for i in json.loads(fp.read_text())) if x]
        rows += items
        ts = pd.Series([x["created_utc"] for x in items]).dropna()
        rates.append({"sample_end": end, "n": len(items), "rate_per_h": estimate_rate(ts, k, end)})
    posts = pd.DataFrame(rows, columns=SCHEMA).drop_duplicates("post_id").sort_values("created_utc")
    return posts.reset_index(drop=True), pd.DataFrame(rates).sort_values("sample_end").reset_index(drop=True)


def scrape(days: int = 180, dense_days: int = 10, per_day: int = 2, k: int = 20, actor: str = "kaito",
           budget_usd: float = 3.8, workers: int = 5, now: datetime | None = None):
    now = now or datetime.now(timezone.utc)
    plan = sample_plan(now, days, dense_days, per_day)[::-1]   # newest first
    print(f"[apify] {len(plan)} sample points", flush=True)
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_call, end, k, actor) for end in plan]
        for _ in as_completed(futs):
            done += 1
            if done % 25 == 0:
                u = usage_usd()
                print(f"[apify] {done}/{len(plan)} calls, account usage ${u:.2f}", flush=True)
                if u > budget_usd:
                    print("[apify] budget guard hit - stopping", flush=True)
                    for f in futs:
                        f.cancel()
                    break
    return load_cache(actor, k)


def load_legacy() -> pd.DataFrame:
    """Posts from the first scrape pass (unparenthesised OR query: the hourly time filter did not bind,
    so these are valid BTC posts with correct timestamps but NOT hour-targeted samples). Used only
    for daily sentiment series, never for attention/rate or hourly model rows."""
    rows = []
    for fp in sorted((RAW / "apify_v1_legacy").glob("kaito_*.json")):
        rows += [x for x in (normalize_item(i, None) for i in json.loads(fp.read_text())) if x]
    df = pd.DataFrame(rows, columns=SCHEMA).drop_duplicates("post_id")
    df["origin"] = "apify_legacy"
    return df
