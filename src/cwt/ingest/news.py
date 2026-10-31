"""Recent, relevant news on the asset: Tavily (primary, as the brief specifies) + Exa (optional).

News is used for the *sentiment outlook* page and for an event-tagging layer; it is NOT a model
feature, because the brief restricts the forecaster to X sentiment and price only. Keeping news
out of the feature set also avoids a subtle look-ahead problem: news APIs return articles by
publish date but update/republish timestamps can move, which leaks future text into the past.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from ..config import ASSET, EXA_API_KEY, RAW, TAVILY_API_KEY


def tavily_news(queries=None, days: int = 7, max_results: int = 10) -> pd.DataFrame:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY missing. Put it in .env (see .env.example).")
    from tavily import TavilyClient
    c = TavilyClient(api_key=TAVILY_API_KEY)
    rows = []
    for q in queries or ASSET.news_queries:
        r = c.search(query=q, topic="news", days=days, search_depth="advanced",
                     max_results=max_results, include_answer=False)
        for it in r.get("results", []):
            rows.append({"query": q, "title": it.get("title"), "url": it.get("url"),
                         "content": it.get("content"), "published": it.get("published_date"),
                         "score": it.get("score"), "source": "tavily"})
    return _finish(rows, "tavily")


def exa_news(queries=None, days: int = 7, n: int = 10) -> pd.DataFrame:
    if not EXA_API_KEY:
        return pd.DataFrame()
    import requests
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for q in queries or ASSET.news_queries:
        r = requests.post("https://api.exa.ai/search", headers={"x-api-key": EXA_API_KEY},
                          json={"query": q, "numResults": n, "startPublishedDate": since,
                                "category": "news", "contents": {"text": {"maxCharacters": 1500}}},
                          timeout=60).json()
        for it in r.get("results", []):
            rows.append({"query": q, "title": it.get("title"), "url": it.get("url"),
                         "content": it.get("text"), "published": it.get("publishedDate"),
                         "score": it.get("score"), "source": "exa"})
    return _finish(rows, "exa")


def _finish(rows, tag) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("url")
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce", format="mixed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    (RAW / "news").mkdir(exist_ok=True)
    df.to_json(RAW / "news" / f"{tag}_{stamp}.json", orient="records", date_format="iso")
    return df


def score_news(df: pd.DataFrame, model) -> pd.DataFrame:
    """Score headline+snippet with the same crypto sentiment model used for posts."""
    if df.empty:
        return df
    s = model.score((df["title"].fillna("") + ". " + df["content"].fillna("")).tolist())
    return pd.concat([df.reset_index(drop=True), s], axis=1)
