"""Post-level -> hourly bar aggregation (shared by historical backtest and live Apify mode).

Every hourly row summarises posts with created_utc in [t, t+1h). Sums are stored (not means) so that
hourly rows from different files/runs can be re-aggregated exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nlp.bots import bot_flags, influence_weight


def score_posts(posts: pd.DataFrame, model) -> pd.DataFrame:
    posts = posts.drop_duplicates(subset=["text", "author"]).copy()
    posts = posts[~posts["is_repost"].astype(bool)]  # reposts duplicate text; engagement counts reach
    s = model.score(posts["text"].tolist())
    posts[["p_bear", "p_bull", "sent"]] = s.values
    posts["bot"] = bot_flags(posts).values
    posts["w"] = influence_weight(posts)
    posts["hour"] = posts["created_utc"].dt.floor("h")
    return posts


def hourly_sums(p: pd.DataFrame) -> pd.DataFrame:
    h = p["human"] = ~p["bot"]
    p["sent2"] = p["sent"] ** 2
    p["ws"] = p["w"] * p["sent"]
    for c in ["sent", "sent2", "p_bull", "p_bear", "ws", "w"]:
        p[f"h_{c}"] = p[c] * h
    p["strong_bull"] = (p["p_bull"] > 0.6).astype(float)
    p["strong_bear"] = (p["p_bear"] > 0.6).astype(float)
    p["h_strong_bull"] = p["strong_bull"] * h
    p["h_strong_bear"] = p["strong_bear"] * h
    agg = p.groupby("hour").agg(
        n=("sent", "size"), n_h=("human", "sum"),
        s_sent=("sent", "sum"), s_sent2=("sent2", "sum"), s_bull=("p_bull", "sum"), s_bear=("p_bear", "sum"),
        s_ws=("ws", "sum"), s_w=("w", "sum"),
        sh_sent=("h_sent", "sum"), sh_sent2=("h_sent2", "sum"), sh_bull=("h_p_bull", "sum"),
        sh_bear=("h_p_bear", "sum"), sh_ws=("h_ws", "sum"), sh_w=("h_w", "sum"),
        n_sbull=("strong_bull", "sum"), n_sbear=("strong_bear", "sum"),
        nh_sbull=("h_strong_bull", "sum"), nh_sbear=("h_strong_bear", "sum"),
        authors=("author", "nunique"))
    return agg


def combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts)
    num = df.drop(columns=["authors"]).groupby(level=0).sum()
    num["authors"] = df["authors"].groupby(level=0).sum()  # approx across chunk borders
    return num.sort_index()


def finalize(s: pd.DataFrame) -> pd.DataFrame:
    """Turn sums into interpretable hourly sentiment measures."""
    out = pd.DataFrame(index=s.index)
    out["n_posts"] = s["n"]
    out["n_human"] = s["n_h"]
    out["bot_share"] = 1 - s["n_h"] / s["n"]
    out["sent_mean"] = s["s_sent"] / s["n"]
    out["sent_std"] = np.sqrt((s["s_sent2"] / s["n"] - out["sent_mean"] ** 2).clip(0))  # disagreement
    out["bull_mean"] = s["s_bull"] / s["n"]
    out["bear_mean"] = s["s_bear"] / s["n"]
    out["sent_w"] = s["s_ws"] / s["s_w"]
    nh = s["n_h"].replace(0, np.nan)
    out["h_sent_mean"] = s["sh_sent"] / nh
    out["h_sent_std"] = np.sqrt((s["sh_sent2"] / nh - out["h_sent_mean"] ** 2).clip(0))
    out["h_bull_mean"] = s["sh_bull"] / nh
    out["h_bear_mean"] = s["sh_bear"] / nh
    out["h_sent_w"] = s["sh_ws"] / s["sh_w"].replace(0, np.nan)
    # Antweiler-Frank (2004) bullishness index on confident posts, human-only
    out["h_bullishness"] = np.log((1 + s["nh_sbull"]) / (1 + s["nh_sbear"]))
    out["bullishness"] = np.log((1 + s["n_sbull"]) / (1 + s["n_sbear"]))
    out["authors"] = s["authors"]
    return out
