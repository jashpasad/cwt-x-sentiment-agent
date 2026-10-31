"""Central configuration. Every tunable lives here so a reviewer can audit the whole
experiment design in one file. Nothing in this file is fitted on test data."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW, PROC, MODELS = DATA / "raw", DATA / "processed", DATA / "models"
OUT = ROOT / "outputs"
CHARTS, REPORTS = OUT / "charts", OUT / "reports"
DB_PATH = DATA / "cwt.duckdb"

for _p in (RAW, PROC, MODELS, CHARTS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@dataclass(frozen=True)
class AssetSpec:
    name: str = "Bitcoin"
    ticker: str = "BTC"
    cashtag: str = "$BTC"
    price_symbol: str = "BTCUSDT"          # Binance spot (bulk archive, no key needed)
    coinbase_product: str = "BTC-USD"     # fallback live price source
    stocktwits_symbol: str = "BTC.X"
    # X search queries: ticker, cashtag and keyword (the brief asks for all three)
    x_queries: tuple = ("$BTC", "#BTC", "#Bitcoin", "bitcoin", "BTC price")
    news_queries: tuple = ("Bitcoin price news", "Bitcoin ETF flows", "Bitcoin regulation",
                           "Bitcoin macro outlook Federal Reserve")


@dataclass(frozen=True)
class ApifySpec:
    # Tweet Scraper V2 (Pay Per Result) by apidojo. Internal actor id 61RPP7dywgiy0JPD0.
    actor_id: str = "apidojo/tweet-scraper"
    actor_internal_id: str = "61RPP7dywgiy0JPD0"
    max_items_per_day_query: int = 480   # 20/hour x 24; 180 d x 5 queries x 480 = 432k posts ~= US$173 at $0.40/1k
    sort: str = "Latest"                 # per-hour windows; if Latest under-returns, actor docs suggest Top
    lang: str = "en"


@dataclass(frozen=True)
class ExperimentSpec:
    bar: str = "1h"                       # hourly bars; social signal decays within hours (Guegan & Renault 2021)
    horizons: tuple = (0, 1, 2, 3, 4, 5)  # n+0 ... n+5 (see docs/METHODOLOGY.md for the exact timing contract)
    min_posts_per_day: int = 300          # coverage gate: days with fewer collected posts are not modelled
    train_min_days: int = 60              # first walk-forward training window
    retrain_every_hours: int = 24 * 7     # weekly refit
    purge_hours: int = 6                  # >= max horizon + 1 so no label overlaps the test start
    embargo_hours: int = 24
    calib_frac: float = 0.2               # tail of each train window used for split-conformal calibration
    conformal_alpha: float = 0.2          # 80% prediction intervals
    cost_bps: float = 5.0                 # one-way cost per unit turnover (Binance VIP0 maker 7.5bps w/ BNB ~ 5-7.5)
    target_vol_annual: float = 0.40       # vol-targeted sizing
    seed: int = 7
    lgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        objective="huber", alpha=1.0, verbose=-1))


ASSET = AssetSpec()
APIFY = ApifySpec()
EXP = ExperimentSpec()

APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
