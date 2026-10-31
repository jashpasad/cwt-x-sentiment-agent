"""Run the budgeted Apify X scrape (180 days) and write data/raw/apify_posts.parquet + apify_rates.parquet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt.config import RAW  # noqa: E402
from cwt.ingest.apify_x import scrape  # noqa: E402

if __name__ == "__main__":
    posts, rates = scrape(days=180, dense_days=7, per_day=1, k=20, budget_usd=4.75)
    posts.to_parquet(RAW / "apify_posts.parquet")
    rates.to_parquet(RAW / "apify_rates.parquet")
    print("posts:", len(posts), "samples:", len(rates), "range:", posts.created_utc.min(), posts.created_utc.max())
