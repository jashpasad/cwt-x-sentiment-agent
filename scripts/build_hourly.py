"""Score every historical post and aggregate to hourly bars.
Output: data/processed/sent_hourly_hist.parquet (+ DuckDB table `sent_hourly`)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt.aggregate import combine, finalize, hourly_sums, score_posts  # noqa: E402
from cwt.config import MODELS, PROC, RAW  # noqa: E402
from cwt.ingest.historical import iter_parts  # noqa: E402
from cwt.nlp.sentiment import CryptoSentiment  # noqa: E402


def main(parts_dir=None):
    model = CryptoSentiment.load(MODELS / "cryptolex.joblib")
    sums, t0, n_in, n_kept = [], time.time(), 0, 0
    samples = []
    for df in iter_parts(Path(parts_dir) if parts_dir else None):
        n_in += len(df)
        p = score_posts(df, model)
        n_kept += len(p)
        sums.append(hourly_sums(p))
        samples.append(p.sample(min(3000, len(p)), random_state=1)[
            ["created_utc", "text", "sent", "p_bull", "p_bear", "bot", "source"]])
        print(f"{n_in:,} read, {n_kept:,} kept, {time.time()-t0:.0f}s", flush=True)
    h = finalize(combine(sums))
    h.to_parquet(PROC / "sent_hourly_hist.parquet")
    pd.concat(samples).to_parquet(PROC / "post_sample_scored.parquet")
    print("hours:", len(h), "posts read:", n_in, "kept (dedup, no reposts):", n_kept)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RAW / "parts"))
