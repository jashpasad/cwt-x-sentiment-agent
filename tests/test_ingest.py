"""Regression tests for live-ingestion pitfalls found during development."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cwt.ingest.apify_x import QUERY, build_input, estimate_rate, normalize_item, sample_plan  # noqa: E402


def test_or_group_is_parenthesised():
    # Without parentheses X binds since_time/until_time only to the LAST OR term (bug seen live).
    assert QUERY.startswith("(") and QUERY.endswith(")")
    inp = build_input(datetime(2026, 10, 30, 15, tzinfo=timezone.utc))
    assert int(inp["until_time"]) - int(inp["since_time"]) == 3600


def test_mock_items_dropped():
    assert normalize_item({"id": 1, "text": "From KaitoEasyAPI, a reminder: ..."}, None) is None
    ok = normalize_item({"id": 2, "text": "btc", "createdAt": "Thu Oct 29 14:59:59 +0000 2026"}, None)
    assert ok and ok["created_utc"].hour == 14


def test_rate_estimator():
    end = pd.Timestamp("2026-10-30 15:00", tz="UTC")
    t = pd.Series(pd.date_range(end - pd.Timedelta(minutes=10), periods=20, freq="30s"))
    assert 100 < estimate_rate(t, 20, end) < 130          # 19 posts in ~10 min ~= 114/h
    assert estimate_rate(t.iloc[:5], 20, end) == 5        # window exhausted -> count is the rate


def test_sample_plan_dense_and_sparse():
    p = sample_plan(datetime(2026, 10, 31, 23, 30, tzinfo=timezone.utc), 180, 7, 1)
    assert len(p) == 7 * 24 + 173 and p[-1].hour == 23
