.PHONY: setup data sentiment hourly backtest charts report live test all
setup:      ; pip install -r requirements.txt
audit:      ; python scripts/audit_provenance.py
data:       ; python -c "import sys;sys.path.insert(0,'src');from cwt.ingest.historical import extract;print('dropped rows:',extract())"
sentiment:  ; python scripts/train_sentiment.py
hourly:     ; python scripts/build_hourly.py
backtest:   ; python scripts/run_backtest.py
charts:     ; python scripts/make_charts.py
report:     ; python scripts/build_report.py
live:       ; python scripts/run_live.py --days 180
test:       ; python -m pytest -q tests
vol:        ; python scripts/vol_study.py
scrape:     ; python scripts/scrape_x.py
all: audit data sentiment hourly backtest vol charts scrape live report
