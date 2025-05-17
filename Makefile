.PHONY: ingest valuation test

ingest:
	python -m app.orchestrator

valuation:
	python -m app.orchestrator

test:
	python test_all.py
