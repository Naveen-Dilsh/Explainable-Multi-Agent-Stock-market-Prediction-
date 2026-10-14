PY ?= python

.DEFAULT_GOAL := help
.PHONY: help all data models router sentiment risk coordinator study demo test clean \
        live-data live-sentiment live-coordinator

help:            ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t18

all: data models router sentiment risk coordinator study   ## the whole pipeline. No network, no cost.

data:            ## clean, label, build features, make the walk-forward folds
	$(PY) run.py data
models:          ## baselines, XGBoost, the two-stage router, calibration
	$(PY) run.py models
router:          ## train the deployment router once, freeze it, explain it
	$(PY) run.py router
sentiment:       ## news sentiment per stock per day. Reads the cache only
	$(PY) run.py sentiment
risk:            ## the seven risk rules. No model, no API
	$(PY) run.py risk
coordinator:     ## narrate each decision. Reads the cache only
	$(PY) run.py coordinator
study:           ## grade the comprehension study
	$(PY) run.py study

demo:            ## open the interactive demo in a browser
	$(PY) -m streamlit run demo/app.py

test:            ## run the tests
	$(PY) -m pytest tests -q

clean:           ## remove this project's caches. Never data or results.
	rm -rf __pycache__ tests/__pycache__ .pytest_cache

# Costs money or needs the network. Nothing above depends on it.
live-data:        ## re-download prices from Yahoo Finance
	$(PY) run.py data --live
live-sentiment:   ## call Gemini for stock-days not yet cached
	$(PY) run.py sentiment --live
live-coordinator: ## call Gemini to narrate decisions not yet cached
	$(PY) run.py coordinator --live

# Not ported - these stay as notebooks:
#   nb 05, 11, 13, 15-17   feature gallery, neural first look, agent combinations
#   nb 21                  render the decision cards as PNGs
#   N1-N10                 the 1,350-run neural architecture search
