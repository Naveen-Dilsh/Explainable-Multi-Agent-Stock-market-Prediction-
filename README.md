# Explainable Multi-Agent Framework for Stock Trading

A clean, runnable version of the research pipeline. It predicts the market state
(rising / falling / without direction) twenty trading days ahead, and issues every
decision together with the rules that produced it.

The central design choice: **the learned model never produces the answer.** It
answers only "will the state change?", and that answer selects between two
procedures a reader can check by hand — carry today's state forward, or apply the
threshold rule to the hundred days already known.

## Install

```bash
pip install -r requirements.txt
```

The project is self-contained: `data/` and `results/` live inside it, so it runs
anywhere with no configuration. Point it at another copy with:

```bash
export DATA_ROOT=/path/to/project
```

A Gemini API key is needed only for the two live steps. Put it in `<DATA_ROOT>/.env`:

```
GEMINI_API_KEY=...
```

## Run

```bash
make            # list the targets
make all        # the whole pipeline: no network, no cost
make test       # the test suite
make demo       # the interactive Streamlit demo
```

The steps, in order:

| | |
|---|---|
| `data` | clean, label, build features, make the walk-forward folds |
| `models` | baselines, XGBoost, the two-stage router, calibration |
| `router` | train the deployment router once, freeze it, explain it |
| `sentiment` | news sentiment per stock per day |
| `risk` | the seven risk rules |
| `coordinator` | narrate each decision in one paragraph |
| `study` | grade the comprehension study |

`models` answers "does the design work?" by retraining in every walk-forward
fold. `router` answers "what does the shipped system say?" by training once on
everything up to 2024 and freezing it. Both are needed and they are not the same
question. Or call steps directly:

```bash
python run.py router risk
```

Steps that cost money or need the network are **opt-in and depend on nothing**:

```bash
make live-data           # re-download prices from Yahoo
make live-sentiment      # call Gemini for uncached dates
make live-coordinator    # call Gemini to narrate uncached cards
```

Without `--live` these read from cache and refuse to call out, so an ordinary
`make all` can never spend money. `make clean` removes only this project's
caches — never `data/`, `results/` or the API caches, which represent several
hundred real paid calls.

## Files

| | |
|---|---|
| `config.py` | paths and every constant: thresholds, horizon, seeds, tickers |
| `data.py` | download, clean, regime labels, the 26 features, walk-forward folds |
| `models.py` | baselines, XGBoost, the two-stage router, calibration, scoring |
| `agents.py` | the frozen router, SHAP, counterfactuals, sentiment, the coordinator |
| `risk.py` | the seven risk rules. No model, no API |
| `userstudy.py` | comprehension grading |
| `run.py` | the CLI, and the only file that reads or writes anything |
| `demo/` | the Streamlit app: pick a day, see all four agents explain it |
| `data/` | raw prices, the aligned panel, labels, features, folds, news |
| `results/` | predictions, scores, the trained router, the agent caches |

Two paths are deliberately kept out of version control by `.gitignore`:
`.env`, which holds the API key, and `data/user_study/responses/`, which holds
the raw participant answers that Appendix J states are not published.

Everything except `run.py` is plain functions over DataFrames. That separation is
what makes the pipeline testable without touching the disk.

## What the numbers are

| | |
|---|---|
| Majority baseline | 41.3% |
| Persistence baseline | 77.8% |
| **Mechanical baseline** | **83.7%** — the real bar; needs no model at all |
| Two-stage router | 84.9% — **+1.2** over that bar |
| Oracle (perfect routing) | 94.6% — the ceiling |

The mechanical baseline is high because the state twenty days out is defined by a
120-day window, and a hundred of those days have already happened. Reporting
against the majority baseline instead would have claimed a 43-point gain for the
same model.

Fold 7 (2021) contains no state changes at all, so it cannot discriminate between
models and is excluded. The code finds it rather than assuming it.

On the sealed period (381 days after 31 December 2024, opened once at the end) the
model scored 69.8% against the baseline's 73.8%. The advantage did not generalise,
and that is reported as a finding rather than adjusted away. `tests/test_agents.py`
asserts both figures, so no later change can quietly turn that into a win.

The Coordinator never decides anything. The action, the position size and the
stop and target are copied out of the Risk Agent's record *after* the language
model has replied, so "the narration never contradicts the decision" holds by
construction. `make coordinator` counts the contradictions every run; it is 0.

## Tests

```bash
make test
```

`tests/test_risk.py` holds the thirteen scenarios that pin down the risk rules.
They come from notebook 19, whose scenario cell does not compile — a newer table
was pasted over an older one. The newer table is the correct one and lives here now.

`tests/test_models.py` checks the pipeline against `tests/golden/`, captured from
the notebook output before the port began.

`tests/test_agents.py` pins the calibrated cut-off, the sealed result, and the
two guarantees that matter for explanation: a counterfactual it reports really
does flip the decision, and every feature a card can name has a plain-English
phrase.