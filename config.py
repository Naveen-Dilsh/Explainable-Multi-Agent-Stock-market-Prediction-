"""Paths and constants. The single place anything configurable lives.

DATA_ROOT defaults to this folder, so the project is self-contained: its own
data/ and results/ travel with the code. Point it elsewhere with the DATA_ROOT
environment variable.
"""
import os
from pathlib import Path

import pandas as pd

# ─────────────────────────── where things live ───────────────────────────
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent))

DATA = DATA_ROOT / "data"
RESULTS = DATA_ROOT / "results"
FIGURES = DATA_ROOT / "figures"
ENV_FILE = DATA_ROOT / ".env"

RAW = DATA / "raw"
PANEL = DATA / "clean" / "panel.csv"
REGIME_LABELS = DATA / "clean" / "regime_labels.csv"
REGIME_EPISODES = DATA / "clean" / "regime_episodes.csv"
NEWS = RAW / "news_merged.csv"
DATASET = DATA / "features" / "dataset_h20.csv"
FOLDS = DATA / "splits" / "folds_h20.csv"

PREDICTIONS = RESULTS / "predictions_h20.csv"
CHOSEN_MODEL = RESULTS / "chosen_model_h20.csv"
MODEL_SUMMARY = RESULTS / "model_summary_h20.csv"
BASELINE_SUMMARY = RESULTS / "baseline_summary_h20.csv"

AGENT = RESULTS / "regime_agent"
ROUTER = AGENT / "regime_router.json"
ROUTER_SETTINGS = AGENT / "regime_agent_settings.csv"
REGIME_CARDS = AGENT / "regime_agent_cards.json"
RELIABILITY_BANDS = AGENT / "reliability_bands.csv"
SHAP_IMPORTANCE = AGENT / "shap_global_importance.csv"
SEALED_PREDICTIONS = AGENT / "sealed_predictions.csv"

SENTIMENT_CACHE = RESULTS / "sentiment_agent" / "sentiment_cache.json"
SENTIMENT_RUN = RESULTS / "sentiment_agent" / "sentiment_agent_run.json"
RISK_RUN = RESULTS / "risk_agent" / "risk_agent_run.json"
COORDINATOR_CACHE = RESULTS / "coordinator" / "coordinator_cache.json"
COORDINATOR_CARDS = RESULTS / "coordinator" / "coordinator_cards.json"
CARDS_DIR = DATA_ROOT / "demo" / "cards"
STUDY_RESPONSES = DATA / "user_study" / "responses" / "responses_2026-09-26.csv"
STUDY_RESULTS = RESULTS / "user_study" / "study_results.json"

# ──────────────────────────── the market state ────────────────────────────
# A day is labelled from the trailing 120-day NIFTY return: above +6% rising,
# below -6% falling, otherwise without direction. A new state must hold for 10
# consecutive days before it is accepted. See Notes 4.3.
LOOKBACK = 120
THRESHOLD = 0.06
CONFIRM_DAYS = 10
HORIZON = 20                     # trading days ahead that the system predicts

CLASSES = ["BEAR", "SIDEWAYS", "BULL"]
CODE = {c: i for i, c in enumerate(CLASSES)}
DECODE = {i: c for c, i in CODE.items()}

# ──────────────────────────── validation ────────────────────────────
TRAIN_END = pd.Timestamp("2024-12-31")
SEALED_FROM = pd.Timestamp("2025-01-01")   # never touched during development
TEST_YEARS = range(2015, 2025)             # ten expanding walk-forward folds
PURGE = HORIZON                            # rows dropped between train and test
DEGENERATE_FOLD = 7                        # 2021 has no state changes; excluded
SEEDS = [0, 1, 2, 3, 4]
RANDOM_STATE = 42

# ──────────────────────────── data sources ────────────────────────────
STOCKS = ["IOC", "BHARTIARTL", "ASHOKLEY", "ADANIPORTS", "INDHOTEL"]
TICKERS = {s: f"{s}.NS" for s in STOCKS}
MARKET = {"NIFTY50": "^NSEI", "INDIAVIX": "^INDIAVIX"}
START_DATE = "2010-01-01"
END_DATE = "2026-08-20"

# ──────────────────────────── risk rules ────────────────────────────
TARGET_DAILY_RISK = 1.5    # % daily risk a full position may carry
NO_CONTEXT = 0.70          # market drifting, no support either way
OPPOSED = 0.40             # trading against the prevailing market
CHANGE_PENALTY = 0.80      # a state change is expected, the reading is shaky
MIN_SENT_CONF = 0.50       # minimum sentiment confidence to trade at all
MIN_POSITION = 0.10        # below this the trade is dust, reject
PER_STOCK_WEIGHT = 0.20    # each of the 5 stocks owns 20% of capital at full size
MAX_EXPOSURE = 1.00        # never more than 100% invested
SL_MULT, TP_MULT = 1.0, 2.0    # stop / target in units of the 20-day sigma
VOL_WINDOW = 20

# ──────────────────────────── language model ────────────────────────────
LLM_MODEL = "gemini-3.1-flash-lite"
MIN_GAP_S = 4.5            # free tier allows 15 requests a minute
FRESH_DAYS = 7             # headline window
MIN_FRESH = 3              # fewer than this and the agent abstains
N_HEADLINES = 15

# Confidence bands for the reliability table: how far the router's probability
# sits from its own cut-off, scaled so 0 is on the line and 1 is as far as
# possible. A band no better than the others would mean confidence is decoration.
CONFIDENCE_BANDS = [
    (0.0, 0.2, "borderline"),
    (0.2, 0.5, "moderate"),
    (0.5, 0.8, "confident"),
    (0.8, 1.01, "very confident"),
]


def chosen_model():
    """The model choice recorded by `run.py models`, as a key -> value dict.

    The file is a headerless two-column CSV:
        chosen_prediction_column,pred_xgb_no_ret120
        n_features,25
        features,ret_5|ret_20|...
    """
    if not CHOSEN_MODEL.exists():
        raise FileNotFoundError(f"{CHOSEN_MODEL} not found. Run: python run.py models")
    return pd.read_csv(CHOSEN_MODEL, header=None, index_col=0)[1].to_dict()


def chosen_features():
    """The model's 25 features. One source of truth.

    Before this existed the ret_120 exclusion was hard-coded by hand in six
    different notebooks, and this file was written but never read.
    """
    return chosen_model()["features"].split("|")
