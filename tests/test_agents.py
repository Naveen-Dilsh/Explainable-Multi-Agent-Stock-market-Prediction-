"""The four agents, checked against the numbers the thesis reports.

The expensive part is training the router once; every other test reuses it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents as A  # noqa: E402
import config as c  # noqa: E402
import models as M  # noqa: E402

needs_data = pytest.mark.skipif(not c.DATASET.exists(),
                                reason="no dataset; run `make data` first")


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(c.DATASET, parse_dates=["Date"]).set_index("Date")


@pytest.fixture(scope="module")
def trained(dataset):
    """The frozen deployment router, trained once for the whole module."""
    dev = dataset.loc[dataset.index < c.SEALED_FROM]
    features = c.chosen_features()
    router, cutoff, base_rate = A.train_router(dev, features)
    return A.RegimeAgent(router, dev, features, cutoff), base_rate


# ───────────────────────────── the router ─────────────────────────────
@needs_data
def test_cutoff_is_calibrated_to_the_historical_change_rate(trained):
    """The cut-off is a percentile of the change rate, not an accuracy search.

    21.5% of days precede a regime change, and the reported cut-off is 0.531.
    """
    agent, base_rate = trained
    assert base_rate == pytest.approx(0.215, abs=0.001)
    assert agent.cutoff == pytest.approx(0.531, abs=0.001)


@needs_data
def test_sealed_period_is_reported_as_it_came_out(dataset, trained):
    """The advantage did not generalise. 69.8% against the bar's 73.8%.

    This test exists to stop a later change quietly turning a negative result
    into a positive one.
    """
    agent, _ = trained
    sealed = dataset.loc[dataset.index >= c.SEALED_FROM]
    pred = pd.Series([agent.decide(r)[0] for _, r in sealed.iterrows()], index=sealed.index)

    assert 100 * (pred == sealed["target"]).mean() == pytest.approx(69.8, abs=0.1)
    assert 100 * (M.mechanical(sealed) == sealed["target"]).mean() == pytest.approx(73.8, abs=0.1)


@needs_data
def test_router_only_ever_answers_will_it_change(trained):
    """Stage 1 is binary. The three-way answer comes from a rule, not the model."""
    agent, _ = trained
    assert agent.router.n_classes_ == 2


# ────────────────────────── the explanation ──────────────────────────
@needs_data
def test_counterfactual_really_flips_the_decision(dataset, trained):
    """A counterfactual that does not flip the answer is a false claim."""
    agent, _ = trained
    row = dataset.iloc[-1]
    sv = agent.shap_row(row)

    checked = 0
    for feature in sv.abs().sort_values(ascending=False).head(6).index:
        cf = agent.counterfactual(row, feature)
        if cf is None:
            continue
        checked += 1
        probe = row.copy()
        probe[feature] = cf["flip_at"]
        assert agent.decide(probe)[1] != agent.decide(row)[1], feature
    assert checked > 0, "no feature produced a counterfactual to check"


@needs_data
def test_every_feature_has_a_phrase(trained):
    """A card that falls back to a raw feature name is not plain English."""
    agent, _ = trained
    assert [f for f in agent.features if f not in A.PHRASE] == []


@needs_data
def test_grouped_evidence_accounts_for_every_feature(trained):
    """The eight families must partition the features, not sample them."""
    agent, _ = trained
    covered = [f for fs in A.FAMILY.values() for f in fs]
    assert sorted(covered) == sorted(agent.features)
    assert len(covered) == len(set(covered)), "a feature appears in two families"


# ─────────────────────────── the Sentiment Agent ───────────────────────────
def _news(dates, ticker="IOC.NS"):
    return pd.DataFrame({"ticker": ticker,
                         "published": pd.to_datetime(dates),
                         "title": [f"headline {i}" for i in range(len(dates))]})


def test_headlines_from_the_future_are_never_read():
    news = _news(["2025-01-01", "2025-01-05", "2025-01-09"])
    hl = A.fresh_headlines(news, "IOC.NS", "2025-01-06")
    assert list(hl["published"].dt.strftime("%Y-%m-%d")) == ["2025-01-05", "2025-01-01"]


def test_thin_news_abstains_instead_of_guessing():
    """Fewer than MIN_FRESH headlines is answered NEUTRAL, with no model call."""
    news = _news(["2025-01-05"])
    out, source = A.sentiment_agent(news, "IOC.NS", "2025-01-06", {}, live=True)
    assert (source, out["decision"], out["abstained"]) == ("abstained", "NEUTRAL", True)
    assert out["confidence"] == 0.0


def test_sentiment_never_calls_out_without_live():
    news = _news(["2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"])
    out, source = A.sentiment_agent(news, "IOC.NS", "2025-01-06", {}, live=False)
    assert (out, source) == (None, "pending")


# ─────────────────────────── the Coordinator ───────────────────────────
RISK = {"stock": "IOC", "date": "2025-01-06", "verdict": "SHRINK", "action": "BUY",
        "position_size": 0.4, "stop_loss_pct": 4.5, "take_profit_pct": 8.9,
        "rules_fired": [{"rule": "R4_MARKET_FIT", "detail": "no market support"}]}
REGIME = {"decision": "SIDEWAYS", "regime_today": "BULL", "will_change": True,
          "p_change": 0.61, "reasons": ["the fear index is at 15.2"]}
NEWS = {"decision": "UP", "confidence": 0.7, "explanation": "demand is recovering"}
REPLY = {"synthesis": "x" * 40, "confidence_label": "Medium"}


def test_the_narration_cannot_contradict_the_risk_agent():
    """The action and size are copied after the call, never asked for."""
    card = A.build_card(REGIME, NEWS, RISK, REPLY)
    assert not A.contradicts(card, RISK)
    assert card["risk"]["action"] == RISK["action"]
    assert card["risk"]["position_size"] == RISK["position_size"]


def test_a_card_that_disagrees_is_caught():
    card = A.build_card(REGIME, NEWS, RISK, REPLY)
    assert A.contradicts(card, {**RISK, "action": "SELL"})
    assert A.contradicts(card, {**RISK, "position_size": 1.0})


def test_coordinator_never_calls_out_without_live():
    assert A.coordinate(REGIME, NEWS, RISK, {}, live=False) == (None, "pending")


def test_both_cache_key_schemas_are_honoured():
    """Older entries were keyed STOCK|DATE|ACTION|SIZE|VERDICT. They still count."""
    long_key = {"IOC|2025-01-06|BUY|0.400|SHRINK": REPLY}
    card, source = A.coordinate(REGIME, NEWS, RISK, long_key, live=False)
    assert source == "cached"
    assert not A.contradicts(card, RISK)


def test_a_bare_cached_reply_is_wrapped_not_trusted():
    """Half the cache holds only the model's reply. The record is rebuilt around it."""
    card, source = A.coordinate(REGIME, NEWS, RISK, {"IOC|2025-01-06": REPLY}, live=False)
    assert source == "cached"
    assert card["risk"]["verdict"] == "SHRINK"
    assert card["synthesis"] == REPLY["synthesis"]
