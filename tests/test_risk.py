"""The risk rule book, pinned down.

These thirteen scenarios come from cell 9 of notebooks/19_risk_agent.ipynb. That
cell does not compile: a newer table was pasted over an older one, leaving a
stray `]),` mid-list and ten orphaned rows from the retired 2-sigma generation.
The newer table is the correct one and is reproduced here.

Expected sizes are: min(1, 1.5 / vol), then x1.00 if the news agrees with the
market, x0.40 if it opposes it, x0.70 if the market is drifting; then x0.80 if a
state change is expected; then capped by the room left in the book.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import risk  # noqa: E402


def R(decision, will_change):
    return {"decision": decision, "will_change": will_change,
            "p_change": 0.7 if will_change else 0.1}


def S(decision, confidence):
    return {"decision": decision, "confidence": confidence}


#          name                              regime              sentiment          expo  vol  verdict   size  must_fire
SCENARIOS = [
    ("agree: BUY in a rising market",      R("BULL", False),   S("UP", 0.8),      0.00,  1.2, "APPROVE", 1.00, ["R3_VOL_SIZING", "R7_STOP_LOSS"]),
    ("agree: SELL in a falling market",    R("BEAR", False),   S("DOWN", 0.9),    0.00,  1.0, "APPROVE", 1.00, ["R3_VOL_SIZING", "R7_STOP_LOSS"]),
    ("oppose: BUY into a falling market",  R("BEAR", False),   S("UP", 0.8),      0.00,  2.5, "SHRINK",  0.24, ["R4_MARKET_FIT"]),
    ("oppose: SELL into a rising market",  R("BULL", False),   S("DOWN", 0.9),    0.00,  1.0, "SHRINK",  0.40, ["R4_MARKET_FIT"]),
    ("drifting market gives no support",   R("SIDEWAYS", False), S("DOWN", 0.9),  0.00,  1.0, "SHRINK",  0.70, ["R4_MARKET_FIT"]),
    ("change expected trims further",      R("BULL", True),    S("UP", 0.8),      0.00,  2.5, "SHRINK",  0.48, ["R4B_CHANGE_EXPECTED"]),
    ("neutral news -> reject",             R("BULL", False),   S("NEUTRAL", 0.9), 0.00,  1.2, "REJECT",  0.00, ["R1_NO_DIRECTION"]),
    ("weak signal -> reject",              R("BULL", False),   S("UP", 0.30),     0.00,  1.2, "REJECT",  0.00, ["R2_LOW_CONFIDENCE"]),
    ("high volatility -> small",           R("BULL", False),   S("UP", 0.9),      0.00,  5.0, "SHRINK",  0.30, ["R3_VOL_SIZING"]),
    ("extreme volatility -> dust",         R("BULL", False),   S("UP", 0.9),      0.00, 20.0, "REJECT",  0.00, ["R6_MIN_SIZE"]),
    ("nearly full book -> shrink",         R("BULL", False),   S("UP", 0.8),      0.95,  1.2, "SHRINK",  0.25, ["R5_EXPOSURE_LIMIT"]),
    ("fully invested -> reject",           R("BULL", False),   S("UP", 0.8),      1.00,  1.2, "REJECT",  0.00, ["R5_EXPOSURE_LIMIT"]),
    ("oppose + tight book -> both cut",    R("BEAR", False),   S("UP", 0.9),      0.96,  2.5, "SHRINK",  0.20, ["R4_MARKET_FIT", "R5_EXPOSURE_LIMIT"]),
]


@pytest.mark.parametrize(
    "name,regime,sentiment,exposure,vol,verdict,size,must_fire",
    SCENARIOS,
    ids=[s[0] for s in SCENARIOS],
)
def test_scenario(name, regime, sentiment, exposure, vol, verdict, size, must_fire):
    r = risk.risk_agent(regime, sentiment, vol, {"current_exposure": exposure})
    fired = [f["rule"] for f in r["rules_fired"]]

    assert r["verdict"] == verdict
    assert r["position_size"] == pytest.approx(size, abs=1e-3)
    for rule in must_fire:
        assert rule in fired, f"{rule} did not fire; got {fired}"


@pytest.mark.parametrize("name,regime,sentiment,exposure,vol,verdict,size,must_fire",
                         SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_contract(name, regime, sentiment, exposure, vol, verdict, size, must_fire):
    """Invariants that must hold for every decision the agent can produce."""
    r = risk.risk_agent(regime, sentiment, vol, {"current_exposure": exposure})

    assert 0.0 <= r["position_size"] <= 1.0
    assert r["explanation"], "every decision must carry an explanation"

    if r["verdict"] == "REJECT":
        assert r["action"] == "NONE"
        assert r["position_size"] == 0.0
        assert r["stop_loss_pct"] is None and r["take_profit_pct"] is None
    else:
        assert r["action"] in ("BUY", "SELL")
        assert r["stop_loss_pct"] > 0 and r["take_profit_pct"] > 0
        # The target is twice the stop, but both are rounded to one decimal
        # independently, so 2 x sl can sit up to 0.15 away from tp.
        assert r["take_profit_pct"] == pytest.approx(2 * r["stop_loss_pct"], abs=0.16)


def test_no_volatility_is_rejected():
    r = risk.risk_agent(R("BULL", False), S("UP", 0.9), None)
    assert r["verdict"] == "REJECT"
    assert "R3_VOL_SIZING" in [f["rule"] for f in r["rules_fired"]]
