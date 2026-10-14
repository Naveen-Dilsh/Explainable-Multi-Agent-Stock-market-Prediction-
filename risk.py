"""Risk Agent — deterministic trade approval (APPROVE / SHRINK / REJECT).

Seven rules, no model and no API. The explanation is simply which rules fired
and the numbers that triggered them.

    regime    {"decision", "regime_today", "will_change", "p_change", "reasons"}
    sentiment {"decision", "confidence"}
    vol       trailing 20-day realised volatility, % per day
    portfolio {"current_exposure"}  fraction of capital already deployed

Lifted unchanged from demo/risk_agent.py, which was already the canonical copy.
Only the constants moved out, into config.py.
"""
import math

import numpy as np
import pandas as pd

from config import (
    CHANGE_PENALTY,
    HORIZON,
    MAX_EXPOSURE,
    MIN_POSITION,
    MIN_SENT_CONF,
    NO_CONTEXT,
    OPPOSED,
    PER_STOCK_WEIGHT,
    SL_MULT,
    TARGET_DAILY_RISK,
    TP_MULT,
    VOL_WINDOW,
)


def realised_vol(panel, stock, as_of, window=VOL_WINDOW):
    """Trailing realised volatility for one stock, in % per day.

    Strictly backward looking: only closes up to and including `as_of` are used.
    Replaces the retired Technical Agent's predicted volatility.
    """
    col = f"{stock}_Close"
    if col not in panel.columns:
        return None
    closes = panel.loc[:pd.Timestamp(as_of), col].dropna()
    if len(closes) < window + 1:
        return None
    v = float(closes.pct_change().tail(window).std() * 100)
    return v if np.isfinite(v) and v > 0 else None


def risk_agent(regime, sentiment, vol, portfolio=None):
    """Deterministic rule-based gatekeeper. Mirrors Step 19 exactly."""
    portfolio = portfolio or {"current_exposure": 0.0}
    exposure = float(portfolio["current_exposure"])
    fired = []

    def result(action, verdict, size, sl=None, tp=None):
        return {"action": action, "verdict": verdict, "position_size": round(size, 3),
                "stop_loss_pct": None if sl is None else round(sl, 1),
                "take_profit_pct": None if tp is None else round(tp, 1),
                "rules_fired": fired,
                "explanation": " ".join(f'[{f["rule"]}] {f["detail"]}' for f in fired)}

    # R1 — no direction, no trade
    if sentiment["decision"] in ("NEUTRAL", "?"):
        fired.append({"rule": "R1_NO_DIRECTION",
                      "detail": "Sentiment Agent found no clear direction - no honest basis to trade.",
                      "plain": "The news didn't point clearly up or down, so there's nothing to act on."})
        return result("NONE", "REJECT", 0.0)

    # R2 — weak signal, no trade
    if float(sentiment["confidence"]) < MIN_SENT_CONF:
        fired.append({"rule": "R2_LOW_CONFIDENCE",
                      "detail": f'Sentiment confidence {sentiment["confidence"]:.2f} is below the '
                                f'{MIN_SENT_CONF:.2f} minimum - signal too weak to act on.',
                      "plain": f'The news agent was only {sentiment["confidence"]:.0%} sure — below the '
                               f'{MIN_SENT_CONF:.0%} needed to risk money on it.'})
        return result("NONE", "REJECT", 0.0)

    if vol is None:
        fired.append({"rule": "R3_VOL_SIZING",
                      "detail": "No volatility estimate available - cannot size the position.",
                      "plain": "We don't know how much this stock normally moves, so we can't judge a safe size."})
        return result("NONE", "REJECT", 0.0)

    action = "BUY" if sentiment["decision"] == "UP" else "SELL"

    # R3 — volatility targeting: riskier stock -> smaller position
    size = min(1.0, TARGET_DAILY_RISK / vol)
    fired.append({"rule": "R3_VOL_SIZING",
                  "detail": f"Trailing daily volatility is {vol:.2f}%; sizing to a "
                            f"{TARGET_DAILY_RISK:.1f}% risk target gives {size:.0%} of the allocation.",
                  "plain": f"This stock has been moving about {vol:.2f}% a day recently. We only want to "
                           f"risk about {TARGET_DAILY_RISK:.1f}% a day, so we buy {size:.0%} of a normal amount."})

    # R4 — how much to believe the news, given what the market is doing.
    #
    # Restores the direction-aware logic of the original Regime+Sentiment combiner
    # (notebook 13, rules C2/C3/C4/C5). The earlier version of this rule asked only
    # "is the market bad?", so it capped a BUY into a falling market and a SELL into a
    # falling market identically, and left a SELL into a RISING market completely
    # uncapped. Trading against the prevailing market is now penalised explicitly.
    mood = regime["decision"]
    agrees   = (action == "BUY" and mood == "BULL") or (action == "SELL" and mood == "BEAR")
    opposes  = (action == "BUY" and mood == "BEAR") or (action == "SELL" and mood == "BULL")

    if agrees:
        mult, why = 1.00, f"the news and the market agree (both point {mood})"
        plain_why = "the news and the market are pointing the same way"
    elif opposes:
        mult, why = OPPOSED, f"news says {action} but the market is {mood} - trading against it"
        plain_why = ("we would be buying while the market falls" if action == "BUY"
                     else "we would be selling while the market rises")
    else:                                   # market is SIDEWAYS - no support either way
        mult, why = NO_CONTEXT, "the market is drifting - it offers no support either way"
        plain_why = "the market is drifting, so it gives us no help either way"

    if mult < 1.0:
        size *= mult
        fired.append({"rule": "R4_MARKET_FIT",
                      "detail": f"{why}; position scaled by {mult:.2f}.",
                      "plain": f"Because {plain_why}, we cut the position to "
                               f"{mult:.0%} of what it would otherwise be."})

    # R4b — an expected change makes the market reading itself less reliable
    if regime["will_change"]:
        size *= CHANGE_PENALTY
        fired.append({"rule": "R4B_CHANGE_EXPECTED",
                      "detail": f"a regime change is expected ({regime['p_change']:.0%}); "
                                f"position scaled by {CHANGE_PENALTY:.2f}.",
                      "plain": f"The market looks like it is about to change, so we trim the "
                               f"position further, to {CHANGE_PENALTY:.0%} of that."})

    # R5 — portfolio exposure limit
    room = (MAX_EXPOSURE - exposure) / PER_STOCK_WEIGHT
    if room <= 0:
        fired.append({"rule": "R5_EXPOSURE_LIMIT",
                      "detail": f"Portfolio is already {exposure:.0%} invested - no room for a new position.",
                      "plain": f"Our money is already {exposure:.0%} tied up in other trades — there's "
                               f"nothing left to invest."})
        return result("NONE", "REJECT", 0.0)
    if size > room:
        size = room
        fired.append({"rule": "R5_EXPOSURE_LIMIT",
                      "detail": f"Portfolio is {exposure:.0%} invested; position shrunk to {size:.0%} "
                                f"so total exposure stays under {MAX_EXPOSURE:.0%}.",
                      "plain": f"We've already committed {exposure:.0%} of our money elsewhere, so this "
                               f"trade is trimmed to {size:.0%} to stay within budget."})

    # R6 — dust filter
    if size < MIN_POSITION:
        fired.append({"rule": "R6_MIN_SIZE",
                      "detail": f"Resulting position ({size:.0%}) is below the {MIN_POSITION:.0%} "
                                f"minimum - not worth trading costs. Rejected.",
                      "plain": f"What's left ({size:.0%}) is too small to bother with — the trading fees "
                               f"would eat any gain."})
        return result("NONE", "REJECT", 0.0)

    # R7 — exits scale with REALISED risk
    sigma20 = vol * math.sqrt(HORIZON)
    sl, tp = SL_MULT * sigma20, TP_MULT * sigma20
    fired.append({"rule": "R7_STOP_LOSS",
                  "detail": f"Stop-loss {sl:.1f}% / take-profit {tp:.1f}% = {SL_MULT:.0f}x / "
                            f"{TP_MULT:.0f}x the trailing 20-day sigma ({sigma20:.1f}%).",
                  "plain": f"Over a month this stock typically swings about {sigma20:.1f}%. So we sell "
                           f"automatically if it drops {sl:.1f}% (cut the loss), or if it gains "
                           f"{tp:.1f}% (take the win)."})

    verdict = "APPROVE" if size >= 0.999 else "SHRINK"
    return result(action, verdict, size, sl, tp)


def evaluate(stocks, sentiments, regime, panel, as_of, opening_exposure=0.0):
    """Run the rule book over every stock for one day, sharing one portfolio.

    Positions are taken in list order and each approved trade consumes exposure,
    so R5 sees a book that fills up as the day is worked through.
    """
    out, exposure = {}, float(opening_exposure)
    for s in stocks:
        vol = realised_vol(panel, s, as_of)
        r = risk_agent(regime, sentiments[s], vol,
                       {"current_exposure": min(exposure, MAX_EXPOSURE)})
        r.update(stock=s, date=str(pd.Timestamp(as_of).date()),
                 realised_vol=None if vol is None else round(vol, 2),
                 regime=regime["decision"], will_change=regime["will_change"])
        exposure += r["position_size"] * PER_STOCK_WEIGHT
        out[s] = r
    for s in stocks:
        out[s]["book_after"] = round(min(exposure, MAX_EXPOSURE), 4)
    return out


def one_liner(r):
    """One plain-English sentence assembled from the rules that fired."""
    if r["verdict"] == "REJECT":
        last = r["rules_fired"][-1]
        return f'we are not trading {r["stock"]} today. {last.get("plain", last["detail"])}'
    verb = "buy" if r["action"] == "BUY" else "sell"
    trimmed = " (less than usual, because of the rules below)" if r["verdict"] == "SHRINK" else ""
    return (f'{verb} {r["stock"]} at {r["position_size"]:.0%} of what it would normally get'
            f'{trimmed}. Sell automatically if it drops {r["stop_loss_pct"]:.1f}% '
            f'or gains {r["take_profit_pct"]:.1f}%.')


# ---------------------------------------------------------------- outcome check
def trade_outcome(panel, stock, as_of, action, stop_pct, target_pct, horizon=HORIZON):
    """Walk the REAL price forward from the decision day and see which barrier it hits first.

    Only possible because every date in the demo is historical - the outcome is already known.
    Returns None when the trade has no direction or the data runs out.
    """
    if action not in ("BUY", "SELL") or not stop_pct or not target_pct:
        return None
    col = f"{stock}_Close"
    if col not in panel.columns:
        return None
    px = panel[col]
    as_of = pd.Timestamp(as_of)
    if as_of not in px.index:
        return None
    i = px.index.get_loc(as_of)
    entry = float(px.iloc[i])
    fwd = px.iloc[i + 1: i + 1 + horizon]
    if len(fwd) < 2:
        return None

    if action == "BUY":
        stop_lvl, tgt_lvl = entry * (1 - stop_pct / 100), entry * (1 + target_pct / 100)
        hit_stop, hit_tgt = fwd <= stop_lvl, fwd >= tgt_lvl
    else:                                     # short
        stop_lvl, tgt_lvl = entry * (1 + stop_pct / 100), entry * (1 - target_pct / 100)
        hit_stop, hit_tgt = fwd >= stop_lvl, fwd <= tgt_lvl

    d_stop = fwd.index[hit_stop][0] if hit_stop.any() else None
    d_tgt = fwd.index[hit_tgt][0] if hit_tgt.any() else None
    if d_tgt is not None and (d_stop is None or d_tgt <= d_stop):
        result, when = "TARGET", d_tgt
    elif d_stop is not None:
        result, when = "STOP", d_stop
    else:
        result, when = "OPEN", fwd.index[-1]

    final = float(fwd.iloc[-1])
    pnl = (final - entry) / entry * 100 * (1 if action == "BUY" else -1)
    return {"entry": entry, "stop_lvl": stop_lvl, "target_lvl": tgt_lvl,
            "result": result, "when": when, "days_taken": int(fwd.index.get_loc(when)) + 1,
            "pnl_pct": round(pnl, 2), "path": fwd, "complete": len(fwd) >= horizon}
