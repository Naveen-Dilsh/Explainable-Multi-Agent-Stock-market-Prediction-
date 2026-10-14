"""The three learned-or-narrated agents. The fourth, Risk, is in risk.py.

Ported from notebooks 12, 14, 18 and 20. Nothing here reads or writes a file;
run.py does. The only exception is the environment variable holding the API key,
which `call_llm` reads at the moment of calling out.

    RegimeAgent    one day in, one explained decision out      nb 12, 18
    sentiment      news headlines in, a direction out          nb 14
    coordinate     three agent outputs in, one paragraph out   nb 20

The Regime Agent is the only one that holds state, because the router is trained
once and frozen. The other two are functions over a cache dict.

Both language-model steps take `live=False` and return a "pending" marker rather
than calling out, so a plain `make` can never spend money.
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from config import (
    CONFIDENCE_BANDS,
    FRESH_DAYS,
    LLM_MODEL,
    MIN_FRESH,
    N_HEADLINES,
    RANDOM_STATE,
    THRESHOLD,
)

# ─────────────────────────── stage 1: the router ───────────────────────────
def make_router(seed=RANDOM_STATE):
    """The same detector as models.make_detector, kept here so the frozen
    deployment router and the walk-forward one cannot drift apart by accident.
    """
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", scale_pos_weight=3.5,
        random_state=seed, n_jobs=4, verbosity=0,
    )


def train_router(dev, features, seed=RANDOM_STATE):
    """Train the deployment router on everything up to TRAIN_END, then freeze it.

    The cut-off is calibrated on an inner 80/20 slice of `dev` so that the router
    flags changes at the rate history actually shows. Sealed data is never seen.

    Returns (router, cutoff, base_rate).
    """
    y = (dev["regime_now"] != dev["target"]).astype(int)
    base_rate = float(y.mean())

    cut = int(0.8 * len(dev))
    inner = make_router(seed).fit(dev.iloc[:cut][features], y.iloc[:cut])
    p_inner = inner.predict_proba(dev.iloc[cut:][features])[:, 1]
    cutoff = float(np.quantile(p_inner, 1 - base_rate))

    return make_router(seed).fit(dev[features], y), cutoff, base_rate


def load_router(path):
    r = XGBClassifier()
    r.load_model(str(path))
    return r


def reliability_bands(p_change, cutoff, truth_change, bands=CONFIDENCE_BANDS):
    """How often the router is right at each confidence level.

    A band no better than the others would mean the reported confidence is
    decoration. This is what lets a card say "when this confident, correct X%".
    """
    margin = np.abs(p_change - cutoff) / max(cutoff, 1 - cutoff)
    says = p_change >= cutoff
    rows = []
    for lo, hi, name in bands:
        m = (margin >= lo) & (margin < hi)
        if m.sum() < 5:
            continue
        rows.append({"band": name, "n": int(m.sum()),
                     "accuracy": round(float((says[m] == truth_change[m]).mean()), 3)})
    return pd.DataFrame(rows)


# ──────────────────────── how a feature reads in words ────────────────────────
# One phrase per feature. Before this, three notebooks and the demo each kept
# their own copy and they had already drifted apart.
PHRASE = {
    "flip_margin": lambda v: f"we sit {v:.1f} standard deviations from the nearest regime boundary",
    "ret_100": lambda v: f"the market is {v*100:+.1f}% over 100 days",
    "ret_60": lambda v: f"the market is {v*100:+.1f}% over 60 days",
    "ret_20": lambda v: f"the market is {v*100:+.1f}% over 20 days",
    "ret_5": lambda v: f"the market is {v*100:+.1f}% this week",
    "px_vs_sma200": lambda v: f"price sits {v*100:+.1f}% against its 200-day average",
    "px_vs_sma50": lambda v: f"price sits {v*100:+.1f}% against its 50-day average",
    "sma50_vs_sma200": lambda v: f"the 50-day average is {v*100:+.1f}% against the 200-day",
    "vix_level": lambda v: f"the fear index is at {v:.1f}",
    "vix_chg_20": lambda v: f"fear has moved {v*100:+.0f}% in 20 days",
    "vol_20": lambda v: f"recent volatility is {v*100:.2f}% daily",
    "vol_60": lambda v: f"two-month volatility is {v*100:.2f}% daily",
    "bb_width": lambda v: f"the Bollinger bands are {v*100:.1f}% wide",
    "drawdown": lambda v: f"we are {abs(v)*100:.1f}% below the peak",
    "days_since_high": lambda v: f"it has been {int(v)} days since the last high",
    "breadth_above_sma200": lambda v: f"{int(round(v*5))} of 5 stocks are above their 200-day average",
    "avg_correlation": lambda v: f"the stocks move together at {v:.2f} correlation",
    "adx_14": lambda v: f"trend strength is {v:.0f}",
    "trend_r2_60": lambda v: f"the trend is {v:.0%} straight",
    "rsi_14": lambda v: f"RSI is {v:.0f}",
    "macd_hist": lambda v: f"MACD momentum is {v*1000:+.2f}",
    "days_in_regime": lambda v: f"this regime is {int(v)} days old",
    "in_bull": lambda v: "we are in a bull regime" if v else "we are not in a bull regime",
    "in_bear": lambda v: "we are in a bear regime" if v else "we are not in a bear regime",
    "in_sideways": lambda v: "we are drifting sideways" if v else "we are not drifting sideways",
}

BINARY = ("in_bull", "in_bear", "in_sideways")

# Twenty-five SHAP numbers are not an explanation. Grouping them into eight
# families gives a reader the kind of evidence before the detail.
FAMILY = {
    "trend": ["ret_5", "ret_20", "ret_60", "ret_100", "px_vs_sma50", "px_vs_sma200",
              "sma50_vs_sma200", "macd_hist", "trend_r2_60", "adx_14"],
    "fear": ["vix_level", "vix_chg_20"],
    "turbulence": ["vol_20", "vol_60", "bb_width"],
    "stress": ["drawdown", "days_since_high"],
    "agreement": ["breadth_above_sma200", "avg_correlation"],
    "regime age": ["in_bull", "in_sideways", "in_bear", "days_in_regime"],
    "fragility": ["flip_margin"],
    "momentum": ["rsi_14"],
}

SAYS = {
    "trend": "the market's direction",
    "fear": "how frightened the market is",
    "turbulence": "how choppy trading has been",
    "stress": "how far below the peak we are",
    "agreement": "whether the stocks are moving together",
    "regime age": "how long the current regime has run",
    "fragility": "how close we are to a regime boundary",
    "momentum": "short-term momentum",
}


def phrase(feature, value):
    return PHRASE.get(feature, lambda v: f"{feature} is {v:.3f}")(value)


def level(pct):
    """A percentile said in words. 'VIX is 27.9' means nothing on its own."""
    for edge, word in ((0.95, "extremely high"), (0.80, "high"),
                       (0.60, "somewhat above normal"), (0.40, "about normal"),
                       (0.20, "somewhat below normal"), (0.05, "low")):
        if pct >= edge:
            return word
    return "extremely low"


def band_of(margin):
    """How far the router's answer sits from its own cut-off, said in words."""
    for lo, hi, name in CONFIDENCE_BANDS:
        if lo <= margin < hi:
            return name
    return CONFIDENCE_BANDS[-1][2]


def mechanical_one(row, threshold=THRESHOLD):
    """The arithmetic branch for a single day. models.mechanical, one row."""
    if row["ret_100"] > threshold:
        return "BULL"
    if row["ret_100"] < -threshold:
        return "BEAR"
    return "SIDEWAYS"


# ────────────────────────── the Regime Agent (nb 12, 18) ──────────────────────────
class RegimeAgent:
    """The frozen router plus everything needed to explain one of its answers.

    Holds state because the router is trained once: `dev` is kept only to read
    percentiles and counterfactual grids off the training distribution, never to
    fit anything further.
    """

    def __init__(self, router, dev, features, cutoff, reliability=None):
        self.router = router
        self.features = list(features)
        self.cutoff = float(cutoff)
        self.dist = dev[self.features]
        self.explainer = shap.TreeExplainer(router)
        self.reliability = reliability or {}

    # ---- the decision ----
    def probability(self, rows):
        """P(the regime changes within the horizon), for one row or many."""
        x = self.frame(rows)
        return self.router.predict_proba(x[self.features])[:, 1]

    def decide(self, row):
        """Returns (decision, rule_used, p_change)."""
        p = float(self.probability(row)[0])
        if p >= self.cutoff:
            return mechanical_one(row), "arithmetic", p
        return row["regime_now"], "persistence", p

    # ---- the explanation ----
    def shap_row(self, row):
        sv = self.explainer.shap_values(self.frame(row)[self.features])[0]
        return pd.Series(sv, index=self.features)

    def importance(self):
        """Mean |SHAP| over the training period: what the router uses at all."""
        sv = self.explainer.shap_values(self.dist)
        return (pd.Series(np.abs(sv).mean(axis=0), index=self.features)
                .sort_values(ascending=False))

    def percentile(self, feature, value):
        return float((self.dist[feature] <= value).mean())

    def margin(self, p_change):
        """Distance from the cut-off, scaled so 0 is on the line and 1 is as far as it gets."""
        return abs(p_change - self.cutoff) / max(self.cutoff, 1 - self.cutoff)

    def band(self, p_change):
        return band_of(self.margin(p_change))

    def counterfactual(self, row, feature, n=60):
        """The nearest value of one feature that would flip the answer.

        Everything else is held at today's value, so this reads as "if only this
        had been different". Returns None when no value in the historical range
        flips it, which is itself worth saying: the answer does not hinge there.
        """
        x = self.frame(row)
        says_now = float(self.probability(row)[0]) >= self.cutoff

        lo, hi = self.dist[feature].quantile(0.01), self.dist[feature].quantile(0.99)
        grid = np.linspace(lo, hi, n)
        probe = pd.concat([x] * n, ignore_index=True)
        probe[feature] = grid
        flips = grid[(self.router.predict_proba(probe[self.features])[:, 1]
                      >= self.cutoff) != says_now]
        if len(flips) == 0:
            return None

        now = float(row[feature])
        nearest = float(flips[np.argmin(np.abs(flips - now))])
        return {"feature": feature, "current": now, "flip_at": nearest,
                "direction": "higher" if nearest > now else "lower",
                "new_answer": "stable" if says_now else "a change"}

    def explain(self, row, top_n=4):
        """One day in, one fully explained decision out."""
        decision, rule, p = self.decide(row)
        sv = self.shap_row(row)
        top = sv.reindex(sv.abs().sort_values(ascending=False).index).head(top_n)
        band = self.band(p)

        reasons = []
        for f in top.index:
            txt = phrase(f, row[f])
            if f not in BINARY:
                pct = self.percentile(f, row[f])
                txt += f" ({level(pct)}, {pct:.0%} percentile)"
            reasons.append(txt)

        cf = next((c for c in (self.counterfactual(row, f) for f in top.index)
                   if c is not None), None)

        return {
            "agent": "regime",
            "scope": "MARKET (NIFTY 50)",
            "date": str(row.name.date()),
            "decision": decision,
            "regime_today": row["regime_now"],
            "will_change": bool(p >= self.cutoff),
            "p_change": round(p, 3),
            "cutoff": round(self.cutoff, 3),
            "confidence_band": band,
            "historical_accuracy_at_this_confidence": self.reliability.get(band),
            "rule_used": rule,
            "shap": {f: round(float(v), 4) for f, v in top.items()},
            "grouped": {k: round(float(v), 4) for k, v in grouped(sv).head(4).items()},
            "reasons": reasons,
            "counterfactual": cf,
        }

    def frame(self, rows):
        """One Series or a DataFrame, always out as a float DataFrame of features."""
        if isinstance(rows, pd.Series):
            return rows[self.features].to_frame().T.astype(float)
        return rows[self.features].astype(float)


def grouped(sv):
    """SHAP values summed into the eight evidence families."""
    return pd.Series({k: sv[[f for f in v if f in sv.index]].sum()
                      for k, v in FAMILY.items()}).sort_values(key=abs, ascending=False)


def render(e, width=78):
    """A card as plain text. The same content the user study showed people."""
    L = ["=" * width, f"  REGIME AGENT  -  {e['date']}", "=" * width,
         f"  DECISION      : {e['decision']}     (today it is {e['regime_today']})",
         f"  Expects change: {'YES' if e['will_change'] else 'no'}  "
         f"({e['p_change']:.0%} vs a {e['cutoff']:.0%} threshold)",
         f"  Confidence    : {e['confidence_band'].upper()}"]
    if e["historical_accuracy_at_this_confidence"]:
        L.append(f"                  when this confident, historically correct "
                 f"{e['historical_accuracy_at_this_confidence']:.0%} of the time")
    L += [f"  Rule applied  : {e['rule_used']}", "", "  WHAT KIND OF EVIDENCE"]
    for k, v in e["grouped"].items():
        L.append(f"    {k:<12s} {v:+.4f}   "
                 f"{'toward CHANGE' if v > 0 else 'toward STABLE':<14s}  ({SAYS[k]})")
    L += ["", "  THE SPECIFICS"]
    for f, r in zip(e["shap"], e["reasons"]):
        L.append(f"    [{e['shap'][f]:+.3f}] {r}")
    if e["counterfactual"]:
        c = e["counterfactual"]
        L += ["", "  WHAT WOULD CHANGE THE ANSWER",
              f"    if {c['feature']} were {c['flip_at']:.3f} instead of "
              f"{c['current']:.3f} ({c['direction']}),",
              f"    the agent would report {c['new_answer']} instead."]
    L.append("=" * width)
    return "\n".join(L)


# ───────────────────────── the Sentiment Agent (nb 14) ─────────────────────────
# What the company actually does. Without this the model reads a crude rally as
# good news for a refiner, when for IOC crude is a cost.
STOCK_INFO = {
    "IOC.NS": "Indian Oil Corporation - India's largest oil refiner and fuel retailer. "
              "IMPORTANT: crude oil is a COST for a refiner, so a crude rally SQUEEZES margins "
              "and is bad news; falling crude is good news. Also sensitive to government fuel "
              "price controls and LPG subsidy policy.",
    "BHARTIARTL.NS": "Bharti Airtel - India's second largest telecom operator. Driven by "
                     "subscriber additions, ARPU (average revenue per user), tariff increases, "
                     "5G rollout costs and spectrum auction spending. Competition with Jio matters.",
    "ASHOKLEY.NS": "Ashok Leyland - commercial vehicle manufacturer (trucks and buses). "
                   "Cyclical: driven by infrastructure spending, freight demand, steel input "
                   "costs and interest rates. Sensitive to the economic cycle.",
    "ADANIPORTS.NS": "Adani Ports and SEZ - India's largest private port operator. Driven by "
                     "cargo volumes, trade activity and capacity expansion. Also exposed to "
                     "Adani group governance news, which can move the stock independently of "
                     "the port business.",
    "INDHOTEL.NS": "Indian Hotels Company (Taj Hotels) - luxury hotel chain. Driven by "
                   "occupancy rates, average room rates, tourism demand and business travel. "
                   "Seasonal, and sensitive to travel disruption.",
}

SENTIMENT_PROMPT = """You are the SENTIMENT AGENT in an explainable multi-agent stock trading system.
Read the recent news headlines for ONE stock and decide whether news sentiment points
UP or DOWN for this stock over roughly the next month.

Be honest and conservative:
- Only say UP or DOWN when the headlines clearly support it.
- If headlines are mixed, stale, or uninformative, answer NEUTRAL. NEUTRAL is a good answer.
- Think about what the news means for THIS company's profits.

Stock: {stock}
Business context: {context}

Recent headlines (newest first, all within the last {fresh_days} days):
{headlines}

Respond with ONLY this JSON (no other text):
{{
  "decision": "UP" or "DOWN" or "NEUTRAL",
  "confidence": <number 0.0-1.0>,
  "explanation": "<2-4 plain-English sentences a non-expert understands, citing the specific headlines that drove the decision>",
  "evidence": [
    {{"headline": "<exact headline>", "sentiment": "positive" or "negative" or "neutral", "why": "<one short clause>"}}
  ]
}}
Include only the 3-5 most influential headlines in "evidence"."""


def fresh_headlines(news, ticker, as_of, fresh_days=FRESH_DAYS, n=N_HEADLINES):
    """Headlines in the window before `as_of`, newest first. Never future ones.

    The window is the one correction that mattered most: reading everything ever
    published about a stock scored worse than reading only the last week.
    """
    as_of = pd.Timestamp(as_of)
    w = news[(news["ticker"] == ticker)
             & (news["published"] <= as_of)
             & (news["published"] > as_of - pd.Timedelta(days=fresh_days))]
    return w.sort_values("published", ascending=False).head(n)


def sentiment_agent(news, ticker, as_of, cache, live=False):
    """One stock on one day. Returns (decision, source).

    source is "cached", "abstained", "live" or "pending" - the last meaning the
    answer is not cached and `live` was not given, so nothing was called.
    """
    as_of = pd.Timestamp(as_of)
    key = f"{ticker}|{as_of.date()}"
    if key in cache:
        return cache[key], "cached"

    hl = fresh_headlines(news, ticker, as_of)

    # Abstaining needs no model, so it costs nothing. It is still cached, because
    # 75 of the cache's existing entries are abstentions and the rest of the
    # pipeline reads the cache as the record of which stock-days were covered.
    if len(hl) < MIN_FRESH:
        out = {"stock": ticker, "as_of": str(as_of.date()), "decision": "NEUTRAL",
               "confidence": 0.0,
               "explanation": f"Only {len(hl)} headline(s) in the last {FRESH_DAYS} days - "
                              f"not enough fresh news to form an honest view.",
               "evidence": [], "n_headlines_read": len(hl), "abstained": True}
        cache[key] = out
        return out, "abstained"

    if not live:
        return None, "pending"

    lines = "\n".join(f"- [{r.published:%Y-%m-%d}] {r.title}" for _, r in hl.iterrows())
    prompt = SENTIMENT_PROMPT.format(stock=ticker, context=STOCK_INFO[ticker],
                                     fresh_days=FRESH_DAYS, headlines=lines)
    try:
        out = call_llm(prompt)
    except Exception as e:
        return {"stock": ticker, "as_of": str(as_of.date()), "decision": "NEUTRAL",
                "confidence": 0.0,
                "explanation": f"Language model call failed: {type(e).__name__}",
                "evidence": [], "error": True}, f"error:{type(e).__name__}"

    out.update(stock=ticker, as_of=str(as_of.date()), n_headlines_read=len(hl),
               abstained=False,
               oldest_headline_age_days=int((as_of - hl["published"].min()).days))
    cache[key] = out
    return out, "live"


# ─────────────────────────── the Coordinator (nb 20) ───────────────────────────
COORDINATOR_PROMPT = '''You are the COORDINATOR in an explainable multi-agent stock trading system. You do not
decide the trade -- a rule-based Risk Agent already has. Your only job is to explain, in plain
English a non-technical reader can follow, WHY the system reached this decision, by weaving
together what each of the three agents found.

Stock: {stock}   Date: {date}

REGIME AGENT (market-wide conditions):
  today's regime: {regime_today}
  20-day forecast: {regime_decision} (change expected: {will_change}, {p_change:.0%} probability)
  reasons: {regime_reasons}

NEWS AGENT (this stock specifically):
  direction: {news_decision} (confidence {news_confidence:.2f})
  reasoning: {news_explanation}

RISK AGENT (already decided -- do not contradict this):
  verdict: {verdict}   action: {action}   position size: {size:.0%}
  rules that fired: {rules_fired}

Write a JSON object with exactly these two fields:
{{"synthesis": "<3-5 sentences, plain English, explaining why this action and size follow from
   the three agents above -- must be consistent with the Risk Agent's verdict, never suggest a
   different action>",
 "confidence_label": "<one of: High, Medium, Low -- how much the three agents agree with each other>"}}
Respond with ONLY the JSON object, no other text.'''


def regime_reasons(row, sv, k=3):
    """The k strongest SHAP features, phrased for a non-technical reader."""
    top = sv.reindex(sv.abs().sort_values(ascending=False).index).head(k)
    return [phrase(f, row[f]) for f in top.index]


def coordinator_prompt(regime, news, risk):
    return COORDINATOR_PROMPT.format(
        stock=risk["stock"], date=risk["date"],
        regime_today=regime["regime_today"], regime_decision=regime["decision"],
        will_change=regime["will_change"], p_change=regime["p_change"],
        regime_reasons="; ".join(regime["reasons"]),
        news_decision=news["decision"], news_confidence=float(news["confidence"]),
        news_explanation=news.get("explanation", ""),
        verdict=risk["verdict"], action=risk["action"], size=risk["position_size"],
        rules_fired=", ".join(f["rule"] for f in risk["rules_fired"]) or "none")


def cached_reply(cache, stock, date):
    """Look a cached narration up tolerantly.

    The cache grew two key schemas - "STOCK|DATE" and, for a while,
    "STOCK|DATE|ACTION|SIZE|VERDICT" - and two value shapes: a whole card, or
    just the language model's {synthesis, confidence_label} reply. All four
    combinations name a paid-for narration of the same day, so all four are
    honoured. Returns the reply, which build_card wraps the same way either way.
    """
    key = f"{stock}|{date}"
    hit = cache.get(key)
    if hit is None:
        hit = next((v for k, v in cache.items() if k.startswith(key + "|")), None)
    return hit


def build_card(regime, news, risk, reply):
    """Wrap a language-model reply in the agents' own records.

    The action, size and levels are copied from the Risk Agent here, after the
    call. "Never contradicts the Risk Agent" is therefore true by construction,
    not by a test that happens to pass.
    """
    return {
        "stock": risk["stock"], "date": risk["date"], "regime": regime,
        "news": {"decision": news["decision"], "confidence": news["confidence"],
                 "explanation": news.get("explanation", "")},
        "risk": {"verdict": risk["verdict"], "action": risk["action"],
                 "position_size": risk["position_size"],
                 "stop_loss_pct": risk["stop_loss_pct"],
                 "take_profit_pct": risk["take_profit_pct"],
                 "rules_fired": risk["rules_fired"]},
        "synthesis": reply["synthesis"], "confidence_label": reply["confidence_label"],
        "source": "REAL",
    }


def coordinate(regime, news, risk, cache, live=False):
    """Narrate one decision. Returns (card, source).

    The narration itself is cached; the surrounding record is rebuilt from the
    agents' current output every time, so a card can never drift out of step
    with the decision it describes. See build_card.
    """
    if news is None or news.get("decision") in (None, "?"):
        return None, "no_news"

    reply = cached_reply(cache, risk["stock"], risk["date"])
    if reply is not None:
        return build_card(regime, news, risk, reply), "cached"
    if not live:
        return None, "pending"

    try:
        reply = call_llm(coordinator_prompt(regime, news, risk))
    except Exception as e:
        return None, f"error:{type(e).__name__}"

    card = build_card(regime, news, risk, reply)
    cache[f'{risk["stock"]}|{risk["date"]}'] = card
    return card, "live"


def contradicts(card, risk):
    """Guard: a card's copied fields must still match the Risk Agent's record."""
    r = card["risk"]
    return (r["action"] != risk["action"]
            or abs(r["position_size"] - risk["position_size"]) > 1e-9)


# ──────────────────────── the one place that calls out ────────────────────────
def call_llm(prompt, model=LLM_MODEL, temperature=0.0, retries=3, gap_s=3.0):
    """Ask Gemini for JSON. Reached only when a caller passed live=True.

    Needs GEMINI_API_KEY in the environment; run.py loads .env before calling.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature))
            txt = re.sub(r"^```(?:json)?|```$", "", r.text.strip(), flags=re.M).strip()
            return json.loads(txt)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(gap_s)
