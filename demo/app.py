"""Explainable Trading Agent demo - four agents, one trading day at a time.

    Technical Agent   market-wide   "what kind of market is this?"
    Sentiment Agent   per stock     "which direction for THIS stock?"
    Risk Agent        per stock     "should we trade, and how big?"
    Coordinator       per stock     one paragraph a non-expert can follow

Run from the project root:

    streamlit run demo/app.py

Every number comes from the package: the frozen router is loaded, not retrained,
so the demo cannot disagree with `make router`. Nothing here calls out unless the
"Call the LLM for news" box is ticked.

The plain-English wording below (PLAIN_FAMILY, MEANING, WHY, SHORT_LABEL, PHRASE)
is this screen's own copy, deliberately gentler than the phrasing agents.py uses
for the printed cards. It is presentation, so it lives with the presentation.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents as A           # noqa: E402
import config as cfg         # noqa: E402
import risk as RA            # noqa: E402

STOCKS = cfg.STOCKS
THRESHOLD, SEALED_FROM = cfg.THRESHOLD, cfg.SEALED_FROM
FRESH_DAYS, MIN_FRESH, N_HEADLINES = cfg.FRESH_DAYS, cfg.MIN_FRESH, cfg.N_HEADLINES

st.set_page_config(page_title="Regime + News Demo", page_icon="📊", layout="wide")

st.markdown("""<style>
  .agent-hd{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;
            padding:.55rem .85rem;margin:-.4rem 0 .9rem;border-radius:6px;}
  .agent-ic{font-size:1.25rem;line-height:1;}
  .agent-nm{font-weight:800;font-size:1.02rem;letter-spacing:.04em;text-transform:uppercase;}
  .agent-rl{font-size:.83rem;opacity:.72;font-style:italic;}
  .agent-tag{margin-left:auto;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;
             opacity:.65;font-weight:600;}
  .ctx-hd{font-weight:700;font-size:.78rem;letter-spacing:.13em;text-transform:uppercase;
          opacity:.6;margin:.4rem 0 .5rem;}
</style>""", unsafe_allow_html=True)

REGIME_C, NEWS_C, RISK_C, COORD_C = "#3D7BE0", "#D08B2C", "#2E9E5B", "#8B5CF6"


def agent_header(icon, name, role, color, tag=""):
    st.markdown(
        f'<div class="agent-hd" style="border-left:5px solid {color};background:{color}1A">'
        f'<span class="agent-ic">{icon}</span>'
        f'<span class="agent-nm" style="color:{color}">{name}</span>'
        f'<span class="agent-rl">{role}</span>'
        f'<span class="agent-tag">{tag}</span></div>', unsafe_allow_html=True)


# ─────────────────────────── loading ───────────────────────────
@st.cache_resource
def load_everything():
    """Load the data and the frozen router. No training happens here.

    The router on disk was fitted by `run.py router` on everything up to
    2024-12-31 and saved. Loading it rather than refitting is what guarantees
    this screen shows the same system the thesis reports.
    """
    if not cfg.ROUTER.exists():
        st.error("The router has not been trained yet. Run:  python run.py router")
        st.stop()

    data = pd.read_csv(cfg.DATASET, parse_dates=["Date"]).set_index("Date")
    panel = pd.read_csv(cfg.PANEL, parse_dates=["Date"]).set_index("Date")
    news = pd.read_csv(cfg.NEWS, parse_dates=["published"])

    settings = pd.read_csv(cfg.ROUTER_SETTINGS, header=None, index_col=0)[1]
    cutoff = float(settings["cutoff"])
    reliability = (pd.read_csv(cfg.RELIABILITY_BANDS).set_index("band")["accuracy"].to_dict()
                   if cfg.RELIABILITY_BANDS.exists() else {})

    dev = data.loc[data.index < SEALED_FROM]
    agent = A.RegimeAgent(A.load_router(cfg.ROUTER), dev, cfg.chosen_features(),
                          cutoff, reliability)
    return data, panel, news, agent


DATA, PANEL, NEWS, AGENT = load_everything()
FEATURES, DEV, CUTOFF = AGENT.features, AGENT.dist, AGENT.cutoff
RELIABILITY = AGENT.reliability
FAMILY = A.FAMILY
level = A.level


# ─────────────────────────── the agents, wired to the package ───────────────────────────
def regime_agent(row, overrides=None):
    """The Technical Agent's answer, in the shape this screen draws.

    `overrides` is the what-if slider: it changes what the router sees, but the
    arithmetic branch still reads the real row, because the slider is asking
    "would the router still route the same way?" and not "what if the market had
    been different?".
    """
    x = AGENT.frame(row)
    for k, v in (overrides or {}).items():
        x[k] = v

    p = float(AGENT.router.predict_proba(x[FEATURES])[:, 1][0])
    change = p >= CUTOFF
    decision = A.mechanical_one(row) if change else row["regime_now"]
    sv = pd.Series(AGENT.explainer.shap_values(x)[0], index=FEATURES)
    m = AGENT.margin(p)
    return dict(p=p, change=change, decision=decision, shap=sv,
                margin=m, band=A.band_of(m))


def counterfactual(row, feature):
    return AGENT.counterfactual(row, feature, n=50)


def _read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def news_agent(ticker, as_of, live=False):
    """The Sentiment Agent, with the cache read and written around it."""
    cache = _read(cfg.SENTIMENT_CACHE)
    before = len(cache)
    out, how = A.sentiment_agent(NEWS, ticker, as_of, cache, live=live)
    if len(cache) != before:
        _write(cfg.SENTIMENT_CACHE, cache)

    if out is None:            # not cached and not allowed to call out
        n = len(A.fresh_headlines(NEWS, ticker, as_of))
        return dict(stock=ticker, decision="?", confidence=0.0, n_headlines_read=n,
                    evidence=[], abstained=False,
                    explanation="Not yet analysed - tick 'call the LLM' to run it."), "pending"
    return out, how


def regime_block(reg, row):
    """Reshape this screen's regime answer into the Coordinator's contract."""
    return {"decision": reg["decision"], "regime_today": row["regime_now"],
            "will_change": bool(reg["change"]), "p_change": round(reg["p"], 3),
            "reasons": A.regime_reasons(row, reg["shap"])}


def coordinate(reg_block, news, risk, live=False):
    cache = _read(cfg.COORDINATOR_CACHE)
    before = len(cache)
    card, source = A.coordinate(reg_block, news, risk, cache, live=live)
    if len(cache) != before:
        _write(cfg.COORDINATOR_CACHE, cache)
    return card, source

# ─────────────────── this screen's own plain-English wording ───────────────────
# plain-English headers for the evidence rows - what the group is actually measuring,
# so a reader never has to know the words "turbulence", "breadth" or "regime"
PLAIN_FAMILY = {
    "trend": "Which way the market has been heading",
    "fear": "How nervous investors are",
    "turbulence": "How jumpy prices have been",
    "stress": "How far the market is below its peak",
    "agreement": "Whether the 5 stocks are moving together",
    "regime age": "How long this mood has already lasted",
    "fragility": "How close we are to tipping into another mood",
    "momentum": "Short-term buying pressure",
}

# plain definitions, for the expandable rows - what the word means, and why it predicts anything
MEANING = {
    "trend": "Which way the market has been heading, and how far it has travelled - measured over "
             "a week, a month, three months and a hundred days, plus how today's prices compare "
             "with their own long-run averages.",
    "fear": "How nervous investors are. This comes from the India VIX, an index that rises when "
            "traders pay more to protect themselves against a fall.",
    "turbulence": "How much prices jump around from one day to the next - the size of the daily "
                  "swings, regardless of direction.",
    "stress": "How far the market has fallen from its best-ever point, and how long ago that peak "
              "was.",
    "agreement": "Whether the five stocks are moving as one group or going their own separate ways.",
    "regime age": "How long the current market mood has already been running, and which mood it is.",
    "fragility": "How close the market is to the line that separates one regime from another.",
    "momentum": "Whether short-term buying pressure is building or fading.",
}
WHY = {
    "trend": "A strong, well-established trend tends to keep going. One that is flattening out often "
             "comes before the mood changes.",
    "fear": "Calm markets usually stay in the same mood. Sudden spikes in fear tend to arrive around "
            "turning points.",
    "turbulence": "Calm markets tend to stay calm. A sudden burst of jumpiness often shows up just "
                  "before the mood shifts.",
    "stress": "A market that is a long way below its peak, and has been for a long time, is under "
              "pressure and more likely to shift.",
    "agreement": "When all five stocks move together it usually means one big force is pushing "
                 "everything at once - often a sign of stress. When they drift apart, conditions "
                 "are settled.",
    "regime age": "Market moods do not last forever. The longer one has already run, the more "
                  "likely it is to be near its end.",
    "fragility": "If the market is sitting right on the boundary, even an ordinary month can tip it "
                 "into a different regime. Far from the line, the answer is effectively locked in.",
    "momentum": "A fast-moving but noisy signal - it reacts quickly, but it is wrong often enough "
                "that the model leans on it only lightly.",
}
# short plain-English names for the SHAP bar chart, so the axis is not raw column names
SHORT_LABEL = {
    "days_in_regime": "how old this regime is", "days_since_high": "days since the last high",
    "vix_level": "the fear index", "vix_chg_20": "change in fear",
    "avg_correlation": "stocks moving together", "breadth_above_sma200": "stocks above trend",
    "ret_5": "this week's move", "ret_20": "last month's move", "ret_60": "3-month move",
    "ret_100": "100-day move", "px_vs_sma50": "price vs 50-day average",
    "px_vs_sma200": "price vs 200-day average", "sma50_vs_sma200": "short vs long average",
    "drawdown": "how far below the peak", "vol_20": "recent choppiness",
    "vol_60": "2-month choppiness", "bb_width": "trading range width",
    "flip_margin": "closeness to a boundary", "rsi_14": "momentum",
    "macd_hist": "trend momentum", "adx_14": "trend strength", "trend_r2_60": "how straight the trend is",
    "in_bull": "currently bull", "in_bear": "currently bear", "in_sideways": "currently sideways",
}
PHRASE = {
    "flip_margin": lambda v: f"the market sits {v:.1f} typical monthly moves away from tipping into another regime",
    "ret_100": lambda v: f"the market has moved {v*100:+.1f}% over the last 100 days",
    "ret_60": lambda v: f"the market has moved {v*100:+.1f}% over the last 3 months",
    "ret_20": lambda v: f"the market has moved {v*100:+.1f}% over the last month",
    "ret_5": lambda v: f"the market has moved {v*100:+.1f}% this week",
    "px_vs_sma200": lambda v: f"prices are {v*100:+.1f}% away from their long-term (200-day) average",
    "px_vs_sma50": lambda v: f"prices are {v*100:+.1f}% away from their recent (50-day) average",
    "sma50_vs_sma200": lambda v: f"the recent average is {v*100:+.1f}% away from the long-term one"
                                 + (" (short-term strength)" if v > 0 else " (short-term weakness)"),
    "vix_level": lambda v: f"the market's fear gauge reads {v:.1f}"
                           + (" - investors are nervous" if v > 20 else " - investors are relaxed"),
    "vix_chg_20": lambda v: f"fear has {'risen' if v > 0 else 'fallen'} {abs(v)*100:.0f}% in the last month",
    "vol_20": lambda v: f"prices have been {'jumpy' if v*100 > 1.2 else 'calm'} lately, "
                        f"moving about {v*100:.2f}% on an average day",
    "vol_60": lambda v: f"over three months prices have been {'jumpy' if v*100 > 1.2 else 'calm'}, "
                        f"moving about {v*100:.2f}% on an average day",
    "bb_width": lambda v: f"prices are swinging in a {'wide' if v*100 > 8 else 'narrow'} band "
                          f"({v*100:.1f}% from top to bottom)",
    "drawdown": lambda v: f"the market is {abs(v)*100:.1f}% below its highest point",
    "days_since_high": lambda v: f"it has been {int(v)} days since the market last hit a new high",
    "breadth_above_sma200": lambda v: f"{int(round(v*5))} of the 5 stocks are above their own long-term average",
    "avg_correlation": lambda v: f"the 5 stocks are moving {'closely together' if v > 0.4 else 'fairly independently'} "
                                 f"(correlation {v:.2f})",
    "adx_14": lambda v: f"the market has a {'strong' if v > 25 else 'weak'} sense of direction (strength {v:.0f})",
    "trend_r2_60": lambda v: f"the recent trend has been {v:.0%} steady rather than zig-zagging",
    "rsi_14": lambda v: f"short-term momentum is {'hot' if v > 60 else 'cold' if v < 40 else 'neutral'} ({v:.0f} out of 100)",
    "macd_hist": lambda v: f"momentum is {'building' if v > 0 else 'fading'}",
    "days_in_regime": lambda v: f"the current market mood has lasted {int(v)} days so far",
    "in_bull": lambda v: "the market is currently rising (bull)" if v else "the market is not currently rising",
    "in_bear": lambda v: "the market is currently falling (bear)" if v else "the market is not currently falling",
    "in_sideways": lambda v: "the market is currently drifting sideways" if v else "the market is not drifting sideways",
}


# ─────────────────────────── UI ───────────────────────────
st.title("📊 Explainable Trading Agent Demo")
st.caption("Four agents. Pick any trading day and see exactly why the system decided what it did.")

sealed = DATA[DATA.index >= SEALED_FROM]
dates = list(sealed.index)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    picked = st.select_slider("Trading day", options=dates,
                              value=dates[len(dates) // 2],
                              format_func=lambda d: d.strftime("%d %b %Y"))
with c2:
    live = st.checkbox("Call the LLM for news", value=False,
                       help="Uncached dates need a Gemini call (a few seconds per stock).")
with c3:
    st.metric("Days available", f"{len(dates)}")

row = sealed.loc[picked]
reg = regime_agent(row)

# what actually happened 20 trading days later (we have the data - so we can mark the homework)
_pos = DATA.index.get_indexer([picked])[0]
_future_date = DATA.index[_pos + 20] if _pos + 20 < len(DATA) else None
_truth = row["target"] if pd.notna(row.get("target")) else None

st.markdown('<div class="ctx-hd">The market - context for both agents</div>', unsafe_allow_html=True)

# --- price chart + the window the label actually looks at ---
COL = {"BULL": "#2ca02c", "SIDEWAYS": "#ff9e2c", "BEAR": "#d62728"}
win = DATA.loc[picked - pd.Timedelta(days=400): picked + pd.Timedelta(days=120)]

pos = DATA.index.get_indexer([picked])[0]

# The 120 days that decide the FUTURE label (target, 20 trading days ahead): 100 already known
# + 20 not yet known. Falls back to an estimated future date near the end of the sealed period,
# where the real 20-days-ahead row does not exist yet.
future_pos = pos + 20
if future_pos < len(DATA):
    future_date = DATA.index[future_pos]
else:
    avg_gap = (DATA.index[-1] - DATA.index[0]) / (len(DATA) - 1)
    future_date = picked + avg_gap * 20
decomp_start = DATA.index[max(0, future_pos - 119)]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 4.6), sharex=False,
                              gridspec_kw={"height_ratios": [3, 1.15]})

ep = (win["regime_now"] != win["regime_now"].shift()).cumsum()
for _, g in win["regime_now"].groupby(ep):
    ax.axvspan(g.index[0], g.index[-1], color=COL[g.iloc[0]], alpha=.18, lw=0)
px = PANEL.loc[win.index, "NIFTY_Close"]
ax.plot(px.index, px.values, color="black", lw=1.1, zorder=5)

lo, hi = px.min(), px.max()
span = hi - lo

# known half: decomp_start -> picked
ax.add_patch(plt.Rectangle((decomp_start, lo), picked - decomp_start, hi - lo,
                           fill=True, facecolor="#0047AB", alpha=.10,
                           edgecolor="#0047AB", lw=2.2, zorder=4))
# unknown half: picked -> future_date
ax.add_patch(plt.Rectangle((picked, lo), future_date - picked, hi - lo,
                           fill=False, edgecolor="#7a7a7a", lw=2.0, hatch="////", zorder=4))

ax.annotate("", xy=(future_date, hi), xytext=(decomp_start, hi),
            arrowprops=dict(arrowstyle="<->", color="#0047AB", lw=1.8))
ax.text(decomp_start + (future_date - decomp_start) / 2, hi,
        "  the 120 days that decide the answer - 100 known + 20 unknown  ",
        color="#0047AB", fontsize=9, fontweight="bold", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#0047AB", lw=1))

ax.text(decomp_start + (picked - decomp_start) / 2, lo + span * 0.03, "100 known days",
        fontsize=8, fontweight="bold", ha="center", va="bottom", color="#0b2e63")
ax.text(picked + (future_date - picked) / 2, lo + span * 0.03, "20 unknown",
        fontsize=8, fontweight="bold", ha="center", va="bottom", color="#555555")

ax.set_ylim(lo - span * 0.04, hi + span * 0.16)

ax.plot([picked, picked], [lo, hi], color="black", ls="--", lw=1.6, zorder=6)
ax.scatter([picked], [PANEL.loc[picked, "NIFTY_Close"]], s=80, color="black", zorder=7)
ax.set_title(f"NIFTY 50 around {picked:%d %b %Y}   (green BULL · orange SIDEWAYS · red BEAR)",
             fontsize=11, fontweight="bold")
ax.grid(alpha=.25)

# --- what each window length would say ---
windows = [("5 days", "ret_5"), ("20 days", "ret_20"), ("60 days", "ret_60"),
           ("100 days", "ret_100"), ("120 days ← the rule", "ret_120")]
vals = [row[c] * 100 for _, c in windows]
cols = ["#2ca02c" if v > 6 else "#d62728" if v < -6 else "#ff9e2c" for v in vals]
cols[-1] = "#0047AB"
y = np.arange(len(windows))
ax2.barh(y, vals, color=cols, height=.62)
for yi, v in zip(y, vals):
    says = "BULL" if v > 6 else "BEAR" if v < -6 else "SIDEWAYS"
    ax2.text(v + (0.4 if v >= 0 else -0.4), yi, f"{v:+.1f}%  → {says}",
             va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
ax2.axvline(6, color="#2ca02c", ls="--", lw=1.3)
ax2.axvline(-6, color="#d62728", ls="--", lw=1.3)
ax2.axvline(0, color="black", lw=.8)
ax2.text(6, len(windows) - .35, " +6% = BULL", color="#2ca02c", fontsize=8, fontweight="bold")
ax2.text(-6, len(windows) - .35, "-6% = BEAR ", color="#d62728", fontsize=8,
         fontweight="bold", ha="right")
ax2.set_yticks(y)
ax2.set_yticklabels([w for w, _ in windows], fontsize=9)
ax2.set_xlim(min(min(vals) * 1.7, -9), max(max(vals) * 1.7, 9))
ax2.set_title("What each lookback length would say", fontsize=10, fontweight="bold")
ax2.grid(axis="x", alpha=.25)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# --- diagnose WHY the label is what it is: two different lags ---
def raw_label(pct):
    return "BULL" if pct > 6 else "BEAR" if pct < -6 else "SIDEWAYS"

raw_now = raw_label(row["ret_120"] * 100)          # what the threshold says right now
short_says = raw_label(row["ret_60"] * 100)        # what a shorter window would say
label = row["regime_now"]                          # the CONFIRMED label actually used

if raw_now != label:
    # cause 1: the threshold has already flipped, the 10-day confirmation has not accepted it
    st.warning(
        f"⏳ **Waiting for confirmation.** The 120-day return is "
        f"**{row['ret_120']*100:+.1f}%**, which on its own reads **{raw_now}**. But a new regime "
        f"is only accepted after the rule agrees with itself for **10 days running**, so the label "
        f"is still **{label}**. This filter exists to stop the label flickering - the cost is that "
        f"it accepts a genuine change about 10 days late.")
elif short_says != label:
    # cause 2: the windows themselves disagree - old conditions still inside the 120 days
    st.warning(
        f"⏳ **The window is lagging.** The last 60 days read **{short_says}**, but the rule uses "
        f"120 days - which still contains older, different conditions - so the label is "
        f"**{label}**. A backward-looking label recognises a regime only once it has been "
        f"running long enough to dominate its own window.")

# ---------------- regime panel ----------------
with st.container(border=True):
    agent_header('🧭', 'Technical Agent', 'what kind of market is this?', REGIME_C,
                 'market-wide')
    a, b, c = st.columns(3)
    a.metric("Decision", reg["decision"])
    b.metric("Change expected", "YES" if reg["change"] else "no",
             f"{reg['p']:.0%} vs {CUTOFF:.0%}")
    acc = RELIABILITY.get(reg["band"])
    c.metric("Confidence", reg["band"].title(),
             f"right {acc:.0%} historically" if acc else None)

    # --- the answer, explained as the rule trace that actually produced it ---
    # (SHAP below explains only the "will it change?" step; the final answer comes from a
    #  deterministic rule, so it has a trace rather than feature attributions)
    _r100 = row["ret_100"] * 100
    if reg["change"]:
        _why = (f"I say **{reg['decision']}** because I **do** expect the market to change in the "
                f"next 20 days ({reg['p']:.0%} chance, and I act above {CUTOFF:.0%}). "
                f"So I stop copying today and do the sum instead: the market has moved "
                f"**{_r100:+.1f}%** over the last 100 days, which is "
                + ("**above the +6% line → BULL**." if _r100 > 6 else
                   "**below the −6% line → BEAR**." if _r100 < -6 else
                   "**between −6% and +6% → SIDEWAYS**."))
    else:
        _why = (f"I say **{reg['decision']}** because I do **not** expect the market to change in "
                f"the next 20 days (only a {reg['p']:.0%} chance, and I act above {CUTOFF:.0%}). "
                f"So I carry today's **{row['regime_now']}** forward unchanged.")
    st.info(_why)
    st.caption("That is the whole answer. Everything below explains just one part of it - "
               "how the system arrived at that chance-of-change figure.")

    sv = reg["shap"]
    grouped = pd.Series({k: sv[[f for f in v if f in sv.index]].sum()
                         for k, v in FAMILY.items()}).sort_values(key=abs, ascending=False)
    def strength(v):
        """How loudly this piece of evidence is speaking, in words rather than log-odds."""
        a = abs(v)
        return "very strong" if a >= 1.0 else "strong" if a >= 0.5 else \
               "moderate" if a >= 0.2 else "weak"

    def points_to(v):
        """What this evidence is arguing for, in words a reader can act on."""
        return "the mood to CHANGE" if v > 0 else "today's mood to CONTINUE"

    def rarity(pct):
        """Say how unusual a value is, counting in whichever direction reads naturally."""
        if pct >= 0.5:
            return f"higher than {pct:.0%} of days in the past"
        return f"lower than {1 - pct:.0%} of days in the past"

    st.markdown("**The four things that mattered most**")
    st.caption("“Mood” means whether the market is rising (BULL), falling (BEAR) or drifting "
               "(SIDEWAYS). Click any row to see what it means and what it is saying today.")
    for k, v in grouped.head(4).items():
        arrow = "🔺" if v > 0 else "🔻"
        with st.expander(f"{arrow}  **{PLAIN_FAMILY[k]}** - a {strength(v)} reason to expect "
                         f"{points_to(v)}"):
            st.markdown(f"**What it means.**  {MEANING[k]}")
            st.markdown(f"**Why it predicts anything.**  {WHY[k]}")
            st.markdown("**What it is saying today**")
            members = [f for f in FAMILY[k] if f in sv.index and abs(sv[f]) >= 0.005]
            members.sort(key=lambda f: -abs(sv[f]))
            if members:
                for f in members:
                    txt = PHRASE.get(f, lambda x: f"{f} = {x:.3f}")(row[f])
                    st.markdown(f"- {txt.capitalize()}  \n"
                                f"  <span style='opacity:.5'>a {strength(sv[f])} reason to expect "
                                f"{points_to(sv[f]).lower()} ({sv[f]:+.2f})</span>",
                                unsafe_allow_html=True)
            else:
                st.markdown("- *Nothing in this group moved today's decision noticeably.*")
            st.caption(f"These add up to {v:+.2f} - a {strength(v)} reason to expect "
                       f"{points_to(v).lower()}. Bigger numbers mean a stronger push. "
                       f"(Technical name for this group: “{k}”.)")

    st.markdown("**The individual facts behind those four**")
    top = sv.reindex(sv.abs().sort_values(ascending=False).index).head(4)
    for f, v in top.items():
        pct = float((DEV[f] <= row[f]).mean())
        txt = PHRASE.get(f, lambda x: f"{f} = {x:.3f}")(row[f])
        tail = "" if f.startswith("in_") else f" - {rarity(pct)}"
        st.markdown(f"- {txt.capitalize()}{tail}.  \n"
                    f"  <span style='opacity:.5'>A {strength(v)} reason to expect "
                    f"{points_to(v).lower()} ({v:+.2f})</span>",
                    unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(11, 2.9))
    t = top[::-1]
    ax2.barh(range(len(t)), t.values,
             color=["#d62728" if v > 0 else "#4c72b0" for v in t.values], height=.62)
    ax2.axvline(0, color="black", lw=.9)
    ax2.set_yticks(range(len(t)))
    # plain-English labels, with the raw feature name kept small in brackets
    ax2.set_yticklabels(
        [f"{SHORT_LABEL.get(f, f)}\n({f})" for f in t.index], fontsize=7.5)
    ax2.set_xlabel("← stays the same        how strongly it pushes        a change comes →",
                   fontsize=7.5)
    ax2.tick_params(labelsize=8)
    ax2.grid(axis="x", alpha=.3)
    st.pyplot(fig2, use_container_width=True)
    st.caption("Bars are SHAP values - how much each fact moved this particular decision. "
               "Red pushes toward a change, blue toward staying the same.")

    st.markdown("**🔀 What would have to be different?**")
    st.caption("The system's answer rests on the numbers below. Pick one and see how far it would "
               "have to move before the answer flips.")

    def fmt_val(f, v):
        """Show a value the way a person would say it, not the way it is stored."""
        if f in ("days_in_regime", "days_since_high"):
            return f"{v:.0f} days"
        if f == "breadth_above_sma200":
            return f"{v*5:.0f} of the 5 stocks"
        if f in ("vix_level", "adx_14", "rsi_14"):
            return f"{v:.1f}"
        if f == "avg_correlation":
            return f"{v:.2f}"
        if f.startswith(("ret_", "px_vs", "sma50", "vol_", "bb_", "drawdown", "vix_chg", "trend_r2")):
            return f"{v*100:+.1f}%"
        return f"{v:.2f}"

    cf_feat = st.selectbox("Which number do you want to test?", top.index.tolist(),
                           format_func=lambda f: SHORT_LABEL.get(f, f), key="cf")
    nice = SHORT_LABEL.get(cf_feat, cf_feat)
    cf = counterfactual(row, cf_feat)
    if cf:
        new_answer = "expect a CHANGE" if cf["new_answer"] == "a change" else "expect things to STAY THE SAME"
        st.info(f"Right now, **{nice}** is **{fmt_val(cf_feat, cf['current'])}**.  \n"
                f"If it were **{fmt_val(cf_feat, cf['flip_at'])}** instead "
                f"({cf['direction']}), the system would **{new_answer}**.")
    else:
        st.info(f"Nothing you could set **{nice}** to would change the answer - "
                f"today's decision does not rest on it.")

    lo, hi = float(DEV[cf_feat].quantile(.01)), float(DEV[cf_feat].quantile(.99))
    val = st.slider(f"Try it yourself - drag {nice}", lo, hi, float(row[cf_feat]),
                    (hi - lo) / 100, key="slider",
                    format="%.2f")
    st.caption(f"Now set to **{fmt_val(cf_feat, val)}** "
               f"(really it is {fmt_val(cf_feat, float(row[cf_feat]))})")
    alt = regime_agent(row, overrides={cf_feat: val})
    if alt["change"] != reg["change"]:
        st.success(f"⇢ **The answer flipped.** With {nice} at {fmt_val(cf_feat, val)}, the system "
                   f"would say **{alt['decision']}** - it now "
                   f"{'expects a change' if alt['change'] else 'expects no change'} "
                   f"({alt['p']:.0%} chance vs the {CUTOFF:.0%} line).")
    else:
        st.caption(f"Answer unchanged - still **{alt['decision']}**, with a {alt['p']:.0%} chance "
                   f"of a change (the line is {CUTOFF:.0%}).")

# ---------------- news panel ----------------
with st.container(border=True):
    agent_header('📰', 'Sentiment Agent', 'which direction for this stock?', NEWS_C,
                 f'per stock · {FRESH_DAYS}-day window')

    results = {}
    for s in STOCKS:
        t = f"{s}.NS"
        out, how = news_agent(t, picked, live=live)
        results[s] = out
        icon = {"UP": "🟢", "DOWN": "🔴", "NEUTRAL": "⚪", "?": "⏳"}[out["decision"]]
        badge = {"cached": "", "live": " ·  fresh call", "abstained": " ·  abstained",
                 "pending": " ·  not analysed", "error": " ·  error"}[how]
        with st.expander(f"{icon} **{s}** - {out['decision']} "
                         f"({out['confidence']:.2f}) · {out['n_headlines_read']} headlines{badge}"):
            st.write(out["explanation"])
            for e in out.get("evidence", [])[:3]:
                st.markdown(f"- *[{e.get('sentiment','?')}]* “{e.get('headline','')}” - "
                            f"{e.get('why','')}")

st.write("")
with st.container(border=True):
    agent_header('🛡️', 'Risk Agent', 'should we trade, and how big?', RISK_C,
                 'seven rules · consumes both agents above')

    st.caption("This agent never picks a direction - the Sentiment Agent already did. It only decides "
               "**whether the trade is worth taking, and how much money to put in.**")

    opening = st.slider("How much of our money is already tied up in other trades?", 0, 100, 0, 2,
                        key="exposure", format="%d%%",
                        help="Start at 0% for an empty portfolio. Drag it up and watch trades get "
                             "trimmed or refused as the money runs out.") / 100

    reg_block = regime_block(reg, row)
    verdicts = RA.evaluate(STOCKS, results, reg_block, PANEL, picked, opening_exposure=opening)
    ICON = {"APPROVE": "🟢", "SHRINK": "🟡", "REJECT": "⚪"}
    PLAIN_VERDICT = {"APPROVE": "Full size", "SHRINK": "Smaller size", "REJECT": "No trade"}
    PLAIN_ACTION = {"BUY": "Buy", "SELL": "Sell", "NONE": "-"}

    st.dataframe(pd.DataFrame([{
        "Stock": s_,
        "News says": results[s_]["decision"],
        "Market is": reg["decision"],
        "Moves per day": f'{verdicts[s_]["realised_vol"]:.2f}%' if verdicts[s_]["realised_vol"] else "-",
        "Decision": PLAIN_VERDICT[verdicts[s_]["verdict"]],
        "Buy or sell": PLAIN_ACTION[verdicts[s_]["action"]],
        "How much": f'{verdicts[s_]["position_size"]:.0%}',
        "Sell if it drops": f'{verdicts[s_]["stop_loss_pct"]:.1f}%' if verdicts[s_]["stop_loss_pct"] else "-",
        "Sell if it gains": f'{verdicts[s_]["take_profit_pct"]:.1f}%' if verdicts[s_]["take_profit_pct"] else "-",
    } for s_ in STOCKS]), hide_index=True, use_container_width=True)
    st.caption(f'**“How much”** is a share of what this stock would normally get, not of everything '
               f'we own.  ·  After today we would have '
               f'**{verdicts[STOCKS[0]]["book_after"]:.0%}** of our money invested, and we never '
               f'go past {RA.MAX_EXPOSURE:.0%}.')

    st.markdown("**Why it decided that**")
    pick = st.selectbox("Pick a stock to see the reasoning", STOCKS, key="why")
    v = verdicts[pick]
    st.markdown(f'{ICON[v["verdict"]]} **{PLAIN_VERDICT[v["verdict"]]}** - {RA.one_liner(v)}')
    for f in v["rules_fired"]:
        st.markdown(f'- {f.get("plain", f["detail"])}  \n'
                    f'  <span style="opacity:.45">rule {f["rule"].split("_")[0]} · {f["detail"]}</span>',
                    unsafe_allow_html=True)

    # ---- did the stop-loss / take-profit actually work? we know: it is all in the past ----
    oc = RA.trade_outcome(PANEL, pick, picked, v["action"],
                          v["stop_loss_pct"], v["take_profit_pct"])
    if oc is not None:
        st.markdown("**Did those two exit levels actually work?**")
        VERD = {
            "TARGET": ("🟢", f'**Hit the target** after {oc["days_taken"]} trading days - the trade '
                             f'would have been closed for a win.'),
            "STOP":   ("🔴", f'**Hit the stop-loss** after {oc["days_taken"]} trading days - the trade '
                             f'would have been closed to cut the loss.'),
            "OPEN":   ("⚪", f'**Neither level was reached** in {oc["days_taken"]} trading days. The '
                             f'price stayed between them the whole time, so the trade would still be open.'),
        }
        icon, verdict_txt = VERD[oc["result"]]
        st.markdown(f'{icon} {verdict_txt}')

        path, entry = oc["path"], oc["entry"]
        rel = (path / entry - 1) * 100          # % move from entry, day by day
        xs = np.arange(1, len(rel) + 1)
        up_is_good = v["action"] == "BUY"
        tgt_line = v["take_profit_pct"] if up_is_good else -v["take_profit_pct"]
        stop_line = -v["stop_loss_pct"] if up_is_good else v["stop_loss_pct"]

        figt, axt = plt.subplots(figsize=(11, 3.1))
        axt.axhspan(min(stop_line, tgt_line), max(stop_line, tgt_line), color="#9e9e9e", alpha=.07)
        axt.axhline(0, color="black", lw=1.1)
        axt.axhline(tgt_line, color="#2ca02c", ls="--", lw=1.6)
        axt.axhline(stop_line, color="#d62728", ls="--", lw=1.6)
        axt.plot(xs, rel.values, color="#0047AB", lw=1.9, marker="o", ms=3)

        if oc["result"] in ("TARGET", "STOP"):
            hit_x = oc["days_taken"]
            axt.scatter([hit_x], [rel.iloc[hit_x - 1]], s=150, zorder=6,
                        color="#2ca02c" if oc["result"] == "TARGET" else "#d62728",
                        edgecolor="white", lw=1.6)

        pad = max(abs(tgt_line), abs(stop_line)) * 0.35
        axt.set_ylim(min(stop_line, tgt_line) - pad, max(stop_line, tgt_line) + pad)
        axt.text(0.4, tgt_line, f"  take-profit  +{v['take_profit_pct']:.1f}%", color="#2ca02c",
                 fontsize=8.5, fontweight="bold", va="bottom")
        axt.text(0.4, stop_line, f"  stop-loss  −{v['stop_loss_pct']:.1f}%", color="#d62728",
                 fontsize=8.5, fontweight="bold", va="top")
        axt.set_xlabel("trading days after the decision", fontsize=9)
        axt.set_ylabel("price move\nsince entry (%)", fontsize=9)
        axt.set_title(f"What {pick} actually did after {picked:%d %b %Y}"
                      f"   ({'buying' if up_is_good else 'selling'} - "
                      f"{'up' if up_is_good else 'down'} is good)",
                      fontsize=10, fontweight="bold")
        axt.grid(alpha=.25)
        plt.tight_layout()
        st.pyplot(figt, use_container_width=True)
        st.caption(f'After {oc["days_taken"]} trading days the position was '
                   f'**{oc["pnl_pct"]:+.2f}%** '
                   f'{"up" if oc["pnl_pct"] >= 0 else "down"}. '
                   "This is real price history - every date in this demo is in the past, so the "
                   "outcome of the exit rules is already known and can be checked rather than assumed.")

st.write("")
with st.container(border=True):
    agent_header('🎯', 'Coordinator', 'one paragraph a non-expert can follow', COORD_C,
                 'narrates · never re-decides')

    card, src = coordinate(reg_block, results[pick], verdicts[pick], live=live)
    badge = {"cached": "from cache", "live": "fresh call", "pending": "not yet narrated",
             "no_news": "no news to narrate"}.get(src, src)

    if card:
        bad = A.contradicts(card, verdicts[pick])
        v_ = verdicts[pick]
        n_ = results[pick]

        st.caption(f"Everything below is about **{pick}** on **{picked:%d %b %Y}**. "
                   "The three agents each answered one question; this is how their answers combined.")

        # --- the chain: what each agent said, in the order they were asked ---
        AGREE_MEANS = {
            "High": "nothing pulled against the decision - it was clear-cut",
            "Medium": "the agents partly pulled in different directions, so the trade was "
                      "reduced or held back",
            "Low": "the agents contradicted each other",
        }
        # The market line should give REASONS, not the router's probability. The SHAP values
        # explain "will it change?", so cite the facts pushing the SAME way as that verdict -
        # those are genuinely why the market call came out as it did.
        sv_ = reg["shap"]
        want_positive = reg["change"]          # change expected -> cite what pushed toward change
        aligned = [f for f in sv_.index
                   if (sv_[f] > 0) == want_positive and abs(sv_[f]) >= 0.05]
        aligned.sort(key=lambda f: -abs(sv_[f]))
        # one fact per family, so three near-duplicates (vol_20/vol_60/bb_width) don't all appear
        FAM_OF = {f: fam for fam, feats in FAMILY.items() for f in feats}
        seen_fam, picked_feats = set(), []
        for f in aligned:
            fam = FAM_OF.get(f, f)
            if fam in seen_fam:
                continue
            seen_fam.add(fam)
            picked_feats.append(f)
            if len(picked_feats) == 3:
                break
        reasons = [PHRASE.get(f, lambda x: f"{f} = {x:.2f}")(row[f]) for f in picked_feats]
        if reasons:
            reason_txt = ", ".join(reasons[:-1]) + (" and " if len(reasons) > 1 else "") + reasons[-1]
        else:
            reason_txt = "no single fact stood out"

        market_line = (f'**{reg["decision"]}**, and it looks like **changing soon** - because '
                       if reg["change"] else
                       f'**{reg["decision"]}**, and it looks like **staying that way** - because ')
        market_line += reason_txt + "."

        news_line = f'**{n_["decision"]}**'
        if n_.get("abstained"):
            news_line += " - the agent stood back, there was too little fresh news to judge"
        elif n_["decision"] == "NEUTRAL":
            news_line += (f' - it read {n_["n_headlines_read"]} headlines and found nothing '
                          f'clearly good or bad')
        else:
            news_line += (f' - from {n_["n_headlines_read"]} headlines, and it was '
                          f'{n_["confidence"]:.0%} sure')

        risk_line = f'**{PLAIN_VERDICT[v_["verdict"]]}**'
        if v_["rules_fired"]:
            last_plain = v_["rules_fired"][-1].get("plain", "")
            risk_line += f' - {last_plain}' if last_plain else ""

        chain_rows = [
            ("🧭", "The market", market_line),
            ("📰", f"The news about {pick}", news_line),
            ("🛡️", "The risk check", risk_line),
        ]
        for icon, who, what in chain_rows:
            st.markdown(f'{icon}&nbsp; **{who}:** {what}', unsafe_allow_html=True)

        st.markdown("&nbsp;&nbsp;&nbsp;⬇︎ **so the system decided:**", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Do what?", PLAIN_ACTION[card["risk"]["action"]]
                  if card["risk"]["action"] != "NONE" else "Nothing")
        k2.metric("How much?", f'{card["risk"]["position_size"]:.0%}',
                  help="A share of what this stock would normally get - not of everything we own.")
        k3.metric("Sell if it drops / gains",
                  f'{card["risk"]["stop_loss_pct"] or 0:.0f}% / {card["risk"]["take_profit_pct"] or 0:.0f}%'
                  if card["risk"]["stop_loss_pct"] else "-")
        k4.metric("How clear-cut?", card["confidence_label"],
                  help=AGREE_MEANS.get(card["confidence_label"], ""))
        st.caption(f'*“{card["confidence_label"]}” means {AGREE_MEANS.get(card["confidence_label"], "")}.*')

        st.markdown("**In plain English**")
        st.markdown(f'> {card["synthesis"]}')
        st.caption(f'{badge} · '
                   + ("⚠️ **contradicts the Risk Agent**" if bad
                      else "✅ the action and size above are copied straight from the Risk Agent - "
                           "this agent only writes the explanation, it never changes the decision"))
    elif src == "pending":
        st.info(f'**{pick}** on {picked:%d %b %Y} has not been narrated yet. '
                'Tick **“Call the LLM for news”** above to generate it.')
    else:
        st.warning(f'Nothing to narrate for **{pick}** - {badge}.')

# --- did it get it right? we know, because this data was held back ---
# Kept to the end on purpose: the agents explain themselves first, and only then
# is the answer marked. Put at the top it tells a reader the verdict before they
# have read a single reason.
if _truth is not None and _future_date is not None:
    st.markdown("---")
    st.markdown('<div class="ctx-hd">Marking the homework</div>', unsafe_allow_html=True)
    correct = (reg["decision"] == _truth)
    v1, v2, v3 = st.columns([1.3, 1.3, 2.4])
    v1.metric("System predicts", reg["decision"], help="what it says will be true in 20 trading days")
    v2.metric("Actually happened", _truth,
              help=f"the real regime on {_future_date:%d %b %Y}")
    if correct:
        v3.success(f"✅ **Correct** - on {_future_date:%d %b %Y} the market really was **{_truth}**.")
    else:
        v3.error(f"❌ **Wrong** - it said **{reg['decision']}**, but on "
                 f"{_future_date:%d %b %Y} the market was **{_truth}**.")
    st.caption("The model was trained only on data up to 31 Dec 2024 and has never seen this period. "
               "Because we already know what happened, every prediction here can be marked. "
               "Across all 381 days: **69.8% correct** - against **73.8%** for a two-line "
               "arithmetic rule, which still wins.")

st.markdown("---")
st.caption("""Technical Agent: XGBoost router + two arithmetic rules, exact SHAP via TreeExplainer.  
Sentiment Agent: Gemini 3.1 Flash Lite over 8,033 headlines with a 7-day freshness rule.  
Risk Agent: seven deterministic rules over 20-day trailing realised volatility.  
Coordinator: Gemini narrates the three agents; the action and size are copied, never generated.  
All data after 2024-12-31 was held out during model development.""")
