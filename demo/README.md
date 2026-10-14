# Demo — the four agents, one trading day at a time

Pick **any trading day** between 2025-01-02 and 2026-07-22 and see the decision, the
reasoning behind it, and what would have had to be different for the answer to change.
Every date shown is after 2024-12-31, which was held out entirely during model development,
so each prediction can be marked against what actually happened.

## Run it

From the project root:

```bash
python -m streamlit run demo/app.py
```

Then open http://localhost:8501

It needs `results/regime_agent/regime_router.json`, which `python run.py router` writes.
If it is missing the app says so rather than training a router of its own.

## What it shows

| Panel | Contents |
|---|---|
| Price chart | NIFTY 50 shaded by regime, with the 120-day window split into the 100 days already known and the 20 still unknown |
| 🧭 Technical Agent | the decision and the rule that produced it, evidence grouped into eight families, the specific values with percentile context, a SHAP bar chart |
| 🔀 Counterfactual | the value that would flip the decision, plus a live slider — drag it and watch the answer change |
| 📰 Sentiment Agent | per stock: direction, confidence, the headlines it read and why each mattered |
| 🛡️ Risk Agent | the seven rules, the position size, the stop and target, and whether those two levels actually worked |
| 🎯 Coordinator | one paragraph a non-expert can follow, with the action and size copied from the Risk Agent |

## Where the numbers come from

The app imports the package and **loads the frozen router** rather than training its own.
That is deliberate: the demo cannot drift away from what `make router` reports. The plain
English on screen is this file's own wording, kept gentler than the phrasing `agents.py`
uses for the printed cards.

## The language model

News decisions are cached in `results/sentiment_agent/sentiment_cache.json` and narrations
in `results/coordinator/coordinator_cache.json`. Cached dates load instantly and cost nothing.
For a date that is not cached, tick **"Call the LLM for news"** — it calls Gemini once per
stock and caches the result, so it is only ever slow once. That needs `GEMINI_API_KEY` in
`.env` at the project root.

Without the tick, uncached stocks show as *not analysed* rather than silently guessing.

## Honesty notes built into the app

- The Sentiment Agent **abstains** rather than guessing when fewer than 3 headlines exist in
  the preceding 7 days — shown explicitly as "abstained".
- Confidence is reported with its **measured historical accuracy**, so a reader can tell when
  to distrust it. Those numbers differ a lot: "very confident" was right 89% of the time,
  "confident" only 54%.
- The sealed-period caption reports 69.8% against the arithmetic rule's 73.8%. The rule wins,
  and the app says so.

## Files

```
demo/
  app.py       the Streamlit application
  cards/       exported static decision cards (used in the comprehension study)
  README.md    this file
```
