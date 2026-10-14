#!/usr/bin/env python
"""Run a pipeline step.

    python run.py data
    python run.py models router risk        # several, in order
    python run.py --help

Steps: data, models, router, sentiment, risk, coordinator, study.

This is the only file that reads or writes anything. Everything else is plain
functions over DataFrames.

Steps that cost money or need the network (data --live, sentiment, coordinator)
refuse to call out unless --live is given, so a plain run can never spend money.
"""
import argparse
import json
import sys

import pandas as pd

import config as c


def _load_dataset():
    return pd.read_csv(c.DATASET, parse_dates=["Date"]).set_index("Date")


def _load_folds():
    return pd.read_csv(c.FOLDS, parse_dates=["train_start", "train_end",
                                             "test_start", "test_end"])


def _save(df, path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kw)
    print(f"  wrote {path.relative_to(c.DATA_ROOT)}  {df.shape}")


def _save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"  wrote {path.relative_to(c.DATA_ROOT)}  ({len(obj)} entries)")


def _load_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else (
        {} if default is None else default)


def _regime_agent():
    """Rebuild the Regime Agent around the saved router. Does not retrain."""
    import agents as A

    if not c.ROUTER.exists():
        sys.exit(f"  router not found at {c.ROUTER}. Run: python run.py router")
    data = _load_dataset()
    dev = data.loc[data.index < c.SEALED_FROM]
    features = c.chosen_features()
    cutoff = float(pd.read_csv(c.ROUTER_SETTINGS, header=None, index_col=0)[1]["cutoff"])
    bands = (pd.read_csv(c.RELIABILITY_BANDS).set_index("band")["accuracy"].to_dict()
             if c.RELIABILITY_BANDS.exists() else {})
    agent = A.RegimeAgent(A.load_router(c.ROUTER), dev, features, cutoff, bands)
    return data, agent


# ──────────────────────────────── steps ────────────────────────────────
def step_data(args):
    """Clean, label, build features, and make the walk-forward folds."""
    import data as D

    if args.live:
        print("downloading from Yahoo ...")
        raw = D.download()
        for name, df in raw.items():
            _save(df, c.RAW / f"{name}_raw.csv")
        panel = D.build_panel(raw)
        _save(panel, c.PANEL)
    else:
        panel = pd.read_csv(c.PANEL, parse_dates=["Date"]).set_index("Date")
        print(f"  using existing panel.csv  {panel.shape}   (--live to re-download)")

    labels = D.label_regimes(panel)
    _save(labels, c.REGIME_LABELS)
    print(f"  {D.count_episodes(labels['regime_raw'])} raw episodes -> "
          f"{D.count_episodes(labels['regime'])} after {c.CONFIRM_DAYS}-day confirmation")

    dataset = D.build_dataset(panel, labels)
    _save(dataset, c.DATASET)

    dev = dataset.loc[dataset.index < c.SEALED_FROM]
    folds = D.make_folds(dev.index)
    problems = D.check_folds(folds)
    if problems:
        sys.exit("  SPLIT CHECKS FAILED: " + "; ".join(problems))
    print(f"  {len(folds)} folds, purge {c.PURGE} rows, all split checks passed")
    _save(D.fold_table(dev.index, folds), c.FOLDS, index=False)


def step_models(args):
    """Baselines, the single-stage models, and the two-stage router."""
    import models as M

    data, folds = _load_dataset(), _load_folds()
    all_features = [f for f in data.columns if f not in ("regime_now", "target")]
    no_ret120 = [f for f in all_features if f != "ret_120"]

    preds = M.run_baselines(data, folds)
    print(f"  baselines done, {len(preds)} rows")

    print("  training ...")
    preds["pred_logreg"] = M.run_walkforward(data, folds, M.make_logreg, all_features, scale=True)
    preds["pred_xgb"] = M.run_walkforward(data, folds, M.make_xgb, all_features)
    preds["pred_xgb_no_ret120"] = M.run_walkforward(data, folds, M.make_xgb, no_ret120)
    preds["pred_xgb_recency"] = M.run_walkforward(data, folds, M.make_xgb, all_features,
                                                  weight_fn=M.recency_weights)
    preds["pred_xgb_balanced"] = M.run_walkforward(data, folds, M.make_xgb, all_features,
                                                   weight_fn=M.balanced_weights)

    # Pre-registered ablation: drop ret_120 if it costs less than a point.
    full = M.evaluate(preds, "pred_xgb")
    ablated = M.evaluate(preds, "pred_xgb_no_ret120")
    drop_it = abs(full - ablated) < 1.0
    chosen_features = no_ret120 if drop_it else all_features
    chosen_col = "pred_xgb_no_ret120" if drop_it else "pred_xgb"
    print(f"  ret_120 ablation: {full:.2f}% with, {ablated:.2f}% without "
          f"({full - ablated:+.2f} pp) -> {'drop' if drop_it else 'keep'} it")

    preds["pred_two_stage"] = M.run_two_stage(data, folds, chosen_features, calibrated=False)
    preds["pred_two_stage_cal_mech"] = M.run_two_stage(data, folds, chosen_features,
                                                       calibrated=True)
    preds["pred_oracle"] = M.oracle(data).reindex(preds.index)

    degenerate = M.degenerate_folds(preds)
    print(f"  degenerate folds (no state changes): {degenerate}")

    # Keep any prediction columns written by work outside this pipeline - the
    # neural notebooks add their own. We own our columns and nobody else's.
    if c.PREDICTIONS.exists():
        existing = pd.read_csv(c.PREDICTIONS, parse_dates=["Date"]).set_index("Date")
        foreign = [col for col in existing.columns if col not in preds.columns]
        if foreign:
            preds = preds.join(existing[foreign], how="left")
            print(f"  kept {len(foreign)} column(s) from other work: {', '.join(foreign)}")

    pred_cols = [col for col in preds.columns if col.startswith("pred_")]
    scores = M.scoreboard(preds, pred_cols, drop_folds=tuple(degenerate))

    _save(preds, c.PREDICTIONS)
    _save(scores, c.MODEL_SUMMARY, index=False)
    _save(pd.Series({"chosen_prediction_column": chosen_col,
                     "n_features": len(chosen_features),
                     "features": "|".join(chosen_features)}),
          c.CHOSEN_MODEL, header=False)
    print()
    print(scores.to_string(index=False))


def step_router(args):
    """Train the deployment router once, freeze it, and explain it.

    Walk-forward in `models` answers "does the design work?". This answers
    "what does the shipped system say?", so the router is trained on everything
    up to TRAIN_END and never retrained. The sealed period is scored but never
    fitted on.
    """
    import agents as A
    import models as M

    data = _load_dataset()
    dev = data.loc[data.index < c.SEALED_FROM]
    sealed = data.loc[data.index >= c.SEALED_FROM]
    features = c.chosen_features()

    print(f"  training on {len(dev):,} rows "
          f"{dev.index.min().date()} -> {dev.index.max().date()}")
    router, cutoff, base_rate = A.train_router(dev, features)
    print(f"  historical change rate {base_rate:.1%} -> calibrated cut-off {cutoff:.3f}")

    # Reliability is measured on the sealed period: the only data the frozen
    # router has never seen, so the only honest place to ask how often it is right.
    p_sealed = router.predict_proba(sealed[features])[:, 1]
    truth_change = (sealed["regime_now"] != sealed["target"]).to_numpy()
    bands = A.reliability_bands(p_sealed, cutoff, truth_change)
    _save(bands, c.RELIABILITY_BANDS, index=False)

    agent = A.RegimeAgent(router, dev, features, cutoff,
                          bands.set_index("band")["accuracy"].to_dict())

    pred = pd.Series([agent.decide(r)[0] for _, r in sealed.iterrows()], index=sealed.index)
    flagged = p_sealed >= cutoff
    mech = M.mechanical(sealed)
    acc = float((pred == sealed["target"]).mean())
    bar = float((mech == sealed["target"]).mean())
    print(f"  SEALED {sealed.index.min().date()} -> {sealed.index.max().date()}"
          f"  ({len(sealed)} days)")
    print(f"    mechanical bar {bar:7.1%}")
    print(f"    regime agent   {acc:7.1%}   ({acc - bar:+.1%} vs the bar)")
    print(f"    changes: {truth_change.mean():.1%} actual, {flagged.mean():.1%} flagged")

    _save(pd.DataFrame({"regime_now": sealed["regime_now"], "prediction": pred,
                        "truth": sealed["target"], "p_change": p_sealed.round(3),
                        "flagged": flagged}), c.SEALED_PREDICTIONS)
    _save(agent.importance().to_frame("mean_abs_shap"), c.SHAP_IMPORTANCE)

    picks = list(sealed.index[::max(1, len(sealed) // 10)][:10])
    cards = []
    for d in picks:
        e = agent.explain(sealed.loc[d])
        e["actual_20d_later"] = sealed.loc[d, "target"]
        e["correct"] = bool(e["decision"] == e["actual_20d_later"])
        e["card_text"] = A.render(e)
        cards.append(e)
    _save_json(cards, c.REGIME_CARDS)

    _save(pd.Series({
        "model": "two_stage_cal_mech",
        "router": "XGBoost binary (will the regime change in 20 days?)",
        "trained_on": f"{dev.index.min().date()} to {dev.index.max().date()} "
                      f"({len(dev)} rows)",
        "n_features": len(features),
        "features": "|".join(features),
        "base_rate": round(base_rate, 4),
        # Full precision on purpose: the demo loads this value and routes with it,
        # so a rounded copy could decide a borderline day differently from training.
        "cutoff": cutoff,
        "regime_threshold": c.THRESHOLD,
        "horizon_days": c.HORIZON,
        "explainer": f"shap.TreeExplainer (exact), shap {A.shap.__version__}",
        "sealed_accuracy": round(acc, 4),
        "sealed_bar": round(bar, 4),
    }), c.ROUTER_SETTINGS, header=False)

    c.ROUTER.parent.mkdir(parents=True, exist_ok=True)
    router.save_model(str(c.ROUTER))
    print(f"  wrote {c.ROUTER.relative_to(c.DATA_ROOT)}")
    print(f"\n{cards[-1]['card_text']}")


def _cached_dates(cache):
    """The decision dates the sentiment cache already covers."""
    return sorted({k.split("|")[1] for k in cache})


def step_sentiment(args):
    """News sentiment per stock per date. Cached; --live to call Gemini."""
    import agents as A

    news = pd.read_csv(c.NEWS, parse_dates=["published"])
    cache = _load_json(c.SENTIMENT_CACHE)
    print(f"  {len(news):,} headlines, {len(cache)} cached decisions")

    dates = _cached_dates(cache) or [str(d["date"]) for d in _load_json(c.REGIME_CARDS, [])]
    if not dates:
        sys.exit("  no dates to score. Run: python run.py router")

    run, counts = [], {}
    for d in dates:
        for t in c.TICKERS.values():
            out, source = A.sentiment_agent(news, t, d, cache, live=args.live)
            counts[source] = counts.get(source, 0) + 1
            if out is not None:
                run.append(out)
    print("  " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if counts.get("pending"):
        print("  pending means not cached and --live was not given. Nothing was called.")

    _save_json(cache, c.SENTIMENT_CACHE)
    _save_json(run, c.SENTIMENT_RUN)


def step_risk(args):
    """Apply the seven risk rules. Deterministic: no model, no API, no cost.

    Each stock is judged on its own, opening at zero exposure, which is how the
    reported run was produced. risk.evaluate - the same rules sharing one book
    across the five stocks, so R5 sees it fill up - is the demo's runtime path
    and is not used here, because a shared book makes a decision depend on which
    stocks happened to have news that day.
    """
    import risk as R

    data, agent = _regime_agent()
    panel = pd.read_csv(c.PANEL, parse_dates=["Date"]).set_index("Date")
    sentiment = _load_json(c.SENTIMENT_CACHE)
    if not sentiment:
        sys.exit("  no sentiment cached. Run: python run.py sentiment")

    decisions, skipped = [], 0
    for d in _cached_dates(sentiment):
        day = pd.Timestamp(d)
        if day not in data.index:
            continue
        regime = agent.explain(data.loc[day])
        for stock, ticker in c.TICKERS.items():
            sent = sentiment.get(f"{ticker}|{d}")
            vol = R.realised_vol(panel, stock, day)
            if sent is None or vol is None:
                skipped += 1
                continue
            r = R.risk_agent(regime, sent, vol)
            r.update(stock=stock, date=d, regime=regime["decision"],
                     will_change=regime["will_change"], realised_vol=round(vol, 2),
                     sentiment_source="REAL")
            r["plain_english"] = R.one_liner(r)
            decisions.append(r)

    verdicts = pd.Series([r["verdict"] for r in decisions]).value_counts().to_dict()
    actions = pd.Series([r["action"] for r in decisions]).value_counts().to_dict()
    print(f"  {len(decisions)} decisions ({skipped} stock-days skipped for want of "
          f"news or volatility)")
    print("  verdicts: " + ", ".join(f"{v} {k}" for k, v in verdicts.items()))
    print("  actions : " + ", ".join(f"{v} {k}" for k, v in actions.items()))

    _save_json({
        "note": "Risk Agent built around the Regime Agent only. The rules are "
                "pinned by tests/test_risk.py; run `make test` for their status.",
        "parameters": {k: getattr(c, k) for k in
                       ("TARGET_DAILY_RISK", "NO_CONTEXT", "OPPOSED", "CHANGE_PENALTY",
                        "MIN_SENT_CONF", "MIN_POSITION", "PER_STOCK_WEIGHT",
                        "MAX_EXPOSURE", "SL_MULT", "TP_MULT", "HORIZON")},
        "verdict_counts": verdicts,
        "action_counts": actions,
        "n_real_decisions": len(decisions),
        "real_decisions": decisions,
    }, c.RISK_RUN)


def step_coordinator(args):
    """Narrate each risk decision in one paragraph. Cached; --live to call Gemini.

    The action, size and levels are copied from the Risk Agent, never asked for,
    so the narration cannot contradict the decision it describes.
    """
    import agents as A

    data, agent = _regime_agent()
    sentiment = _load_json(c.SENTIMENT_CACHE)
    cache = _load_json(c.COORDINATOR_CACHE)
    risk_run = _load_json(c.RISK_RUN, {})
    decisions = risk_run.get("real_decisions", [])
    if not decisions:
        sys.exit("  no risk decisions. Run: python run.py risk")
    print(f"  {len(decisions)} decisions, {len(cache)} cards already cached")

    cards, counts, contradictions = [], {}, 0
    for r in decisions:
        day = pd.Timestamp(r["date"])
        if day not in data.index:
            continue
        row = data.loc[day]
        reg = agent.explain(row)
        reg["reasons"] = A.regime_reasons(row, agent.shap_row(row))
        news = sentiment.get(f'{r["stock"]}.NS|{r["date"]}')
        card, source = A.coordinate(reg, news, r, cache, live=args.live)
        counts[source] = counts.get(source, 0) + 1
        if card is not None:
            contradictions += A.contradicts(card, r)
            cards.append(card)

    print("  " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if counts.get("pending"):
        print("  pending means not cached and --live was not given. Nothing was called.")
    print(f"  contradictions against the Risk Agent's own record: {contradictions}")

    _save_json(cache, c.COORDINATOR_CACHE)
    _save_json(cards, c.COORDINATOR_CARDS)


def step_study(args):
    """Grade the comprehension study."""
    import json

    import userstudy as U

    if not c.STUDY_RESPONSES.exists():
        sys.exit(f"  responses not found at {c.STUDY_RESPONSES}")
    result = U.grade(pd.read_csv(c.STUDY_RESPONSES))
    c.STUDY_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    c.STUDY_RESULTS.write_text(json.dumps(result, indent=2))
    print(f"  wrote {c.STUDY_RESULTS.relative_to(c.DATA_ROOT)}")
    print(f"  {result['comprehension']['correct']}/{result['n_answers']} correct "
          f"= {100 * result['comprehension']['accuracy']:.1f}%")


STEPS = {
    "data": step_data,
    "models": step_models,
    "router": step_router,
    "sentiment": step_sentiment,
    "risk": step_risk,
    "coordinator": step_coordinator,
    "study": step_study,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("steps", nargs="+", choices=list(STEPS),
                   help="one or more steps, run in the order given")
    p.add_argument("--live", action="store_true",
                   help="allow network and language-model calls (costs money)")
    args = p.parse_args()

    if args.live and c.ENV_FILE.exists():
        from dotenv import load_dotenv
        load_dotenv(c.ENV_FILE)

    for name in args.steps:
        print(f"\n=== {name} ===")
        STEPS[name](args)
    print("\ndone")


if __name__ == "__main__":
    main()
