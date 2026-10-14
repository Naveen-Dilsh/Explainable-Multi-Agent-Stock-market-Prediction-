"""Baselines, models, the two-stage router and the scoring.

Ported from notebooks 07-10. Nothing here reads or writes a file; run.py does.

The two-stage design is the contribution: the learned part only answers "will the
state change?", and its answer picks between two procedures a reader can check -
carry today's state forward, or apply the threshold rule to the 100 days already
known. Neither branch is a model.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import CODE, DECODE, DEGENERATE_FOLD, RANDOM_STATE, THRESHOLD


# ──────────────────────────────── baselines ────────────────────────────────
def predict_majority(train, test):
    """Always guess whichever state was commonest in training."""
    return pd.Series(train["target"].value_counts().idxmax(), index=test.index)


def predict_persistence(train, test):
    """Guess that nothing changes: the state in 20 days is today's state."""
    return test["regime_now"].copy()


def mechanical(df, threshold=THRESHOLD):
    """Apply the labelling rule to the part of the window already known.

    The state 20 days out is decided by the window t-100 .. t+20. One hundred of
    those 120 days have already happened, so this needs no model at all - and at
    83.7% it is the real bar to beat.
    """
    out = pd.Series("SIDEWAYS", index=df.index)
    out[df["ret_100"] > threshold] = "BULL"
    out[df["ret_100"] < -threshold] = "BEAR"
    return out


def predict_mechanical(train, test):
    return mechanical(test)


BASELINES = {
    "majority": predict_majority,
    "persistence": predict_persistence,
    "mechanical": predict_mechanical,
}


# ───────────────────────────────── models ─────────────────────────────────
def make_xgb():
    """Conservative settings, fixed in advance and never tuned."""
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="multi:softprob", num_class=3,
        random_state=RANDOM_STATE, n_jobs=4, verbosity=0,
    )


def make_logreg():
    return LogisticRegression(max_iter=3000, C=1.0, random_state=RANDOM_STATE)


def make_detector():
    """Stage 1: a binary "will the state change?" classifier.

    scale_pos_weight stops the 78/22 imbalance pushing it straight to
    "nothing ever changes".
    """
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", scale_pos_weight=3.5,
        random_state=RANDOM_STATE, n_jobs=4, verbosity=0,
    )


def make_specialist():
    """Optional stage 2b: a 3-class model trained only on transition days."""
    return XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        objective="multi:softprob", num_class=3,
        random_state=RANDOM_STATE, n_jobs=4, verbosity=0,
    )


def recency_weights(y_train, index, half_life_days=500):
    """Recent rows count more. A 500-day half-life is about two years."""
    return 0.5 ** (np.arange(len(y_train))[::-1] / half_life_days)


def balanced_weights(y_train, index):
    """Up-weight rare classes so BEAR is not simply ignored."""
    return compute_sample_weight("balanced", y_train)


# ──────────────────────────────── running ────────────────────────────────
def run_walkforward(data, folds, make_model, features, weight_fn=None, scale=False):
    """Train a fresh model per fold and predict that fold's test year.

    `scale` standardises using training statistics only - fitting the scaler on
    everything would leak test statistics backwards.
    """
    out = []
    for _, f in folds.iterrows():
        train = data.loc[f["train_start"]: f["train_end"]]
        test = data.loc[f["test_start"]: f["test_end"]]

        X_train, X_test = train[features].to_numpy(), test[features].to_numpy()
        y_train = train["target"].map(CODE).to_numpy()

        if scale:
            scaler = StandardScaler().fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        model = make_model()
        if weight_fn is not None:
            model.fit(X_train, y_train, sample_weight=weight_fn(y_train, train.index))
        else:
            model.fit(X_train, y_train)

        out.append(pd.Series(model.predict(X_test), index=test.index).map(DECODE))

    return pd.concat(out).sort_index()


def run_baselines(data, folds):
    """All three baselines over every fold, plus the fold bookkeeping columns."""
    frames = []
    for _, f in folds.iterrows():
        train = data.loc[f["train_start"]: f["train_end"]]
        test = data.loc[f["test_start"]: f["test_end"]]
        row = pd.DataFrame(index=test.index)
        row["fold"] = f["fold"]
        row["year"] = test.index[0].year
        row["regime_now"] = test["regime_now"]
        row["target"] = test["target"]
        row["ret_100"] = test["ret_100"]
        row["is_transition"] = (test["regime_now"] != test["target"]).astype(int)
        for name, fn in BASELINES.items():
            row[f"pred_{name}"] = fn(train, test)
        frames.append(row)
    return pd.concat(frames).sort_index()


# ────────────────────────────── the two stages ──────────────────────────────
GRID = np.round(np.arange(0.20, 0.86, 0.02), 3)


def pick_cutoff(train, features, grid=GRID):
    """Pick the switch probability by searching an inner split of training only.

    Superseded by calibrated_cutoff, kept because notebook 09 reports both.
    """
    cut = int(0.8 * len(train))
    inner_train, inner_val = train.iloc[:cut], train.iloc[cut:]
    y_inner = (inner_train["regime_now"] != inner_train["target"]).astype(int)
    if y_inner.nunique() < 2:
        return 0.5

    det = make_detector().fit(inner_train[features], y_inner)
    p_val = det.predict_proba(inner_val[features])[:, 1]
    mech_val = mechanical(inner_val)

    best_t, best_acc = 0.5, -1.0
    for t in grid:
        combined = np.where(p_val >= t, mech_val, inner_val["regime_now"])
        acc = (combined == inner_val["target"]).mean()
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t


def calibrated_cutoff(train, features, flag_multiple=1.0):
    """Set the cut-off so the detector flags changes at the historical rate.

    This is the chosen rule. It is NOT a search for the best accuracy: the
    criterion was fixed before the sweep was run, which is why the 1.25x
    multiple that scored 0.2 higher was not adopted.

    Returns (cutoff, base_rate).
    """
    y_train = (train["regime_now"] != train["target"]).astype(int)
    base_rate = y_train.mean()
    target_rate = min(0.95, base_rate * flag_multiple)

    cut = int(0.8 * len(train))
    inner_train, inner_val = train.iloc[:cut], train.iloc[cut:]
    if y_train.iloc[:cut].nunique() < 2:
        return 0.5, base_rate

    det = make_detector().fit(inner_train[features], y_train.iloc[:cut])
    p_val = det.predict_proba(inner_val[features])[:, 1]
    return float(np.quantile(p_val, 1 - target_rate)), base_rate


def run_two_stage(data, folds, features, flag_multiple=1.0, calibrated=True):
    """The router: predict whether the state changes, then apply a procedure.

    change expected  ->  the threshold rule on the 100 known days
    no change        ->  carry today's state forward
    """
    out = []
    for _, f in folds.iterrows():
        train = data.loc[f["train_start"]: f["train_end"]]
        test = data.loc[f["test_start"]: f["test_end"]]
        y_train = (train["regime_now"] != train["target"]).astype(int)

        if calibrated:
            cutoff, _ = calibrated_cutoff(train, features, flag_multiple)
        else:
            cutoff = pick_cutoff(train, features)

        det = make_detector().fit(train[features], y_train)
        says_change = det.predict_proba(test[features])[:, 1] >= cutoff
        out.append(pd.Series(np.where(says_change, mechanical(test), test["regime_now"]),
                             index=test.index))
    return pd.concat(out).sort_index()


def oracle(data):
    """A perfect router with the same two branches. The ceiling, not a model."""
    changes = data["regime_now"] != data["target"]
    return pd.Series(np.where(changes, mechanical(data), data["regime_now"]),
                     index=data.index)


# ──────────────────────────────── scoring ────────────────────────────────
def degenerate_folds(preds):
    """Folds with no state changes at all, which cannot discriminate anything.

    Computed rather than assumed; fold 7 (2021) is the one that qualifies.
    """
    bad = []
    for fold, g in preds.groupby("fold"):
        if (g["regime_now"] != g["target"]).sum() == 0:
            bad.append(int(fold))
    return bad


def evaluate(preds, pred_col, drop_folds=(DEGENERATE_FOLD,)):
    """Headline accuracy: the mean of the per-fold accuracies, dropping fold 7."""
    accs = [(g[pred_col] == g["target"]).mean()
            for fold, g in preds.groupby("fold") if fold not in drop_folds]
    return 100 * float(np.mean(accs))


def per_fold_accuracy(preds, pred_col):
    """Accuracy in each fold, for the per-fold tables and figures."""
    return pd.DataFrame([
        {"fold": int(fold), "year": int(g["year"].iloc[0]), "n": len(g),
         "accuracy": round(100 * (g[pred_col] == g["target"]).mean(), 2),
         "transitions_%": round(100 * g["is_transition"].mean(), 2)}
        for fold, g in preds.groupby("fold")
    ])


def scoreboard(preds, pred_cols, drop_folds=(DEGENERATE_FOLD,)):
    """One row per model, matching results/model_summary_h20.csv.

    `skill` is the margin over the mechanical baseline. It is the only honest
    comparison: against the majority baseline every model looks spectacular.
    """
    keep = preds[~preds["fold"].isin(drop_folds)]
    stable = keep[keep["is_transition"] == 0]
    moving = keep[keep["is_transition"] == 1]
    is_bear = keep["target"] == "BEAR"

    rows = []
    for col in pred_cols:
        per_fold = [(g[col] == g["target"]).mean()
                    for fold, g in preds.groupby("fold") if fold not in drop_folds]
        rows.append({
            "model": col.replace("pred_", ""),
            "headline_9f": round(100 * float(np.mean(per_fold)), 2),
            "spread": round(100 * float(np.std(per_fold, ddof=1)), 2),
            "stable": round(100 * (stable[col] == stable["target"]).mean(), 2),
            "transition": round(100 * (moving[col] == moving["target"]).mean(), 2),
            "bear_recall": round(100 * ((keep[col] == "BEAR") & is_bear).sum()
                                 / max(int(is_bear.sum()), 1), 2),
        })

    out = pd.DataFrame(rows).sort_values("headline_9f", ascending=False)
    bar = out.loc[out["model"] == "mechanical", "headline_9f"]
    out["skill"] = (out["headline_9f"] - float(bar.iloc[0])).round(2) if len(bar) else np.nan
    cols = ["model", "headline_9f", "skill", "spread", "stable", "transition", "bear_recall"]
    return out[cols].reset_index(drop=True)
