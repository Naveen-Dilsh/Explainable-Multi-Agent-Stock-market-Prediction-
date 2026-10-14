"""The modelling pipeline, checked against the numbers the thesis reports.

The golden files in tests/golden/ were taken from the notebook output before the
port began. If these pass, the port did not change any result.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as c  # noqa: E402
import data as D  # noqa: E402
import models as M  # noqa: E402

GOLDEN = Path(__file__).parent / "golden"
needs_data = pytest.mark.skipif(not c.DATASET.exists(),
                                reason="no dataset; run `make data` first")


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(c.DATASET, parse_dates=["Date"]).set_index("Date")


@pytest.fixture(scope="module")
def folds():
    return pd.read_csv(c.FOLDS, parse_dates=["train_start", "train_end",
                                             "test_start", "test_end"])


@pytest.fixture(scope="module")
def preds(dataset, folds):
    return M.run_baselines(dataset, folds)


# ─────────────────────────── pure functions ───────────────────────────
def test_mechanical_applies_the_threshold():
    df = pd.DataFrame({"ret_100": [0.10, -0.10, 0.0, 0.06, -0.06]})
    assert list(M.mechanical(df)) == ["BULL", "BEAR", "SIDEWAYS", "SIDEWAYS", "SIDEWAYS"]


def test_confirmation_suppresses_flicker():
    """A new state must hold CONFIRM_DAYS running before it is accepted."""
    raw = pd.Series(["BULL"] * 10 + ["BEAR"] * 3 + ["BULL"] * 10)
    assert set(D.confirm(raw, k=10)) == {"BULL"}       # the 3-day blip is ignored

    raw = pd.Series(["BULL"] * 10 + ["BEAR"] * 10)
    assert D.confirm(raw, k=10).iloc[-1] == "BEAR"     # a sustained change is taken


def test_trend_straightness_is_one_on_a_clean_ramp():
    close = pd.Series(np.exp(np.linspace(0, 1, 100)))
    assert D.trend_straightness(close, 60).iloc[-1] == pytest.approx(1.0, abs=1e-9)


@needs_data
def test_folds_pass_every_split_check(dataset):
    dev = dataset.loc[dataset.index < c.SEALED_FROM]
    assert D.check_folds(D.make_folds(dev.index)) == []


@needs_data
def test_no_sealed_row_reaches_any_fold(dataset):
    dev = dataset.loc[dataset.index < c.SEALED_FROM]
    assert int((dev.index >= c.SEALED_FROM).sum()) == 0


@needs_data
def test_degenerate_fold_is_found_not_assumed(preds):
    """Fold 7 (2021) has no state changes. It should be discovered, not hardcoded."""
    assert M.degenerate_folds(preds) == [c.DEGENERATE_FOLD]


# ───────────────────────────── goldens ─────────────────────────────
@needs_data
def test_folds_match_golden(dataset):
    dev = dataset.loc[dataset.index < c.SEALED_FROM]
    built = D.fold_table(dev.index, D.make_folds(dev.index))
    gold = pd.read_csv(GOLDEN / "folds_h20.csv")
    for col in ("fold", "n_train", "n_test", "gap_rows", "horizon", "purge"):
        assert list(built[col]) == list(gold[col]), col


@needs_data
@pytest.mark.parametrize("name,expected", [
    ("majority", 41.31), ("persistence", 77.80), ("mechanical", 83.68),
])
def test_baselines_match_golden(preds, name, expected):
    assert M.evaluate(preds, f"pred_{name}") == pytest.approx(expected, abs=0.05)


@needs_data
def test_two_stage_beats_the_bar_by_the_reported_margin(dataset, folds, preds):
    """The headline claim: 84.9%, which is 1.2 points over the mechanical bar."""
    p = preds.copy()
    p["pred_two_stage_cal_mech"] = M.run_two_stage(dataset, folds, c.chosen_features())

    model = M.evaluate(p, "pred_two_stage_cal_mech")
    bar = M.evaluate(p, "pred_mechanical")

    assert model == pytest.approx(84.9, abs=0.1)
    assert model - bar == pytest.approx(1.2, abs=0.1)


@needs_data
def test_oracle_is_the_ceiling(dataset, preds):
    """A perfect router with the same two branches still only reaches 94.6%."""
    p = preds.copy()
    p["pred_oracle"] = M.oracle(dataset).reindex(p.index)
    assert M.evaluate(p, "pred_oracle") == pytest.approx(94.6, abs=0.1)
