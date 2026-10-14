"""Market data, regime labels, features and walk-forward folds.

Ported from notebooks 01-04 and 06. Every function here is causal: a row dated D
is built only from prices up to and including D. Nothing in this file reads or
writes a file; run.py does that.
"""
import numpy as np
import pandas as pd

from config import (
    CONFIRM_DAYS,
    END_DATE,
    HORIZON,
    LOOKBACK,
    MARKET,
    PURGE,
    START_DATE,
    STOCKS,
    TEST_YEARS,
    THRESHOLD,
    TICKERS,
)


# ──────────────────────────────── download ────────────────────────────────
def download():
    """Fetch raw OHLCV from Yahoo. Returns {name: DataFrame}.

    Yahoo is unreachable from Colab, so this only runs locally. It is also the
    one step whose output is not reproducible, because a later call will not
    return 2010-2026 identically.
    """
    import yfinance as yf

    out = {}
    for name, ticker in {**TICKERS, **MARKET}.items():
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out[name] = df
    return out


# ───────────────────────────────── align ─────────────────────────────────
def build_panel(raw):
    """Intersect the seven calendars into one wide frame.

    Only days on which every series traded survive, so no row is ever built
    from a stale price carried forward.
    """
    frames = []
    for name, df in raw.items():
        prefix = {"NIFTY50": "NIFTY", "INDIAVIX": "VIX"}.get(name, name)
        cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        frames.append(df[cols].add_prefix(f"{prefix}_"))

    panel = pd.concat(frames, axis=1, join="inner").dropna()
    panel.index.name = "Date"
    return panel.sort_index()


# ───────────────────────────────── labels ─────────────────────────────────
def make_raw_labels(close, lookback=LOOKBACK, threshold=THRESHOLD):
    """The market state from the trailing return alone, before confirmation."""
    trailing = close.pct_change(lookback)
    label = pd.Series(np.nan, index=close.index, dtype=object)
    label[trailing > threshold] = "BULL"
    label[trailing < -threshold] = "BEAR"
    label[(trailing >= -threshold) & (trailing <= threshold)] = "SIDEWAYS"
    return label, trailing


def confirm(raw, k=CONFIRM_DAYS):
    """Switch state only after the raw label has agreed k days running.

    Causal: the value at position i depends only on raw[:i+1]. Without this the
    label flickers whenever the trailing return sits on a threshold - 237
    episodes become 49.
    """
    values = raw.to_numpy()
    out = np.empty(len(values), dtype=object)
    current = values[0]
    for i in range(len(values)):
        if i >= k - 1:
            window = values[i - k + 1: i + 1]
            if window[0] != current and all(w == window[0] for w in window):
                current = window[0]
        out[i] = current
    return pd.Series(out, index=raw.index)


def count_episodes(series):
    """How many separate runs of the same label there are."""
    return int((series != series.shift()).sum())


def label_regimes(panel, lookback=LOOKBACK, threshold=THRESHOLD,
                  confirm_days=CONFIRM_DAYS, horizon=HORIZON):
    """Full labelling: raw label, confirmed state, and the shifted target."""
    close = panel["NIFTY_Close"]
    raw, trailing = make_raw_labels(close, lookback, threshold)
    valid = raw.notna()
    regime = confirm(raw[valid], confirm_days)

    return pd.DataFrame({
        f"trailing_return_{lookback}d": trailing[valid],
        "regime_raw": raw[valid],
        "regime": regime,
        f"target_regime_t{horizon}": regime.shift(-horizon),
    })


# ─────────────────────────────── indicators ───────────────────────────────
def rsi(close, n=14):
    """Relative Strength Index: 0 = only falls recently, 100 = only rises."""
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd_histogram(close, fast=12, slow=26, signal=9):
    """MACD histogram, divided by price so it is comparable across years."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return (line - sig) / close


def adx(high, low, close, n=14):
    """Average Directional Index: how strong the trend is, ignoring direction."""
    up, down = high.diff(), -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def trend_straightness(close, window=60):
    """R-squared of a straight line fitted to log price.

    Near 1 the market is moving in a clean line; near 0 it is wandering.
    """
    x = np.arange(window, dtype=float)
    x -= x.mean()
    sxx = (x ** 2).sum()

    def r2(y):
        yc = y - y.mean()
        ss_tot = (yc ** 2).sum()
        if ss_tot <= 0:
            return 0.0
        slope = (x * yc).sum() / sxx
        return 1 - ((yc - slope * x) ** 2).sum() / ss_tot

    return np.log(close).rolling(window).apply(r2, raw=True)


# ──────────────────────────────── features ────────────────────────────────
def build_features(panel, regime, horizon=HORIZON,
                   lookback=LOOKBACK, threshold=THRESHOLD):
    """The 26 features the model may see. Causal by construction."""
    close, high, low = panel["NIFTY_Close"], panel["NIFTY_High"], panel["NIFTY_Low"]
    vix = panel["VIX_Close"]
    daily = close.pct_change()
    f = {}

    # A. trend
    for n in (5, 20, 60, 120):
        f[f"ret_{n}"] = close.pct_change(n)
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    f["px_vs_sma50"] = close / sma50 - 1
    f["px_vs_sma200"] = close / sma200 - 1
    f["sma50_vs_sma200"] = sma50 / sma200 - 1
    f["macd_hist"] = macd_histogram(close)

    # A2. the decomposition. The state `horizon` days out is set by the window
    # t-100 .. t+20, and 100 of those 120 days are already known here, so hand
    # that part over directly instead of making the model rediscover it.
    f["ret_100"] = close.pct_change(lookback - horizon)
    known_growth = 1 + f["ret_100"]
    need_bull = (1 + threshold) / known_growth - 1
    need_bear = (1 - threshold) / known_growth - 1
    nearest_line = pd.concat([need_bull.abs(), need_bear.abs()], axis=1).min(axis=1)
    typical_move = daily.rolling(20).std() * np.sqrt(horizon)
    # how far the next 20 days must travel to cross a threshold, in sigmas.
    # small = sitting on the line; large = the answer is already locked in.
    f["flip_margin"] = (nearest_line / typical_move.replace(0, np.nan)).clip(upper=10)

    # B. trend strength
    f["adx_14"] = adx(high, low, close)
    f["trend_r2_60"] = trend_straightness(close, 60)

    # C. volatility and fear
    f["vix_level"] = vix
    f["vix_chg_20"] = vix.pct_change(20)
    f["vol_20"] = daily.rolling(20).std()
    f["vol_60"] = daily.rolling(60).std()
    f["bb_width"] = 4 * close.rolling(20).std() / close.rolling(20).mean()

    # D. stress
    running_max = close.cummax()
    f["drawdown"] = close / running_max - 1
    pos = np.arange(len(close), dtype=float)
    last_high = pd.Series(np.where((close >= running_max).to_numpy(), pos, np.nan),
                          index=close.index).ffill()
    f["days_since_high"] = pos - last_high

    # E. breadth: do the individual stocks agree?
    stock_close = panel[[f"{s}_Close" for s in STOCKS]]
    f["breadth_above_sma200"] = (stock_close > stock_close.rolling(200).mean()).mean(axis=1)
    n = len(STOCKS)
    corr = stock_close.pct_change().rolling(60).corr().to_numpy().reshape(-1, n, n)
    iu = np.triu_indices(n, k=1)
    f["avg_correlation"] = pd.Series(corr[:, iu[0], iu[1]].mean(axis=1), index=stock_close.index)

    # F. momentum
    f["rsi_14"] = rsi(close)

    features = pd.DataFrame(f, index=panel.index)

    # G. the current state, handed over explicitly as the persistence baseline
    reg = regime.reindex(panel.index)
    for name in ("BULL", "SIDEWAYS", "BEAR"):
        features[f"in_{name.lower()}"] = (reg == name).astype(float)
    episode = (reg != reg.shift()).cumsum()
    features["days_in_regime"] = reg.groupby(episode).cumcount().astype(float) + 1
    features.loc[reg.isna(), "days_in_regime"] = np.nan

    return features


def build_dataset(panel, labels, horizon=HORIZON):
    """Features plus `regime_now` and `target`, ready to model."""
    regime = labels["regime"]
    data = build_features(panel, regime, horizon=horizon)
    data["regime_now"] = regime.reindex(data.index)
    data["target"] = labels[f"target_regime_t{horizon}"].reindex(data.index)
    return data.dropna()


# ────────────────────────────── walk-forward ──────────────────────────────
def make_folds(index, test_years=TEST_YEARS, purge=PURGE):
    """Expanding-window folds with a purged gap before each test set.

    The purge equals the prediction horizon: a training row dated D has a target
    at D+horizon, so without the gap the last training labels would be computed
    partly from days inside the test year.

    Returns [(fold_id, train_positions, test_positions)] as integer positions.
    """
    pos = np.arange(len(index))
    folds = []
    for i, year in enumerate(test_years, start=1):
        is_test = index.year == year
        if not is_test.any():
            continue
        test_pos = pos[is_test]
        train_pos = pos[: max(0, test_pos[0] - purge)]
        if len(train_pos):
            folds.append((i, train_pos, test_pos))
    return folds


def fold_table(index, folds, horizon=HORIZON, purge=PURGE):
    """One row per fold, matching data/splits/folds_h20.csv."""
    rows = [{
        "fold": fid,
        "train_start": index[tr[0]].date(), "train_end": index[tr[-1]].date(),
        "n_train": len(tr),
        "test_start": index[te[0]].date(), "test_end": index[te[-1]].date(),
        "n_test": len(te),
        "gap_rows": te[0] - tr[-1] - 1,
    } for fid, tr, te in folds]
    out = pd.DataFrame(rows)
    out["horizon"], out["purge"] = horizon, purge
    return out


def check_folds(folds, horizon=HORIZON):
    """The five split checks from notebook 06. Returns a list of problems."""
    problems = []

    if not all(tr[-1] < te[0] for _, tr, te in folds):
        problems.append("a training window does not end before its test set")

    if any(len(np.intersect1d(tr, te)) for _, tr, te in folds):
        problems.append("a row appears in both train and test of one fold")

    # a training row dated D knows about D+horizon; that must stay before the test set
    if any(tr[-1] + horizon >= te[0] for _, tr, te in folds):
        problems.append("purge too small: training knowledge reaches into the test set")

    seen = np.concatenate([te for _, _, te in folds])
    if len(seen) != len(np.unique(seen)):
        problems.append("test sets overlap each other")

    sizes = [len(tr) for _, tr, _ in folds]
    if not all(b > a for a, b in zip(sizes, sizes[1:])):
        problems.append("the training window does not expand every fold")

    return problems
