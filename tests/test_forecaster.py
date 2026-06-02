"""
tests/test_forecaster.py
Regression tests for the ReturnForecaster train/inference feature-parity fix.

These lock in the v4.2 contract that was previously broken:
  - Online single-row predict() must MATCH the batch prediction for that row
    (the prior bug z-scored a single row to all-zeros, yielding one constant
    prediction for every market state).
  - Online predictions must VARY across rows.
  - Inference must reuse STORED training normalisation stats, not recompute.
  - When a row can't be resolved against the cache, predict() returns a
    neutral 0.0 instead of feeding the model a degenerate feature vector.
  - The model machinery recovers a known signal (sanity check on the code,
    not a claim about real-data predictability).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.ml_forecaster import ReturnForecaster, _raw_feature_frame, FEATURE_COLS


def _features(n=1500, seed=0) -> pd.DataFrame:
    """Minimal engineered feature frame, indexed 0..n-1 like the real loader."""
    rng    = np.random.default_rng(seed)
    prices = 50_000 + np.cumsum(rng.normal(0, 25, n))
    df = pd.DataFrame({
        "open_time":           pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open":                prices,
        "high":                prices + rng.uniform(0, 50, n),
        "low":                 prices - rng.uniform(0, 50, n),
        "close":               prices,
        "volume":              rng.uniform(1, 10, n),
        "taker_buy_base_vol":  rng.uniform(0.4, 0.6, n) * rng.uniform(1, 10, n),
        "taker_buy_quote_vol": rng.uniform(0.4, 0.6, n) * rng.uniform(1, 10, n) * prices,
        "n_trades":            rng.integers(10, 100, n),
    })
    from src.data_loader import engineer_features
    return engineer_features(df)


@pytest.fixture(scope="module")
def fitted():
    feats = _features()
    fc = ReturnForecaster(horizon_bars=5).fit(feats)
    assert fc.is_fitted
    return fc, feats


def test_online_matches_batch_exactly(fitted):
    """The core regression: predict(row) == predict_series(df)[row] for every row."""
    fc, feats = fitted
    batch = fc.predict_series(feats)
    idxs = list(range(100, 400, 7))
    online = np.array([fc.predict(feats.iloc[i]) for i in idxs])
    assert np.allclose(online, batch.iloc[idxs].values, atol=0.0, rtol=0.0), \
        "online single-row predictions must equal the vectorised batch predictions"


def test_predictions_vary_on_signal_data():
    """The old bug produced ONE constant value for every row even when the data
    carried signal (because a single row z-scored to all zeros). On data with a
    planted signal the fixed path must produce varying online predictions that
    match the batch. (On pure-noise data a constant prediction is correct and
    expected, so variation is only asserted where signal exists.)"""
    rng = np.random.default_rng(1)
    n = 4000
    ofi = rng.normal(0, 0.3, n)
    sig = 0.6 * ofi
    next_ret = 0.0008 * sig + rng.normal(0, 0.0005, n)
    log_ret = np.empty(n); log_ret[0] = 0.0; log_ret[1:] = next_ret[:-1]
    close = 60000 * np.exp(np.cumsum(log_ret))
    df = pd.DataFrame({
        "close": close, "log_ret": log_ret, "ofi_proxy": ofi,
        "buy_ratio": np.clip(0.5 + rng.normal(0, 0.1, n), 0, 1),
        "vol_5m": np.abs(rng.normal(0.001, 0.0003, n)),
        "vol_30m": np.abs(rng.normal(0.001, 0.0002, n)),
        "realised_vol": np.abs(rng.normal(0.5, 0.1, n)),
        "trend_signal": rng.normal(0, 0.001, n),
        "trade_intensity": np.abs(rng.normal(100, 20, n)),
        "ret_1m": rng.normal(0, 0.001, n),
        "ret_5m": rng.normal(0, 0.002, n), "ret_15m": rng.normal(0, 0.003, n),
    })
    fc = ReturnForecaster(horizon_bars=1).fit(df)
    online = np.array([fc.predict(df.iloc[i]) for i in range(200, 600)])
    batch = fc.predict_series(df).iloc[200:600].values
    assert len(np.unique(online)) > 1, "predictions must vary when the data has signal"
    assert np.allclose(online, batch, atol=0.0, rtol=0.0)


def test_inference_reuses_stored_stats(fitted):
    """Normalisation stats are fitted once and stored, not recomputed per call."""
    fc, _ = fitted
    assert fc._feat_mean is not None and fc._feat_std is not None
    assert len(fc._feat_mean) == len(fc._feature_cols_fitted)
    assert np.all(fc._feat_std > 0)


def test_raw_features_are_pure_and_unnormalised():
    """_raw_feature_frame must NOT self-normalise (so a 1-row call can't zero out)."""
    feats = _features(n=400)
    X, cols = _raw_feature_frame(feats)
    # Raw direct features should equal the source columns (no z-scoring applied).
    for c in FEATURE_COLS:
        if c in feats.columns:
            assert np.allclose(X[c].fillna(0).values, feats[c].fillna(0).values)


def test_uncached_row_returns_neutral_zero(fitted):
    """A row not present in the prime cache yields neutral 0.0, never a garbage vector."""
    fc, _ = fitted
    orphan = pd.Series({c: 0.123 for c in FEATURE_COLS}, name=10_000_000)  # index not in cache
    with pytest.warns(RuntimeWarning):
        v = fc.predict(orphan)
    assert v == 0.0


def test_unfitted_predict_is_safe():
    fc = ReturnForecaster()
    assert fc.predict(pd.Series({c: 0.0 for c in FEATURE_COLS}, name=0)) == 0.0


def test_model_recovers_known_signal():
    """Sanity check on the MACHINERY (not a real-data claim): with a planted,
    properly-lagged signal the model must build >1 tree and score positive IC."""
    rng = np.random.default_rng(0)
    n = 6000
    ofi = rng.normal(0, 0.3, n)
    buy = np.clip(0.5 + rng.normal(0, 0.1, n), 0, 1)
    r1  = rng.normal(0, 0.001, n)
    sig = 0.4 * ofi + 0.3 * (buy - 0.5) * 4 - 0.5 * (r1 / r1.std())
    next_ret = 0.0006 * sig + rng.normal(0, 0.0006, n)
    log_ret = np.empty(n); log_ret[0] = 0.0; log_ret[1:] = next_ret[:-1]
    close = 60000 * np.exp(np.cumsum(log_ret))
    df = pd.DataFrame({
        "close": close, "log_ret": log_ret, "ofi_proxy": ofi, "buy_ratio": buy,
        "vol_5m": np.abs(rng.normal(0.001, 0.0003, n)),
        "vol_30m": np.abs(rng.normal(0.001, 0.0002, n)),
        "realised_vol": np.abs(rng.normal(0.5, 0.1, n)),
        "trend_signal": rng.normal(0, 0.001, n),
        "trade_intensity": np.abs(rng.normal(100, 20, n)),
        "ret_1m": r1, "ret_5m": rng.normal(0, 0.002, n), "ret_15m": rng.normal(0, 0.003, n),
    })
    fc = ReturnForecaster(horizon_bars=1).fit(df)
    assert fc.information_coefficient > 0.1, "should recover a planted signal"
    nonzero = (fc.feature_importance_df()["importance"] > 0).sum()
    assert nonzero >= 2, "importance should spread across features when signal exists"
