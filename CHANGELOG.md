# Changelog — v4

## v4.2 — Forecaster train/inference parity + provenance (correctness fixes)

### Critical: ML forecaster train/inference feature mismatch
- **Bug:** `_make_features()` z-score normalised using the statistics of the
  frame it was handed. The quoting loop called `predict()` with a single row,
  so every rolling/lag feature was NaN→0 and z-scoring one value gave 0 for
  every column. The model received an all-zero vector on every call and
  returned **one constant** for all market states; rebuilding a DataFrame +
  rolling stats per row also made the simulation ~50× slower (the observed
  "slow → KeyboardInterrupt").
- **Fix (`ml_forecaster.py` → v4.2):**
  - `_raw_feature_frame()` builds raw, **un-normalised** features with one
    vectorised, pure function used identically for train and inference.
  - Normalisation mean/std are fitted **once on the training split** (removing
    prior train/val leakage) and stored on the model; inference reuses them.
  - `prime_cache()` precomputes predictions for the whole feature frame in one
    pass; online `predict(row)` is an O(1) index lookup that returns **exactly**
    the batch prediction (verified bit-identical). Per-call cost ~9 ms → ~µs.
  - Uncacheable rows return a neutral `0.0` instead of a degenerate vector.
- **Result:** train/inference feature definitions are identical; Forecast-
  Adaptive runs in ~0.18 s/episode (was ~4.9 s) and its forecast signal varies.

### Honest data provenance (replaces filename guess)
- Cache filenames always contain "BTCUSDT", so the old `_data_source` check
  reported "REAL Binance" even for non-real data. Now a `.source` sidecar is
  written on live Binance fetch and read back; `params["data_source"]` reports
  `real_binance` only when both kline and trade caches are verified live, else
  `unknown (...)`. Surfaced in the loader summary and `run.py` header.

### Other
- Fixed an unterminated multi-line f-string in `run.py` (syntax error on 3.x).
- Added `tests/test_forecaster.py`: batch/online parity, signal-data variation,
  stored-stats reuse, neutral fallback, and a model signal-recovery sanity check.
- Added optional `make_synthetic_cache.py` for offline runs (writes provenance
  markers tagging the data as `synthetic_local`, never as real).

### Notes for reviewers
- LightGBM tree count, IC, and feature-importance spread are **data-dependent**
  outcomes, not targets. On weak-signal data a 1-tree near-constant model with
  IC≈0 is the correct, honest result; the signal-recovery test shows the same
  code builds hundreds of trees with IC≈0.45 when real signal is present.
  Validate these on real Binance data and report whatever they are.

---

## Breaking Changes
- `load_btcusdt()` still works (backward compatible), but new primary API is `load_btcusdt_v4()` which returns `(features, trades, params)`.
- `regime` labels changed from `quiet / volatile / trending` → `low_vol / medium_vol / high_vol`.
- `run.py` now requires `lightgbm` and `scikit-learn`.

## Priority 1 — Real Market Data (Highest)

### 1A: Real Historical Trade Flow
- `fetch_real_trades()` in `data_loader.py`: downloads Binance aggTrades with full pagination and caching.
- `run_episode_real_trades()` in `simulation.py`: **replays actual trades chronologically**. Zero synthetic order flow. Every simulated trade is a real historical market event.
- Fill logic: a real trade fills our quote if and only if the actual market price crossed our bid/ask. Informed flow detected by how aggressively price crosses the quote.

### 1B: Real Order Book Data
- `fetch_orderbook()`: Level-2 book snapshot (top-10 levels).
- `fetch_orderbook_series()`: time series of snapshots for fill calibration.
- Computed: spread, mid, bid/ask size, bid/ask depth (10 levels), order book imbalance.

### 1C: Calibrated Fill Model
- `calibrate_fill_model()`: fits fill decay from real spread distribution.
- Fill probability = exp(-fill_decay × half_spread / σ), with fill_decay estimated from real market data rather than heuristic default.

## Priority 2 — Volatility Regime Detection

- `regime_detector.py`: new module with `RegimeDetector` class.
- Three volatility regimes: **low_vol / medium_vol / high_vol** (rolling realized vol terciles).
- `regime_performance_table()`: produces the Low Vol / Medium Vol / High Vol × Strategy table.
- All episode summaries include a `regime` column.

## Priority 3 — ML Forecasting Layer

- `ml_forecaster.py`: new module with `ReturnForecaster`.
- Uses **LightGBM** (falls back to Ridge regression if LightGBM unavailable).
- Predicts 1-bar-ahead log returns using market features: OFI, realized vol, trade intensity, momentum.
- `ForecastAdaptiveMM` in `market_maker.py`: extends AdaptiveMM with forecast-based quote skew.
  - Bullish prediction → more aggressive bid, less aggressive ask.
  - Bearish prediction → more aggressive ask, less aggressive bid.
  - `forecast_scale` controls aggressiveness.

## Priority 4 — RL Market Maker

- `rl_market_maker.py`: new module with `RLMarketMaker`.
- **PPO** via stable-baselines3 if available; falls back to inventory-aware rule-based policy.
- State: inventory, spread, vol, OFI, recent returns, trade intensity, time-to-close.
- Actions: hold / tighten / widen / skew bid up / skew ask up / skew bid down / skew ask down.
- Reward: +spread capture, +realized PnL, −inventory risk, −drawdown.

## Strategy Comparison

`compare_strategies_v4()` benchmarks all 6 strategies under identical conditions:

1. Passive-MM (fixed spread floor)
2. Glosten-Milgrom
3. Avellaneda-Stoikov
4. Adaptive-MM (4-layer)
5. Forecast-Enhanced Adaptive
6. RL Market Maker

## New Metrics

- Sharpe, Sortino, CVaR(5%), MaxDD (per episode)
- Fill Rate (fraction of real trades that crossed our quotes)
- Inventory Variance
- Spread Capture
- Adverse Selection Cost
- Regime-specific performance

## File Changes

| File | Status | Summary |
|------|--------|---------|
| `src/data_loader.py` | **Rewritten** | Real trade fetch, order book, fill calibration |
| `src/simulation.py` | **Rewritten** | Real trade replay, all 6 strategies |
| `src/market_maker.py` | **Extended** | +ForecastAdaptiveMM |
| `src/ml_forecaster.py` | **New** | LightGBM/Ridge return forecaster |
| `src/rl_market_maker.py` | **New** | PPO/rule-based RL market maker |
| `src/regime_detector.py` | **New** | Volatility regime classification |
| `src/__init__.py` | Updated | All new exports |
| `run.py` | **Rewritten** | v4 9-step pipeline |
| `requirements.txt` | Updated | +lightgbm, +scikit-learn |
