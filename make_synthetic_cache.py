"""
make_synthetic_cache.py  —  LOCAL REPRODUCTION ONLY (NOT real Binance data)

Generates synthetic BTCUSDT klines + aggTrades parquet files in the schema
expected by src/data_loader.py, so the full pipeline can be exercised offline
in an environment where api.binance.com is unreachable.

IMPORTANT: every number produced downstream from this cache is SYNTHETIC and
must NOT be reported as a validation of the research claims. It exists solely
to reproduce the runtime failure and verify the code fixes.
"""
import numpy as np
import pandas as pd
from pathlib import Path

CACHE = Path("data_cache")
CACHE.mkdir(exist_ok=True)

N_CANDLES = 5000
N_TRADES  = 10000
SEED      = 7

rng = np.random.default_rng(SEED)

# ── 1. Klines (1-minute) ────────────────────────────────────────────────
# Build a price path with mild autocorrelation in returns so that the
# forecaster has *some* (weak, realistic) signal to pick up, plus
# volatility clustering so the regime labels are non-degenerate.
start_price = 60000.0
mins = N_CANDLES

# volatility clustering via simple GARCH-like process
vol = np.zeros(mins)
vol[0] = 0.0008
for t in range(1, mins):
    vol[t] = np.sqrt(1e-8 + 0.92 * vol[t-1]**2 + 0.05 * (rng.normal()*vol[t-1])**2)

# returns with weak short-term mean reversion (AR(1) with negative coef)
eps = rng.normal(size=mins) * vol
ret = np.zeros(mins)
for t in range(1, mins):
    ret[t] = -0.06 * ret[t-1] + eps[t]

close = start_price * np.exp(np.cumsum(ret))
open_ = np.concatenate([[start_price], close[:-1]])
# intrabar high/low
hl_spread = np.abs(rng.normal(size=mins)) * vol * close
high = np.maximum(open_, close) + hl_spread
low  = np.minimum(open_, close) - hl_spread

# volume correlated with |return| (activity rises with moves)
base_vol = 40.0
volume = base_vol * (1.0 + 5.0 * np.abs(ret) / (vol + 1e-9)) * np.exp(rng.normal(0, 0.3, mins))
volume = np.clip(volume, 1.0, None)

# taker buy fraction correlated with contemporaneous return (buy pressure → up)
buy_frac = np.clip(0.5 + 8.0 * ret + rng.normal(0, 0.05, mins), 0.05, 0.95)
taker_buy_base_vol = volume * buy_frac
taker_buy_quote_vol = taker_buy_base_vol * close
quote_asset_volume = volume * close
n_trades = np.clip((volume * rng.uniform(8, 15, mins)).astype(int), 1, None)

open_time = pd.date_range("2024-01-01", periods=mins, freq="1min", tz="UTC")

klines = pd.DataFrame({
    "open_time": open_time,
    "open": open_, "high": high, "low": low, "close": close,
    "volume": volume,
    "taker_buy_base_vol": taker_buy_base_vol,
    "taker_buy_quote_vol": taker_buy_quote_vol,
    "n_trades": n_trades,
})
kpath = CACHE / f"BTCUSDT_1m_{N_CANDLES}.parquet"
klines.to_parquet(kpath)
print(f"[synthetic] klines  -> {kpath}  ({len(klines)} rows)")

# ── 2. aggTrades ────────────────────────────────────────────────────────
# Sample trade times across the kline window; price follows the minute close
# with sub-minute noise; side correlated with local return sign.
n = N_TRADES
t_idx = np.sort(rng.integers(0, mins, size=n))
minute_close = close[t_idx]
px_noise = rng.normal(0, 1.0, n) * (vol[t_idx] * minute_close * 0.5)
price = minute_close + px_noise

# qty: heavy-tailed (lognormal) so the "informed = qty>2x rolling mean" path triggers
qty = np.exp(rng.normal(-2.0, 1.0, n))  # median ~0.135 BTC, occasional whales
local_ret = ret[t_idx]
p_buy = np.clip(0.5 + 6.0 * local_ret, 0.05, 0.95)
side = np.where(rng.random(n) < p_buy, "buy", "sell")
ts = open_time[t_idx] + pd.to_timedelta(rng.integers(0, 60_000, n), unit="ms")

trades = pd.DataFrame({
    "timestamp": ts,
    "price": price.astype(float),
    "qty": qty.astype(float),
    "side": side,
    "agg_id": np.arange(n),
}).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

tpath = CACHE / f"BTCUSDT_real_trades_{N_TRADES}.parquet"
trades.to_parquet(tpath)
# Honest provenance markers so the pipeline reports this as NON-real data.
import json as _json
(CACHE / f"BTCUSDT_1m_{N_CANDLES}.parquet.source").write_text(_json.dumps({"source": "synthetic_local"}))
(CACHE / f"BTCUSDT_real_trades_{N_TRADES}.parquet.source").write_text(_json.dumps({"source": "synthetic_local"}))
print(f"[synthetic] trades  -> {tpath}  ({len(trades)} rows)")
print(f"[synthetic] buy frac = {(trades['side']=='buy').mean():.3f}  "
      f"price range = {trades['price'].min():.0f}-{trades['price'].max():.0f}")
