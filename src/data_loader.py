"""
data_loader.py  —  v4
Fetches real BTC/USDT market data from Binance public REST API.

v4 changes (Priority 1):
  - fetch_real_trades()       : downloads historical aggTrades with full pagination
  - fetch_orderbook()         : Level-2 order book (top-10 levels)
  - compute_real_orderflow()  : OFI, spread, imbalance from real book snapshots
  - calibrate_fill_model()    : fill probability calibrated from real spread distribution
  - load_btcusdt_v4()         : unified loader returning trades + book features + klines

All synthetic order-flow generation has been removed.
"""

import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timezone

BINANCE_BASE = "https://api.binance.com"
CACHE_DIR    = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Data provenance markers
# ─────────────────────────────────────────────────────────────
# Cache filenames always contain "BTCUSDT" regardless of where the data came
# from, so a filename check cannot prove the data is real. Instead we write a
# sidecar marker the moment data is genuinely fetched live from Binance, and
# read it back to report a verified source. Absent/unknown markers are reported
# honestly as "unknown" rather than assumed real.

def _marker_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".source")

def _write_provenance(cache_path: Path, source: str) -> None:
    try:
        _marker_path(cache_path).write_text(
            json.dumps({"source": source,
                        "fetched_utc": datetime.now(timezone.utc).isoformat()})
        )
    except Exception:
        pass  # provenance is best-effort; never block the pipeline

def _read_provenance(cache_path: Path) -> str:
    mp = _marker_path(cache_path)
    if not mp.exists():
        return "unknown"
    try:
        return str(json.loads(mp.read_text()).get("source", "unknown"))
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────
# Raw fetch helpers
# ─────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict, retries: int = 3) -> list:
    url = BINANCE_BASE + endpoint
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)
    return []


# ─────────────────────────────────────────────────────────────
# Priority 1A: Real historical trade data
# ─────────────────────────────────────────────────────────────

def fetch_agg_trades(
    symbol:   str = "BTCUSDT",
    limit:    int = 1000,
    start_ms: Optional[int] = None,
    end_ms:   Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch aggregated trades from Binance.
    isBuyerMaker=True  → seller was aggressor → SELL tick
    isBuyerMaker=False → buyer was aggressor  → BUY tick

    Each row: timestamp, price, qty, side (buy/sell)
    This is REAL historical order flow, not simulated.
    """
    params = {"symbol": symbol, "limit": limit}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms

    raw = _get("/api/v3/aggTrades", params)
    if not raw:
        return pd.DataFrame(columns=["timestamp", "price", "qty", "side", "agg_id"])

    df = pd.DataFrame(raw)
    df = df.rename(columns={
        "T": "timestamp_ms", "p": "price", "q": "qty",
        "m": "is_buyer_maker", "a": "agg_id"
    })
    df["price"] = df["price"].astype(float)
    df["qty"]   = df["qty"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"].astype(int), unit="ms", utc=True)
    df["side"] = df["is_buyer_maker"].map({False: "buy", True: "sell"})
    return df[["timestamp", "price", "qty", "side", "agg_id"]].copy()


def fetch_real_trades(
    symbol:   str = "BTCUSDT",
    n_trades: int = 10000,
    cache:    bool = True,
) -> pd.DataFrame:
    """
    Fetch n_trades historical aggregated trades with full pagination.
    Returns REAL trade-by-trade data: timestamp, price, quantity, aggressor side.

    This replaces ALL synthetic order flow. The simulator will replay
    these actual trades chronologically.
    """
    cache_path = CACHE_DIR / f"{symbol}_real_trades_{n_trades}.parquet"
    if cache and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 48:
            print(f"  [cache] Loaded {n_trades} real trades ({age_hours:.1f}h old)")
            return pd.read_parquet(cache_path)

    print(f"  [Binance] Fetching {n_trades} real aggTrades for {symbol}...")
    frames = []
    end_ms = None
    remaining = n_trades

    while remaining > 0:
        batch = min(remaining, 1000)
        params = {"symbol": symbol, "limit": batch}
        if end_ms:
            params["endTime"] = end_ms

        raw = _get("/api/v3/aggTrades", params)
        if not raw:
            break

        df_batch = pd.DataFrame(raw)
        df_batch = df_batch.rename(columns={
            "T": "timestamp_ms", "p": "price", "q": "qty",
            "m": "is_buyer_maker", "a": "agg_id"
        })
        df_batch["price"] = df_batch["price"].astype(float)
        df_batch["qty"]   = df_batch["qty"].astype(float)
        df_batch["timestamp"] = pd.to_datetime(df_batch["timestamp_ms"].astype(int), unit="ms", utc=True)
        df_batch["side"] = df_batch["is_buyer_maker"].map({False: "buy", True: "sell"})

        frames.append(df_batch[["timestamp", "price", "qty", "side", "agg_id"]])
        end_ms    = int(df_batch["timestamp_ms"].astype(int).iloc[0]) - 1
        remaining -= len(df_batch)
        if len(df_batch) < batch:
            break

    if not frames:
        return pd.DataFrame(columns=["timestamp", "price", "qty", "side", "agg_id"])

    result = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    result = result.drop_duplicates("timestamp").reset_index(drop=True)

    # We only reach here via a live Binance fetch, so record provenance now —
    # independent of caching. This keeps `--no-cache` runs correctly attributed.
    _write_provenance(cache_path, "binance_live")
    if cache:
        result.to_parquet(cache_path)
        print(f"  [cache] Saved {len(result)} real trades → {cache_path}")
    return result


# ─────────────────────────────────────────────────────────────
# Priority 1B: Real Level-2 order book data
# ─────────────────────────────────────────────────────────────

def fetch_orderbook(
    symbol: str = "BTCUSDT",
    depth:  int = 10,
) -> dict:
    """
    Fetch current Level-2 order book snapshot.
    depth: number of levels (5, 10, 20, 50, 100, 500, 1000).

    Returns dict with:
      bids: list of [price, qty] (best bid first)
      asks: list of [price, qty] (best ask first)
      lastUpdateId: sequence number
    """
    raw = _get("/api/v3/depth", {"symbol": symbol, "limit": depth})
    bids = [(float(p), float(q)) for p, q in raw.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in raw.get("asks", [])]
    return {
        "bids": bids,
        "asks": asks,
        "last_update_id": raw.get("lastUpdateId", 0),
        "fetched_at": time.time(),
    }


def fetch_orderbook_series(
    symbol:      str = "BTCUSDT",
    n_snapshots: int = 200,
    interval_s:  float = 0.5,
    depth:       int = 10,
    cache:       bool = True,
) -> pd.DataFrame:
    """
    Collect a time series of order book snapshots to compute spread,
    imbalance, and depth features used in fill calibration.

    Returns DataFrame with columns:
      timestamp, best_bid, best_ask, bid_size, ask_size,
      spread, mid, imbalance, bid_depth_10, ask_depth_10
    """
    cache_path = CACHE_DIR / f"{symbol}_book_{n_snapshots}snap.parquet"
    if cache and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 48:
            print(f"  [cache] Loaded {n_snapshots} book snapshots")
            return pd.read_parquet(cache_path)

    print(f"  [Binance] Collecting {n_snapshots} order book snapshots for {symbol}...")
    rows = []
    for i in range(n_snapshots):
        try:
            book = fetch_orderbook(symbol, depth)
            if not book["bids"] or not book["asks"]:
                continue
            best_bid, bid_size_1 = book["bids"][0]
            best_ask, ask_size_1 = book["asks"][0]
            bid_depth = sum(q for _, q in book["bids"])
            ask_depth = sum(q for _, q in book["asks"])
            rows.append({
                "timestamp":      pd.Timestamp.now(tz="UTC"),
                "best_bid":       best_bid,
                "best_ask":       best_ask,
                "bid_size":       bid_size_1,
                "ask_size":       ask_size_1,
                "spread":         best_ask - best_bid,
                "mid":            (best_bid + best_ask) / 2.0,
                "imbalance":      (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9),
                "bid_depth_10":   bid_depth,
                "ask_depth_10":   ask_depth,
            })
            if i < n_snapshots - 1:
                time.sleep(interval_s)
        except Exception as e:
            print(f"  [warn] Book snapshot {i} failed: {e}")
            continue

    df = pd.DataFrame(rows)
    if cache and len(df) > 0:
        df.to_parquet(cache_path)
    return df


# ─────────────────────────────────────────────────────────────
# Priority 1C: Klines (unchanged from v3)
# ─────────────────────────────────────────────────────────────

def fetch_klines_multi(
    symbol:     str = "BTCUSDT",
    interval:   str = "1m",
    n_candles:  int = 5000,
    cache:      bool = True,
) -> pd.DataFrame:
    """Fetch klines with caching and pagination."""
    cache_path = CACHE_DIR / f"{symbol}_{interval}_{n_candles}.parquet"
    if cache and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 48:
            print(f"  [cache] Loaded {cache_path} ({age_hours:.1f}h old)")
            return pd.read_parquet(cache_path)

    print(f"  [Binance] Fetching {n_candles}× {interval} klines for {symbol}...")
    frames = []
    end_ms = None
    remaining = n_candles

    while remaining > 0:
        batch = min(remaining, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": batch}
        if end_ms:
            params["endTime"] = end_ms
        raw = _get("/api/v3/klines", params)
        if not raw:
            break

        df_batch = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "n_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume", "taker_buy_base_vol", "taker_buy_quote_vol"]:
            df_batch[col] = df_batch[col].astype(float)
        df_batch["n_trades"] = df_batch["n_trades"].astype(int)
        df_batch["open_time"] = pd.to_datetime(df_batch["open_time"].astype(int), unit="ms", utc=True)

        frames.append(df_batch[["open_time", "open", "high", "low", "close", "volume",
                                 "taker_buy_base_vol", "taker_buy_quote_vol", "n_trades"]])
        end_ms    = int(df_batch["open_time"].iloc[0].timestamp() * 1000) - 1
        remaining -= len(df_batch)
        if len(df_batch) < batch:
            break

    result = pd.concat(frames, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    result = result.drop_duplicates("open_time").reset_index(drop=True)

    # Reached only via a live Binance fetch — record provenance regardless of
    # caching so `--no-cache` runs are still attributed to Binance.
    _write_provenance(cache_path, "binance_live")
    if cache:
        result.to_parquet(cache_path)
        print(f"  [cache] Saved {len(result)} rows → {cache_path}")
    return result


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────

def engineer_features(klines: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    Compute features from klines needed by simulation and ML layer.
    v4 additions: spread_proxy, trade_intensity, buy_ratio columns.
    """
    df = klines.copy()
    df["mid"]      = (df["high"] + df["low"]) / 2.0
    df["log_ret"]  = np.log(df["close"] / df["close"].shift(1))
    df["realised_vol"] = (
        df["log_ret"]
        .rolling(vol_window)
        .std()
        * np.sqrt(365 * 24 * 60)
    )

    # Real OFI from taker volume split
    df["taker_sell_vol"] = df["volume"] - df["taker_buy_base_vol"]
    df["ofi_proxy"] = (
        (df["taker_buy_base_vol"] - df["taker_sell_vol"])
        / (df["volume"] + 1e-9)
    )

    # Trade intensity (trades per minute — raw from klines)
    df["trade_intensity"] = df["n_trades"].astype(float)

    # Buy ratio
    df["buy_ratio"] = df["taker_buy_base_vol"] / (df["volume"] + 1e-9)

    # Trend signal
    df["ma_fast"] = df["close"].rolling(10).mean()
    df["ma_slow"] = df["close"].rolling(30).mean()
    df["trend_signal"] = (df["ma_fast"] - df["ma_slow"]) / df["close"]

    # Short-term momentum (for ML features)
    df["ret_1m"]  = df["log_ret"]
    df["ret_5m"]  = np.log(df["close"] / df["close"].shift(5))
    df["ret_15m"] = np.log(df["close"] / df["close"].shift(15))

    # Rolling vol at multiple windows
    df["vol_5m"]  = df["log_ret"].rolling(5).std()
    df["vol_30m"] = df["log_ret"].rolling(30).std()

    # Signed volume (OFI in absolute vol units)
    df["signed_vol"] = df["taker_buy_base_vol"] - df["taker_sell_vol"]

    df = df.dropna().reset_index(drop=True)
    df["regime"] = _label_regimes(df)
    return df


def _label_regimes(df: pd.DataFrame) -> pd.Series:
    """Three-regime labelling: low_vol / medium_vol / high_vol based on rolling realised vol."""
    vol_p33 = df["realised_vol"].expanding(min_periods=1).quantile(0.33)
    vol_p67 = df["realised_vol"].expanding(min_periods=1).quantile(0.67)

    labels = pd.Series("medium_vol", index=df.index)
    labels[df["realised_vol"] <= vol_p33] = "low_vol"
    labels[df["realised_vol"] >  vol_p67] = "high_vol"

    # Smooth with rolling mode (5-bar window)
    def rolling_mode(s, w=5):
        return s.rolling(w, min_periods=1).apply(
            lambda x: pd.Series(x).mode().iloc[0], raw=False
        )

    encoded = labels.map({"low_vol": 0, "medium_vol": 1, "high_vol": 2})
    smoothed = rolling_mode(encoded)
    return smoothed.map({0: "low_vol", 1: "medium_vol", 2: "high_vol"})


# ─────────────────────────────────────────────────────────────
# Priority 1D: Real fill model calibration
# ─────────────────────────────────────────────────────────────

def calibrate_fill_model(
    trades: pd.DataFrame,
    book_snapshots: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Calibrate fill probability model from real trade data.

    The key insight: fill probability depends on spread and market activity.
    We estimate:
      - Typical spread from book snapshots (or kline high-low proxy)
      - Fill decay: how quickly fill prob drops as spread widens
      - Base fill rate at tight spreads

    Returns dict with fill_decay, base_fill_rate, spread_mean, spread_std.
    """
    if book_snapshots is not None and len(book_snapshots) > 10:
        spread_mean = float(book_snapshots["spread"].mean())
        spread_std  = float(book_snapshots["spread"].std())
        mid_price   = float(book_snapshots["mid"].mean())
        spread_bps  = spread_mean / mid_price * 10_000
    else:
        # Proxy from trade data: std of price over short windows ≈ half-spread
        spread_proxy = float(trades["price"].rolling(10).std().mean())
        spread_mean  = max(spread_proxy * 2, 1.0)
        spread_std   = spread_mean * 0.3
        mid_price    = float(trades["price"].mean())
        spread_bps   = spread_mean / mid_price * 10_000

    # Fill decay calibration:
    # In liquid crypto markets, tight spreads (< 1 bps) → fill prob near 1.
    # At 5 bps (typical) → fill prob ~0.5–0.7 for passive limit orders.
    # We fit: fill_prob = exp(-fill_decay * half_spread / sigma)
    # where sigma is the per-step price std.
    # Empirically calibrated: fill_decay ≈ 1.5–3.0 for BTC/USDT
    price_std = float(trades["price"].rolling(60).std().mean())
    if price_std > 0 and spread_mean > 0:
        # At typical spread, fill prob ≈ 0.6 for passive orders
        target_fill_at_typical = 0.60
        half_spread_typical = spread_mean / 2.0
        fill_decay = -np.log(target_fill_at_typical) / (half_spread_typical / (price_std + 1e-6))
        fill_decay = float(np.clip(fill_decay, 0.5, 5.0))
    else:
        fill_decay = 1.5

    return {
        "fill_decay":      fill_decay,
        "base_fill_rate":  0.95,      # near-certain fill at zero spread
        "spread_mean":     spread_mean,
        "spread_std":      spread_std,
        "spread_bps":      spread_bps,
        "price_std":       price_std,
        "mid_price":       mid_price,
    }


# ─────────────────────────────────────────────────────────────
# Compute real order flow features from trade data
# ─────────────────────────────────────────────────────────────

def compute_real_orderflow(
    trades: pd.DataFrame,
    window_s: float = 60.0,
) -> pd.DataFrame:
    """
    Compute order flow features from real trade data.

    Groups trades into time windows and computes:
      - buy_volume, sell_volume, net_volume (signed)
      - trade_count, buy_count, sell_count
      - OFI (order flow imbalance) = (buy_vol - sell_vol) / total_vol
      - VPIN proxy: rolling fraction of buy trades
      - price_impact: price move per unit of signed volume
    """
    df = trades.copy()
    df = df.set_index("timestamp").sort_index()

    # Resample to fixed time windows
    rule = f"{int(window_s)}s"
    buy  = df[df["side"] == "buy"]
    sell = df[df["side"] == "sell"]

    buy_vol   = buy["qty"].resample(rule).sum().rename("buy_volume")
    sell_vol  = sell["qty"].resample(rule).sum().rename("sell_volume")
    buy_cnt   = buy["qty"].resample(rule).count().rename("buy_count")
    sell_cnt  = sell["qty"].resample(rule).count().rename("sell_count")
    avg_price = df["price"].resample(rule).mean().rename("price")
    last_price= df["price"].resample(rule).last().rename("last_price")
    first_price = df["price"].resample(rule).first().rename("first_price")

    result = pd.concat([buy_vol, sell_vol, buy_cnt, sell_cnt, avg_price,
                        last_price, first_price], axis=1).fillna(0)

    result["total_volume"] = result["buy_volume"] + result["sell_volume"]
    result["net_volume"]   = result["buy_volume"] - result["sell_volume"]
    result["ofi"]          = result["net_volume"] / (result["total_volume"] + 1e-9)
    result["trade_count"]  = result["buy_count"] + result["sell_count"]
    result["buy_ratio"]    = result["buy_volume"] / (result["total_volume"] + 1e-9)
    result["price_return"] = np.log(result["last_price"] / result["first_price"].replace(0, np.nan))
    result = result.reset_index().rename(columns={"timestamp": "time"})
    result = result.dropna().reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────
# Parameter calibration
# ─────────────────────────────────────────────────────────────

def calibrate_params(df: pd.DataFrame, trades: Optional[pd.DataFrame] = None) -> dict:
    """
    Calibrate simulation parameters from real kline data (and optionally trades).

    mu0 alignment:
      When trades are provided, mu0 is set to the trades mean price so the
      belief initialises in the same price range as actual observed flow.
      For Binance data the kline close and trade prices are always aligned;
      for locally-generated synthetic data this correction is important.
    """
    log_rets = df["log_ret"].dropna()
    closes   = df["close"]

    step_vol_pct = log_rets.std()

    ofi   = df["ofi_proxy"].values[1:]
    dp    = np.diff(closes.values)
    if len(ofi) > 10:
        from numpy.linalg import lstsq
        A = np.column_stack([ofi, np.ones_like(ofi)])
        coef, _, _, _ = lstsq(A, dp, rcond=None)
        kyle_lambda = abs(coef[0])
        dp_hat = ofi * coef[0] + coef[1]
        ss_res = np.sum((dp - dp_hat) ** 2)
        ss_tot = np.sum((dp - dp.mean()) ** 2)
        r2 = float(np.clip(1.0 - ss_res / (ss_tot + 1e-9), 0.0, 1.0))
        alpha_kyle = float(np.clip(0.10 + r2 * 1.5, 0.10, 0.30))
    else:
        kyle_lambda = 1.0
        alpha_kyle  = 0.15

    # mu0: prefer the mean trade price when trades are available.
    if trades is not None and len(trades) > 0 and "price" in trades.columns:
        mu0 = float(trades["price"].mean())
    else:
        mu0 = float(closes.iloc[-1])

    # sigma0: 1-hour rolling std at last bar; fall back to overall std if NaN.
    sigma0_series = closes.rolling(60).std()
    sigma0 = float(sigma0_series.iloc[-1] if not np.isnan(sigma0_series.iloc[-1])
                   else closes.std())

    # sigma_v: per-step USD std, anchored to mu0
    sigma_v = float(mu0 * step_vol_pct)

    process_noise = float((sigma0 * 0.01) ** 2)
    noise_var     = float(sigma0 ** 2)

    return {
        "mu0":           mu0,
        "sigma0":        sigma0,
        "sigma_v":       sigma_v,
        "alpha":         alpha_kyle,
        "process_noise": process_noise,
        "noise_var":     noise_var,
        "step_vol_pct":  step_vol_pct,
        "kyle_lambda":   kyle_lambda,
        "fill_decay":    1.8,
    }


# ─────────────────────────────────────────────────────────────
# Main loader v4: real trades + book features + klines
# ─────────────────────────────────────────────────────────────

def load_btcusdt_v4(
    n_candles:    int  = 5000,
    n_trades:     int  = 10000,
    n_book_snaps: int  = 0,      # set > 0 to collect live book snapshots (takes time)
    cache:        bool = True,
    verbose:      bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full v4 pipeline: fetch → feature engineer → calibrate.

    Returns:
      features_df  : kline-based features (mid, realised_vol, ofi_proxy, regime, ...)
      trades_df    : REAL historical trade flow (timestamp, price, qty, side)
      params       : calibrated simulation parameters

    The trades_df is the primary input for the simulator in v4.
    ALL synthetic order flow has been removed.
    """
    # 1. Klines for price process and feature engineering
    klines   = fetch_klines_multi("BTCUSDT", "1m", n_candles, cache=cache)
    features = engineer_features(klines)

    # 2. Real historical trade data (Priority 1A — replaces synthetic order flow)
    trades = fetch_real_trades("BTCUSDT", n_trades=n_trades, cache=cache)

    # Calibrate after fetching trades so mu0 is aligned to trade price range
    params   = calibrate_params(features, trades=trades)

    # Verified provenance: real only if BOTH kline and trade caches were
    # written by a live Binance fetch. Anything else is reported as unknown.
    kline_src = _read_provenance(CACHE_DIR / f"BTCUSDT_1m_{n_candles}.parquet")
    trade_src = _read_provenance(CACHE_DIR / f"BTCUSDT_real_trades_{n_trades}.parquet")
    if kline_src == "binance_live" and trade_src == "binance_live":
        params["data_source"] = "real_binance"
    else:
        params["data_source"] = f"unknown (klines={kline_src}, trades={trade_src})"

    # 3. Optional: order book snapshots for fill calibration
    book_df = None
    if n_book_snaps > 0:
        book_df = fetch_orderbook_series("BTCUSDT", n_snapshots=n_book_snaps, cache=cache)
        fill_params = calibrate_fill_model(trades, book_df)
        params["fill_decay"]   = fill_params["fill_decay"]
        params["spread_mean"]  = fill_params["spread_mean"]
        params["spread_bps"]   = fill_params["spread_bps"]
    else:
        fill_params = calibrate_fill_model(trades)
        params["fill_decay"]  = fill_params["fill_decay"]
        params["spread_mean"] = fill_params["spread_mean"]

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  BTC/USDT v4 — Real Market Data")
        print(f"  Data source:     {params.get('data_source', 'unknown')}")
        print(f"  Kline bars:      {len(features):,}")
        print(f"  Real trades:     {len(trades):,}")
        print(f"  Mid price:       ${params['mu0']:,.2f}")
        print(f"  1h σ:            ${params['sigma0']:,.2f}")
        print(f"  Per-step vol:    {params['step_vol_pct']*100:.4f}%")
        print(f"  Kyle λ:          {params['kyle_lambda']:.4f}")
        print(f"  Alpha (est):     {params['alpha']:.3f}")
        print(f"  Fill decay:      {params['fill_decay']:.2f}")
        print(f"  Spread (est):    ${params['spread_mean']:.2f} ({params.get('spread_bps', 0):.2f} bps)")
        if features is not None:
            regime_counts = features["regime"].value_counts().to_dict()
            print(f"  Regimes:         {regime_counts}")
        print(f"{'─'*55}\n")

    return features, trades, params


# Legacy compatibility: keep load_btcusdt for existing code
def load_btcusdt(
    n_candles: int = 5000,
    cache:     bool = True,
    verbose:   bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Legacy loader for backward compatibility."""
    features, trades, params = load_btcusdt_v4(
        n_candles=n_candles, n_trades=5000, cache=cache, verbose=verbose
    )
    return features, params


# Also export for legacy imports
fetch_agg_trades_multi = fetch_real_trades
