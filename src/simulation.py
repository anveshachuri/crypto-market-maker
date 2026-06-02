"""
simulation.py  —  v4
Episode runner for the BTC/USDT market-making simulation.

v4 changes (Priority 1):
  - Real trade replay: simulator consumes actual Binance aggTrades chronologically
  - No synthetic order flow: TraderPopulation is retained for multi-episode
    hybrid runs but real trades are used in replay mode
  - fill model: calibrated from real spread data (fill_decay from params)
  - All 6 strategies: PassiveMM, GM, AS, AdaptiveMM, ForecastAdaptiveMM, RLMarketMaker

Price process:
  - ReplayProcess: real Binance mid prices
  - HybridProcess: calibrated synthetic (for multi-episode runs)

Real trade replay (run_episode_real_trades):
  - Iterates through actual historical trades chronologically
  - Each trade is a real market event: price, qty, side
  - No synthetic order arrival model needed
  - Fill model: exponential decay calibrated from real spread data

Public API:
  run_episode_real_trades()  : replay actual trades (PRIMARY v4 path)
  run_episode_replay()       : replay real prices w/ synthetic order flow
  run_episode_hybrid()       : synthetic prices (multi-episode stats)
  run_many_episodes()        : multi-episode runner
  compare_strategies_v4()    : all 6 strategies on identical data
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import copy

from .price_process  import ReplayProcess, HybridProcess
from .traders        import TraderPopulation
from .belief         import GaussianBelief
from .market_maker   import (
    GlostenMilgromMM, AvellanedaStoikovMM, AdaptiveMM,
    PassiveMM, OracleMM, ForecastAdaptiveMM,
)
from .rl_market_maker import RLMarketMaker


@dataclass
class CryptoSimConfig:
    """
    Simulation configuration for v4.
    """
    n_steps: int = 500

    mu0:     float = 50_000.0
    sigma0:  float = 200.0
    sigma_v: float = 25.0

    noise_var:     float = 100.0
    process_noise: float = 10.0

    alpha:         float = 0.25
    momentum_frac: float = 0.08
    buy_prob:      float = 0.5

    mm_type:     str   = "adaptive"
    gamma:       float = 0.1
    kappa_mm:    float = 1.5
    min_spread:  float = 1.0
    max_spread:  float = 500.0
    inv_limit:   float = 0.5

    penalty_lambda:  float = 0.02
    inv_scale:       float = 0.3
    as_multiplier:   float = 2.0
    vol_gamma:       float = 0.5
    ofi_sensitivity: float = 0.3
    fill_decay:      float = 1.8   # v4: calibrated from real spread data

    seed: Optional[int] = 42

    # v4: RL and forecast params
    rl_train_steps:   int   = 20_000
    forecast_scale:   float = 5.0      # how aggressively to skew quotes on forecast signal

    @classmethod
    def from_calibration(cls, params: dict, **overrides) -> "CryptoSimConfig":
        sigma0 = params["sigma0"]
        mu0    = params["mu0"]
        target_spread_usd = 0.0002 * mu0
        gamma = float(np.clip(target_spread_usd / (sigma0 ** 2 + 1e-6), 0.001, 0.05))

        cfg = cls(
            mu0            = mu0,
            sigma0         = sigma0,
            sigma_v        = params["sigma_v"],
            noise_var      = params["noise_var"],
            process_noise  = params["process_noise"],
            alpha          = params["alpha"],
            gamma          = gamma,
            fill_decay     = params.get("fill_decay", 1.8),
            min_spread     = max(1.0,  mu0 * 0.00005),
            max_spread     = min(500.0, sigma0 * 5.0),
            inv_limit      = 0.5,
            penalty_lambda = 0.05,
            inv_scale      = 0.15,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


def _build_mm(cfg: CryptoSimConfig, forecaster=None, rl_model=None):
    """Construct the market maker for a given cfg.mm_type."""
    if cfg.mm_type == "passive":
        return PassiveMM(min_spread=cfg.min_spread, max_spread=cfg.max_spread)
    elif cfg.mm_type == "gm":
        return GlostenMilgromMM(alpha=cfg.alpha, min_spread=cfg.min_spread, max_spread=cfg.max_spread)
    elif cfg.mm_type == "as":
        return AvellanedaStoikovMM(
            gamma=cfg.gamma, kappa=cfg.kappa_mm, alpha=cfg.alpha,
            min_spread=cfg.min_spread, max_spread=cfg.max_spread, inv_limit=cfg.inv_limit,
        )
    elif cfg.mm_type == "forecast":
        mm = ForecastAdaptiveMM(
            gamma=cfg.gamma, kappa=cfg.kappa_mm, alpha=cfg.alpha,
            min_spread=cfg.min_spread, max_spread=cfg.max_spread,
            inv_limit=cfg.inv_limit, penalty_lambda=cfg.penalty_lambda,
            inv_scale=cfg.inv_scale, as_multiplier=cfg.as_multiplier,
            vol_gamma=cfg.vol_gamma, ofi_sensitivity=cfg.ofi_sensitivity,
            fill_decay=cfg.fill_decay, forecast_scale=cfg.forecast_scale,
        )
        if forecaster is not None:
            mm.set_forecaster(forecaster)
        return mm
    elif cfg.mm_type == "rl":
        return rl_model if rl_model is not None else RLMarketMaker(
            gamma=cfg.gamma, inv_limit=cfg.inv_limit,
            min_spread=cfg.min_spread, max_spread=cfg.max_spread,
        )
    else:  # "adaptive"
        return AdaptiveMM(
            gamma=cfg.gamma, kappa=cfg.kappa_mm, alpha=cfg.alpha,
            min_spread=cfg.min_spread, max_spread=cfg.max_spread,
            inv_limit=cfg.inv_limit, penalty_lambda=cfg.penalty_lambda,
            inv_scale=cfg.inv_scale, as_multiplier=cfg.as_multiplier,
            vol_gamma=cfg.vol_gamma, ofi_sensitivity=cfg.ofi_sensitivity,
            fill_decay=cfg.fill_decay,
        )


# ─────────────────────────────────────────────────────────────
# Priority 1: Real trade replay (PRIMARY v4 simulation path)
# ─────────────────────────────────────────────────────────────

def run_episode_real_trades(
    cfg:        CryptoSimConfig,
    features:   pd.DataFrame,
    trades:     pd.DataFrame,
    offset:     int = 0,
    forecaster: object = None,
    rl_model:   object = None,
) -> pd.DataFrame:
    """
    Run one episode using REAL historical trade data.

    Quoting reference price fix (v4.1):
      The MM quotes are centered on the LAST OBSERVED TRADE PRICE (V),
      not on belief.mu. This matches how real market makers operate:
      they anchor to the last traded price / best bid-ask mid, then apply
      spread and inventory skew from that anchor.

      belief.mu continues to track the "true value" estimate (Bayesian
      inference from trade direction) which is used for:
        - alpha_hat (informed-flow toxicity estimate)
        - mu_error (research metric: how far our prior was from reality)
        - belief.sigma (uncertainty floor for spreads)

      Using belief.mu as the quoting anchor caused systematic inventory
      build-up because the Bayesian filter's variance collapses quickly,
      making it overconfident in a stale price estimate. The last trade
      price V is always a better anchor in a real-trade replay.

    Inventory guard:
      When |inventory| >= inv_limit, the MM stops quoting on the side
      that would increase it further (sets that quote to unprofitable level).
      This prevents runaway inventory from a persistent directional trend.
    """
    rng   = np.random.default_rng(cfg.seed)

    # Slice trades first so we can initialise belief at actual first trade price
    n_trades = len(trades)
    start    = offset % max(1, n_trades - cfg.n_steps)
    trade_ep = trades.iloc[start : start + cfg.n_steps].reset_index(drop=True)
    if len(trade_ep) < 10:
        trade_ep = trades.iloc[: cfg.n_steps].reset_index(drop=True)

    first_trade_price = float(trade_ep["price"].iloc[0]) if len(trade_ep) > 0 else cfg.mu0

    # Belief: tracks directional inference — NOT used as quoting anchor
    belief = GaussianBelief(
        mu0=first_trade_price, sigma0=cfg.sigma0, alpha=cfg.alpha,
        noise_var=cfg.noise_var, process_noise=cfg.process_noise,
    )

    mm     = _build_mm(cfg, forecaster=forecaster, rl_model=rl_model)
    oracle = OracleMM(tick=cfg.min_spread * 0.1)

    regimes = (
        features["regime"].values
        if "regime" in features.columns
        else np.full(len(features), "medium_vol")
    )
    ofi_arr = (
        features["ofi_proxy"].values
        if "ofi_proxy" in features.columns
        else np.zeros(len(features))
    )

    n_ep              = len(trade_ep)
    inventory         = 0.0;  cash = 0.0
    oracle_inv        = 0.0;  oracle_cash = 0.0
    spread_revenue    = 0.0
    adverse_sel_cost  = 0.0
    inventory_mtm_pnl = 0.0
    V_prev            = first_trade_price
    _fill_decay       = cfg.fill_decay
    _sigma_v          = cfg.sigma_v          # stable price-step vol for fill calibration
    _inv_limit        = cfg.inv_limit
    _min_spread       = cfg.min_spread

    records: List[Dict[str, Any]] = []

    for t, trade_row in trade_ep.iterrows():
        t_remaining = max((n_ep - t) / n_ep, 1e-3)

        V         = float(trade_row["price"])   # last real trade price — quoting anchor
        real_qty  = float(trade_row["qty"])
        real_side = str(trade_row["side"])

        feat_idx = min(start + t, len(features) - 1)
        regime   = regimes[feat_idx]
        ofi      = float(ofi_arr[feat_idx])

        belief.time_step()
        if hasattr(mm, "observe_price"):
            mm.observe_price(V)      # feed last trade price, not belief.mu

        if hasattr(mm, "update_features") and feat_idx < len(features):
            mm.update_features(features.iloc[feat_idx])

        alpha_hat = belief.alpha_hat

        # ── Quote around V (last trade price), not stale belief.mu ──
        bid_raw, ask_raw = mm.quote(
            V, belief.sigma, inventory, t_remaining, alpha_hat, ofi
        )

        # ── Inventory guard: neutralise the side that grows inventory ──
        # When at inventory limit, make one side uncompetitive so it won't fill.
        if inventory >= _inv_limit:
            # Already long: don't take more buys → lower bid far from market
            bid = V - cfg.max_spread * 2   # unreachable
            ask = ask_raw
        elif inventory <= -_inv_limit:
            # Already short: don't take more sells → raise ask far from market
            bid = bid_raw
            ask = V + cfg.max_spread * 2
        else:
            bid = bid_raw
            ask = ask_raw

        # Guarantee minimum spread (never negative)
        if ask - bid < _min_spread:
            mid_q = (bid + ask) / 2.0
            bid   = mid_q - _min_spread / 2.0
            ask   = mid_q + _min_spread / 2.0

        o_bid, o_ask = oracle.quote(V)
        half_spread  = (ask - bid) / 2.0
        # fill_prob: uses cfg.sigma_v (calibrated price-step volatility), not belief.sigma
        # belief.sigma is value uncertainty (collapses fast); sigma_v is price volatility (stable)
        fill_prob    = float(np.exp(-_fill_decay * half_spread / (_sigma_v + 1e-6)))

        # ── Fill model (probabilistic) ──────────────────────────────────────
        # Each real aggTrade is an order arrival event. Whether it fills our
        # passive quote depends on our spread width:
        #   P(fill | arrival) = exp(-fill_decay × half_spread / σ)
        # Wider spread → lower fill probability (we are deeper in the queue).
        #
        # Informed-flow detection from trade size:
        # Trades with qty > 2× recent rolling mean are classified as informed.
        # Large-size aggression carries stronger directional information.
        #
        # When inventory guard has neutralised one side (quotes moved far from
        # market), the fill_prob for that side is effectively zero (the guard
        # already set bid/ask to an unreachable level, so V will never reach it).
        # We set fill_prob=0 explicitly to avoid spurious fill classifications.
        # ─────────────────────────────────────────────────────────────────────
        trade_occurred = False
        actual_size    = 0.0
        trade_side     = "none"
        trader_type    = "none"
        trade_pnl      = 0.0
        oracle_pnl     = 0.0

        # Informed detection: qty > 2× rolling mean qty in recent window
        _recent_qty = float(trade_ep["qty"].iloc[max(0, t - 20):t + 1].mean())
        is_informed = real_qty > 2.0 * _recent_qty

        # Informed traders fill with near-certainty (they demand liquidity regardless)
        # Uninformed traders: probabilistic fill based on spread
        _effective_fill_prob = 0.95 if is_informed else fill_prob
        trader_type_candidate = "informed" if is_informed else "uninformed"

        filled = rng.random() < _effective_fill_prob

        if filled:
            if real_side == "buy":
                trade_side     = "buy"
                trader_type    = trader_type_candidate
                trade_occurred = True
                actual_size    = real_qty

                # We sold at our ask; execution price = ask (passive limit order)
                cash      += ask * actual_size
                inventory -= actual_size
                trade_pnl  = (ask - V) * actual_size
                oracle_cash += o_ask * actual_size
                oracle_inv  -= actual_size
                oracle_pnl   = (o_ask - V) * actual_size
                belief.update("buy", bid, ask)
                if is_informed:
                    # Adverse: we sold at ask but true value was higher → inventory shortfall
                    adverse_sel_cost += max(0, V - ask) * actual_size
                else:
                    spread_revenue   += half_spread * actual_size

            else:  # real_side == "sell"
                trade_side     = "sell"
                trader_type    = trader_type_candidate
                trade_occurred = True
                actual_size    = real_qty

                # We bought at our bid
                cash      -= bid * actual_size
                inventory += actual_size
                trade_pnl  = (V - bid) * actual_size
                oracle_cash -= o_bid * actual_size
                oracle_inv  += actual_size
                oracle_pnl   = (V - o_bid) * actual_size
                belief.update("sell", bid, ask)
                if is_informed:
                    adverse_sel_cost += max(0, bid - V) * actual_size
                else:
                    spread_revenue   += half_spread * actual_size

        inv_mtm_step       = inventory * (V - V_prev)
        inventory_mtm_pnl += inv_mtm_step
        mtm_pnl    = cash + inventory * V
        oracle_mtm = oracle_cash + oracle_inv * V

        records.append({
            "t":                  t,
            "V":                  V,
            "mu":                 belief.mu,
            "sigma":              belief.sigma,
            "alpha_hat":          belief.alpha_hat,
            "ofi":                ofi,
            "bid":                bid,
            "ask":                ask,
            "spread":             ask - bid,
            "spread_bps":         (ask - bid) / V * 10_000 if V > 0 else 0.0,
            "fill_prob":          fill_prob,
            "side":               trade_side,
            "trader_type":        trader_type,
            "trade_size":         actual_size,
            "real_trade_qty":     real_qty,
            "trade_occurred":     trade_occurred,
            "trade_pnl":          trade_pnl,
            "inventory":          inventory,
            "cash":               cash,
            "mtm_pnl":            mtm_pnl,
            "oracle_mtm":         oracle_mtm,
            "regret":             oracle_mtm - mtm_pnl,
            "mu_error":           belief.mu - V,
            "mu_error_bps":       (belief.mu - V) / V * 10_000 if V > 0 else 0.0,
            "t_remaining":        t_remaining,
            "regime":             regime,
            "spread_revenue":     spread_revenue,
            "adverse_sel_cost":   adverse_sel_cost,
            "inventory_mtm_pnl":  inventory_mtm_pnl,
            "momentum_cost":      0.0,
        })
        V_prev = V

    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────
# Legacy episode runners (retained from v3)
# ─────────────────────────────────────────────────────────────

def _run_episode_core(
    cfg:        CryptoSimConfig,
    process,
    ofi_series: Optional[np.ndarray] = None,
    forecaster: object = None,
    rl_model:   object = None,
) -> pd.DataFrame:
    """Shared episode runner with synthetic order flow (hybrid/multi-episode mode)."""
    rng      = np.random.default_rng(cfg.seed)
    traders  = TraderPopulation(
        alpha=cfg.alpha, momentum_frac=cfg.momentum_frac,
        buy_prob_uninformed=cfg.buy_prob, seed=cfg.seed,
    )
    belief   = GaussianBelief(
        mu0=cfg.mu0, sigma0=cfg.sigma0, alpha=cfg.alpha,
        noise_var=cfg.noise_var, process_noise=cfg.process_noise,
    )
    mm     = _build_mm(cfg, forecaster=forecaster, rl_model=rl_model)
    oracle = OracleMM(tick=cfg.min_spread * 0.1)
    _fill_decay = cfg.fill_decay

    inventory = 0.0; cash = 0.0
    oracle_inv = 0.0; oracle_cash = 0.0
    spread_revenue    = 0.0
    adverse_sel_cost  = 0.0
    inventory_mtm_pnl = 0.0
    momentum_cost     = 0.0
    records: List[Dict[str, Any]] = []
    V_prev = cfg.mu0

    for t in range(cfg.n_steps):
        t_remaining = max((cfg.n_steps - t) / cfg.n_steps, 1e-3)
        V           = process.step()
        belief.time_step()
        mid_prev = belief.mu
        traders.observe_mid(mid_prev)
        if hasattr(mm, "observe_price"):
            mm.observe_price(mid_prev)

        alpha_hat = belief.alpha_hat
        ofi = float(ofi_series[t]) if ofi_series is not None and t < len(ofi_series) else belief.order_flow_imbalance

        bid, ask   = mm.quote(belief.mu, belief.sigma, inventory, t_remaining, alpha_hat, ofi)
        o_bid, o_ask = oracle.quote(V)
        half_spread = (ask - bid) / 2.0
        side, size, trader_type = traders.arrive(V, bid, ask)

        fill_prob = float(np.exp(-_fill_decay * half_spread / (belief.sigma + 1e-6)))
        if trader_type == "informed":
            filled = (side != "none")
        else:
            filled = (side != "none") and (rng.random() < fill_prob)

        trade_pnl = oracle_pnl = 0.0
        trade_occurred = False
        actual_size    = 0.0

        if filled and side != "none":
            actual_size    = size
            trade_occurred = True
            if side == "buy":
                cash += ask * actual_size; inventory -= actual_size
                trade_pnl = (ask - V) * actual_size
                oracle_cash += o_ask * actual_size; oracle_inv -= actual_size
                oracle_pnl   = (o_ask - V) * actual_size
                belief.update("buy", bid, ask)
                if trader_type == "informed":
                    adverse_sel_cost += max(0, (V - ask)) * actual_size
                elif trader_type == "momentum":
                    momentum_cost    += max(0, (V - ask)) * actual_size
                else:
                    spread_revenue   += half_spread * actual_size
            else:
                cash -= bid * actual_size; inventory += actual_size
                trade_pnl = (V - bid) * actual_size
                oracle_cash -= o_bid * actual_size; oracle_inv += actual_size
                oracle_pnl   = (V - o_bid) * actual_size
                belief.update("sell", bid, ask)
                if trader_type == "informed":
                    adverse_sel_cost += max(0, (bid - V)) * actual_size
                elif trader_type == "momentum":
                    momentum_cost    += max(0, (bid - V)) * actual_size
                else:
                    spread_revenue   += half_spread * actual_size

        inv_mtm_step       = inventory * (V - V_prev)
        inventory_mtm_pnl += inv_mtm_step
        mtm_pnl    = cash + inventory * V
        oracle_mtm = oracle_cash + oracle_inv * V
        regime     = getattr(process, "regime", "medium_vol")

        records.append({
            "t": t, "V": V, "mu": belief.mu, "sigma": belief.sigma,
            "alpha_hat": belief.alpha_hat, "ofi": ofi,
            "bid": bid, "ask": ask, "spread": ask - bid,
            "spread_bps": (ask - bid) / V * 10_000,
            "fill_prob": fill_prob, "side": side,
            "trader_type": trader_type if trade_occurred else "none",
            "trade_size": actual_size, "trade_occurred": trade_occurred,
            "trade_pnl": trade_pnl, "inventory": inventory, "cash": cash,
            "mtm_pnl": mtm_pnl, "oracle_mtm": oracle_mtm,
            "regret": oracle_mtm - mtm_pnl,
            "mu_error": belief.mu - V,
            "mu_error_bps": (belief.mu - V) / V * 10_000,
            "t_remaining": t_remaining, "regime": regime,
            "spread_revenue": spread_revenue,
            "adverse_sel_cost": adverse_sel_cost,
            "inventory_mtm_pnl": inventory_mtm_pnl,
            "momentum_cost": momentum_cost,
        })
        V_prev = V

    return pd.DataFrame(records)


def run_episode_replay(
    cfg: CryptoSimConfig, features: pd.DataFrame, offset: int = 0,
    forecaster: object = None, rl_model: object = None,
) -> pd.DataFrame:
    prices  = features["mid"].values
    regimes = features["regime"].values if "regime" in features.columns else np.array(["medium_vol"] * len(features))
    ofi_ser = features["ofi_proxy"].values if "ofi_proxy" in features.columns else np.zeros(len(features))
    start   = offset % max(1, len(prices) - cfg.n_steps)
    prices_ep  = np.concatenate([prices[start:],  prices[:start]])[:cfg.n_steps]
    regimes_ep = np.concatenate([regimes[start:], regimes[:start]])[:cfg.n_steps]
    ofi_ep     = np.concatenate([ofi_ser[start:], ofi_ser[:start]])[:cfg.n_steps]
    process = ReplayProcess(prices=prices_ep, regimes=regimes_ep, seed=cfg.seed)
    return _run_episode_core(cfg, process, ofi_series=ofi_ep, forecaster=forecaster, rl_model=rl_model)


def run_episode_hybrid(
    cfg: CryptoSimConfig, features: pd.DataFrame, seed_offset: int = 0,
    forecaster: object = None, rl_model: object = None,
) -> pd.DataFrame:
    seed    = (cfg.seed + seed_offset) if cfg.seed is not None else None
    process = HybridProcess.from_features(features, seed=seed)
    return _run_episode_core(cfg, process, forecaster=forecaster, rl_model=rl_model)


def _summarize_episode(df: pd.DataFrame, episode_idx: int) -> dict:
    """Compute summary statistics for one episode."""
    trades = df[df["trade_occurred"]]
    n_tr   = len(trades)
    pnl_series = df["mtm_pnl"].values
    running_max = np.maximum.accumulate(pnl_series)
    max_dd = float(np.max(running_max - pnl_series))
    returns = df["mtm_pnl"].diff().dropna()
    sharpe  = float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(252))
    neg_ret = returns[returns < 0]
    sortino = float(returns.mean() / (neg_ret.std() + 1e-9) * np.sqrt(252))
    cvar_5  = float(returns.quantile(0.05)) if len(returns) > 20 else 0.0
    fill_rate = n_tr / max(len(df), 1)
    inv_variance = float(df["inventory"].var())
    regime  = df["regime"].mode()[0] if "regime" in df.columns and len(df) > 0 else "medium_vol"

    return {
        "episode":             episode_idx,
        "final_pnl":           df["mtm_pnl"].iloc[-1],
        "final_oracle_pnl":    df["oracle_mtm"].iloc[-1],
        "final_regret":        df["regret"].iloc[-1],
        "n_trades":            n_tr,
        "mean_spread":         df["spread"].mean(),
        "mean_spread_bps":     df["spread_bps"].mean(),
        "sharpe":              sharpe,
        "sortino":             sortino,
        "cvar_5":              cvar_5,
        "max_drawdown":        max_dd,
        "fill_rate":           fill_rate,
        "mean_fill_prob":      fill_rate,   # backward compat alias
        "inventory_mtm_pnl":  float(df["inventory_mtm_pnl"].iloc[-1]) if "inventory_mtm_pnl" in df.columns else 0.0,
        "momentum_cost":       float(df["momentum_cost"].iloc[-1]) if "momentum_cost" in df.columns else 0.0,
        "spread_revenue":      float(df["spread_revenue"].iloc[-1]) if "spread_revenue" in df.columns else 0.0,
        "max_drawdown_bps":    max_dd / df["V"].mean() * 10_000 if df["V"].mean() > 0 else 0.0,
        "pnl_per_trade":       df["mtm_pnl"].iloc[-1] / max(n_tr, 1),
        "mean_sigma":          float(df["sigma"].mean()),
        "final_sigma":         float(df["sigma"].iloc[-1]),
        "rmse_mu":             float(np.sqrt((df["mu_error"] ** 2).mean())) if "mu_error" in df.columns else 0.0,
        "inv_variance":        inv_variance,
        "spread_capture":      df["spread_revenue"].iloc[-1] if "spread_revenue" in df.columns else 0.0,
        "adverse_sel_cost":    df["adverse_sel_cost"].iloc[-1] if "adverse_sel_cost" in df.columns else 0.0,
        "max_inventory":       df["inventory"].abs().max(),
        "final_inventory":     df["inventory"].iloc[-1],
        "rmse_mu_bps":         float(np.sqrt((df["mu_error_bps"] ** 2).mean())),
        "pnl_per_trade":       df["mtm_pnl"].iloc[-1] / max(n_tr, 1),
        "regime":              regime,
    }


def run_many_episodes(
    cfg:        CryptoSimConfig,
    features:   pd.DataFrame,
    n_episodes: int = 50,
    verbose:    bool = True,
    trades:     Optional[pd.DataFrame] = None,
    forecaster: object = None,
    rl_model:   object = None,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Multi-episode runner. Uses real trades if provided, else hybrid mode.
    """
    episodes  = []
    summaries = []

    for i in range(n_episodes):
        if trades is not None and len(trades) >= cfg.n_steps:
            df = run_episode_real_trades(
                cfg, features, trades, offset=i * cfg.n_steps,
                forecaster=forecaster, rl_model=rl_model,
            )
        else:
            df = run_episode_hybrid(
                cfg, features, seed_offset=i * 1000,
                forecaster=forecaster, rl_model=rl_model,
            )
        episodes.append(df)
        summaries.append(_summarize_episode(df, i))

        if verbose and (i + 1) % 10 == 0:
            s = summaries[-1]
            print(f"  Ep {i+1}/{n_episodes} | PnL=${s['final_pnl']:,.0f} | "
                  f"Regret=${s['final_regret']:,.0f} | Trades={s['n_trades']} | "
                  f"Sharpe={s['sharpe']:.2f}")

    return episodes, pd.DataFrame(summaries)


def compare_strategies_v4(
    cfg:        CryptoSimConfig,
    features:   pd.DataFrame,
    n_episodes: int = 30,
    verbose:    bool = True,
    trades:     Optional[pd.DataFrame] = None,
    forecaster: object = None,
    rl_model:   object = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compare all 6 strategies under IDENTICAL market conditions.

    Strategies:
      1. PassiveMM
      2. Glosten-Milgrom
      3. Avellaneda-Stoikov
      4. AdaptiveMM
      5. Forecast-Enhanced Adaptive (ForecastAdaptiveMM)
      6. RL Market Maker

    Returns dict mapping strategy_name → summary DataFrame.
    """
    strategy_configs = [
        ("passive",  "Passive-MM"),
        ("gm",       "Glosten-Milgrom"),
        ("as",       "Avellaneda-Stoikov"),
        ("adaptive", "Adaptive-MM"),
        ("forecast", "Forecast-Adaptive"),
        ("rl",       "RL-MarketMaker"),
    ]

    results = {}
    for mm_type, label in strategy_configs:
        if verbose:
            print(f"\n── {label} ──")

        cfg2 = copy.copy(cfg)
        cfg2.mm_type = mm_type

        # RL strategy needs pre-trained model
        rl_m = rl_model if mm_type == "rl" else None
        fc   = forecaster if mm_type == "forecast" else None

        _, summary = run_many_episodes(
            cfg2, features, n_episodes, verbose=verbose,
            trades=trades, forecaster=fc, rl_model=rl_m,
        )
        summary["strategy"] = label
        results[label] = summary

        if verbose:
            print(f"  PnL: ${summary['final_pnl'].mean():,.0f} ± ${summary['final_pnl'].std():,.0f} | "
                  f"Sharpe: {summary['sharpe'].mean():.2f} | "
                  f"MaxDD: ${summary['max_drawdown'].mean():,.0f}")

    return results


# Legacy: keep run_three_way_comparison
def run_three_way_comparison(
    cfg: CryptoSimConfig, features: pd.DataFrame,
    n_episodes: int = 50, verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    results = {}
    for mm_type, label in [("gm", "Glosten-Milgrom"), ("as", "Avellaneda-Stoikov"), ("adaptive", "Adaptive-MM")]:
        if verbose:
            print(f"\n── {label} ──")
        cfg2 = copy.copy(cfg)
        cfg2.mm_type = mm_type
        _, summary = run_many_episodes(cfg2, features, n_episodes, verbose=verbose)
        summary["strategy"] = label
        results[label] = summary
    return results
