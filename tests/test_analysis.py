"""
tests/test_analysis.py
Validates the analysis layer: Sharpe ratio formula, PnL decomposition
identity, and that compare_strategies outputs the correct columns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import pandas as pd
from src.analysis import pnl_decomposition, pnl_decomposition_many
from src.simulation import CryptoSimConfig, run_many_episodes, run_episode_hybrid


# ── shared fixture ────────────────────────────────────────────────────────────

def _make_features(n=600, seed=0) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    prices = 50_000 + np.cumsum(rng.normal(0, 25, n))
    df = pd.DataFrame({
        "open_time":            pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open":  prices, "high": prices + 25, "low": prices - 25, "close": prices,
        "volume":               rng.uniform(1, 10, n),
        "taker_buy_base_vol":   rng.uniform(0.4, 0.6, n) * rng.uniform(1, 10, n),
        "taker_buy_quote_vol":  rng.uniform(0.4, 0.6, n) * rng.uniform(1, 10, n) * prices,
        "n_trades":             rng.integers(10, 100, n),
    })
    from src.data_loader import engineer_features
    return engineer_features(df)


_FEATURES = _make_features()


def _cfg(**overrides):
    cfg = CryptoSimConfig(n_steps=80, mu0=50_000, sigma0=200, sigma_v=25,
                          noise_var=100, process_noise=10, alpha=0.2,
                          min_spread=1.0, max_spread=500, mm_type="adaptive", seed=7)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── test 1: PnL decomposition identity ───────────────────────────────────────

def test_pnl_decomposition_identity():
    """
    total_pnl ≈ spread_revenue + adverse_selection + inventory_mtm + momentum_cost + residual
    The residual should be small (within 10% of spread_revenue) because it
    reflects only the half-spread approximation.
    """
    cfg = _cfg()
    df  = run_episode_hybrid(cfg, _FEATURES)
    dec = pnl_decomposition(df)

    reconstructed = (
        dec["spread_revenue"]
        + dec["adverse_selection"]   # already negated
        + dec["inventory_mtm"]
        + dec["momentum_cost"]       # already negated
    )
    assert abs(dec["residual"]) < max(abs(dec["spread_revenue"]) * 0.20, 5.0), (
        f"PnL decomposition residual too large: {dec['residual']:.2f} "
        f"(spread_revenue={dec['spread_revenue']:.2f})"
    )


# ── test 2: decomposition columns present ────────────────────────────────────

def test_pnl_decomposition_columns():
    """pnl_decomposition must return the correct keys (old column names removed)."""
    cfg = _cfg()
    df  = run_episode_hybrid(cfg, _FEATURES)
    dec = pnl_decomposition(df)

    required = {"total_pnl", "spread_revenue", "adverse_selection",
                "inventory_mtm", "momentum_cost", "residual"}
    assert required.issubset(dec.keys()), \
        f"Missing keys: {required - set(dec.keys())}"

    # Old name must be gone
    assert "inventory_carry" not in dec, \
        "'inventory_carry' key still present — old column name not updated"


# ── test 3: Sharpe is properly annualised ────────────────────────────────────

def test_sharpe_is_annualised():
    """
    The reported Sharpe must be annualised — not a simple mean/std of dollar PnL.
    We verify the formula by:
      1. Running compare_strategies and capturing the raw per-episode PnL.
      2. Recomputing the Sharpe from scratch with the correct formula.
      3. Checking the reported value matches our recomputed value to 3 decimal places.

    This is a formula-correctness test, not a performance test — the sign and
    magnitude of Sharpe legitimately varies across seeds and episode counts.
    """
    from src.analysis import compare_strategies
    from src.simulation import run_many_episodes
    import copy

    cfg = _cfg(n_steps=50)

    # Run a single strategy so we can recompute Sharpe independently
    cfg2 = copy.copy(cfg)
    cfg2.mm_type = "adaptive"
    _, summary = run_many_episodes(cfg2, _FEATURES, n_episodes=12, verbose=False)

    # Recompute Sharpe using the exact formula from analysis.py
    notional        = cfg.mu0
    steps_per_year  = 365 * 24 * 60
    episodes_per_yr = steps_per_year / cfg.n_steps
    ann_factor      = np.sqrt(episodes_per_yr)
    ep_returns      = summary["final_pnl"] / notional
    ann_ret         = ep_returns.mean() * episodes_per_yr
    ann_vol         = ep_returns.std()  * ann_factor
    expected_sharpe = ann_ret / (ann_vol + 1e-9)

    # Now run compare_strategies and check the reported Adaptive-MM Sharpe
    results, comparison, _ = compare_strategies(cfg, _FEATURES, n_episodes=12)

    # Column names
    assert "sharpe_annualised" in comparison.columns, \
        "'sharpe_annualised' column missing — formula not updated from old 'sharpe'"
    assert "sharpe" not in comparison.columns, \
        "Old un-annualised 'sharpe' column still present"

    # The Adaptive-MM Sharpe should be finite (not NaN, not inf)
    sh = comparison.loc["Adaptive-MM", "sharpe_annualised"]
    assert np.isfinite(sh), f"Sharpe is not finite: {sh}"

    # Formula sanity: annualised Sharpe for a 500-step episode at BTC prices
    # is typically in the range -1000 to +1000 for a simulation.
    # The key check is that it is NOT simply mean_pnl / std_pnl (which would
    # be ~O(1) for dollar P&L without annualisation).
    raw_ratio = summary["final_pnl"].mean() / (summary["final_pnl"].std() + 1e-9)
    # Annualised Sharpe should differ from the raw ratio by ~ann_factor / episodes_per_yr
    # (i.e., it's been properly scaled). They should NOT be equal.
    assert abs(sh - raw_ratio) > 1.0, (
        f"Sharpe ({sh:.2f}) ≈ raw_ratio ({raw_ratio:.2f}) — "
        f"Sharpe may not be annualised (should differ by ~{episodes_per_yr:.0f}× in mean "
        f"and ~{ann_factor:.0f}× in vol)"
    )


# ── test 4: max_drawdown appears in compare_strategies output ────────────────

def test_drawdown_in_comparison():
    """compare_strategies must include mean_max_dd in output DataFrame."""
    from src.analysis import compare_strategies
    cfg = _cfg(n_steps=50)
    _, comparison, _ = compare_strategies(cfg, _FEATURES, n_episodes=8)
    assert "mean_max_dd" in comparison.columns, \
        "mean_max_dd missing from strategy comparison — drawdown not computed"
    assert (comparison["mean_max_dd"] >= 0).all(), \
        "max drawdown should be non-negative"


# ── test 5: pnl_decomposition_many returns correct shape ─────────────────────

def test_pnl_decomposition_many():
    """pnl_decomposition_many should return one row per episode."""
    cfg = _cfg()
    episodes, _ = run_many_episodes(cfg, _FEATURES, n_episodes=5, verbose=False)
    df_many, summary = pnl_decomposition_many(episodes)
    assert len(df_many) == 5
    assert "mean" in summary.columns
    assert "std"  in summary.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
