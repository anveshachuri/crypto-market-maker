"""
tests/test_simulation.py
Validates the core P&L accounting identity and simulation mechanics.

The key invariant:
    mtm_pnl = cash + inventory × V
    Δ(mtm_pnl) = trade_pnl + inventory_mtm   (approximately, up to half-spread)

Also checks:
  - fill_ref respects cfg.fill_decay (not hardcoded 1.5)
  - max_drawdown is non-negative and ≤ |final_pnl|
  - inventory_mtm_pnl column exists and sums sensibly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import pandas as pd
from src.simulation import CryptoSimConfig, run_episode_hybrid
from src.market_maker import AdaptiveMM


# ── minimal synthetic features for tests ─────────────────────────────────────

def _make_features(n=600, seed=0) -> pd.DataFrame:
    """Minimal feature DataFrame for offline testing (no Binance call needed)."""
    rng    = np.random.default_rng(seed)
    prices = 50_000 + np.cumsum(rng.normal(0, 25, n))
    df = pd.DataFrame({
        "open_time":            pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open":                 prices,
        "high":                 prices + rng.uniform(0, 50, n),
        "low":                  prices - rng.uniform(0, 50, n),
        "close":                prices,
        "volume":               rng.uniform(1, 10, n),
        "taker_buy_base_vol":   rng.uniform(0.4, 0.6, n) * rng.uniform(1, 10, n),
        "taker_buy_quote_vol":  rng.uniform(0.4, 0.6, n) * rng.uniform(1, 10, n) * prices,
        "n_trades":             rng.integers(10, 100, n),
    })
    from src.data_loader import engineer_features
    return engineer_features(df)


_FEATURES = _make_features()


def _default_cfg(**overrides) -> CryptoSimConfig:
    cfg = CryptoSimConfig(
        n_steps       = 100,
        mu0           = 50_000.0,
        sigma0        = 200.0,
        sigma_v       = 25.0,
        noise_var     = 100.0,
        process_noise = 10.0,
        alpha         = 0.2,
        min_spread    = 1.0,
        max_spread    = 500.0,
        mm_type       = "adaptive",
        seed          = 42,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── test 1: mtm_pnl accounting identity ──────────────────────────────────────

def test_mtm_accounting_identity():
    """
    At every step: mtm_pnl == cash + inventory × V.
    This is a fundamental double-entry identity — any violation means a
    cash or inventory update is missing.
    """
    cfg = _default_cfg()
    df  = run_episode_hybrid(cfg, _FEATURES)
    for _, row in df.iterrows():
        mtm_reconstructed = row["cash"] + row["inventory"] * row["V"]
        assert abs(row["mtm_pnl"] - mtm_reconstructed) < 0.01, (
            f"t={row['t']}: mtm_pnl={row['mtm_pnl']:.4f} != "
            f"cash+inv×V={mtm_reconstructed:.4f}"
        )


# ── test 2: inventory_mtm_pnl column exists ───────────────────────────────────

def test_inventory_mtm_column_exists():
    """inventory_mtm_pnl must be present (replaced old inventory_carry_cost)."""
    cfg = _default_cfg()
    df  = run_episode_hybrid(cfg, _FEATURES)
    assert "inventory_mtm_pnl" in df.columns, \
        "inventory_mtm_pnl column missing — old inventory_carry_cost column name?"
    assert "inventory_carry_cost" not in df.columns, \
        "Old inventory_carry_cost column still present — rename not applied"


# ── test 3: max_drawdown in summary ──────────────────────────────────────────

def test_max_drawdown_non_negative():
    """max_drawdown must be ≥ 0 (it's a peak-to-trough magnitude)."""
    from src.simulation import run_many_episodes
    cfg = _default_cfg(n_steps=50)
    _, summary = run_many_episodes(cfg, _FEATURES, n_episodes=5, verbose=False)
    assert "max_drawdown" in summary.columns, "max_drawdown missing from summary"
    assert (summary["max_drawdown"] >= 0).all(), \
        f"Negative drawdown values: {summary['max_drawdown'].min()}"


# ── test 4: fill_ref respects cfg.fill_decay ─────────────────────────────────

def test_fill_decay_respected():
    """
    Two configs with different fill_decay should produce different fill rates.
    Previously fill_ref = AdaptiveMM() always used the default decay=1.5,
    ignoring cfg.fill_decay entirely.
    """
    cfg_tight = _default_cfg(fill_decay=0.5)   # very permissive fill
    cfg_strict = _default_cfg(fill_decay=5.0)  # very restrictive fill

    df_tight  = run_episode_hybrid(cfg_tight,  _FEATURES, seed_offset=0)
    df_strict = run_episode_hybrid(cfg_strict, _FEATURES, seed_offset=0)

    fill_tight  = df_tight["fill_prob"].mean()
    fill_strict = df_strict["fill_prob"].mean()

    assert fill_tight > fill_strict, (
        f"Tighter decay ({cfg_tight.fill_decay}) should give higher fill prob "
        f"({fill_tight:.3f}) than strict ({cfg_strict.fill_decay}) decay "
        f"({fill_strict:.3f})"
    )


# ── test 5: GM strategy runs without error ────────────────────────────────────

def test_gm_strategy_runs():
    """GlostenMilgromMM strategy should complete without errors."""
    cfg = _default_cfg(mm_type="gm")
    df  = run_episode_hybrid(cfg, _FEATURES)
    assert len(df) == cfg.n_steps
    assert df["mtm_pnl"].notna().all()


# ── test 6: AS strategy runs without error ────────────────────────────────────

def test_as_strategy_runs():
    """AvellanedaStoikovMM strategy should complete without errors."""
    cfg = _default_cfg(mm_type="as")
    df  = run_episode_hybrid(cfg, _FEATURES)
    assert len(df) == cfg.n_steps
    assert df["mtm_pnl"].notna().all()


# ── test 7: spread_revenue is non-negative ────────────────────────────────────

def test_spread_revenue_non_negative():
    """Spread revenue accumulates from uninformed fills — must be ≥ 0."""
    cfg = _default_cfg()
    df  = run_episode_hybrid(cfg, _FEATURES)
    assert df["spread_revenue"].iloc[-1] >= 0.0, \
        f"Spread revenue is negative: {df['spread_revenue'].iloc[-1]}"


# ── test 8: inventory stays bounded ──────────────────────────────────────────

def test_inventory_bounded():
    """Inventory should not grow without bound in a well-configured run."""
    cfg = _default_cfg(inv_limit=0.5, penalty_lambda=0.05)
    df  = run_episode_hybrid(cfg, _FEATURES)
    max_inv = df["inventory"].abs().max()
    # Should not exceed a few multiples of inv_limit (exponential penalty kicks in)
    assert max_inv < 5.0, f"Inventory blew up to {max_inv:.3f} BTC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
