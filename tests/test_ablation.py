"""
tests/test_ablation.py
Validates the ablation study machinery: PassiveMM quoting, AdaptiveMM
ablation flags, variant construction, and summary/significance tables.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import pandas as pd
from src.market_maker import AdaptiveMM, PassiveMM
from src.ablation import (
    run_ablation_study,
    ablation_summary,
    ablation_significance,
    ABLATION_VARIANTS,
    BUILD_UP_SEQUENCE,
)
from src.simulation import CryptoSimConfig


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _cfg():
    return CryptoSimConfig(
        n_steps=60, mu0=50_000, sigma0=200, sigma_v=25,
        noise_var=100, process_noise=10, alpha=0.2,
        min_spread=1.0, max_spread=500, mm_type="adaptive", seed=7,
    )


# ── test 1: PassiveMM always quotes a fixed spread ────────────────────────────

def test_passive_mm_fixed_spread():
    """PassiveMM should quote the same spread regardless of market state."""
    mm = PassiveMM(half_spread_bps=5.0, min_spread=1.0, max_spread=500.0)
    bids, asks = [], []
    for mu in [40_000, 50_000, 60_000, 55_000]:
        bid, ask = mm.quote(mu=mu, sigma=200.0, inventory=0.5, ofi=0.8)
        bids.append(bid)
        asks.append(ask)
        spread_bps = (ask - bid) / mu * 10_000
        assert abs(spread_bps - 10.0) < 0.1, (
            f"PassiveMM spread_bps={spread_bps:.2f}, expected 10 bps (2×half_spread_bps)"
        )


def test_passive_mm_ignores_inventory():
    """PassiveMM quotes the same bid/ask regardless of inventory."""
    mm = PassiveMM(half_spread_bps=5.0)
    bid0, ask0 = mm.quote(mu=50_000, inventory=0.0)
    bid1, ask1 = mm.quote(mu=50_000, inventory=0.5)
    assert bid0 == bid1 and ask0 == ask1, \
        "PassiveMM should not adjust for inventory"


def test_passive_mm_ignores_ofi():
    """PassiveMM quotes identically regardless of OFI signal."""
    mm = PassiveMM(half_spread_bps=5.0)
    bid0, ask0 = mm.quote(mu=50_000, ofi=0.0)
    bid1, ask1 = mm.quote(mu=50_000, ofi=0.9)
    assert bid0 == bid1 and ask0 == ask1, \
        "PassiveMM should not adjust for OFI"


# ── test 2: AdaptiveMM ablation flags ─────────────────────────────────────────

def test_no_inventory_penalty_identical_for_zero_inventory():
    """When inventory=0, penalty flag doesn't matter."""
    mm_on  = AdaptiveMM(use_inventory_penalty=True,  gamma=0.1, min_spread=0, max_spread=1e6)
    mm_off = AdaptiveMM(use_inventory_penalty=False, gamma=0.1, min_spread=0, max_spread=1e6)
    b_on,  a_on  = mm_on.quote(mu=50_000, sigma=200, inventory=0)
    b_off, a_off = mm_off.quote(mu=50_000, sigma=200, inventory=0)
    assert b_on == b_off and a_on == a_off


def test_inventory_penalty_moves_quotes():
    """With inventory != 0, enabling the penalty should shift quotes."""
    mm_on  = AdaptiveMM(use_inventory_penalty=True,  penalty_lambda=0.1, inv_scale=0.1,
                         gamma=0.1, min_spread=0, max_spread=1e6, use_ofi_skew=False)
    mm_off = AdaptiveMM(use_inventory_penalty=False, penalty_lambda=0.1, inv_scale=0.1,
                         gamma=0.1, min_spread=0, max_spread=1e6, use_ofi_skew=False)
    b_on,  a_on  = mm_on.quote(mu=50_000, sigma=200, inventory=0.5)
    b_off, a_off = mm_off.quote(mu=50_000, sigma=200, inventory=0.5)
    assert b_on != b_off or a_on != a_off, \
        "Inventory penalty flag should affect quotes when inventory != 0"


def test_toxicity_adj_widens_spread():
    """Enabling toxicity adjustment should widen spread when alpha_hat > alpha_base."""
    mm_on  = AdaptiveMM(use_toxicity_adj=True,  alpha_base=0.2, as_multiplier=3.0,
                         gamma=0.1, min_spread=0, max_spread=1e6, use_ofi_skew=False)
    mm_off = AdaptiveMM(use_toxicity_adj=False, alpha_base=0.2, as_multiplier=3.0,
                         gamma=0.1, min_spread=0, max_spread=1e6, use_ofi_skew=False)
    b_on,  a_on  = mm_on.quote( mu=50_000, sigma=200, alpha_hat=0.7, inventory=0)
    b_off, a_off = mm_off.quote(mu=50_000, sigma=200, alpha_hat=0.7, inventory=0)
    assert (a_on - b_on) > (a_off - b_off), \
        "Toxicity adjustment should widen spread when alpha_hat > alpha_base"


def test_vol_regime_widens_spread():
    """Enabling vol regime layer should widen spread when local_vol > 0."""
    mm_on  = AdaptiveMM(use_vol_regime=True,  vol_gamma=2.0,
                         gamma=0.1, min_spread=0, max_spread=1e6, use_ofi_skew=False)
    mm_off = AdaptiveMM(use_vol_regime=False, vol_gamma=2.0,
                         gamma=0.1, min_spread=0, max_spread=1e6, use_ofi_skew=False)
    # Inject price history to give a non-zero local_vol
    for p in [50_000 + i * 50 for i in range(25)]:
        mm_on.observe_price(p)
    b_on,  a_on  = mm_on.quote(mu=50_000, sigma=200, inventory=0)
    b_off, a_off = mm_off.quote(mu=50_000, sigma=200, inventory=0)
    assert (a_on - b_on) >= (a_off - b_off), \
        "Vol regime layer should widen spread when local_vol > 0"


def test_ofi_skew_flag():
    """Disabling OFI skew should make quotes symmetric regardless of OFI."""
    mm_on  = AdaptiveMM(use_ofi_skew=True,  ofi_sensitivity=0.5,
                         gamma=0.1, min_spread=0, max_spread=1e6)
    mm_off = AdaptiveMM(use_ofi_skew=False, ofi_sensitivity=0.5,
                         gamma=0.1, min_spread=0, max_spread=1e6)
    b_on,  a_on  = mm_on.quote(mu=50_000, sigma=200, ofi=+0.9, inventory=0)
    b_off, a_off = mm_off.quote(mu=50_000, sigma=200, ofi=+0.9, inventory=0)
    # With OFI off and no inventory: bid and ask should be symmetric about r
    r_off = (b_off + a_off) / 2
    assert abs((a_off - r_off) - (r_off - b_off)) < 0.01, \
        "With OFI skew disabled, quotes should be symmetric about reservation price"
    # With OFI on: ask should be wider on buy-pressure side
    assert (a_on - b_on) > (a_off - b_off), \
        "OFI skew should widen ask under buy pressure"


# ── test 3: ABLATION_VARIANTS structure ──────────────────────────────────────

def test_ablation_variants_include_passive_and_full():
    """ABLATION_VARIANTS must include both Passive-Fixed and Full-Adaptive."""
    labels = [label for label, _ in ABLATION_VARIANTS]
    assert "Passive-Fixed"  in labels, "Passive-Fixed missing from ablation variants"
    assert "Full-Adaptive"  in labels, "Full-Adaptive missing from ablation variants"


def test_build_up_sequence_ordered():
    """BUILD_UP_SEQUENCE must start with Passive-Fixed and end with Full-Adaptive."""
    assert BUILD_UP_SEQUENCE[0]  == "Passive-Fixed",  "First must be Passive-Fixed"
    assert BUILD_UP_SEQUENCE[-1] == "Full-Adaptive", "Last must be Full-Adaptive"


# ── test 4: ablation study runs and produces summaries ───────────────────────

def test_ablation_run_minimal():
    """run_ablation_study should complete with 3 episodes per variant."""
    cfg = _cfg()
    # Only run first 3 variants for speed
    mini_variants = ABLATION_VARIANTS[:3]
    results = run_ablation_study(cfg, _FEATURES, n_episodes=3,
                                 variants=mini_variants, verbose=False)
    assert len(results) == 3, "Should return one entry per variant"
    for label, df in results.items():
        assert len(df) == 3, f"{label}: expected 3 episodes"
        assert "final_pnl" in df.columns


def test_ablation_summary_shape():
    """ablation_summary should return one row per variant."""
    cfg = _cfg()
    mini_variants = ABLATION_VARIANTS[:3]
    results = run_ablation_study(cfg, _FEATURES, n_episodes=3,
                                 variants=mini_variants, verbose=False)
    table = ablation_summary(results, cfg, n_bootstrap=50)
    assert len(table) == 3
    assert "mean_pnl" in table.columns
    assert "sharpe_annualised" in table.columns
    assert "pnl_ci_lo" in table.columns
    assert "pnl_ci_hi" in table.columns


def test_ablation_significance_shape():
    """ablation_significance should return one row per consecutive build-up pair."""
    cfg = _cfg()
    # Run all build-up variants
    build_variants = [(l, f) for l, f in ABLATION_VARIANTS if l in BUILD_UP_SEQUENCE]
    results = run_ablation_study(cfg, _FEATURES, n_episodes=3,
                                 variants=build_variants, verbose=False)
    sig = ablation_significance(results)
    # Should have n_build-up - 1 rows (one per consecutive pair)
    expected_rows = sum(1 for i in range(len(BUILD_UP_SEQUENCE) - 1)
                        if BUILD_UP_SEQUENCE[i] in results and BUILD_UP_SEQUENCE[i+1] in results)
    assert len(sig) == expected_rows


def test_full_adaptive_pnl_column_exists():
    """Full-Adaptive variant must produce a valid PnL column."""
    cfg = _cfg()
    full_variant = [v for v in ABLATION_VARIANTS if v[0] == "Full-Adaptive"]
    results = run_ablation_study(cfg, _FEATURES, n_episodes=2,
                                 variants=full_variant, verbose=False)
    df = results["Full-Adaptive"]
    assert df["final_pnl"].notna().all(), "Full-Adaptive should have non-NaN PnL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
