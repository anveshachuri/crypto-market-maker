"""
tests/test_market_maker.py
Validates each market maker's quoting formulas and the key fixes:
  - GM break-even spread formula
  - AS reservation price and spread
  - AdaptiveMM OFI asymmetry (ask rises under buy pressure, bid stays neutral)
  - local_vol uses BTC 24/7 minutes (1440), not US equity session (390)
  - fill probability model
  - spread bounds always respected
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.market_maker import GlostenMilgromMM, AvellanedaStoikovMM, AdaptiveMM


# ── test 1: GM break-even spread ─────────────────────────────────────────────

def test_gm_spread_formula():
    """
    GM half-spread = α·σ / (1−α).
    At alpha=0.3, sigma=10: half_spread = 0.3*10/0.7 ≈ 4.286
    """
    alpha, sigma = 0.3, 10.0
    mm = GlostenMilgromMM(alpha=alpha, min_spread=0.0, max_spread=1000.0)
    bid, ask = mm.quote(mu=100.0, sigma=sigma)
    half_spread = (ask - bid) / 2.0
    expected    = alpha * sigma / (1.0 - alpha)
    assert abs(half_spread - expected) < 0.01, (
        f"GM half_spread={half_spread:.4f}, expected={expected:.4f}"
    )


def test_gm_spread_increases_with_alpha():
    """Higher informed fraction → wider GM spread."""
    mm = GlostenMilgromMM(min_spread=0.0, max_spread=1000.0)
    spreads = []
    for alpha in [0.1, 0.2, 0.3, 0.4]:
        bid, ask = mm.quote(mu=100.0, sigma=5.0, alpha_hat=alpha)
        spreads.append(ask - bid)
    assert spreads == sorted(spreads), f"GM spread should increase with alpha: {spreads}"


# ── test 2: AS reservation price ─────────────────────────────────────────────

def test_as_reservation_price():
    """
    AS reservation price: r = μ − q·γ·σ²·T
    Positive inventory → r below μ (MM leans short to unload).
    Negative inventory → r above μ.
    """
    mm = AvellanedaStoikovMM(gamma=0.1, kappa=1.5)
    r_long  = mm.reservation_price(mu=100.0, sigma=5.0, inventory=+1.0, t_remaining=1.0)
    r_short = mm.reservation_price(mu=100.0, sigma=5.0, inventory=-1.0, t_remaining=1.0)
    assert r_long  < 100.0, "Long inventory should push reservation price below mid"
    assert r_short > 100.0, "Short inventory should push reservation price above mid"


def test_as_spread_increases_with_time():
    """AS optimal spread is larger with more time remaining (more uncertainty)."""
    mm = AvellanedaStoikovMM(gamma=0.1, kappa=1.5)
    s_early = mm.optimal_spread(sigma=5.0, t_remaining=1.0)
    s_late  = mm.optimal_spread(sigma=5.0, t_remaining=0.1)
    assert s_early > s_late, "Spread should be wider earlier in episode"


# ── test 3: AdaptiveMM OFI is ASYMMETRIC ─────────────────────────────────────

def test_adaptive_ofi_asymmetric():
    """
    Under buy pressure (OFI > 0):
      - ask should rise (deters informed buyers)
      - bid should NOT rise by the same amount (that would attract sell-side AS)
    Under sell pressure (OFI < 0):
      - bid should fall
      - ask should NOT fall by the same amount
    """
    mm = AdaptiveMM(ofi_sensitivity=0.3, min_spread=0.0, max_spread=1e6)
    mu, sigma = 50_000.0, 200.0

    bid_neutral, ask_neutral = mm.quote(mu=mu, sigma=sigma, ofi=0.0)
    bid_buy,     ask_buy     = mm.quote(mu=mu, sigma=sigma, ofi=+0.8)
    bid_sell,    ask_sell    = mm.quote(mu=mu, sigma=sigma, ofi=-0.8)

    # Under buy pressure: ask rises, bid stays the same (neutral)
    assert ask_buy > ask_neutral, "Ask should rise under buy pressure"
    assert abs(bid_buy - bid_neutral) < 0.01, (
        f"Bid should not move under buy pressure; got Δbid={bid_buy - bid_neutral:.4f}"
    )

    # Under sell pressure: bid falls, ask stays the same
    assert bid_sell < bid_neutral, "Bid should fall under sell pressure"
    assert abs(ask_sell - ask_neutral) < 0.01, (
        f"Ask should not move under sell pressure; got Δask={ask_sell - ask_neutral:.4f}"
    )

    # Symmetry: buy-pressure ask-rise ≈ sell-pressure bid-fall
    ask_rise  = ask_buy  - ask_neutral
    bid_fall  = bid_neutral - bid_sell
    assert abs(ask_rise - bid_fall) < 0.01, (
        f"OFI response should be symmetric in magnitude: "
        f"ask_rise={ask_rise:.4f}, bid_fall={bid_fall:.4f}"
    )


# ── test 4: local_vol uses BTC 24/7 annualisation ────────────────────────────

def test_local_vol_btc_annualisation():
    """
    local_vol should scale by sqrt(1440) not sqrt(390).
    We inject a known per-step std and check the annualised output.
    BTC is 24/7: 1 day = 1440 minutes.
    """
    mm = AdaptiveMM(vol_window=10)
    per_step_std = 100.0   # USD per 1-minute bar
    # Feed prices that produce exactly per_step_std of absolute differences
    rng = np.random.default_rng(0)
    prices = [50_000.0]
    for _ in range(15):
        prices.append(prices[-1] + rng.choice([-per_step_std, +per_step_std]))
        mm.observe_price(prices[-1])

    actual_vol  = mm.local_vol
    # std(±100) = 100; expected daily vol = 100 * sqrt(1440) ≈ 3795
    expected_approx = per_step_std * np.sqrt(1440)

    # Allow 30% tolerance (due to small sample variance in std estimate)
    assert abs(actual_vol - expected_approx) / expected_approx < 0.30, (
        f"local_vol={actual_vol:.1f}, expected≈{expected_approx:.1f}. "
        f"Possible cause: still using sqrt(390) (US equity) instead of sqrt(1440)"
    )


# ── test 5: fill probability model ───────────────────────────────────────────

def test_fill_probability_range():
    """Fill probability must be in (0, 1] for any half_spread ≥ 0."""
    mm = AdaptiveMM(fill_decay=1.5)
    for hs in [0.0, 1.0, 10.0, 100.0, 1000.0]:
        fp = mm.fill_probability(half_spread=hs, sigma=50.0)
        assert 0.0 < fp <= 1.0, f"fill_prob={fp} out of range for hs={hs}"


def test_fill_probability_decreases_with_spread():
    """Wider spread → lower fill probability."""
    mm = AdaptiveMM(fill_decay=1.5)
    probs = [mm.fill_probability(hs, sigma=50.0) for hs in [1, 5, 25, 100]]
    assert probs == sorted(probs, reverse=True), \
        f"Fill probs should decrease with spread: {probs}"


def test_fill_at_zero_spread():
    """At zero half_spread, fill probability should be 1.0."""
    mm = AdaptiveMM(fill_decay=1.5)
    assert mm.fill_probability(0.0, sigma=50.0) == pytest.approx(1.0)


# ── test 6: spread bounds always respected ────────────────────────────────────

def test_spread_bounds():
    """Quoted spread must always be within [min_spread, max_spread]."""
    mm = AdaptiveMM(min_spread=1.0, max_spread=500.0, gamma=0.1)
    rng = np.random.default_rng(3)
    for _ in range(100):
        mu        = rng.uniform(40_000, 60_000)
        sigma     = rng.uniform(10, 500)
        inventory = rng.uniform(-2, 2)
        ofi       = rng.uniform(-1, 1)
        alpha_hat = rng.uniform(0.05, 0.9)
        bid, ask  = mm.quote(mu=mu, sigma=sigma, inventory=inventory,
                              ofi=ofi, alpha_hat=alpha_hat)
        spread = ask - bid
        assert spread >= 1.0,   f"Spread {spread:.4f} < min_spread=1.0"
        assert spread <= 1000.0, f"Spread {spread:.4f} > max_spread×2=1000.0"


# ── test 7: AdaptiveMM widens spread when alpha_hat rises ────────────────────

def test_adaptive_widens_on_toxicity():
    """Higher alpha_hat (more informed flow) → wider AdaptiveMM spread."""
    mm = AdaptiveMM(as_multiplier=2.0, alpha_base=0.3, min_spread=0.0, max_spread=1e6)
    bid_lo, ask_lo = mm.quote(mu=50_000, sigma=200.0, alpha_hat=0.3)
    bid_hi, ask_hi = mm.quote(mu=50_000, sigma=200.0, alpha_hat=0.7)
    assert (ask_hi - bid_hi) > (ask_lo - bid_lo), \
        "Higher toxicity should produce wider spread"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
