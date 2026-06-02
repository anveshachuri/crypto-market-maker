"""
tests/test_belief.py
Validates GaussianBelief correctness against GridBelief (the exact
Bayesian reference), and checks key invariants.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.belief import GaussianBelief, GridBelief


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_gaussian(mu0=100.0, sigma0=3.0, alpha=0.3):
    return GaussianBelief(mu0=mu0, sigma0=sigma0, alpha=alpha,
                          noise_var=1.0, process_noise=0.02)


def _make_grid(mu0=100.0, sigma0=3.0, alpha=0.3):
    return GridBelief(v_min=mu0 - 5 * sigma0, v_max=mu0 + 5 * sigma0,
                      n_grid=200, mu0=mu0, sigma0=sigma0, alpha=alpha)


# ── test 1: variance never goes negative ─────────────────────────────────────

def test_variance_never_negative():
    """Kalman variance must stay non-negative after any number of updates."""
    b = _make_gaussian()
    rng = np.random.default_rng(0)
    for _ in range(200):
        side = "buy" if rng.random() < 0.5 else "sell"
        b.update(side, bid=99.0, ask=101.0)
        b.time_step()
        assert b.var >= 0.0, f"Variance went negative: {b.var}"
        assert b.sigma >= 0.0


# ── test 2: process noise prevents variance collapse ─────────────────────────

def test_process_noise_prevents_collapse():
    """After many updates, variance should stabilise (not collapse to zero)."""
    b = _make_gaussian(sigma0=3.0)
    for _ in range(500):
        b.update("buy", bid=99.0, ask=101.0)
        b.time_step()
    # Variance should remain well above zero — process noise keeps it open
    assert b.var > 1e-4, f"Variance collapsed to {b.var}"


# ── test 3: Gaussian mu moves in the right direction ─────────────────────────

def test_belief_direction():
    """A sequence of buy orders should push mu up; sells should push it down."""
    b_buy  = _make_gaussian(mu0=100.0)
    b_sell = _make_gaussian(mu0=100.0)
    for _ in range(20):
        b_buy.update("buy",  bid=99.0, ask=101.0)
        b_sell.update("sell", bid=99.0, ask=101.0)
    assert b_buy.mu  > 100.0, "Buy pressure should raise mu"
    assert b_sell.mu < 100.0, "Sell pressure should lower mu"


# ── test 4: GridBelief mean moves in the same direction as Gaussian ───────────

def test_grid_vs_gaussian_direction():
    """GridBelief (exact Bayes) and GaussianBelief should agree on direction."""
    g = _make_gaussian()
    grd = _make_grid()
    sides = ["buy"] * 10 + ["sell"] * 5
    for s in sides:
        g.update(s, bid=99.0, ask=101.0)
        grd.update(s, bid=99.0, ask=101.0)

    # Both should show net upward belief from more buys than sells
    assert g.mu   > 100.0
    assert grd.mu > 100.0
    # Sign should agree
    assert (g.mu - 100.0) * (grd.mu - 100.0) > 0


# ── test 5: GridBelief mu close to GaussianBelief mu ────────────────────────

def test_grid_gaussian_close():
    """
    GaussianBelief is a linear (Kalman) approximation to the exact grid.
    Their posterior means should be in the same ballpark (within 2 USD on a
    $100 asset with sigma=3) after a small number of balanced updates.
    """
    g   = _make_gaussian(mu0=100.0, sigma0=3.0, alpha=0.2)
    grd = _make_grid(mu0=100.0, sigma0=3.0, alpha=0.2)

    rng = np.random.default_rng(7)
    for _ in range(30):
        side = "buy" if rng.random() < 0.55 else "sell"   # slight buy bias
        g.update(side, bid=99.0, ask=101.0)
        grd.update(side, bid=99.0, ask=101.0)

    assert abs(g.mu - grd.mu) < 2.5, (
        f"GaussianBelief mu={g.mu:.3f} vs GridBelief mu={grd.mu:.3f}: "
        f"divergence {abs(g.mu - grd.mu):.3f} exceeds tolerance"
    )


# ── test 6: alpha_hat stays in [0, 1] ────────────────────────────────────────

def test_alpha_hat_bounded():
    """alpha_hat is an EMA of |OFI| ∈ [0,1], so it must stay in [0,1]."""
    b = _make_gaussian()
    rng = np.random.default_rng(1)
    for _ in range(300):
        side = "buy" if rng.random() < 0.9 else "sell"   # heavy buy imbalance
        b.update(side, bid=98.0, ask=102.0)
        assert 0.0 <= b.alpha_hat <= 1.0, f"alpha_hat out of bounds: {b.alpha_hat}"


# ── test 7: OFI imbalance range ──────────────────────────────────────────────

def test_ofi_bounded():
    """order_flow_imbalance must stay in [-1, 1]."""
    b = _make_gaussian()
    rng = np.random.default_rng(2)
    for _ in range(200):
        side = "buy" if rng.random() < 0.8 else "sell"
        b.update(side, bid=99.0, ask=101.0)
        ofi = b.order_flow_imbalance
        assert -1.0 <= ofi <= 1.0, f"OFI out of range: {ofi}"


# ── test 8: reset restores initial state ─────────────────────────────────────

def test_reset():
    """reset() must restore mu, var, alpha_hat, OFI EMAs to initial values."""
    b = _make_gaussian(mu0=100.0, sigma0=3.0, alpha=0.3)
    for _ in range(50):
        b.update("buy", bid=99.0, ask=101.0)
    b.reset()
    assert b.mu  == pytest.approx(100.0)
    assert b.var == pytest.approx(9.0)
    assert b.alpha_hat == pytest.approx(0.3)
    assert len(b.history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
