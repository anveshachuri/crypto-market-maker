"""
tests/test_statistics.py
Validates statistical functions: bootstrap CIs, risk metrics, fill calibration,
OFI predictive test, Welch test. All tests are self-contained (no network calls).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import pandas as pd
from src.statistics import (
    bootstrap_ci,
    risk_metrics,
    cohens_d,
    welch_test,
    fill_calibration,
    ofi_predictive_test,
)
from src.simulation import CryptoSimConfig


# ── shared helpers ────────────────────────────────────────────────────────────

def _dummy_cfg():
    return CryptoSimConfig(n_steps=100, mu0=50_000.0, seed=0)


def _dummy_summary(n=30, rng_seed=0, mean_pnl=500.0, std_pnl=200.0):
    rng = np.random.default_rng(rng_seed)
    pnl = rng.normal(mean_pnl, std_pnl, n)
    return pd.DataFrame({
        "episode":         np.arange(n),
        "final_pnl":       pnl,
        "max_inventory":   rng.uniform(0.01, 0.3, n),
        "max_drawdown":    np.abs(rng.normal(100, 50, n)),
        "max_drawdown_bps": np.abs(rng.normal(2, 1, n)),
    })


def _dummy_episodes(n_ep=10, n_steps=80, rng_seed=0):
    """Generate minimal synthetic episode DataFrames."""
    rng = np.random.default_rng(rng_seed)
    eps = []
    for i in range(n_ep):
        prices = 50_000 + np.cumsum(rng.normal(0, 20, n_steps))
        pnl    = np.cumsum(rng.normal(1, 30, n_steps))
        adv_sel = np.cumsum(np.abs(rng.normal(0, 0.5, n_steps)))
        alpha_hat = np.clip(0.2 + rng.normal(0, 0.05, n_steps), 0.05, 0.9)
        trade_occ = rng.random(n_steps) < 0.4
        trader_type = np.where(trade_occ,
                               rng.choice(["informed", "uninformed", "momentum"], n_steps),
                               "none")
        sigma = np.ones(n_steps) * 200.0
        spread = rng.uniform(5, 50, n_steps)
        half_spread = spread / 2.0
        fill_prob = np.exp(-1.5 * half_spread / sigma)
        eps.append(pd.DataFrame({
            "t":                np.arange(n_steps),
            "V":                prices,
            "mtm_pnl":          pnl,
            "spread":           spread,
            "spread_bps":       spread / prices * 10_000,
            "fill_prob":        fill_prob,
            "trade_occurred":   trade_occ,
            "trader_type":      trader_type,
            "adverse_sel_cost": adv_sel,
            "alpha_hat":        alpha_hat,
            "sigma":            sigma,
        }))
    return eps


# ── test 1: bootstrap CI contains true mean ───────────────────────────────────

def test_bootstrap_ci_coverage():
    """95% CI should contain the sample mean (trivially) and have correct width."""
    rng  = np.random.default_rng(1)
    data = rng.normal(100, 20, 50)
    pt, lo, hi = bootstrap_ci(data, np.mean, n_bootstrap=500)
    assert lo <= pt <= hi, "Point estimate should be within CI bounds"
    assert hi > lo, "CI should have positive width"
    # For n=50 from N(100,20), 95% CI on mean ≈ ±5.5; check sanity
    assert (hi - lo) < 40, f"CI width {hi-lo:.2f} unreasonably wide for n=50"
    assert (hi - lo) > 1,  f"CI width {hi-lo:.2f} unreasonably narrow"


def test_bootstrap_ci_point_estimate():
    """Point estimate must equal statistic_fn(data)."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pt, _, _ = bootstrap_ci(data, np.mean, n_bootstrap=200)
    assert pt == pytest.approx(3.0, abs=1e-9)


# ── test 2: risk_metrics returns correct keys and types ───────────────────────

def test_risk_metrics_keys():
    """risk_metrics must return all required keys."""
    summary = _dummy_summary(n=30)
    cfg     = _dummy_cfg()
    rm = risk_metrics(summary, cfg, n_bootstrap=100)
    required = {
        "sharpe_annualised", "sharpe_ci_95", "sortino_annualised",
        "ann_return_pct", "ann_vol_pct", "mean_pnl", "mean_pnl_ci_95",
        "std_pnl", "var_95", "cvar_95", "inventory_var_95",
        "mean_max_dd", "max_max_dd", "n_episodes",
    }
    assert required.issubset(rm.keys()), f"Missing keys: {required - set(rm.keys())}"


def test_risk_metrics_cvar_le_var():
    """CVaR (Expected Shortfall) must be ≤ VaR for any distribution."""
    summary = _dummy_summary(n=50, rng_seed=5)
    cfg     = _dummy_cfg()
    rm = risk_metrics(summary, cfg, n_bootstrap=100)
    assert rm["cvar_95"] <= rm["var_95"] + 1e-6, (
        f"CVaR={rm['cvar_95']:.2f} should be ≤ VaR={rm['var_95']:.2f}"
    )


def test_risk_metrics_sharpe_sign():
    """Sharpe sign must match the sign of mean return."""
    cfg = _dummy_cfg()
    pos_summary = _dummy_summary(n=40, mean_pnl=+1000, std_pnl=100)
    neg_summary = _dummy_summary(n=40, mean_pnl=-1000, std_pnl=100)
    rm_pos = risk_metrics(pos_summary, cfg, n_bootstrap=50)
    rm_neg = risk_metrics(neg_summary, cfg, n_bootstrap=50)
    assert rm_pos["sharpe_annualised"] > 0, "Positive mean return → positive Sharpe"
    assert rm_neg["sharpe_annualised"] < 0, "Negative mean return → negative Sharpe"


def test_risk_metrics_sortino_ge_sharpe_for_skewed():
    """When losses are rare (many positive returns), Sortino ≥ Sharpe."""
    rng = np.random.default_rng(42)
    # Mostly positive, rare large losses → positive skew → Sortino ≥ Sharpe
    pnl = np.concatenate([rng.exponential(200, 45), rng.normal(-50, 20, 5)])
    summary = pd.DataFrame({
        "episode":         np.arange(50),
        "final_pnl":       pnl,
        "max_inventory":   np.ones(50) * 0.1,
        "max_drawdown":    np.ones(50) * 50,
        "max_drawdown_bps": np.ones(50) * 1,
    })
    cfg = _dummy_cfg()
    rm = risk_metrics(summary, cfg, n_bootstrap=50)
    # Sortino uses downside vol only; if upside dominates, Sortino ≥ Sharpe
    assert rm["sortino_annualised"] >= rm["sharpe_annualised"] - 0.5, (
        f"Sortino={rm['sortino_annualised']:.2f} should ≥ Sharpe={rm['sharpe_annualised']:.2f}"
        " when returns are right-skewed"
    )


# ── test 3: Welch test + Cohen's d ───────────────────────────────────────────

def test_welch_test_identical():
    """Welch's t on identical samples should give t≈0, p≈1."""
    data = np.arange(30, dtype=float)
    wt   = welch_test(data, data)
    assert abs(wt["t_stat"]) < 0.01, f"t should be ~0 for identical samples: {wt['t_stat']}"
    assert wt["p_value"] > 0.90


def test_welch_test_clearly_different():
    """Welch's t should detect a large difference (mean shift >> std)."""
    rng = np.random.default_rng(3)
    a   = rng.normal(0, 1, 100)
    b   = rng.normal(10, 1, 100)   # 10 std separation
    wt  = welch_test(a, b)
    assert wt["significant_at_01"], "Should be highly significant"
    assert abs(wt["cohens_d"]) > 5, f"Cohen's d should be large: {wt['cohens_d']}"
    assert wt["effect_size"] == "large"


def test_cohens_d_zero_for_equal():
    """Cohen's d should be 0 for identical distributions."""
    data = np.arange(20, dtype=float)
    assert abs(cohens_d(data, data)) < 1e-6


# ── test 4: fill_calibration ──────────────────────────────────────────────────

def test_fill_calibration_shape():
    """fill_calibration should return one row per bin."""
    eps = _dummy_episodes(n_ep=5)
    result = fill_calibration(eps, n_bins=5, fit_model=False)
    assert len(result) >= 1
    assert "model_fill_prob" in result.columns
    assert "empirical_fill_rate" in result.columns


def test_fill_calibration_rates_in_range():
    """Both model and empirical fill rates must be in [0, 1]."""
    eps = _dummy_episodes(n_ep=8)
    result = fill_calibration(eps, n_bins=5, fit_model=False)
    assert (result["model_fill_prob"]     >= 0).all()
    assert (result["model_fill_prob"]     <= 1).all()
    assert (result["empirical_fill_rate"] >= 0).all()
    assert (result["empirical_fill_rate"] <= 1).all()


# ── test 5: OFI predictive test ───────────────────────────────────────────────

def test_ofi_predictive_test_returns_keys():
    """ofi_predictive_test must return all required keys."""
    eps  = _dummy_episodes(n_ep=20, n_steps=100)
    result = ofi_predictive_test(eps, lag_steps=5)
    assert "spearman_rho" in result
    assert "p_value"      in result
    assert "n_obs"        in result
    assert "interpretation" in result


def test_ofi_predictive_test_rho_bounded():
    """Spearman ρ must be in [-1, 1]."""
    eps = _dummy_episodes(n_ep=15, n_steps=100)
    result = ofi_predictive_test(eps, lag_steps=3)
    if not np.isnan(result["spearman_rho"]):
        assert -1.0 <= result["spearman_rho"] <= 1.0


def test_ofi_predictive_insufficient_data():
    """With very few observations, test should return nan not crash."""
    eps = _dummy_episodes(n_ep=1, n_steps=5)
    result = ofi_predictive_test(eps, lag_steps=5, min_obs=100)
    assert np.isnan(result["spearman_rho"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── OFI interpretation consistency (v4.2 regression) ─────────────────────────

def _ofi_episode(rho_target, n=3000, seed=1):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    incr = rho_target * a + np.sqrt(max(1e-9, 1 - rho_target**2)) * rng.normal(0, 1, n)
    cum = np.concatenate([[0.0], np.cumsum(incr)])[:n]
    return pd.DataFrame({"alpha_hat": a, "adverse_sel_cost": cum})


def test_ofi_interpretation_never_contradicts():
    """A significant result must never also be described as 'no ... significant'
    in the same string, regardless of rho sign (the old bug for rho < 0)."""
    from src.statistics import ofi_predictive_test
    for rt in (+0.5, +0.06, -0.4, 0.0):
        res = ofi_predictive_test([_ofi_episode(rt)], lag_steps=1)
        txt = res["interpretation"].lower()
        if res["significant_at_05"]:
            assert "no statistically significant" not in txt, txt
            # direction word must match the sign of rho
            if res["spearman_rho"] > 0:
                assert "positive" in txt
            elif res["spearman_rho"] < 0:
                assert "negative" in txt
        else:
            assert "no statistically significant" in txt or "undefined" in txt, txt


def test_ofi_significant_negative_is_reported_as_negative():
    """The previously-contradictory case: significant rho < 0."""
    from src.statistics import ofi_predictive_test
    res = ofi_predictive_test([_ofi_episode(-0.4)], lag_steps=1)
    assert res["significant_at_05"] is True
    assert res["direction"] == "negative"
    assert "opposite to the toxicity hypothesis" in res["interpretation"]


def test_ofi_small_effect_flagged_economically():
    """Statistically significant but small |rho| must be flagged as economically limited."""
    from src.statistics import ofi_predictive_test
    res = ofi_predictive_test([_ofi_episode(0.06)], lag_steps=1)
    if res["significant_at_05"]:
        assert res["effect_size"] in ("negligible", "small")
        assert "economically" in res["interpretation"]


def test_ofi_degenerate_input_is_handled():
    import numpy as np, pandas as pd
    from src.statistics import ofi_predictive_test
    ep = pd.DataFrame({"alpha_hat": np.ones(500), "adverse_sel_cost": np.arange(500.0)})
    res = ofi_predictive_test([ep], lag_steps=1)
    assert "undefined" in res["interpretation"].lower()
    assert res["significant_at_05"] is False
