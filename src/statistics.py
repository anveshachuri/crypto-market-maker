"""
statistics.py
Rigorous statistical validation for the crypto market-making simulation.

Every claim in the README is backed by a function here. The module is
intentionally independent of the simulation engine — it operates on
summary DataFrames (returned by run_many_episodes / run_ablation_study)
and episode DataFrames (list of per-step records).

Design principles:
  1. Bootstrap CIs preferred over parametric: episode PnL is skewed and
     non-Normal in general; t-intervals are not valid without normality.
     Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap."
  2. All annualisation uses 525,600 min/year (BTC 24/7), NOT 252 days.
  3. Risk-free rate = 0 for crypto (no risk-free asset denominated in USDT).
  4. Effect sizes (Cohen's d) reported alongside p-values — statistical
     significance without effect size is uninformative at scale.
  5. Claims are hedged: simulation-internal validity is distinguished from
     real-world market validity throughout.

Public API:
  bootstrap_ci()          — non-parametric confidence interval for any statistic
  risk_metrics()          — Sharpe, Sortino, CVaR, VaR, drawdown with CIs
  cohens_d()              — effect size between two sample groups
  welch_test()            — Welch's t-test + Cohen's d in one call
  fill_calibration()      — model vs empirical fill probability by spread bin
  ofi_predictive_test()   — does OFI predict future adverse selection? (Spearman)
  out_of_sample_eval()    — walk-forward IS/OOS split validation
  sensitivity_analysis()  — one-at-a-time parameter sensitivity table
"""

import copy
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    data:         np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap:  int   = 2000,
    ci_level:     float = 0.95,
    seed:         int   = 0,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.

    Uses the percentile method (Efron 1979): resample with replacement,
    compute the statistic on each resample, take the (α/2, 1−α/2) percentiles.

    Parameters
    ----------
    data         : 1-D array of observed episode values (e.g. final_pnl)
    statistic_fn : function from array → scalar (e.g. np.mean, sharpe_fn)
    n_bootstrap  : number of bootstrap resamples (2000 is standard)
    ci_level     : confidence level, e.g. 0.95 for 95% CI
    seed         : RNG seed for reproducibility

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)

    Note: bootstrap CIs are appropriate when n ≥ 20–30. Below this, the
    percentile method under-covers; use with caution for n < 20 episodes.
    """
    rng   = np.random.default_rng(seed)
    n     = len(data)
    point = float(statistic_fn(data))
    boots = np.array([
        float(statistic_fn(rng.choice(data, size=n, replace=True)))
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - ci_level) / 2.0
    lo    = float(np.percentile(boots, 100 * alpha))
    hi    = float(np.percentile(boots, 100 * (1.0 - alpha)))
    return point, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPREHENSIVE RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────

def risk_metrics(
    summary:     pd.DataFrame,
    cfg,                        # CryptoSimConfig — for n_steps and mu0
    n_bootstrap: int   = 2000,
    ci_level:    float = 0.95,
    seed:        int   = 0,
) -> Dict[str, Any]:
    """
    Institutional-style risk-adjusted performance metrics with bootstrap CIs.

    Metrics computed:
      sharpe_annualised : (E[r] − 0) / std[r] × √N_episodes_per_year
      sortino_annualised: (E[r] − 0) / downside_std[r] × √N_episodes_per_year
      ann_return_pct    : annualised expected return as % of notional
      ann_vol_pct       : annualised return volatility as % of notional
      var_95            : 5th-percentile episode PnL (Value at Risk, 95%)
      cvar_95           : mean PnL below VaR (Expected Shortfall / CVaR)
      inventory_var_95  : 5th-percentile max_inventory per episode
      mean_max_dd       : mean peak-to-trough MTM drawdown per episode
      max_max_dd        : worst episode drawdown across all episodes

    All return-based metrics treat the full notional (cfg.mu0) as the
    capital base, which is appropriate for a delta-neutral MM funded at 1 BTC.

    Sharpe and Sortino 95% bootstrap CIs are reported as (lo, hi) tuples.
    VaR and CVaR are reported in USD (not annualised returns).
    """
    steps_per_year  = 365 * 24 * 60       # BTC 24/7
    episodes_per_yr = steps_per_year / cfg.n_steps
    ann_factor      = float(np.sqrt(episodes_per_yr))
    notional        = float(cfg.mu0)

    pnl_arr    = summary["final_pnl"].values
    ep_returns = pnl_arr / notional        # fractional returns per episode

    ann_ret  = float(ep_returns.mean() * episodes_per_yr)
    ann_vol  = float(ep_returns.std()  * ann_factor)
    sharpe   = ann_ret / (ann_vol + 1e-9)

    # Sortino: penalise only downside returns
    downside = ep_returns[ep_returns < 0.0]
    if len(downside) >= 2:
        ds_vol  = float(np.sqrt(np.mean(downside ** 2)) * ann_factor)
    else:
        ds_vol  = ann_vol   # fallback: no downside observed
    sortino = ann_ret / (ds_vol + 1e-9)

    # VaR and CVaR at 95% confidence (5th percentile of PnL)
    var_95  = float(np.percentile(pnl_arr, 5))
    tail    = pnl_arr[pnl_arr <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

    # Inventory VaR: worst-case max inventory position at 95th percentile
    inv_var_95 = float(np.percentile(summary["max_inventory"].values, 95))

    # Bootstrap CIs for Sharpe
    def _sharpe_fn(data: np.ndarray) -> float:
        r = data / notional
        return (r.mean() * episodes_per_yr) / (r.std() * ann_factor + 1e-9)

    sh_pt, sh_lo, sh_hi = bootstrap_ci(pnl_arr, _sharpe_fn, n_bootstrap, ci_level, seed)

    # Bootstrap CI for mean PnL
    mean_pt, mean_lo, mean_hi = bootstrap_ci(pnl_arr, np.mean, n_bootstrap, ci_level, seed + 1)

    return {
        "sharpe_annualised":    round(sharpe, 3),
        "sharpe_ci_95":         (round(sh_lo, 3), round(sh_hi, 3)),
        "sortino_annualised":   round(sortino, 3),
        "ann_return_pct":       round(ann_ret * 100, 3),
        "ann_vol_pct":          round(ann_vol * 100, 3),
        "mean_pnl":             round(float(pnl_arr.mean()), 2),
        "mean_pnl_ci_95":       (round(mean_lo, 2), round(mean_hi, 2)),
        "std_pnl":              round(float(pnl_arr.std()), 2),
        "var_95":               round(var_95, 2),
        "cvar_95":              round(cvar_95, 2),
        "inventory_var_95":     round(inv_var_95, 4),
        "mean_max_dd":          round(float(summary["max_drawdown"].mean()), 2),
        "max_max_dd":           round(float(summary["max_drawdown"].max()), 2),
        "n_episodes":           len(pnl_arr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. HYPOTHESIS TESTS WITH EFFECT SIZE
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d effect size: (mean_a − mean_b) / pooled_std.
    Conventions: |d| < 0.2 small, 0.2–0.5 medium, > 0.8 large.
    """
    n_a, n_b = len(a), len(b)
    pooled_std = float(np.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
        / (n_a + n_b - 2 + 1e-9)
    ))
    return float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-9))


def welch_test(
    a:     np.ndarray,
    b:     np.ndarray,
    label: str = "",
) -> Dict[str, Any]:
    """
    Welch's t-test (unequal variances) with Cohen's d effect size.

    Welch's test is more robust than Student's t when sample sizes or
    variances differ between groups — which is typical in strategy comparisons
    where different MMs have different trade frequencies and PnL volatility.

    Returns
    -------
    dict with: t_stat, p_value, cohens_d, significant_at_05, label
    """
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d    = cohens_d(a, b)
    return {
        "label":              label,
        "t_stat":             round(float(t), 3),
        "p_value":            round(float(p), 4),
        "cohens_d":           round(d, 3),
        "significant_at_05":  bool(p < 0.05),
        "significant_at_01":  bool(p < 0.01),
        "effect_size":        "small" if abs(d) < 0.2 else ("medium" if abs(d) < 0.5 else "large"),
        "n_a":                len(a),
        "n_b":                len(b),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. FILL PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def fill_calibration(
    episodes:    List[pd.DataFrame],
    n_bins:      int = 10,
    fit_model:   bool = True,
) -> pd.DataFrame:
    """
    Validate the exponential fill probability model against empirical fill rates.

    The model assumes: P(fill | spread) = exp(−fill_decay × spread/σ)
    This function checks whether the model is well-calibrated by comparing
    the model-predicted fill probability against the empirically observed
    fraction of steps where a trade actually occurred.

    Method
    ------
    1. Pool all steps across episodes.
    2. Bin steps by spread_bps decile.
    3. For each bin:
       - model_fill_prob: mean of the model's P(fill) prediction
       - empirical_fill_rate: fraction of steps where trade_occurred = True
    4. Optionally fit P(fill) = exp(−k × half_spread / σ) to empirical data
       to recover the empirical fill_decay parameter.

    Interpretation
    --------------
    - If model_fill_prob ≈ empirical_fill_rate across bins: well-calibrated.
    - If model systematically over-predicts: fill_decay is too low (should increase).
    - If model systematically under-predicts: fill_decay is too high.

    Note: fill rates in this simulation include all trader types — informed
    traders fill deterministically (they transact whenever profitable). The
    empirical fill rate therefore mixes two different fill mechanisms, which
    means the fitted decay will be lower than the fill_decay for uninformed
    flow alone. This is documented in the README as a known model limitation.
    """
    all_steps = pd.concat(episodes, ignore_index=True)

    # Bin by spread_bps
    try:
        all_steps["spread_bin"] = pd.qcut(all_steps["spread_bps"], q=n_bins, duplicates="drop")
    except ValueError:
        all_steps["spread_bin"] = pd.cut(all_steps["spread_bps"], bins=n_bins)

    result = (
        all_steps.groupby("spread_bin", observed=True)
        .agg(
            model_fill_prob     = ("fill_prob",       "mean"),
            empirical_fill_rate = ("trade_occurred",  "mean"),
            mean_spread_bps     = ("spread_bps",      "mean"),
            mean_hs_sigma_ratio = ("spread",          lambda x:
                                   (x / 2 / (all_steps.loc[x.index, "sigma"] + 1e-6)).mean()),
            informed_frac       = ("trader_type",     lambda x: (x == "informed").mean()),
            n_steps             = ("t",               "count"),
        )
        .reset_index()
    )
    result["calibration_error_abs"] = abs(result["model_fill_prob"] - result["empirical_fill_rate"])
    result["calibration_error_pct"] = (
        result["calibration_error_abs"] / (result["model_fill_prob"] + 1e-6) * 100
    )

    if fit_model and len(result) >= 3:
        # Fit k in exp(-k * x) to empirical data
        x = result["mean_hs_sigma_ratio"].values
        y = result["empirical_fill_rate"].values
        valid = (x > 0) & (y > 0) & (y < 1)
        if valid.sum() >= 2:
            from scipy.optimize import curve_fit
            try:
                def _exp_model(x_val, k):
                    return np.exp(-k * x_val)
                popt, _ = curve_fit(_exp_model, x[valid], y[valid], p0=[1.5], maxfev=1000)
                result["empirical_fill_decay"] = popt[0]
                result["fitted_fill_prob"]     = np.exp(-popt[0] * x)
            except Exception:
                result["empirical_fill_decay"] = np.nan
                result["fitted_fill_prob"]     = np.nan
        else:
            result["empirical_fill_decay"] = np.nan
            result["fitted_fill_prob"]     = np.nan

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. OFI PREDICTIVE VALIDITY TEST
# ─────────────────────────────────────────────────────────────────────────────

def ofi_predictive_test(
    episodes:  List[pd.DataFrame],
    lag_steps: int = 5,
    min_obs:   int = 100,
) -> Dict[str, Any]:
    """
    Tests whether the OFI-based alpha_hat predicts future adverse selection cost.

    Economic hypothesis (Easley et al. 2012):
        High |OFI| (order flow imbalance) is a signal of informed directional
        trading. If our toxicity proxy alpha_hat has predictive power, then
        high alpha_hat at step t should predict higher adverse selection cost
        at step t + lag_steps.

    Method
    ------
    For each step t (excluding the last `lag_steps` steps of each episode):
      - x_t = alpha_hat(t)               ← current toxicity estimate
      - y_t = adv_sel_cost(t+lag) − adv_sel_cost(t)  ← future adv. sel. increment

    Compute Spearman rank correlation ρ(x_t, y_t).
    A Spearman test is used because both series may be skewed.

    Interpretation
    --------------
    - ρ > 0, p < 0.05: alpha_hat is a statistically significant leading
      indicator of adverse selection. This validates the OFI-based approach.
    - ρ ≈ 0 or p ≥ 0.05: alpha_hat does not reliably predict adverse selection
      in this simulation. The spread-widening response is still theoretically
      motivated (see GM break-even), but lacks empirical validation here.

    IMPORTANT CAVEAT:
    This test uses simulated adverse selection labels (trader_type == "informed"),
    which are *model-generated*, not observed from real market data. The test
    therefore measures internal consistency of the simulation — not real-world
    predictive power of OFI. On real data, one would use the Lee-Ready algorithm
    or trade signing to identify aggressive trades and compare with future
    price impact.
    """
    x_vals, y_vals = [], []

    for ep in episodes:
        n = len(ep)
        if n <= lag_steps + 1:
            continue
        alpha_hat   = ep["alpha_hat"].values
        adv_sel_cum = ep["adverse_sel_cost"].values
        # Forward difference: adverse selection accumulated in next lag_steps
        for t in range(n - lag_steps):
            x_vals.append(alpha_hat[t])
            y_vals.append(adv_sel_cum[t + lag_steps] - adv_sel_cum[t])

    if len(x_vals) < min_obs:
        return {
            "spearman_rho":      np.nan,
            "p_value":           np.nan,
            "significant_at_05": False,
            "direction":         "undefined",
            "effect_size":       "undefined",
            "n_obs":             len(x_vals),
            "lag_steps":         lag_steps,
            "interpretation":    "insufficient observations",
        }

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    with warnings.catch_warnings():
        # Degenerate (constant) input is handled explicitly below; don't leak the warning.
        warnings.simplefilter("ignore")
        rho, p = stats.spearmanr(x_arr, y_arr)

    # Compose the interpretation from THREE independent, non-overlapping facts:
    #   (1) statistical significance  -> from the p-value alone
    #   (2) direction of association  -> from the sign of rho alone
    #   (3) economic/effect magnitude -> from |rho| alone
    # Keeping these orthogonal prevents contradictory phrasings such as
    # "significant ... does not significantly predict" when rho < 0.
    rho_f = float(rho)
    p_f   = float(p)
    degenerate = not (np.isfinite(rho_f) and np.isfinite(p_f))

    if degenerate:
        significant = False
        sig         = "undefined (degenerate input)"
        direction   = "undefined"
        effect      = "undefined"
        interp = ("Spearman correlation undefined (constant or degenerate input); "
                  "no relationship can be assessed.")
    else:
        significant = p_f < 0.05
        if   p_f < 0.01: sig = "highly significant (p < 0.01)"
        elif p_f < 0.05: sig = "significant (p < 0.05)"
        else:            sig = "not significant (p >= 0.05)"

        if   rho_f > 0:  direction = "positive"
        elif rho_f < 0:  direction = "negative"
        else:            direction = "zero"

        a = abs(rho_f)
        if   a < 0.10: effect = "negligible"
        elif a < 0.30: effect = "small"
        elif a < 0.50: effect = "moderate"
        else:          effect = "large"

        # Direction-aware, significance-aware sentence (covers all four quadrants).
        if significant and rho_f > 0:
            meaning = (f"alpha_hat is a statistically significant {effect} POSITIVE "
                       f"leading indicator of adverse selection at lag={lag_steps} "
                       f"(consistent with the toxicity hypothesis).")
        elif significant and rho_f < 0:
            meaning = (f"alpha_hat has a statistically significant {effect} NEGATIVE "
                       f"association with future adverse selection at lag={lag_steps} "
                       f"(opposite to the toxicity hypothesis).")
        else:  # not significant (either sign)
            meaning = (f"alpha_hat shows no statistically significant association with "
                       f"future adverse selection at lag={lag_steps} (effect size "
                       f"{effect}).")

        # Separate statistical significance from economic importance explicitly.
        if significant and effect in ("negligible", "small"):
            meaning += (" Note: statistically significant but economically "
                        f"{effect} (|rho|={a:.3f}); the practical edge is limited.")

        interp = f"Spearman rho = {rho_f:.3f} ({sig}). " + meaning

    interp += " NOTE: This test uses simulated trader labels, not real market data."

    return {
        "spearman_rho":      round(rho_f, 4) if np.isfinite(rho_f) else float("nan"),
        "p_value":           round(p_f,   6) if np.isfinite(p_f)   else float("nan"),
        "significant_at_05": bool(significant),
        "direction":         direction,
        "effect_size":       effect,
        "n_obs":             len(x_vals),
        "lag_steps":         lag_steps,
        "interpretation":    interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. OUT-OF-SAMPLE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def out_of_sample_eval(
    features:    pd.DataFrame,
    base_cfg,                   # CryptoSimConfig
    n_episodes:  int   = 30,
    is_frac:     float = 0.70,
    seed:        int   = 42,
) -> Dict[str, Any]:
    """
    Walk-forward out-of-sample validation.

    Split:
      In-sample (IS):      first is_frac of price bars → calibrate parameters
      Out-of-sample (OOS): remaining (1−is_frac) bars → evaluate same parameters

    Design rationale:
      - All parameters (gamma, alpha, sigma0, etc.) are calibrated from IS data
        only. No OOS data leaks into parameter estimation.
      - Performance is then evaluated on OOS price paths using the IS parameters.
      - IS vs OOS Sharpe comparison reveals overfitting: if IS Sharpe >> OOS
        Sharpe, the strategy is data-mined.
      - For a simulation project, the relevant question is whether the adaptive
        layers improve OOS performance vs OOS AS baseline, not whether IS=OOS.

    Minimum data requirement: each split must have ≥ 2×n_steps bars.
    If OOS split is too short, the function raises ValueError.

    Returns
    -------
    dict with:
      is_metrics  : risk_metrics() on IS episodes
      oos_metrics : risk_metrics() on OOS episodes
      is_summary  : raw IS episode summary DataFrame
      oos_summary : raw OOS episode summary DataFrame
      split_info  : {n_is, n_oos, is_end_date, oos_start_date}
      interpretation: string summarising IS vs OOS comparison
    """
    from .simulation import run_many_episodes
    from .data_loader import calibrate_params

    n_total = len(features)
    n_is    = int(n_total * is_frac)
    n_oos   = n_total - n_is
    min_bars = base_cfg.n_steps * 2

    if n_oos < min_bars:
        raise ValueError(
            f"OOS split has only {n_oos} bars (need ≥ {min_bars}). "
            f"Use more data (--candles 5000+) or reduce is_frac."
        )

    is_features  = features.iloc[:n_is].reset_index(drop=True)
    oos_features = features.iloc[n_is:].reset_index(drop=True)

    # Calibrate on IS data only — no OOS leakage
    is_params = calibrate_params(is_features)

    # Build IS config
    is_cfg = copy.copy(base_cfg)
    is_cfg.mu0           = is_params["mu0"]
    is_cfg.sigma0        = is_params["sigma0"]
    is_cfg.sigma_v       = is_params["sigma_v"]
    is_cfg.noise_var     = is_params["noise_var"]
    is_cfg.process_noise = is_params["process_noise"]
    is_cfg.alpha         = is_params["alpha"]
    is_cfg.seed          = seed

    # OOS config: same params as IS (no re-calibration on OOS data)
    oos_cfg = copy.copy(is_cfg)

    print(f"  [OOS] IS: {n_is} bars ({is_frac*100:.0f}%) | OOS: {n_oos} bars ({(1-is_frac)*100:.0f}%)")
    print(f"  [OOS] IS params: alpha={is_params['alpha']:.3f}, sigma0=${is_params['sigma0']:.2f}")

    _, is_summary  = run_many_episodes(is_cfg,  is_features,  n_episodes, verbose=False)
    _, oos_summary = run_many_episodes(oos_cfg, oos_features, n_episodes, verbose=False)

    is_metrics  = risk_metrics(is_summary,  is_cfg,  n_bootstrap=500)
    oos_metrics = risk_metrics(oos_summary, oos_cfg, n_bootstrap=500)

    is_sharpe  = is_metrics["sharpe_annualised"]
    oos_sharpe = oos_metrics["sharpe_annualised"]
    degradation = ((is_sharpe - oos_sharpe) / (abs(is_sharpe) + 1e-9)) * 100

    if abs(degradation) < 20:
        verdict = "STABLE — IS and OOS Sharpe within 20%; low overfitting risk."
    elif degradation > 50:
        verdict = "OVERFITTED — Sharpe degraded >50% OOS. Parameters may be data-mined."
    else:
        verdict = "MODERATE DEGRADATION — some IS/OOS gap; expected for calibrated models."

    interp = (
        f"IS Sharpe={is_sharpe:.2f} vs OOS Sharpe={oos_sharpe:.2f} "
        f"(degradation={degradation:.1f}%). {verdict}"
    )
    print(f"  [OOS] {interp}")

    split_info = {
        "n_is":            n_is,
        "n_oos":           n_oos,
        "is_end_idx":      n_is,
        "oos_start_idx":   n_is,
        "n_episodes":      n_episodes,
    }

    return {
        "is_metrics":    is_metrics,
        "oos_metrics":   oos_metrics,
        "is_summary":    is_summary,
        "oos_summary":   oos_summary,
        "split_info":    split_info,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(
    base_cfg,           # CryptoSimConfig
    features:  pd.DataFrame,
    n_episodes: int = 15,
    seed:       int = 0,
) -> pd.DataFrame:
    """
    One-at-a-time (OAT) parameter sensitivity analysis.

    For each parameter, we vary it over a ±50% range from its baseline
    while holding all others fixed, then record mean PnL and Sharpe.

    Purpose:
      - Parameters with large sensitivity (>20% PnL change per 50% param
        change) require stronger calibration justification.
      - Parameters with low sensitivity are robust — small calibration errors
        do not materially affect performance.

    Parameters varied:
      gamma          : AS risk-aversion (core spread calibration)
      fill_decay     : fill probability decay (model assumption)
      as_multiplier  : toxicity response aggressiveness (Layer 2)
      vol_gamma      : vol regime spread multiplier (Layer 3)
      ofi_sensitivity: OFI skew magnitude (Layer 4)
      penalty_lambda : inventory penalty scale (Layer 1)

    Returns
    -------
    DataFrame with columns: param, value, pct_of_baseline, mean_pnl, std_pnl,
      sharpe_annualised, mean_max_dd, vs_baseline_pnl_pct
    """
    from .simulation import run_many_episodes

    steps_per_year  = 365 * 24 * 60
    episodes_per_yr = steps_per_year / base_cfg.n_steps
    ann_factor      = float(np.sqrt(episodes_per_yr))
    notional        = float(base_cfg.mu0)

    def _sharpe(pnl_arr: np.ndarray) -> float:
        r = pnl_arr / notional
        return float((r.mean() * episodes_per_yr) / (r.std() * ann_factor + 1e-9))

    # Run baseline first
    cfg0 = copy.copy(base_cfg)
    cfg0.seed = seed
    _, base_summary = run_many_episodes(cfg0, features, n_episodes, verbose=False)
    base_pnl    = float(base_summary["final_pnl"].mean())
    base_sharpe = _sharpe(base_summary["final_pnl"].values)

    param_grids = {
        "gamma":           [0.03, 0.05, 0.075, 0.10, 0.15, 0.20],
        "fill_decay":      [0.5,  0.8,  1.0,   1.5,  2.0,  3.0 ],
        "as_multiplier":   [0.0,  0.5,  1.0,   2.0,  3.0,  4.0 ],
        "vol_gamma":       [0.0,  0.1,  0.25,  0.5,  0.75, 1.0 ],
        "ofi_sensitivity": [0.0,  0.1,  0.2,   0.3,  0.4,  0.5 ],
        "penalty_lambda":  [0.0,  0.01, 0.03,  0.05, 0.10, 0.20],
    }

    rows = []
    for param, values in param_grids.items():
        baseline_val = float(getattr(base_cfg, param, np.nan))
        for val in values:
            cfg_i = copy.copy(base_cfg)
            cfg_i.seed = seed
            setattr(cfg_i, param, val)
            try:
                _, summary_i = run_many_episodes(cfg_i, features, n_episodes, verbose=False)
                mean_pnl    = float(summary_i["final_pnl"].mean())
                std_pnl     = float(summary_i["final_pnl"].std())
                sh          = _sharpe(summary_i["final_pnl"].values)
                mean_max_dd = float(summary_i["max_drawdown"].mean())
            except Exception as e:
                mean_pnl = std_pnl = sh = mean_max_dd = np.nan

            pct_of_base = (val / (baseline_val + 1e-9) - 1.0) * 100
            vs_base_pnl = ((mean_pnl - base_pnl) / (abs(base_pnl) + 1.0)) * 100

            rows.append({
                "param":              param,
                "value":              val,
                "baseline_value":     baseline_val,
                "pct_of_baseline":    round(pct_of_base, 1),
                "mean_pnl":           round(mean_pnl, 2),
                "std_pnl":            round(std_pnl, 2),
                "sharpe_annualised":  round(sh, 3),
                "mean_max_dd":        round(mean_max_dd, 2),
                "vs_baseline_pnl_pct": round(vs_base_pnl, 1),
            })

    df = pd.DataFrame(rows)
    df["is_baseline"] = np.abs(df["pct_of_baseline"]) < 1.0
    return df
