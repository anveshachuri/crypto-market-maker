"""
analysis.py
Post-simulation analysis — adapted for crypto (USD/bps units).

Analyses:
  1. PnL decomposition (spread revenue, adverse selection, inventory carry, momentum)
  2. Regime-conditional performance (quiet / volatile / trending)
  3. Three-way strategy comparison (GM, AS, Elite) with t-tests
  4. Adverse selection dynamics (alpha_hat, OFI tracking)
  5. Spread efficiency (fill rate vs spread in bps)
  6. Belief convergence (mu_error in bps)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Dict, Tuple, Optional

from .simulation import CryptoSimConfig, run_many_episodes, run_episode_hybrid, run_three_way_comparison


# ─────────────────────────────────────────────────────────────
# 1. PNL DECOMPOSITION
# ─────────────────────────────────────────────────────────────

def pnl_decomposition(df: pd.DataFrame) -> Dict[str, float]:
    """
    Decompose episode P&L into four interpretable components.

    Identity (approximate — residual reflects the half-spread approximation
    vs actual trade P&L, and is small relative to spread_revenue):
        total_pnl ≈ spread_revenue
                    − adverse_selection_cost
                    + inventory_mtm_pnl      ← signed: positive = tailwind
                    − momentum_cost
                    + residual
    """
    final       = df.iloc[-1]
    total_pnl   = final["mtm_pnl"]
    spread_rev  = final["spread_revenue"]
    adv_sel     = final["adverse_sel_cost"]
    inv_mtm     = final["inventory_mtm_pnl"]   # signed: positive = gain from inventory
    mom_cost    = final["momentum_cost"]
    residual    = total_pnl - (spread_rev - adv_sel + inv_mtm - mom_cost)

    return {
        "total_pnl":         round(total_pnl, 2),
        "spread_revenue":    round(spread_rev, 2),
        "adverse_selection": round(-adv_sel, 2),
        "inventory_mtm":     round(inv_mtm, 2),
        "momentum_cost":     round(-mom_cost, 2),
        "residual":          round(residual, 2),
        "adv_sel_pct":       round(adv_sel / (spread_rev + 1e-6) * 100, 1),
        "inv_mtm_pct":       round(abs(inv_mtm) / (spread_rev + 1e-6) * 100, 1),
    }


def pnl_decomposition_many(episodes: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows    = [pnl_decomposition(ep) for ep in episodes]
    df      = pd.DataFrame(rows)
    summary = pd.DataFrame({"mean": df.mean(), "std": df.std(), "median": df.median()})
    return df, summary


# ─────────────────────────────────────────────────────────────
# 2. REGIME-CONDITIONAL PERFORMANCE
# ─────────────────────────────────────────────────────────────

def regime_performance(episodes: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ep in episodes:
        for regime, grp in ep.groupby("regime"):
            trades = grp[grp["trade_occurred"]]
            n_tr   = len(trades)
            if n_tr == 0:
                continue
            pnl_delta = grp["mtm_pnl"].diff().fillna(0)
            rows.append({
                "regime":              regime,
                "mean_spread_bps":     grp["spread_bps"].mean(),
                "fill_rate":           grp["fill_prob"].mean(),
                "pnl_per_step":        pnl_delta.mean(),
                "pnl_per_trade":       pnl_delta[grp["trade_occurred"]].mean(),
                "adv_sel_per_trade":   (grp["adverse_sel_cost"].diff().fillna(0)
                                         [grp["trader_type"] == "informed"]).mean(),
                "inv_abs_mean":        grp["inventory"].abs().mean(),
                "alpha_hat_mean":      grp["alpha_hat"].mean(),
                "sigma_mean":          grp["sigma"].mean(),
                "mu_error_bps_mean":   grp["mu_error_bps"].abs().mean(),
                "n_steps":             len(grp),
                "n_trades":            n_tr,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.groupby("regime").agg(
        mean_spread_bps     = ("mean_spread_bps",   "mean"),
        fill_rate           = ("fill_rate",          "mean"),
        pnl_per_step        = ("pnl_per_step",       "mean"),
        pnl_per_trade       = ("pnl_per_trade",      "mean"),
        inv_abs_mean        = ("inv_abs_mean",       "mean"),
        alpha_hat_mean      = ("alpha_hat_mean",     "mean"),
        sigma_mean          = ("sigma_mean",         "mean"),
        mu_error_bps_mean   = ("mu_error_bps_mean",  "mean"),
        total_steps         = ("n_steps",            "sum"),
        total_trades        = ("n_trades",           "sum"),
    ).reset_index()


# ─────────────────────────────────────────────────────────────
# 3. THREE-WAY STRATEGY COMPARISON
# ─────────────────────────────────────────────────────────────

def compare_strategies(
    base_cfg:   CryptoSimConfig,
    features:   pd.DataFrame,
    n_episodes: int = 60,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    results = run_three_way_comparison(base_cfg, features, n_episodes, verbose=True)

    # Annualized Sharpe on per-episode returns.
    # Each episode = n_steps minutes of 24/7 BTC trading.
    # Annual episodes = 525,600 / n_steps  →  annualisation factor = sqrt(annual_episodes)
    steps_per_year   = 365 * 24 * 60          # BTC trades 24/7
    episodes_per_yr  = steps_per_year / base_cfg.n_steps
    ann_factor       = np.sqrt(episodes_per_yr)
    notional         = base_cfg.mu0            # 1-BTC-equivalent capital base

    rows = []
    for strategy, summary in results.items():
        ep_returns = summary["final_pnl"] / notional
        ann_ret    = ep_returns.mean() * episodes_per_yr
        ann_vol    = ep_returns.std()  * ann_factor
        sharpe     = ann_ret / (ann_vol + 1e-9)   # r_f = 0 for crypto

        rows.append({
            "strategy":        strategy,
            "mean_pnl":        summary["final_pnl"].mean(),
            "std_pnl":         summary["final_pnl"].std(),
            "sharpe_annualised": round(sharpe, 3),
            "mean_regret":     summary["final_regret"].mean(),
            "mean_max_inv":    summary["max_inventory"].mean(),
            "mean_spread_bps": summary["mean_spread_bps"].mean(),
            "mean_fill_prob":  summary["mean_fill_prob"].mean(),
            "adv_sel_cost":    summary["adverse_sel_cost"].mean(),
            "inv_mtm_pnl":     summary["inventory_mtm_pnl"].mean(),
            "pnl_per_trade":   summary["pnl_per_trade"].mean(),
            "mean_max_dd":     summary["max_drawdown"].mean(),
            "mean_max_dd_bps": summary["max_drawdown_bps"].mean(),
        })

    comparison = pd.DataFrame(rows).set_index("strategy")

    pnls  = {s: results[s]["final_pnl"].values for s in results}
    pairs = [
        ("Glosten-Milgrom", "Avellaneda-Stoikov"),
        ("Avellaneda-Stoikov", "Adaptive-MM"),
        ("Glosten-Milgrom", "Adaptive-MM"),
    ]
    t_tests = {}
    for a, b in pairs:
        if a in pnls and b in pnls:
            t, p = stats.ttest_ind(pnls[a], pnls[b], equal_var=False)
            t_tests[f"{a} vs {b}"] = {"t_stat": round(t, 3), "p_value": round(p, 4)}

    return results, comparison, pd.DataFrame(t_tests).T


# ─────────────────────────────────────────────────────────────
# 4. ADVERSE SELECTION DYNAMICS
# ─────────────────────────────────────────────────────────────

def adverse_selection_analysis(episodes: List[pd.DataFrame]) -> pd.DataFrame:
    alpha_hats = np.array([ep["alpha_hat"].values for ep in episodes])
    spreads    = np.array([ep["spread_bps"].values for ep in episodes])

    return pd.DataFrame({
        "alpha_hat_mean": alpha_hats.mean(axis=0),
        "alpha_hat_p25":  np.percentile(alpha_hats, 25, axis=0),
        "alpha_hat_p75":  np.percentile(alpha_hats, 75, axis=0),
        "spread_bps_mean": spreads.mean(axis=0),
        "spread_bps_p25":  np.percentile(spreads, 25, axis=0),
        "spread_bps_p75":  np.percentile(spreads, 75, axis=0),
    })


# ─────────────────────────────────────────────────────────────
# 5. SPREAD EFFICIENCY
# ─────────────────────────────────────────────────────────────

def spread_fill_tradeoff(episodes: List[pd.DataFrame]) -> pd.DataFrame:
    all_steps = pd.concat(episodes, ignore_index=True)
    all_steps["spread_bin"] = pd.qcut(all_steps["spread_bps"], q=10, duplicates="drop")

    result = all_steps.groupby("spread_bin").agg(
        fill_rate      = ("trade_occurred", "mean"),
        mean_fill_prob = ("fill_prob",      "mean"),
        pnl_per_step   = ("trade_pnl",     "mean"),
        n_steps        = ("t",             "count"),
        informed_frac  = ("trader_type",   lambda x: (x == "informed").mean()),
    ).reset_index()
    result["spread_bps_mid"] = result["spread_bin"].apply(lambda x: x.mid)
    return result


# ─────────────────────────────────────────────────────────────
# 6. BELIEF CONVERGENCE
# ─────────────────────────────────────────────────────────────

def belief_convergence(episodes: List[pd.DataFrame]) -> pd.DataFrame:
    errors = np.array([ep["mu_error_bps"].abs().values for ep in episodes])
    return pd.DataFrame({
        "mean": errors.mean(axis=0),
        "p25":  np.percentile(errors, 25, axis=0),
        "p50":  np.percentile(errors, 50, axis=0),
        "p75":  np.percentile(errors, 75, axis=0),
    })


# ─────────────────────────────────────────────────────────────
# 7. ALPHA SWEEP
# ─────────────────────────────────────────────────────────────

def alpha_sweep(
    base_cfg:   CryptoSimConfig,
    features:   pd.DataFrame,
    alphas:     Optional[List[float]] = None,
    n_episodes: int = 20,
) -> pd.DataFrame:
    if alphas is None:
        alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    rows = []
    for a in alphas:
        import copy
        cfg = copy.copy(base_cfg)
        cfg.alpha = a
        _, summary = run_many_episodes(cfg, features, n_episodes, verbose=False)
        rows.append({
            "alpha":           a,
            "mean_spread_bps": summary["mean_spread_bps"].mean(),
            "mean_pnl":        summary["final_pnl"].mean(),
            "std_pnl":         summary["final_pnl"].std(),
            "mean_regret":     summary["final_regret"].mean(),
            "rmse_mu_bps":     summary["rmse_mu_bps"].mean(),
            "adv_sel_cost":    summary["adverse_sel_cost"].mean(),
            "mean_fill_prob":  summary["mean_fill_prob"].mean(),
        })
        print(f"  α={a:.2f} | spread={rows[-1]['mean_spread_bps']:.1f}bps | "
              f"pnl=${rows[-1]['mean_pnl']:,.0f} | regret=${rows[-1]['mean_regret']:,.0f}")

    return pd.DataFrame(rows)
