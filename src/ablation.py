"""
ablation.py
Systematic ablation study for AdaptiveMM.

Study design: BUILD-UP experiment
  Starting from the Avellaneda-Stoikov baseline, we add each adaptive layer
  one at a time and measure incremental performance improvement.

  Variant 0 — Passive-Fixed          : constant spread, no adaptation
  Variant 1 — AS-Baseline            : Avellaneda-Stoikov, no adaptive layers
  Variant 2 — AS + Inventory         : + exponential inventory penalty (L1)
  Variant 3 — AS + Inv + Vol         : + volatility regime widening (L3)
  Variant 4 — AS + Inv + Vol + Tox   : + OFI toxicity adjustment (L2)
  Variant 5 — Full Adaptive          : + asymmetric OFI skew (L4)

Additional single-layer benchmarks (vs Passive baseline):
  OFI-Only                           : AS + OFI skew only (L4, no others)
  Toxicity-Only                      : AS + toxicity only (L2, no others)

This design answers the question:
  "Which adaptive component is responsible for performance improvement?"

If Full Adaptive >> AS-Baseline but the gain is concentrated in, say,
only the Vol layer, then the OFI and Toxicity layers need stronger justification.

Note on interpretation:
  All results are from a simulation with model-generated trader populations.
  The ablation identifies which layers help within the simulation framework.
  Whether these layers add value on real exchange data depends on whether
  the simulation's adverse selection model is a good proxy for real informed flow.
  See README § "Simulation Realism" for a full discussion.
"""

import copy
import numpy as np
import pandas as pd
import scipy.stats as stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .market_maker import AdaptiveMM, PassiveMM, AvellanedaStoikovMM
from .simulation   import CryptoSimConfig, run_many_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Variant definitions
# ─────────────────────────────────────────────────────────────────────────────

# Each variant is (label, dict of AdaptiveMM ablation-flag overrides).
# "passive" is handled separately (different class).
ABLATION_VARIANTS = [
    # Build-up sequence
    ("Passive-Fixed",         {"_type": "passive"}),
    ("AS-Baseline",           {"use_inventory_penalty": False, "use_vol_regime": False,
                               "use_toxicity_adj": False, "use_ofi_skew": False}),
    ("AS+Inventory",          {"use_inventory_penalty": True,  "use_vol_regime": False,
                               "use_toxicity_adj": False, "use_ofi_skew": False}),
    ("AS+Inv+Vol",            {"use_inventory_penalty": True,  "use_vol_regime": True,
                               "use_toxicity_adj": False, "use_ofi_skew": False}),
    ("AS+Inv+Vol+Toxicity",   {"use_inventory_penalty": True,  "use_vol_regime": True,
                               "use_toxicity_adj": True,  "use_ofi_skew": False}),
    ("Full-Adaptive",         {"use_inventory_penalty": True,  "use_vol_regime": True,
                               "use_toxicity_adj": True,  "use_ofi_skew": True}),
    # Single-layer benchmarks
    ("OFI-Only",              {"use_inventory_penalty": False, "use_vol_regime": False,
                               "use_toxicity_adj": False, "use_ofi_skew": True}),
    ("Toxicity-Only",         {"use_inventory_penalty": False, "use_vol_regime": False,
                               "use_toxicity_adj": True,  "use_ofi_skew": False}),
]

# Which variants form the primary build-up sequence (for the main plot)
BUILD_UP_SEQUENCE = [
    "Passive-Fixed", "AS-Baseline", "AS+Inventory",
    "AS+Inv+Vol", "AS+Inv+Vol+Toxicity", "Full-Adaptive",
]


def _build_variant_mm(cfg: CryptoSimConfig, flags: dict) -> object:
    """
    Construct a market maker from ablation flags.
    If flags has "_type"="passive", returns PassiveMM.
    Otherwise returns AdaptiveMM with the specified layers toggled.
    """
    if flags.get("_type") == "passive":
        # Calibrate passive half-spread to 5 bps — typical for calm BTC/USDT
        return PassiveMM(
            half_spread_bps = 5.0,
            min_spread      = cfg.min_spread,
            max_spread      = cfg.max_spread,
        )

    mm = AdaptiveMM(
        gamma            = cfg.gamma,
        kappa            = cfg.kappa_mm,
        alpha            = cfg.alpha,
        min_spread       = cfg.min_spread,
        max_spread       = cfg.max_spread,
        inv_limit        = cfg.inv_limit,
        penalty_lambda   = cfg.penalty_lambda,
        inv_scale        = cfg.inv_scale,
        as_multiplier    = cfg.as_multiplier,
        vol_gamma        = cfg.vol_gamma,
        ofi_sensitivity  = cfg.ofi_sensitivity,
        fill_decay       = cfg.fill_decay,
        # Apply ablation flags
        use_inventory_penalty = flags.get("use_inventory_penalty", True),
        use_vol_regime        = flags.get("use_vol_regime",        True),
        use_toxicity_adj      = flags.get("use_toxicity_adj",      True),
        use_ofi_skew          = flags.get("use_ofi_skew",          True),
    )
    return mm


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(
    base_cfg:   CryptoSimConfig,
    features:   pd.DataFrame,
    n_episodes: int  = 30,
    variants:   list = None,
    verbose:    bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run all ablation variants and return per-variant episode summaries.

    Parameters
    ----------
    base_cfg   : baseline config (parameters are inherited by all variants)
    features   : feature DataFrame from load_btcusdt()
    n_episodes : episodes per variant (≥ 30 for meaningful statistics)
    variants   : list of (label, flags) tuples — defaults to ABLATION_VARIANTS
    verbose    : print progress

    Returns
    -------
    dict: label → episode summary DataFrame (from run_many_episodes)
    """
    if variants is None:
        variants = ABLATION_VARIANTS

    results: Dict[str, pd.DataFrame] = {}

    for label, flags in variants:
        if verbose:
            print(f"  [Ablation] {label} ...", end="", flush=True)

        # Inject the variant MM directly via mm_type="adaptive" but swap the
        # MM object in the episode runner.
        variant_cfg = copy.copy(base_cfg)
        variant_cfg.seed = base_cfg.seed  # reproducible

        mm = _build_variant_mm(variant_cfg, flags)

        # Run episodes by patching _build_mm to return our custom mm
        summaries = []
        from .simulation import _run_episode_core, HybridProcess
        from .traders    import TraderPopulation
        from .belief     import GaussianBelief
        from .market_maker import OracleMM

        for i in range(n_episodes):
            seed_i  = (variant_cfg.seed + i * 1000) if variant_cfg.seed else None
            process = HybridProcess.from_features(features, seed=seed_i)
            df      = _run_ablation_episode(variant_cfg, process, mm, seed=seed_i)
            trades  = df[df["trade_occurred"]]
            n_tr    = len(trades)
            pnl_series  = df["mtm_pnl"].values
            running_max = np.maximum.accumulate(pnl_series)
            max_dd      = float(np.max(running_max - pnl_series))

            summaries.append({
                "episode":           i,
                "final_pnl":         df["mtm_pnl"].iloc[-1],
                "final_regret":      df["regret"].iloc[-1],
                "n_trades":          n_tr,
                "mean_spread_bps":   df["spread_bps"].mean(),
                "max_inventory":     df["inventory"].abs().max(),
                "mean_fill_prob":    df["fill_prob"].mean(),
                "spread_revenue":    df["spread_revenue"].iloc[-1],
                "adverse_sel_cost":  df["adverse_sel_cost"].iloc[-1],
                "inventory_mtm_pnl": df["inventory_mtm_pnl"].iloc[-1],
                "pnl_per_trade":     df["mtm_pnl"].iloc[-1] / max(n_tr, 1),
                "max_drawdown":      max_dd,
                "max_drawdown_bps":  max_dd / df["V"].mean() * 10_000,
                "variant":           label,
            })

        results[label] = pd.DataFrame(summaries)
        if verbose:
            pnl_mean = results[label]["final_pnl"].mean()
            pnl_std  = results[label]["final_pnl"].std()
            print(f" PnL=${pnl_mean:+,.0f} ± ${pnl_std:,.0f}")

    return results


def _run_ablation_episode(
    cfg,
    process,
    mm,            # pre-built MM instance
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stripped-down episode runner that accepts a pre-built MM object.
    Mirrors _run_episode_core but bypasses _build_mm().
    """
    import numpy as np
    import pandas as pd
    from .traders      import TraderPopulation
    from .belief       import GaussianBelief
    from .market_maker import AdaptiveMM, OracleMM

    rng     = np.random.default_rng(seed)
    traders = TraderPopulation(
        alpha=cfg.alpha, momentum_frac=cfg.momentum_frac,
        buy_prob_uninformed=cfg.buy_prob, seed=seed
    )
    belief  = GaussianBelief(
        mu0=cfg.mu0, sigma0=cfg.sigma0, alpha=cfg.alpha,
        noise_var=cfg.noise_var, process_noise=cfg.process_noise
    )
    oracle  = OracleMM(tick=cfg.min_spread * 0.1)
    # Fill model: use AdaptiveMM with cfg.fill_decay for probability
    fill_ref = AdaptiveMM(fill_decay=cfg.fill_decay)

    inventory = 0.0; cash = 0.0
    oracle_inv = 0.0; oracle_cash = 0.0
    spread_revenue = 0.0; adverse_sel_cost = 0.0
    inventory_mtm_pnl = 0.0; momentum_cost = 0.0
    records = []
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
        ofi       = belief.order_flow_imbalance

        bid, ask   = mm.quote(belief.mu, belief.sigma, inventory, t_remaining, alpha_hat, ofi)
        o_bid, o_ask = oracle.quote(V)
        half_spread  = (ask - bid) / 2.0

        side, size, trader_type = traders.arrive(V, bid, ask)
        fill_prob = fill_ref.fill_probability(half_spread, belief.sigma)

        if trader_type == "informed":
            filled = (side != "none")
        else:
            filled = (side != "none") and (rng.random() < fill_prob)

        trade_pnl = 0.0; oracle_pnl = 0.0; trade_occurred = False; actual_size = 0.0

        if filled and side != "none":
            actual_size = size; trade_occurred = True
            if side == "buy":
                cash += ask * actual_size; inventory -= actual_size
                trade_pnl = (ask - V) * actual_size
                oracle_cash += o_ask * actual_size; oracle_inv -= actual_size
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
        regime     = getattr(process, "regime", "unknown")

        records.append({
            "t":                   t, "V": V, "bid": bid, "ask": ask,
            "spread":              ask - bid,
            "spread_bps":          (ask - bid) / V * 10_000,
            "fill_prob":           fill_prob,
            "side":                side,
            "trader_type":         trader_type if trade_occurred else "none",
            "trade_size":          actual_size,
            "trade_occurred":      trade_occurred,
            "inventory":           inventory,
            "cash":                cash,
            "mtm_pnl":             mtm_pnl,
            "oracle_mtm":          oracle_mtm,
            "regret":              oracle_mtm - mtm_pnl,
            "regime":              regime,
            "spread_revenue":      spread_revenue,
            "adverse_sel_cost":    adverse_sel_cost,
            "inventory_mtm_pnl":   inventory_mtm_pnl,
            "momentum_cost":       momentum_cost,
            "alpha_hat":           belief.alpha_hat,
        })
        V_prev = V

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation summary table
# ─────────────────────────────────────────────────────────────────────────────

def ablation_summary(
    results:    Dict[str, pd.DataFrame],
    base_cfg,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Aggregate ablation results into a comparison table with bootstrap CIs.

    Returns DataFrame with one row per variant and columns:
      mean_pnl, pnl_ci_lo, pnl_ci_hi, sharpe_annualised,
      mean_spread_bps, adv_sel_cost, mean_max_dd,
      incremental_pnl (vs previous build-up step)
    """
    from .statistics import bootstrap_ci, risk_metrics

    rows = []
    prev_pnl = None

    for label, summary in results.items():
        pnl_arr = summary["final_pnl"].values
        _, lo, hi = bootstrap_ci(pnl_arr, np.mean, n_bootstrap)
        rm = risk_metrics(summary, base_cfg, n_bootstrap=n_bootstrap)

        row = {
            "variant":           label,
            "mean_pnl":          round(float(np.mean(pnl_arr)), 2),
            "pnl_ci_lo":         round(lo, 2),
            "pnl_ci_hi":         round(hi, 2),
            "sharpe_annualised": rm["sharpe_annualised"],
            "sortino":           rm["sortino_annualised"],
            "cvar_95":           rm["cvar_95"],
            "mean_spread_bps":   round(float(summary["mean_spread_bps"].mean()), 2),
            "adv_sel_cost":      round(float(summary["adverse_sel_cost"].mean()), 2),
            "mean_max_dd":       round(float(summary["max_drawdown"].mean()), 2),
            "n_episodes":        len(pnl_arr),
        }

        if prev_pnl is not None and label in BUILD_UP_SEQUENCE:
            row["incremental_pnl"] = round(float(np.mean(pnl_arr)) - prev_pnl, 2)
        else:
            row["incremental_pnl"] = np.nan

        if label in BUILD_UP_SEQUENCE:
            prev_pnl = float(np.mean(pnl_arr))

        rows.append(row)

    return pd.DataFrame(rows)


def ablation_significance(
    results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Pairwise Welch's t-tests + Cohen's d between consecutive build-up steps.
    Answers: "is the improvement from adding each layer statistically significant?"
    """
    from .statistics import welch_test
    labels = [l for l, _ in ABLATION_VARIANTS if l in results and l in BUILD_UP_SEQUENCE]
    rows = []
    for i in range(len(labels) - 1):
        a_label, b_label = labels[i], labels[i + 1]
        if a_label not in results or b_label not in results:
            continue
        a_pnl = results[a_label]["final_pnl"].values
        b_pnl = results[b_label]["final_pnl"].values
        test  = welch_test(a_pnl, b_pnl, label=f"{a_label} → {b_label}")
        test["mean_pnl_a"] = round(float(a_pnl.mean()), 2)
        test["mean_pnl_b"] = round(float(b_pnl.mean()), 2)
        rows.append(test)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
