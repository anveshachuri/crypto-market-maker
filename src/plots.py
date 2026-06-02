"""
plots.py
Publication-quality plots for BTC/USDT Bayesian market maker.

All figures adapted for crypto context:
  - Prices in USD, spreads in bps
  - BTC inventory in BTC (not abstract units)
  - Regime labels show actual market conditions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional

PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "quiet":    "#2196F3",
    "volatile": "#F44336",
    "trending": "#FF9800",
    "gm":       "#9E9E9E",
    "as":       "#2196F3",
    "adaptive": "#4CAF50",
}

# Style dict applied per-figure via plt.rc_context() to avoid polluting the
# global matplotlib state (a module-level rcParams.update is a side effect
# that breaks any other matplotlib code in the same Python session).
_RC = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAFA",
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def _save(name: str, tight: bool = True):
    path = PLOT_DIR / f"{name}.png"
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def _fig(*args, **kwargs):
    """Create a figure inside the project rc context."""
    plt.rcParams.update(_RC)
    return plt.figure(*args, **kwargs)


def _subplots(*args, **kwargs):
    """Create subplots inside the project rc context."""
    plt.rcParams.update(_RC)
    return plt.subplots(*args, **kwargs)


# ─────────────────────────────────────────────────────────────
# 1. Episode overview (real price + quotes + inventory + PnL)
# ─────────────────────────────────────────────────────────────

def plot_episode_overview(df: pd.DataFrame, title: str = "Adaptive MM — Single Episode"):
    fig = _fig(figsize=(15, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    regime_colors = {"quiet": "#E3F2FD", "volatile": "#FFEBEE", "trending": "#FFF3E0"}

    def shade_regimes(ax):
        prev_r, start = None, 0
        for i, r in enumerate(df["regime"]):
            if r != prev_r:
                if prev_r is not None:
                    ax.axvspan(start, i, alpha=0.25, color=regime_colors.get(prev_r, "#EEE"), lw=0)
                prev_r, start = r, i
        if prev_r:
            ax.axvspan(start, len(df), alpha=0.25, color=regime_colors.get(prev_r, "#EEE"), lw=0)

    # 1. BTC Price + quotes
    ax1 = fig.add_subplot(gs[0, :])
    shade_regimes(ax1)
    ax1.plot(df["V"],   color="#212121", lw=1.2, label="True mid (BTC/USDT)", zorder=3)
    ax1.plot(df["mu"],  color="#1565C0", lw=1.0, ls="--", alpha=0.8, label="Belief μ")
    ax1.plot(df["bid"], color="#2E7D32", lw=0.8, alpha=0.6, label="MM bid")
    ax1.plot(df["ask"], color="#C62828", lw=0.8, alpha=0.6, label="MM ask")
    ax1.fill_between(df.index, df["bid"], df["ask"], alpha=0.08, color="#9E9E9E")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, ncol=4)

    # Regime legend patch
    for regime, color in regime_colors.items():
        ax1.plot([], [], color=color, alpha=0.7, lw=8, label=regime.capitalize())
    ax1.legend(fontsize=7, ncol=7, loc="upper right")

    # 2. Spread in bps
    ax2 = fig.add_subplot(gs[1, 0])
    shade_regimes(ax2)
    ax2.plot(df["spread_bps"], color="#7B1FA2", lw=1.0)
    ax2.set_ylabel("Spread (bps)")
    ax2.set_title("Quoted Spread")

    # 3. Inventory (BTC)
    ax3 = fig.add_subplot(gs[1, 1])
    shade_regimes(ax3)
    ax3.fill_between(df.index, df["inventory"], alpha=0.5,
                     color=["#C62828" if x < 0 else "#2E7D32" for x in df["inventory"]])
    ax3.axhline(0, color="#212121", lw=0.8)
    ax3.set_ylabel("Inventory (BTC)")
    ax3.set_title("Inventory")

    # 4. MTM PnL (USD)
    ax4 = fig.add_subplot(gs[2, 0])
    shade_regimes(ax4)
    ax4.plot(df["mtm_pnl"],    color="#4CAF50", lw=1.2, label="MM PnL")
    ax4.plot(df["oracle_mtm"], color="#FF9800", lw=1.0, ls="--", alpha=0.7, label="Oracle PnL")
    ax4.axhline(0, color="#212121", lw=0.8)
    ax4.set_ylabel("MTM P&L (USD)")
    ax4.set_title("P&L vs Oracle")
    ax4.legend(fontsize=8)

    # 5. Belief uncertainty + alpha_hat
    ax5 = fig.add_subplot(gs[2, 1])
    shade_regimes(ax5)
    ax5_r = ax5.twinx()
    ax5.plot(df["sigma"],     color="#1565C0", lw=1.0, label="σ (belief)")
    ax5_r.plot(df["alpha_hat"], color="#E65100", lw=1.0, ls="--", label="α̂ (informed est.)")
    ax5.set_ylabel("σ (USD)", color="#1565C0")
    ax5_r.set_ylabel("α̂", color="#E65100")
    ax5.set_title("Belief Uncertainty & Toxicity")
    lines1, l1 = ax5.get_legend_handles_labels()
    lines2, l2 = ax5_r.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, l1 + l2, fontsize=7)

    _save("episode_overview")


# ─────────────────────────────────────────────────────────────
# 2. PnL decomposition
# ─────────────────────────────────────────────────────────────

def plot_pnl_decomposition(summary: pd.DataFrame):
    fig, axes = _subplots(1, 2, figsize=(12, 5))

    components = ["spread_revenue", "adverse_selection", "inventory_mtm", "momentum_cost"]
    labels     = ["Spread Revenue", "Adverse Selection\n(cost)", "Inventory MTM\n(gain/loss)", "Momentum\n(cost)"]
    colors     = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0"]
    means      = [summary.loc[c, "mean"] for c in components]
    stds       = [summary.loc[c, "std"]  for c in components]

    bars = axes[0].bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor="white")
    axes[0].axhline(0, color="#212121", lw=0.8)
    axes[0].set_title("P&L Decomposition (mean ± 1σ)", fontweight="bold")
    axes[0].set_ylabel("USD per episode")
    for bar, val in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"${val:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Waterfall
    total = summary.loc["total_pnl", "mean"]
    wf_labels = ["Spread\nRevenue", "− Adv. Sel.", "+ Inv. MTM", "− Momentum", "Net P&L"]
    wf_vals   = [
        summary.loc["spread_revenue",   "mean"],
        summary.loc["adverse_selection","mean"],
        summary.loc["inventory_mtm",    "mean"],
        summary.loc["momentum_cost",    "mean"],
        total,
    ]
    running = 0
    bottoms = []
    for i, v in enumerate(wf_vals[:-1]):
        bottoms.append(running)
        running += v
    bottoms.append(0)

    wf_colors = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#1565C0"]
    axes[1].bar(wf_labels, wf_vals, bottom=bottoms, color=wf_colors, edgecolor="white")
    axes[1].axhline(0, color="#212121", lw=0.8)
    axes[1].set_title("P&L Waterfall", fontweight="bold")
    axes[1].set_ylabel("USD")
    _save("pnl_decomposition")


# ─────────────────────────────────────────────────────────────
# 3. Regime performance
# ─────────────────────────────────────────────────────────────

def plot_regime_performance(regime_df: pd.DataFrame):
    fig, axes = _subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    metrics = [
        ("mean_spread_bps",   "Mean Spread (bps)",     False),
        ("fill_rate",         "Fill Rate",              False),
        ("pnl_per_trade",     "P&L per Trade (USD)",   False),
        ("inv_abs_mean",      "Mean |Inventory| (BTC)", False),
        ("alpha_hat_mean",    "Est. Informed Fraction", False),
        ("mu_error_bps_mean", "Belief Error (bps)",    False),
    ]

    colors = [STYLE.get(r, "#607D8B") for r in regime_df["regime"]]

    for ax, (col, ylabel, _) in zip(axes, metrics):
        if col not in regime_df.columns:
            ax.set_visible(False)
            continue
        bars = ax.bar(regime_df["regime"], regime_df[col], color=colors, edgecolor="white")
        ax.set_title(ylabel, fontweight="bold")
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, regime_df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Regime-Conditional Performance (BTC/USDT)", fontsize=13, fontweight="bold", y=1.01)
    _save("regime_performance")


# ─────────────────────────────────────────────────────────────
# 4. Strategy comparison
# ─────────────────────────────────────────────────────────────

def plot_strategy_comparison(results: Dict[str, pd.DataFrame], comparison: pd.DataFrame):
    fig, axes = _subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    strategy_colors = {
        "Glosten-Milgrom":    STYLE["gm"],
        "Avellaneda-Stoikov": STYLE["as"],
        "Adaptive-MM":        STYLE["adaptive"],
    }
    colors = [strategy_colors.get(s, "#607D8B") for s in comparison.index]

    metrics = [
        ("mean_pnl",            "Mean P&L (USD)"),
        ("sharpe_annualised",   "Annualised Sharpe"),
        ("mean_regret",         "Mean Regret (USD)"),
        ("mean_spread_bps",     "Mean Spread (bps)"),
        ("adv_sel_cost",        "Adverse Sel. Cost (USD)"),
        ("mean_max_dd",         "Mean Max Drawdown (USD)"),
    ]

    for ax, (col, ylabel) in zip(axes, metrics):
        if col not in comparison.columns:
            ax.set_visible(False)
            continue
        bars = ax.bar(comparison.index, comparison[col], color=colors, edgecolor="white")
        ax.set_title(ylabel, fontweight="bold")
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, comparison[col]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(bar.get_height()) * 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # PnL distribution overlay (last panel)
    ax = axes[-1]
    for strategy, color in strategy_colors.items():
        if strategy in results:
            ax.hist(results[strategy]["final_pnl"], bins=20, alpha=0.5,
                    color=color, label=strategy, edgecolor="white")
    ax.axvline(0, color="#212121", lw=1.0)
    ax.set_xlabel("Final P&L (USD)")
    ax.set_title("P&L Distribution", fontweight="bold")
    ax.legend(fontsize=7)

    plt.suptitle("GM vs AS vs Adaptive-MM — BTC/USDT Market Making", fontsize=13, fontweight="bold", y=1.01)
    _save("strategy_comparison")


# ─────────────────────────────────────────────────────────────
# 5. Adverse selection dynamics
# ─────────────────────────────────────────────────────────────

def plot_adverse_selection(adv_df: pd.DataFrame):
    fig, axes = _subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(adv_df["alpha_hat_mean"], color="#E65100", lw=1.5, label="Mean α̂")
    ax.fill_between(adv_df.index, adv_df["alpha_hat_p25"], adv_df["alpha_hat_p75"],
                    alpha=0.25, color="#E65100")
    ax.set_xlabel("Step")
    ax.set_ylabel("Estimated informed fraction (α̂)")
    ax.set_title("Adverse Selection Detection", fontweight="bold")
    ax.legend()

    ax = axes[1]
    ax.plot(adv_df["spread_bps_mean"], color="#7B1FA2", lw=1.5, label="Mean spread (bps)")
    ax.fill_between(adv_df.index, adv_df["spread_bps_p25"], adv_df["spread_bps_p75"],
                    alpha=0.25, color="#7B1FA2")
    ax.set_xlabel("Step")
    ax.set_ylabel("Spread (bps)")
    ax.set_title("Spread Response to Toxicity", fontweight="bold")
    ax.legend()

    _save("adverse_selection_dynamics")


# ─────────────────────────────────────────────────────────────
# 6. Spread vs fill tradeoff
# ─────────────────────────────────────────────────────────────

def plot_spread_fill_tradeoff(tradeoff_df: pd.DataFrame):
    fig, axes = _subplots(1, 3, figsize=(14, 5))

    ax = axes[0]
    sc = ax.scatter(tradeoff_df["spread_bps_mid"], tradeoff_df["fill_rate"],
                    c=tradeoff_df["informed_frac"], cmap="RdYlGn_r", s=80, zorder=3)
    plt.colorbar(sc, ax=ax, label="Informed fraction")
    ax.set_xlabel("Spread (bps)")
    ax.set_ylabel("Fill rate")
    ax.set_title("Fill Rate vs Spread", fontweight="bold")

    ax = axes[1]
    ax.scatter(tradeoff_df["spread_bps_mid"], tradeoff_df["pnl_per_step"],
               color="#1565C0", s=80, zorder=3)
    ax.axhline(0, color="#212121", lw=0.8)
    ax.set_xlabel("Spread (bps)")
    ax.set_ylabel("P&L per step (USD)")
    ax.set_title("P&L per Step vs Spread", fontweight="bold")

    ax = axes[2]
    ax.bar(range(len(tradeoff_df)), tradeoff_df["informed_frac"],
           color="#F44336", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Spread bin (narrow → wide)")
    ax.set_ylabel("Informed fraction")
    ax.set_title("Adverse Selection by Spread Bin", fontweight="bold")

    _save("spread_fill_tradeoff")


# ─────────────────────────────────────────────────────────────
# 7. Belief convergence
# ─────────────────────────────────────────────────────────────

def plot_belief_convergence(conv_df: pd.DataFrame):
    fig, ax = _subplots(figsize=(10, 5))
    ax.plot(conv_df["mean"], color="#1565C0", lw=1.5, label="Mean |μ error| (bps)")
    ax.fill_between(conv_df.index, conv_df["p25"], conv_df["p75"],
                    alpha=0.25, color="#1565C0", label="IQR")
    ax.plot(conv_df["p50"], color="#1565C0", lw=1.0, ls="--", alpha=0.5, label="Median")
    ax.set_xlabel("Step")
    ax.set_ylabel("Belief error (bps)")
    ax.set_title("Bayesian Belief Convergence — BTC/USDT", fontweight="bold")
    ax.legend()
    _save("belief_convergence")


# ─────────────────────────────────────────────────────────────
# 8. Alpha sweep
# ─────────────────────────────────────────────────────────────

def plot_alpha_sweep(alpha_df: pd.DataFrame):
    fig, axes = _subplots(1, 3, figsize=(14, 5))

    ax = axes[0]
    ax.plot(alpha_df["alpha"], alpha_df["mean_spread_bps"], color="#7B1FA2", lw=2, marker="o")
    ax.set_xlabel("Informed fraction α")
    ax.set_ylabel("Mean spread (bps)")
    ax.set_title("Spread vs α", fontweight="bold")

    ax = axes[1]
    ax.plot(alpha_df["alpha"], alpha_df["mean_pnl"], color="#4CAF50", lw=2, marker="o")
    ax.fill_between(alpha_df["alpha"],
                    alpha_df["mean_pnl"] - alpha_df["std_pnl"],
                    alpha_df["mean_pnl"] + alpha_df["std_pnl"],
                    alpha=0.2, color="#4CAF50")
    ax.axhline(0, color="#212121", lw=0.8)
    ax.set_xlabel("Informed fraction α")
    ax.set_ylabel("Mean P&L (USD)")
    ax.set_title("P&L vs α", fontweight="bold")

    ax = axes[2]
    ax.plot(alpha_df["alpha"], alpha_df["adv_sel_cost"], color="#F44336", lw=2, marker="o", label="Adv. sel. cost")
    ax.plot(alpha_df["alpha"], alpha_df["mean_fill_prob"], color="#2196F3", lw=2, marker="s", label="Fill prob")
    ax.set_xlabel("Informed fraction α")
    ax.set_title("Cost & Fill Rate vs α", fontweight="bold")
    ax.legend()

    plt.suptitle("Sensitivity to Informed Trader Fraction — BTC/USDT", fontsize=12, fontweight="bold", y=1.01)
    _save("alpha_sweep")


# ─────────────────────────────────────────────────────────────
# 9. Real data overview (new: shows raw BTC data before simulation)
# ─────────────────────────────────────────────────────────────

def plot_real_data_overview(features: pd.DataFrame, params: dict):
    fig, axes = _subplots(3, 1, figsize=(14, 10), sharex=True)

    # Price
    ax = axes[0]
    regime_colors = {"quiet": "#E3F2FD", "volatile": "#FFEBEE", "trending": "#FFF3E0"}
    prev_r, start = None, 0
    for i, r in enumerate(features["regime"]):
        if r != prev_r:
            if prev_r:
                ax.axvspan(start, i, alpha=0.3, color=regime_colors.get(prev_r, "#EEE"), lw=0)
            prev_r, start = r, i
    ax.plot(features["mid"], color="#212121", lw=0.8, label="BTC/USDT mid")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"BTC/USDT — {len(features):,} bars | "
                 f"Calibrated: α={params['alpha']:.3f}, σ_step={params['step_vol_pct']*100:.4f}%",
                 fontweight="bold")

    # Realised vol
    ax = axes[1]
    ax.plot(features["realised_vol"], color="#7B1FA2", lw=0.8)
    ax.set_ylabel("Realised vol (ann.)")
    ax.set_title("Annualised Realised Volatility (1m bars)")

    # OFI proxy
    ax = axes[2]
    ax.fill_between(features.index, features["ofi_proxy"], alpha=0.6,
                    color=["#2E7D32" if x > 0 else "#C62828" for x in features["ofi_proxy"]])
    ax.axhline(0, color="#212121", lw=0.8)
    ax.set_ylabel("OFI proxy")
    ax.set_title("Order Flow Imbalance (taker buy/sell ratio)")
    ax.set_xlabel("Bar index")

    # Regime legend
    for regime, color in regime_colors.items():
        axes[0].plot([], [], color=color, alpha=0.7, lw=8, label=regime.capitalize())
    axes[0].legend(fontsize=8, ncol=3)

    _save("real_data_overview")


# ─────────────────────────────────────────────────────────────
# 10. Spread vs realised vol scatter (real data)
# ─────────────────────────────────────────────────────────────

def plot_spread_vs_vol(df: pd.DataFrame):
    fig, ax = _subplots(figsize=(8, 5))
    regime_colors_scatter = {"quiet": "#2196F3", "volatile": "#F44336", "trending": "#FF9800"}
    for regime, color in regime_colors_scatter.items():
        mask = df["regime"] == regime
        if mask.any():
            ax.scatter(df.loc[mask, "sigma"], df.loc[mask, "spread_bps"],
                       alpha=0.3, s=10, color=color, label=regime.capitalize())
    ax.set_xlabel("Belief σ (USD)")
    ax.set_ylabel("Quoted spread (bps)")
    ax.set_title("Spread vs Belief Uncertainty by Regime", fontweight="bold")
    ax.legend()
    _save("spread_vs_vol")
