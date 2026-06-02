"""
run.py  —  v4 (fixed)
Full pipeline: fetch → ML training → RL training → simulate → analyse → plot.

Usage:
    python run.py                    # full v4 pipeline (~50 episodes)
    python run.py --fast             # 20 episodes, rule-based RL
    python run.py --episodes 30      # custom episode count
    python run.py --candles 3000     # fewer kline bars
    python run.py --trades 5000      # number of real trades to fetch
    python run.py --no-cache         # force fresh fetch from Binance
    python run.py --no-rl            # skip PPO training (use rule-based RL)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import (
    load_btcusdt_v4,
    CryptoSimConfig,
    run_episode_real_trades,
    run_many_episodes,
    compare_strategies_v4,
    train_forecaster,
    RLMarketMaker,
    regime_performance_table,
    # Analysis
    pnl_decomposition_many,
    regime_performance,
    adverse_selection_analysis,
    # Statistics
    bootstrap_ci,
    welch_test,
    fill_calibration,
    ofi_predictive_test,
    # Plots
    plot_real_data_overview,
    plot_episode_overview,
    plot_pnl_decomposition,
    plot_regime_performance,
    plot_strategy_comparison,
    plot_adverse_selection,
    plot_spread_fill_tradeoff,
)
from src.analysis import compare_strategies

TABLE_DIR = Path("outputs/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR  = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _banner(step: str, total: int, n: int):
    print(f"\n[{n}/{total}] {step}")
    print("─" * 60)


def _safe(label: str):
    """Decorator-style wrapper: call a function and catch/report errors."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(f"  [warn] {label}: {e}")
                return None
        return wrapper
    return decorator


def main(
    n_episodes:   int  = 50,
    fast:         bool = False,
    cache:        bool = True,
    n_candles:    int  = 5000,
    n_trades:     int  = 10000,
    train_rl:     bool = True,
    n_book_snaps: int  = 0,
):
    TOTAL_STEPS = 8
    t0 = time.time()
    n_ep = 20 if fast else n_episodes

    print("=" * 62)
    print("  Crypto Market Maker  —  v4  (Real Market Data)")
    print("  Strategies: Passive | GM | AS | Adaptive | Forecast | RL")
    print("=" * 62)

    # ── [0] Fetch real data ───────────────────────────────────────────
    _banner("Fetching BTC/USDT data (klines + trades)", TOTAL_STEPS, 0)
    features, trades, params = load_btcusdt_v4(
        n_candles=n_candles, n_trades=n_trades,
        n_book_snaps=n_book_snaps, cache=cache, verbose=True,
    )
    print(f"  ✓ Kline features:   {len(features):,} bars")
    print(f"  ✓ Real trade flow:  {len(trades):,} trades  "
          f"(buy={( trades['side']=='buy').mean():.1%}  sell={(trades['side']=='sell').mean():.1%})")
    print(f"  ✓ Trade price range: ${trades['price'].min():,.0f} – ${trades['price'].max():,.0f}")
    print(f"  ✓ mu0 (belief init): ${params['mu0']:,.2f}  sigma0: ${params['sigma0']:.2f}")

    # Report verified data provenance (not a filename guess). load_btcusdt_v4
    # records whether each cache file was fetched live from Binance.
    _data_source = params.get("data_source", "unknown")
    print(f"\n  DATA SOURCE: {_data_source.upper()}")
    print(f"  NOTE: Statistical results (OFI rho, ML IC, fill rates) depend on the")
    print(f"  data source. Results on real Binance vs synthetic/other data differ")
    print(f"  materially. Always compare results from the same verified source.")

    cfg = CryptoSimConfig.from_calibration(params, n_steps=min(500, len(trades) // 2))

    # ── [1] ML forecaster ────────────────────────────────────────────
    _banner("Training ML return forecaster (LightGBM)", TOTAL_STEPS, 1)
    forecaster = train_forecaster(features, horizon_bars=5, verbose=True)
    if forecaster.is_fitted:
        print(f"  ✓ IC(rank)={forecaster.information_coefficient:.4f}  "
              f"IC(pearson)={forecaster.pearson_ic:.4f}  "
              f"p={forecaster.ic_pvalue:.4f}")
        imp_df = forecaster.feature_importance_df().head(10)
        if len(imp_df) > 0:
            print("\n  Top-10 Feature Importances:")
            for _, row in imp_df.iterrows():
                bar = "█" * int(row["importance_pct"] / 2)
                print(f"    {row['feature']:<30s} {bar:25s} {row['importance_pct']:.1f}%")
    else:
        print("  ⚠ Forecaster not fitted (insufficient data)")
        forecaster = None

    # ── [2] RL market maker ──────────────────────────────────────────
    _banner("RL market maker (PPO / rule-based fallback)", TOTAL_STEPS, 2)
    rl_model = RLMarketMaker(
        gamma=cfg.gamma, inv_limit=cfg.inv_limit,
        min_spread=cfg.min_spread, max_spread=cfg.max_spread,
    )
    if train_rl and not fast:
        rl_model.train(features, params, total_timesteps=cfg.rl_train_steps)
    else:
        rl_model._trained = True
        print(f"  [RL] Rule-based policy active  (pass --no-rl or --fast to keep this)")
    print(f"  ✓ RL model: {rl_model.name}")

    # ── [3] Single replay episode (visual inspection) ─────────────────
    _banner("Single real-trade replay episode", TOTAL_STEPS, 3)
    cfg_adaptive = CryptoSimConfig.from_calibration(
        params, n_steps=min(500, len(trades) // 2), mm_type="adaptive"
    )
    ep_df = run_episode_real_trades(
        cfg_adaptive, features, trades, offset=0,
        forecaster=forecaster, rl_model=None,
    )
    trades_ep = ep_df[ep_df["trade_occurred"]]
    print(f"  Steps: {len(ep_df)}  |  Filled: {len(trades_ep)}  |  "
          f"Fill rate: {len(trades_ep)/len(ep_df):.1%}")
    print(f"  Final PnL:         ${ep_df['mtm_pnl'].iloc[-1]:,.2f}")
    print(f"  Mean spread:       {ep_df['spread_bps'].mean():.2f} bps")
    print(f"  Adverse sel cost:  ${ep_df['adverse_sel_cost'].iloc[-1]:,.2f}")
    print(f"  Spread revenue:    ${ep_df['spread_revenue'].iloc[-1]:,.2f}")
    inv_max = ep_df['inventory'].abs().max()
    print(f"  Max |inventory|:   {inv_max:.4f} BTC")

    try:
        plot_episode_overview(ep_df, title="v4 Real-Trade Replay — AdaptiveMM")
    except Exception as e:
        print(f"  [plot warn] episode_overview: {e}")

    # ── [4] All 6 strategies comparison ──────────────────────────────
    _banner("Comparing all 6 strategies (real trade data)", TOTAL_STEPS, 4)
    all_results = compare_strategies_v4(
        cfg=cfg, features=features, n_episodes=n_ep, verbose=True,
        trades=trades,
        forecaster=forecaster,
        rl_model=rl_model,
    )

    # Summary table
    print("\n  Strategy Performance Summary:")
    hdr = f"  {'Strategy':<25} {'Mean PnL':>10} {'Sharpe':>7} {'Sortino':>7} {'MaxDD':>10} {'Fill%':>7} {'AdvSel':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for strat, s in all_results.items():
        print(f"  {strat:<25} ${s['final_pnl'].mean():>9,.0f} "
              f"{s['sharpe'].mean():>7.2f} "
              f"{s['sortino'].mean():>7.2f} "
              f"${s['max_drawdown'].mean():>9,.0f} "
              f"{s['fill_rate'].mean():>6.1%} "
              f"${s['adverse_sel_cost'].mean():>9,.0f}")

    # Build comparison DataFrame (format plot_strategy_comparison expects)
    steps_per_year  = 365 * 24 * 60
    episodes_per_yr = steps_per_year / cfg.n_steps
    ann_factor      = np.sqrt(episodes_per_yr)
    comp_rows = {}
    for strat, s in all_results.items():
        ep_rets = s["final_pnl"] / (params["mu0"] + 1e-6)
        comp_rows[strat] = {
            "mean_pnl":          s["final_pnl"].mean(),
            "sharpe_annualised": (ep_rets.mean() * episodes_per_yr) / (ep_rets.std() * ann_factor + 1e-9),
            "mean_regret":       s["final_regret"].mean(),
            "mean_spread_bps":   s["mean_spread_bps"].mean(),
            "adv_sel_cost":      s["adverse_sel_cost"].mean(),
            "mean_max_dd":       s["max_drawdown"].mean(),
        }
    comparison_df = pd.DataFrame(comp_rows).T

    try:
        plot_strategy_comparison(all_results, comparison_df)
    except Exception as e:
        print(f"  [plot warn] strategy_comparison: {e}")

    all_summary = pd.concat(all_results.values(), ignore_index=True)
    all_summary.to_csv(TABLE_DIR / "strategy_comparison_v4.csv", index=False)
    comparison_df.to_csv(TABLE_DIR / "strategy_metrics_v4.csv")
    print(f"\n  Saved → {TABLE_DIR}/strategy_comparison_v4.csv")

    # ── [5] Regime-conditional performance ───────────────────────────
    _banner("Regime-conditional performance (Low / Medium / High vol)", TOTAL_STEPS, 5)

    # Regime table: mean PnL per strategy × regime
    regime_table = regime_performance_table(all_results)
    print("\n  Mean PnL by Regime and Strategy:")
    print(regime_table.to_string())
    regime_table.to_csv(TABLE_DIR / "regime_pnl_v4.csv")

    # Detailed by-regime breakdown for each strategy
    print("\n  Detailed regime breakdown:")
    for strat, s in all_results.items():
        if "regime" in s.columns:
            by_r = s.groupby("regime")["final_pnl"].agg(["mean","std","count"])
            print(f"\n  {strat}:")
            print(by_r.to_string())

    # Build the regime_df that plot_regime_performance expects:
    # It expects a DataFrame with one row per regime with aggregated metrics.
    # Use the single AdaptiveMM episode for this (regime_performance needs a list of episode DFs)
    try:
        # Collect several AdaptiveMM episodes to build regime stats
        cfg_rp = CryptoSimConfig.from_calibration(params, n_steps=min(500, len(trades)//2))
        cfg_rp.mm_type = "adaptive"
        ep_list = []
        for i in range(min(n_ep, 10)):
            ep_i = run_episode_real_trades(
                cfg_rp, features, trades,
                offset=i * cfg_rp.n_steps, forecaster=forecaster,
            )
            ep_list.append(ep_i)
        regime_df = regime_performance(ep_list)
        if len(regime_df) > 0:
            print("\n  Regime performance stats (AdaptiveMM):")
            print(regime_df.to_string())
            plot_regime_performance(regime_df)
        regime_df.to_csv(TABLE_DIR / "regime_stats_v4.csv")
    except Exception as e:
        print(f"  [warn] regime_performance plot: {e}")

    # ── [6] Statistical validation ───────────────────────────────────
    _banner("Statistical validation: full risk metrics + bootstrap CIs", TOTAL_STEPS, 6)

    metric_rows = []
    for strat, s in all_results.items():
        metric_rows.append({
            "Strategy":   strat,
            "Mean PnL":   f"${s['final_pnl'].mean():,.0f}",
            "Std PnL":    f"${s['final_pnl'].std():,.0f}",
            "Sharpe":     f"{s['sharpe'].mean():.3f}",
            "Sortino":    f"{s['sortino'].mean():.3f}",
            "CVaR(5%)":   f"${s['cvar_5'].mean():,.0f}",
            "MaxDD":      f"${s['max_drawdown'].mean():,.0f}",
            "FillRate":   f"{s['fill_rate'].mean():.1%}",
            "InvVar":     f"{s['inv_variance'].mean():.4f}",
            "AdvSel":     f"${s['adverse_sel_cost'].mean():,.0f}",
            "SprdCapt":   f"${s['spread_capture'].mean():,.0f}",
        })
    metric_df = pd.DataFrame(metric_rows).set_index("Strategy")
    print("\n  Full Risk Metrics:")
    print(metric_df.to_string())
    metric_df.to_csv(TABLE_DIR / "risk_metrics_v4.csv")

    # Bootstrap CI for the best strategy
    best_strat = max(all_results.items(), key=lambda x: x[1]["final_pnl"].mean())
    print(f"\n  Best strategy: {best_strat[0]}")
    try:
        ci = bootstrap_ci(
            best_strat[1]["final_pnl"].values,
            statistic_fn=np.mean,
            n_bootstrap=2000,
        )
        print(f"  Bootstrap 95% CI for mean PnL: [{ci[0]:,.0f}, {ci[2]:,.0f}]  "
              f"median={ci[1]:,.0f}")
    except Exception as e:
        print(f"  [warn] bootstrap_ci: {e}")

    # Welch t-test: Adaptive vs Passive
    try:
        if "Adaptive-MM" in all_results and "Passive-MM" in all_results:
            test = welch_test(
                all_results["Adaptive-MM"]["final_pnl"].values,
                all_results["Passive-MM"]["final_pnl"].values,
            )
            print(f"  Adaptive vs Passive: t={test.get('t_stat',0):.3f}  "
                  f"p={test.get('p_value',1):.4f}  "
                  f"Cohen's d={test.get('cohens_d',0):.3f}")
    except Exception as e:
        print(f"  [warn] welch_test: {e}")

    # ── [7] OFI predictive test + fill calibration ───────────────────
    _banner("OFI predictive test + fill model calibration", TOTAL_STEPS, 7)

    # OFI predictive test: needs list of episode DataFrames
    try:
        ofi_result = ofi_predictive_test(ep_list, lag_steps=5)
        rho = ofi_result.get("spearman_rho", float("nan"))
        pv  = ofi_result.get("p_value",      float("nan"))
        n   = ofi_result.get("n_obs",        0)
        # Significance/direction/effect come straight from the test result so the
        # printed summary can never contradict the interpretation string.
        sig    = "significant" if ofi_result.get("significant_at_05") else "not significant"
        effect = ofi_result.get("effect_size", "undefined")
        direction = ofi_result.get("direction", "undefined")
        print(f"  OFI predictive test (alpha_hat -> future adverse selection):")
        print(f"    Spearman rho: {rho:.4f}  p={pv:.4f}  ({sig}; {direction}, effect={effect})  n={n:,}")
        print(f"    {ofi_result.get('interpretation', '')}")
    except Exception as e:
        print(f"  [warn] ofi_predictive_test: {e}")

    # Fill calibration: needs list of episode DataFrames
    try:
        fill_df = fill_calibration(ep_list, n_bins=8)
        if isinstance(fill_df, pd.DataFrame) and len(fill_df) > 0:
            print(f"\n  Fill calibration (empirical vs model):")
            print(fill_df.to_string())
            fill_df.to_csv(TABLE_DIR / "fill_calibration_v4.csv")
        elif isinstance(fill_df, dict):
            print(f"  Fill calibration: {fill_df}")
    except Exception as e:
        print(f"  [warn] fill_calibration: {e}")

    # Adverse selection plot
    try:
        from src.analysis import adverse_selection_analysis
        adv_df = adverse_selection_analysis(ep_list)
        plot_adverse_selection(adv_df)
    except Exception as e:
        print(f"  [warn] adverse_selection plot: {e}")

    # ── [8] ML performance report ────────────────────────────────────
    _banner("ML forecaster performance report", TOTAL_STEPS, 8)
    if forecaster is not None and forecaster.is_fitted:
        print(f"  Model:           {forecaster._model_name}")
        print(f"  Horizon:         {forecaster.horizon_bars} bars (≈{forecaster.horizon_bars} min)")
        print(f"  IC (rank):       {forecaster.information_coefficient:.4f}")
        print(f"  IC (pearson):    {forecaster.pearson_ic:.4f}")
        print(f"  p-value:         {forecaster.ic_pvalue:.4f}  "
              f"({'significant' if forecaster.ic_pvalue < 0.05 else 'not significant'})")
        print(f"  Training obs:    {forecaster._n_train:,}")
        print(f"  Validation obs:  {forecaster._n_val:,}")
        print()

        imp_df = forecaster.feature_importance_df()
        if len(imp_df) > 0:
            print("  Feature Importances (top 15):")
            for _, row in imp_df.head(15).iterrows():
                bar = "█" * max(1, int(row["importance_pct"] / 1.5))
                print(f"    {row['feature']:<32s} {bar:<40s} {row['importance_pct']:.2f}%")
            imp_df.to_csv(TABLE_DIR / "feature_importance_v4.csv", index=False)

        # Forecast vs Adaptive PnL diff
        fc_pnl     = all_results.get("Forecast-Adaptive", pd.DataFrame()).get("final_pnl", pd.Series())
        adap_pnl   = all_results.get("Adaptive-MM",       pd.DataFrame()).get("final_pnl", pd.Series())
        if len(fc_pnl) > 0 and len(adap_pnl) > 0:
            diff = fc_pnl.values - adap_pnl.values
            print(f"\n  Forecast-Adaptive vs Adaptive-MM PnL diff:")
            print(f"    Mean diff: ${diff.mean():,.0f}  Std: ${diff.std():,.0f}  "
                  f"Win rate: {(diff > 0).mean():.1%}")
    else:
        print("  Forecaster was not successfully trained.")

    # ── Done ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*62}")
    print(f"  v4 pipeline complete in {elapsed:.1f}s")
    print(f"  Tables → {TABLE_DIR}")
    print(f"  Plots  → {PLOT_DIR}")
    print(f"{'='*62}")

    return all_results, features, trades, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto MM v4 (fixed)")
    parser.add_argument("--fast",        action="store_true", help="20 episodes, skip PPO")
    parser.add_argument("--episodes",    type=int,   default=50)
    parser.add_argument("--candles",     type=int,   default=5000)
    parser.add_argument("--trades",      type=int,   default=10000)
    parser.add_argument("--no-cache",    action="store_true")
    parser.add_argument("--no-rl",       action="store_true")
    parser.add_argument("--book-snaps",  type=int,   default=0)
    args = parser.parse_args()

    main(
        n_episodes   = args.episodes,
        fast         = args.fast,
        cache        = not args.no_cache,
        n_candles    = args.candles,
        n_trades     = args.trades,
        train_rl     = not args.no_rl and not args.fast,
        n_book_snaps = args.book_snaps,
    )
