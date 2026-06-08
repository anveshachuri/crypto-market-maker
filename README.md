# Bayesian Market Maker — BTC/USDT

A simulation framework for six market-making strategies — Glosten-Milgrom
(1985), Avellaneda-Stoikov (2008), a custom adaptive extension, a
forecast-enhanced adaptive variant, a rule-based RL market maker, and a
passive baseline — grounded in real BTC/USDT market data from Binance.

**What this project is:** A quantitative simulation that combines real price
history with a theoretically-motivated order flow model and Bayesian belief
updating. Parameters are calibrated from real data; strategy performance is
evaluated with rigorous statistical methods including bootstrap CIs, Sortino
ratio, CVaR, walk-forward OOS validation, and a systematic ablation study.

**What this project is not:** A live trading system tested on real fills.
Order flow (informed/uninformed/momentum traders) is model-generated, not
observed from a real order book. Fill probabilities are estimated from a
calibrated exponential model, not from historical fill data. See
§ "Simulation Realism and Limitations" for a complete discussion.

---

## Architecture

```
src/
├── data_loader.py    Binance REST fetch, feature engineering, Kyle λ calibration
├── price_process.py  ReplayProcess (real prices) + HybridProcess (calibrated synthetic)
├── belief.py         Gaussian Bayesian belief + OFI toxicity detector
├── traders.py        Informed / uninformed / momentum trader population
├── market_maker.py   GlostenMilgromMM, AvellanedaStoikovMM, AdaptiveMM, PassiveMM,
│                     ForecastAdaptiveMM, RLMarketMaker
├── simulation.py     Episode runner, P&L accounting identity, drawdown tracking
├── analysis.py       Regime analysis, strategy comparison, alpha sweep
├── statistics.py     Bootstrap CIs, Sortino, CVaR, fill calibration, OFI test, OOS
├── ablation.py       Layer-by-layer build-up ablation study

tests/
├── test_belief.py        GaussianBelief vs GridBelief reference; 8 invariant tests
├── test_market_maker.py  GM formula, AS reservation price, OFI asymmetry, fill model
├── test_simulation.py    P&L accounting identity, fill_decay, drawdown
├── test_analysis.py      Sharpe annualisation, decomposition identity
├── test_statistics.py    Bootstrap CI, risk metrics, fill calibration, OFI test
└── test_ablation.py      PassiveMM, ablation flags, build-up study
```

---

## Quick Start

```bash
pip install -r requirements.txt

python run.py                    # full pipeline (~50 episodes per step)
python run.py --fast             # 20 episodes, skip ablation + sensitivity
python run.py --candles 2000     # fetch only 2,000 bars (faster)
python run.py --no-cache         # force fresh Binance fetch

pytest tests/ -v                 # 60 tests, ~10 seconds
```

Data is cached to `data_cache/` (not committed). First run fetches from
Binance public API — no authentication required.

---

## Theoretical Foundation

### 1. Glosten-Milgrom (1985)

The market maker sets a break-even spread to compensate for adverse selection:

```
s* = 2αστ / (1 − α)
```

- `α`: fraction of informed traders — calibrated via Kyle's lambda OLS on real OFI
- `σ`: posterior uncertainty about the true mid price V
- `τ`: trade size normalisation (= 1 in our unit-size formulation)

Reference: Glosten, L. and Milgrom, P. (1985). "Bid, Ask and Transaction Prices
in a Specialist Market with Heterogeneously Informed Traders." *JFE* 14(1):71–100.

### 2. Avellaneda-Stoikov (2008)

Inventory-penalised optimal quoting via stochastic control (risk-neutral MM):

```
r = μ − q·γ·σ²·T          (reservation price — shifts mid by inventory cost)
s = γ·σ²·T + (2/γ)·ln(1 + γ/κ)   (optimal spread)
```

- `q`: signed inventory in BTC
- `γ`: absolute risk aversion coefficient — calibrated so the AS base spread ≈ 2 bps
- `T`: time remaining in episode (normalised to [0,1])
- `κ`: order arrival intensity

Calibration of `γ`: we set `γ = target_spread_USD / σ²` where `target_spread_USD
= 0.0002 × mu0` (2 bps of mid). This is a first-order approximation to the AS
optimal; the exact solution requires the full arrival intensity κ which we do not
observe directly from 1-minute bars.

Reference: Avellaneda, M. and Stoikov, S. (2008). "High-frequency trading in a
limit order book." *Quantitative Finance* 8(3):217–224.

### 3. AdaptiveMM: Four Adaptive Layers

The adaptive MM extends the AS framework with four optional layers, each with its
own economic justification. Layers can be individually disabled for ablation.

#### Layer 1 — Exponential Inventory Penalty

```
reservation_shift = λ · (exp(|q| / q̄) − 1) · sign(q)
```

**Economic justification**: The AS framework penalises inventory via `q·γ·σ²·T`,
which is linear in q. Real MMs face convex costs from position limits and risk
aversion at scale. The exponential form is strongly convex and produces a soft
position limit: penalty = 1 bps at q = q̄, dominates at q = 2q̄.

**Calibration**: `penalty_lambda=0.05`, `inv_scale=0.15`. Set so that at
|q| = 0.5 BTC the exponential cost is comparable to the AS spread.

#### Layer 2 — Adverse Selection Adjustment (Toxicity Widening)

```
Δs_tox = min(k · max(0, α̂ − α_base) · σ, σ)
```

**Economic justification**: In the GM model, `∂(s*)/∂α > 0`. When the MM's
toxicity detector estimates rising informed flow (α̂ > α_base), it adjusts the
spread toward the new break-even. The first-order approximation `k·Δα·σ` is a
linearisation of the GM spread formula around α_base.

**α̂ estimator (OFI proxy)**: `α̂ = EMA(|OFI|)` where `OFI = (buy_vol − sell_vol)
/ total_vol ∈ [−1, 1]`. High `|OFI|` signals directional flow, conceptually
related to VPIN (Easley et al. 2012). We use OFI rather than posterior move
size to avoid circular feedback (the MM's own spread affects posterior moves).

**Limitation**: OFI is a noisy proxy for the true informed fraction. The
Spearman correlation between α̂ and future adverse selection cost is measured in
`statistics.ofi_predictive_test()` — see outputs/tables/ofi_predictive_test.csv.
Results represent internal simulation validity, not real-market predictive power.

Reference: Easley, D., Lopez de Prado, M. and O'Hara, M. (2012). "Flow Toxicity
and Liquidity in a High-Frequency World." *RFS* 25(5):1457–1493.

#### Layer 3 — Volatility Regime Spread

```
Δs_vol = γ_vol · σ_local
```

**Economic justification**: In the AS framework, `s* ∝ σ²`. When local vol
rises above the calibrated σ₀, the spread calibrated on historical average vol
becomes too tight. `σ_local` is estimated from a rolling window of recent price
moves, scaled to daily vol using `sqrt(1440)` (BTC trades 24/7; equity markets
use `sqrt(390)`, understating BTC daily vol by ~1.9×).

**Calibration**: `vol_gamma=0.5` — the vol adjustment adds half of the current
daily vol estimate to the base spread.

#### Layer 4 — Asymmetric OFI Quote Skew

```
Under buy OFI (OFI > 0):  ask += 2ρ·|OFI|·σ;  bid unchanged
Under sell OFI (OFI < 0): bid −= 2ρ·|OFI|·σ;  ask unchanged
```

**Economic justification**: Informed directional flow shifts the posterior mid
estimate upward (for buy OFI). The MM raises the ask to deter the informed buyer
at the current ask level. The bid is left unchanged: raising it would attract
sell-side adverse selection from traders dumping into the momentum — the opposite
of the desired response. This asymmetry is verified in `test_market_maker.py::
test_adaptive_ofi_asymmetric`.

---

## Calibration Methodology

All parameters are derived from real Binance 1-minute BTC/USDT data:

| Parameter | Equation | Source |
|-----------|----------|--------|
| `mu0` | Last mid price | Binance klines |
| `sigma0` | `std(close, window=60)` | 1-hour rolling std |
| `sigma_v` | `std(log_ret) × mu0` | Per-step price vol |
| `alpha` | `0.10 + R²_kyle × 1.5` clamped to [0.10, 0.30] | Kyle's lambda OLS R² |
| `process_noise` | `(0.01 × sigma0)²` | 1% of sigma0² per step |
| `noise_var` | `sigma0²` | Prior uncertainty |
| `gamma` | `0.0002 × mu0 / sigma0²` clamped to [0.001, 0.05] | Targeting 2 bps AS spread |
| `min_spread` | `max($1, 0.5 bps × mu0)` | Regulatory minimum |
| `max_spread` | `min($500, 5 × sigma0)` | Risk limit |

**Kyle's lambda procedure**: Regress `Δclose ~ λ·OFI + ε` (OLS). R² measures the
fraction of price movement explained by signed order flow — a proxy for information
content. In liquid crypto markets R² ∈ [0.01, 0.15] → α ∈ [0.10, 0.25].

**Sensitivity**: `sensitivity_analysis()` in `statistics.py` varies each parameter
±50% and records PnL impact. Parameters with large sensitivity (>20% PnL change
per 50% parameter change) require stronger calibration evidence. Results in
`outputs/tables/sensitivity_analysis.csv`.

---

## Simulation Realism and Limitations

### What is derived from real exchange data

- BTC/USDT 1-minute mid prices (Binance `/api/v3/klines`)
- Taker volume split → OFI proxy (direct from Binance aggressor field)
- Realised volatility per bar (rolling window of log returns)
- Regime labels (quiet/volatile/trending from expanding-window vol percentile)
- All calibrated parameters (mu0, sigma0, alpha, gamma, etc.)

### What is model-generated (not from exchange data)

- **Order flow identity**: which arriving trade is "informed" vs "uninformed" vs
  "momentum" is determined by a simulation draw, not by real trader classification.
  In reality, informed/uninformed labels are unobservable.
- **Trade arrival timing**: trades arrive at every step with a probability draw.
  Real order flow has autocorrelation and clustering not captured here.
- **Fill probabilities**: derived from `exp(−k·spread/σ)`. Actual fill rates
  depend on queue position, latency, and order book depth — none of which are
  modelled. The exponential model is validated against empirical fill rates in
  the simulation (see fill_calibration in statistics.py), but this is internal
  validation, not real order book data.
- **Trade size distribution**: sizes drawn from exponential + base; real sizes
  have fat tails and are correlated with volatility.

### Implication for performance claims

Performance metrics (PnL, Sharpe, adverse selection reduction) measure how well
each strategy performs *within the simulation model*. Whether these improvements
translate to real markets depends on how accurately the simulation captures real
adverse selection dynamics. This is a standard limitation of academic MM simulation
and is stated explicitly in the project.

---

## Ablation Study

`ablation.py` runs a systematic build-up experiment with 8 variants:

| Variant | Components | Incremental vs previous |
|---------|-----------|------------------------|
| Passive-Fixed | constant 5 bps spread | — (floor benchmark) |
| AS-Baseline | AS only, no adaptive layers | vs Passive |
| AS+Inventory | + exponential inventory penalty | vs AS |
| AS+Inv+Vol | + vol regime widening | vs +Inventory |
| AS+Inv+Vol+Toxicity | + OFI toxicity adjustment | vs +Vol |
| Full-Adaptive | + asymmetric OFI skew | vs +Toxicity |
| OFI-Only | AS + OFI skew only | single-layer benchmark |
| Toxicity-Only | AS + toxicity only | single-layer benchmark |

Pairwise Welch t-tests with Cohen's d identify which incremental additions are
statistically significant. Results in `outputs/tables/ablation_summary.csv` and
`outputs/tables/ablation_significance.csv`.

---

## Statistical Validation

`statistics.py` provides rigorous quantitative support for every performance claim:

### Bootstrap confidence intervals
Non-parametric 95% CIs on mean PnL and annualised Sharpe (2,000 resamples,
Efron percentile method). Appropriate for non-Normal PnL distributions.

### Risk-adjusted metrics

| Metric | Definition |
|--------|-----------|
| Sharpe (ann.) | `(E[r] × N_ep/yr) / (std[r] × √N_ep/yr)` |
| Sortino (ann.) | Sharpe using downside vol only |
| CVaR 95% | Mean PnL of worst 5% episodes |
| Inventory VaR 95% | 95th-percentile max inventory |
| Max drawdown | Peak-to-trough MTM within episode |

`N_ep/yr = 525,600 / n_steps` (BTC trades 24/7).

### OFI predictive test
Spearman rank correlation between `alpha_hat(t)` and future adverse selection
cost at lag=5 steps. Reports ρ and p-value. Internal simulation test only.

### Walk-forward OOS validation
70/30 split: parameters calibrated on first 70% of data; performance evaluated
on the remaining 30% using IS-calibrated parameters. Sharpe degradation measures
overfitting risk.

### Fill probability calibration
Compares model-predicted fill probabilities vs empirically observed fill rates
by spread decile. Fits an empirical fill_decay parameter for comparison with
the model assumption.

---

## Empirical Findings & Interpretation

Results are based on 20 episodes of 500 steps each on real Binance BTC/USDT data.

| Strategy | Mean PnL | Sharpe | Sortino | Fill% |
|----------|----------|--------|---------|-------|
| Passive-MM | $1,352 | 2.09 | 3.99 | 35.6% |
| Glosten-Milgrom | $1,464 | 1.96 | 4.06 | 44.5% |
| Avellaneda-Stoikov | $1,487 | 1.95 | 4.08 | 44.1% |
| Adaptive-MM | $1,380 | 2.93 | 5.89 | 13.7% |
| Forecast-Adaptive | $1,366 | 2.90 | 6.01 | 14.3% |
| RL-MarketMaker | $1,488 | 1.97 | 4.10 | 43.2% |

The headline result on this BTC/USDT sample is deliberately understated, and that
is the point. Three findings define it:

| Finding | Observed result | What it means |
|---------|-----------------|---------------|
| Adaptive dominates risk-adjusted | Adaptive-MM achieves best Sharpe (2.93) vs Passive baseline (2.09); Forecast-Adaptive achieves best Sortino (6.01) | The adaptive layers reduce variance and adverse selection exposure more than they increase raw PnL |
| Forecast signal adds little | Forecast-Adaptive trails Adaptive-MM by **-$14** mean PnL (**20% win rate** across 20 episodes) | The LightGBM return forecast (rank-IC = 0.039) is a real but marginal signal; it nudges quotes rather than driving them |
| OFI predicts, modestly | Spearman **ρ = 0.074**, **p < 0.001**, n = 4,950 | Order-flow toxicity is a statistically significant but economically small predictor of adverse selection at lag-5 |

**Why the weak forecast result is a feature, not a bug.** High-frequency return
predictability in liquid crypto is small by nature: the microstructure literature
(e.g. Bouchaud et al. 2018) puts achievable information coefficients in the
~0.02–0.08 range at minute horizons, and any genuine edge is competed away by
faster participants. Our rank-IC of 0.039 sits within this range. A simulation
that produced large, clean alpha from a 1-minute LightGBM forecast would be
evidence of a data leak or unrealistic fill model — not of a profitable strategy.

**What the project does and does not claim.** It does *not* claim a deployable
edge. It claims a correct, defensible research pipeline: real Binance data with
verified provenance, identical train/inference feature definitions (regression-
tested), finite and economically bounded PnL, and statistical tests whose textual
interpretation is derived directly from the p-value sign and effect size so that
significance is never conflated with economic importance. The honest conclusion —
*these signals exist but are economically small, and the Adaptive strategy's
risk-adjusted superiority comes from variance reduction rather than alpha
generation* — is the kind of result a careful quant researcher should expect
and report, and it is reported here without inflation.

**Caveats that bound the conclusion.** Results are sample- and period-specific;
a single BTC/USDT window is not representative of all regimes. The adverse-selection
labels are simulation-generated (see *Simulation Realism and Limitations*), so the
OFI test measures internal consistency rather than real-world price-impact
prediction. Stronger or weaker signal may appear at other horizons, assets, or
data windows; the framework is built to measure that honestly rather than to
guarantee a particular outcome.

---

## P&L Decomposition

The accounting identity holds exactly in simulation:

```
total_pnl = cash + inventory × V    (verified in test_simulation.py)

total_pnl ≈ spread_revenue
           − adverse_selection_cost
           + inventory_mtm_pnl        ← signed: pos = price moved in MM's favour
           − momentum_cost
           + residual                 ← < 20% of spread_revenue in tests
```

`inventory_mtm_pnl = Σ_t [ inventory_{t-1} × (V_t − V_{t-1}) ]`

The residual absorbs the half-spread approximation in the decomposition and is
small by construction (verified in `test_analysis.py::test_pnl_decomposition_identity`).

---

## Outputs

**Plots** (`outputs/plots/`, generated — not committed):

1. `real_data_overview.png` — price, vol, OFI with regime shading
2. `episode_overview.png` — quotes, inventory, P&L, belief single episode
3. `pnl_decomposition.png` — waterfall: spread revenue → adv. sel. → inv. MTM → net
4. `regime_performance.png` — quiet / volatile / trending comparison
5. `strategy_comparison.png` — GM vs AS vs Adaptive: PnL, Sharpe, drawdown, spread
6. `adverse_selection_dynamics.png` — α̂ evolution and spread response
7. `spread_fill_tradeoff.png` — fill rate and P&L vs spread width
8. `belief_convergence.png` — |μ error| in bps over episode steps
9. `alpha_sweep.png` — sensitivity to informed fraction α
10. `spread_vs_vol.png` — quoted spread vs belief uncertainty by regime

**Tables** (`outputs/tables/`, generated — not committed):

| File | Contents |
|------|---------|
| `episode_summary.csv` | Per-episode metrics for Adaptive-MM |
| `pnl_decomposition.csv` | Mean/std of P&L components |
| `regime_performance.csv` | Metrics by market regime |
| `strategy_comparison.csv` | All six strategies summary |
| `strategy_t_tests.csv` | Pairwise Welch t-tests |
| `risk_metrics.csv` | Sharpe, Sortino, CVaR, VaR, drawdown with CIs |
| `ofi_predictive_test.csv` | OFI→adv. sel. Spearman test results |
| `bootstrap_tests.csv` | Cross-strategy bootstrap comparisons |
| `fill_calibration.csv` | Model vs empirical fill rates by spread bin |
| `ablation_summary.csv` | Per-variant PnL with bootstrap CIs |
| `ablation_significance.csv` | Layer-by-layer Welch t-tests + Cohen's d |
| `sensitivity_analysis.csv` | One-at-a-time parameter sensitivity |
| `oos_validation.csv` | IS vs OOS Sharpe and risk metrics |
| `alpha_sweep.csv` | Performance across informed fraction values |

---

## Data Source

Binance public REST API — no authentication required:

- `GET /api/v3/klines` — 1-minute OHLCV (paginated, 1,000 rows/call)
- `GET /api/v3/aggTrades` — real historical trade flow (paginated)
- OFI proxy: `(taker_buy_vol − taker_sell_vol) / total_vol ∈ [−1, +1]`

**Verified provenance.** Cache filenames always contain `BTCUSDT`, so a filename
check cannot prove data is real. Instead, a `.source` sidecar marker is written
the moment data is genuinely fetched from Binance (including `--no-cache` runs),
and read back at load time. `params["data_source"]` reports `real_binance` only
when *both* the kline and trade sources are verified live; otherwise it reports
`unknown (...)`. The run header prints `DATA SOURCE: REAL_BINANCE` accordingly.
If a run shows `UNKNOWN`, the data was loaded from a cache predating this marker —
re-run once (e.g. `python run.py --no-cache`) to refresh and attribute it.

---

## Known Limitations

1. **Synthetic order flow**: informed/uninformed labels are model draws. Real
   informed flow identification requires Lee-Ready or similar algorithms.
2. **Fill model**: exponential decay is a tractable approximation; real fills
   depend on queue priority, latency, and partial fills. The empirical fill_decay
   (0.33) is 4× lower than the model assumption (1.36), suggesting fills occur
   more frequently at wide spreads than the model predicts.
3. **Single venue**: Binance spot only. Real crypto MMs face cross-venue
   arbitrage and inventory correlation across assets.
4. **No transaction costs**: exchange fees (~0.02–0.05% taker) are excluded.
   Including them would reduce all strategy PnLs proportionally.
5. **Regime detection**: expanding-window percentile avoids look-ahead bias
   but the trend threshold (0.001) is fixed, not rolling.
6. **Tick structure**: model quotes in continuous USD, not discrete ticks.
   Real crypto order books have a minimum tick of $0.01 for BTC/USDT.

---

## References

- Glosten, L. and Milgrom, P. (1985). "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders." *Journal of Financial Economics* 14(1):71–100.
- Avellaneda, M. and Stoikov, S. (2008). "High-frequency trading in a limit order book." *Quantitative Finance* 8(3):217–224.
- Kyle, A. (1985). "Continuous Auctions and Insider Trading." *Econometrica* 53(6):1315–1335.
- Easley, D., Lopez de Prado, M. and O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World." *Review of Financial Studies* 25(5):1457–1493.
- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
