"""
market_maker.py
Quoting engine — extended with inventory penalties, adverse selection
adjustments, and volatility-regime spread widening.

Strategies:
  1. GlostenMilgromMM       : pure GM (break-even spread, no inventory mgmt)
  2. AvellanedaStoikovMM    : AS with inventory skew (original)
  3. AdaptiveMM             : AS + four adaptive layers (see class docstring)
  4. PassiveMM              : fixed spread — ablation floor benchmark
  5. OracleMM               : perfect-information oracle for regret analysis

Literature:
  Glosten & Milgrom (1985): "Bid, Ask and Transaction Prices in a Specialist
    Market with Heterogeneously Informed Traders." JFE 14(1):71–100.
  Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book."
    Quantitative Finance 8(3):217–224.
  Kyle (1985): "Continuous Auctions and Insider Trading." Econometrica 53(6):1315.
  Easley et al. (2012): "Flow Toxicity and Liquidity in a High-Frequency World."
    RFS 25(5):1457–1493.  [VPIN methodology — basis for OFI toxicity proxy]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GlostenMilgromMM:
    """
    Pure Glosten-Milgrom. Zero inventory management.
    s* = 2α·σ / (1-α)
    """
    alpha:      float = 0.3
    min_spread: float = 0.02
    max_spread: float = 5.0

    def quote(
        self,
        mu: float,
        sigma: float,
        inventory: float = 0.0,
        t_remaining: float = 1.0,
        alpha_hat: float = None,
        ofi: float = 0.0,
    ) -> Tuple[float, float]:
        a = alpha_hat if alpha_hat is not None else self.alpha
        half_spread = a * sigma / (1.0 - a + 1e-9)
        half_spread = np.clip(half_spread, self.min_spread / 2, self.max_spread / 2)
        return round(mu - half_spread, 4), round(mu + half_spread, 4)

    @property
    def name(self) -> str:
        return "Glosten-Milgrom"


@dataclass
class AvellanedaStoikovMM:
    """
    Avellaneda-Stoikov: inventory skew + uncertainty-linked spread.
    Reservation price: r = μ - q·γ·σ²·T
    Optimal spread:    s = γ·σ²·T + (2/γ)·ln(1 + γ/κ)
    """
    gamma:      float = 0.1
    kappa:      float = 1.5
    alpha:      float = 0.3
    min_spread: float = 0.02
    max_spread: float = 8.0
    inv_limit:  float = 10.0

    def reservation_price(self, mu, sigma, inventory, t_remaining):
        return mu - inventory * self.gamma * (sigma ** 2) * t_remaining

    def optimal_spread(self, sigma, t_remaining):
        s = self.gamma * (sigma ** 2) * t_remaining + (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        return np.clip(s, self.min_spread * 2, self.max_spread * 2)

    def quote(
        self,
        mu: float,
        sigma: float,
        inventory: float = 0.0,
        t_remaining: float = 1.0,
        alpha_hat: float = None,
        ofi: float = 0.0,
    ) -> Tuple[float, float]:
        r  = self.reservation_price(mu, sigma, inventory, t_remaining)
        s  = self.optimal_spread(sigma, t_remaining)
        hs = s / 2.0
        inv_skew = np.sign(inventory) * self.min_spread * 0.5 if abs(inventory) > self.inv_limit else 0.0
        bid = r - hs - inv_skew
        ask = r + hs - inv_skew
        return round(bid, 4), round(ask, 4)

    @property
    def name(self) -> str:
        return "Avellaneda-Stoikov"


@dataclass
class AdaptiveMM:
    """
    Adaptive market maker: AS framework extended with four adaptive layers.

    This class supports individual layer toggling via `use_*` flags, which
    enables the systematic ablation study in ablation.py to isolate each
    component's incremental contribution.

    Layer 1 — Inventory penalty (exponential)
        Reservation price is shifted by: λ·(exp(|q|/q̄) − 1)·sign(q)
        Economic justification: under AS/Stoikov framework, an MM with
        inventory q faces a "gamma-hedging" cost proportional to q²σ²T.
        The exponential form penalises large positions super-linearly,
        reflecting position limits and risk aversion at scale.
        Calibration: penalty_lambda and inv_scale set so penalty = 1 bps
        at q = inv_scale, and becomes dominant (>50 bps) at q = 2×inv_scale.

    Layer 2 — Adverse selection adjustment (toxicity widening)
        When alpha_hat > alpha_base, spread widens by: k·(α̂−α_base)·σ
        Economic justification: in Glosten-Milgrom, the break-even spread
        is s* = 2αστ/(1−α), so d(s*)/d(α) > 0. When the MM detects rising
        informed flow (via OFI imbalance, conceptually similar to VPIN;
        Easley et al. 2012), it adjusts toward the new break-even.
        Limitation: OFI is a noisy proxy for informed fraction. The correction
        is first-order and does not account for the non-linearity in s*(α).

    Layer 3 — Volatility regime spread
        Spread widens by: γ_vol · σ_local
        Economic justification: in the AS framework, s* ∝ σ². When local vol
        spikes above calibrated sigma, the calibrated spread becomes too tight;
        this layer tracks the vol regime and applies a proportional correction.
        σ_local is scaled to daily vol using sqrt(1440): BTC trades 24/7.

    Layer 4 — Order-flow imbalance skew (ASYMMETRIC)
        Under buy OFI: raise ask by 2·ρ·|OFI|·σ; bid unchanged.
        Under sell OFI: lower bid by 2·ρ·|OFI|·σ; ask unchanged.
        Economic justification: informed directional flow raises the posterior
        mid estimate. Raising only the adverse-side quote deters continuation
        of informed trading on that side without inviting the opposite side's
        adverse selection. Symmetric shifting would raise the bid under buy
        pressure — soliciting sell-side adverse selection from momentum traders
        dumping into the move, which increases not decreases inventory risk.

    Fill probability:
        P(fill) = exp(−fill_decay · half_spread / σ)
        At half_spread = 0: P(fill) = 1.0 (guaranteed fill, zero spread)
        At half_spread = σ: P(fill) ≈ exp(−1.5) ≈ 0.22 (default fill_decay)
        Calibration: fill_calibration() in statistics.py validates this
        exponential model against empirically observed fill rates. The decay
        parameter fill_decay is the key degree of freedom; default 1.5 is
        within the empirically plausible range for limit-order queues.
    """
    # AS core params
    gamma:      float = 0.1
    kappa:      float = 1.5
    alpha:      float = 0.3
    min_spread: float = 0.02
    max_spread: float = 12.0
    inv_limit:  float = 2.0

    # Layer 1: inventory penalty
    penalty_lambda: float = 0.01   # base penalty multiplier
    inv_scale:      float = 0.25   # exponential kicks in above 0.25 BTC

    # Layer 2: adverse selection adjustment
    as_multiplier:  float = 2.0    # how aggressively to widen on toxicity
    alpha_base:     float = 0.3    # baseline informed fraction

    # Layer 3: vol regime
    vol_gamma:      float = 0.5    # vol spread multiplier
    vol_window:     int   = 20     # lookback for local vol estimate
    _price_history: list  = field(default_factory=list)

    # Layer 4: OFI skew
    ofi_sensitivity: float = 0.3

    # Fill model
    fill_decay: float = 1.5

    # Ablation flags — set to False to disable a layer for controlled experiments.
    # Used by ablation.py to isolate incremental contributions.
    # Default True = full model. See ablation.py for the build-up study.
    use_inventory_penalty: bool = True   # Layer 1
    use_toxicity_adj:      bool = True   # Layer 2
    use_vol_regime:        bool = True   # Layer 3
    use_ofi_skew:          bool = True   # Layer 4

    def observe_price(self, price: float):
        """Record mid price for local vol estimation."""
        self._price_history.append(price)
        if len(self._price_history) > self.vol_window + 1:
            self._price_history.pop(0)

    @property
    def local_vol(self) -> float:
        """Annualised local vol from recent mid-price moves."""
        if len(self._price_history) < 3:
            return 0.0
        diffs = np.diff(self._price_history)
        # BTC trades 24/7: 1 day = 24 × 60 = 1,440 minutes.
        # Using sqrt(390) (US equity session) understates BTC daily vol by ~1.9×.
        return float(np.std(diffs) * np.sqrt(1_440))  # scale to BTC daily

    def _inventory_penalty(self, inventory: float) -> float:
        """
        Quadratic inventory cost — penalises large positions super-linearly.

        Replaces the previous exponential form exp(|q|/q̄) which overflows
        to IEEE infinity when |q| > ~3×inv_scale (e.g. 3×0.15 = 0.45 BTC
        on real Binance data where whale trades can be 1-50 BTC).

        Quadratic form: λ·(q/q̄)²
          - Same economic motivation: cost accelerates with position size
          - No overflow for any finite inventory
          - Calibration unchanged: at q = inv_scale, penalty = λ (same units)
          - At q = 2×inv_scale, penalty = 4λ (quadratic, not e²λ ≈ 7.4λ)

        The reservation price shift is:
            r_adj = r - sign(q) · penalty
        This skews quotes away from inventory-increasing trades without
        the discontinuous explosion the exponential form produces.
        """
        q_norm = abs(inventory) / (self.inv_scale + 1e-9)
        return float(self.penalty_lambda * q_norm ** 2)

    def fill_probability(self, half_spread: float, sigma: float) -> float:
        """
        Realistic fill probability: decays exponentially with spread/sigma ratio.
        At half_spread = 0: P(fill) = 1.
        At half_spread = sigma: P(fill) ≈ exp(-fill_decay) ≈ 0.22.
        """
        return float(np.exp(-self.fill_decay * half_spread / (sigma + 1e-6)))

    def quote(
        self,
        mu: float,
        sigma: float,
        inventory: float = 0.0,
        t_remaining: float = 1.0,
        alpha_hat: float  = None,
        ofi: float        = 0.0,
    ) -> Tuple[float, float]:
        """
        Returns (bid, ask) incorporating all four adaptive layers.
        Individual layers are gated by use_* flags for ablation experiments.
        """
        if alpha_hat is None:
            alpha_hat = self.alpha

        # ── AS core ──────────────────────────────────────────────────
        r  = mu - inventory * self.gamma * (sigma ** 2) * t_remaining
        s0 = self.gamma * (sigma ** 2) * t_remaining + (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)

        # ── Layer 2: toxicity widening (ABLATION: use_toxicity_adj) ──
        # Capped at 1.0×sigma to prevent runaway widening on alpha_hat spikes.
        # Derived from ∂(s*)/∂α > 0 in Glosten-Milgrom; see class docstring.
        if self.use_toxicity_adj:
            as_adj = min(
                self.as_multiplier * max(0.0, alpha_hat - self.alpha_base) * sigma,
                1.0 * sigma,
            )
        else:
            as_adj = 0.0

        # ── Layer 3: vol regime widening (ABLATION: use_vol_regime) ──
        vol_adj = (self.vol_gamma * self.local_vol) if self.use_vol_regime else 0.0

        total_spread = np.clip(s0 + as_adj + vol_adj, self.min_spread * 2, self.max_spread * 2)
        hs = total_spread / 2.0

        # ── Layer 1: inventory penalty (ABLATION: use_inventory_penalty) ──
        if self.use_inventory_penalty:
            inv_pen  = self._inventory_penalty(inventory)
            inv_skew = np.sign(inventory) * inv_pen if inventory != 0 else 0.0
        else:
            inv_skew = 0.0

        # ── Layer 4: OFI skew, ASYMMETRIC (ABLATION: use_ofi_skew) ──
        if self.use_ofi_skew:
            ofi_adj = self.ofi_sensitivity * abs(ofi) * sigma
            if ofi >= 0:       # buy pressure → raise ask only
                bid = r - hs - inv_skew
                ask = r + hs - inv_skew + 2.0 * ofi_adj
            else:              # sell pressure → lower bid only
                bid = r - hs - inv_skew - 2.0 * ofi_adj
                ask = r + hs - inv_skew
        else:
            bid = r - hs - inv_skew
            ask = r + hs - inv_skew

        # Final clip: OFI applied post-clip, so enforce bound again.
        if (ask - bid) > self.max_spread * 2:
            mid_q = (bid + ask) / 2.0
            bid   = mid_q - self.max_spread
            ask   = mid_q + self.max_spread

        # Hard safety clip: quotes must be finite and within max_spread of mu.
        # This catches any residual overflow from large inventory × gamma terms
        # or vol_adj spikes, without silently corrupting MTM calculations.
        if not (np.isfinite(bid) and np.isfinite(ask)):
            bid = mu - self.max_spread / 2.0
            ask = mu + self.max_spread / 2.0
        bid = float(np.clip(bid, mu - self.max_spread, mu + self.max_spread))
        ask = float(np.clip(ask, mu - self.max_spread, mu + self.max_spread))
        if ask - bid < self.min_spread:
            mid_q = (bid + ask) / 2.0
            bid = mid_q - self.min_spread / 2.0
            ask = mid_q + self.min_spread / 2.0

        return round(bid, 4), round(ask, 4)

    @property
    def name(self) -> str:
        return "Adaptive-MM"


@dataclass
class PassiveMM:
    """
    Fixed-spread passive market maker — floor benchmark for ablation study.

    Quotes at a constant half-spread regardless of market conditions.
    No inventory management, no toxicity detection, no vol adjustment.

    Used in ablation.py to establish the floor: any strategy that fails to
    beat the passive benchmark has no adaptive value whatsoever.

    half_spread_bps: quotes half-spread in basis points of current mid.
    This is set to match the typical mean spread observed from AS baseline,
    so comparisons isolate adaptation rather than spread level.

    Limitation: a real passive MM would face queue position dynamics
    (first-in-first-out priority) that this model ignores. Fill rates
    for passive quotes are typically lower than the exponential model
    predicts at tight spreads where queue depth matters most.
    """
    half_spread_bps: float = 5.0   # calibrated to match AS mean spread
    min_spread:      float = 1.0
    max_spread:      float = 500.0

    def quote(
        self,
        mu: float,
        sigma: float = 0.0,
        inventory: float = 0.0,
        t_remaining: float = 1.0,
        alpha_hat: float = None,
        ofi: float = 0.0,
    ) -> Tuple[float, float]:
        hs = np.clip(
            mu * self.half_spread_bps / 10_000,
            self.min_spread / 2,
            self.max_spread / 2,
        )
        return round(mu - hs, 4), round(mu + hs, 4)

    @property
    def name(self) -> str:
        return "Passive-Fixed"


@dataclass
class OracleMM:
    """Oracle MM: knows V exactly. Used for regret analysis."""
    tick: float = 0.01

    def quote(self, V, sigma=0.0, inventory=0.0, t_remaining=1.0,
              alpha_hat=None, ofi=0.0):
        return round(V - self.tick, 4), round(V + self.tick, 4)

    @property
    def name(self) -> str:
        return "Oracle"


# ─────────────────────────────────────────────────────────────
# v4 Additions: ForecastAdaptiveMM
# ─────────────────────────────────────────────────────────────

@dataclass
class ForecastAdaptiveMM(AdaptiveMM):
    """
    Forecast-Enhanced Adaptive Market Maker  —  v4 Priority 3.

    Extends AdaptiveMM with a predictive ML signal that modifies quotes
    based on short-horizon return forecasts.

    Forecast integration:
      Bullish signal (pred > 0):
        - Bid becomes more aggressive (higher bid)
        - Ask becomes less aggressive (higher ask)
        → Captures upward move; sells more expensively

      Bearish signal (pred < 0):
        - Ask becomes more aggressive (lower ask)
        - Bid becomes less aggressive (lower bid)
        → Captures downward move; buys more cheaply

    The forecast_scale controls how aggressively to adjust.
    At forecast_scale=0, this degenerates to plain AdaptiveMM.
    """
    forecast_scale: float = 2.0   # multiplier for forecast → spread adjustment
    _forecaster: object = field(default=None, repr=False)
    _last_forecast: float = 0.0
    _current_row: object = field(default=None, repr=False)   # latest feature row

    def set_forecaster(self, forecaster) -> None:
        """Attach a trained ReturnForecaster."""
        self._forecaster = forecaster

    def update_features(self, row) -> None:
        """Feed the latest feature row for next prediction."""
        self._current_row = row
        if self._forecaster is not None and self._forecaster.is_fitted:
            try:
                self._last_forecast = self._forecaster.predict(row)
            except Exception:
                self._last_forecast = 0.0

    @property
    def last_forecast(self) -> float:
        return self._last_forecast

    def quote(
        self,
        mu: float,
        sigma: float,
        inventory: float = 0.0,
        t_remaining: float = 1.0,
        alpha_hat: float = None,
        ofi: float = 0.0,
    ) -> Tuple[float, float]:
        # Get base quotes from AdaptiveMM
        bid, ask = super().quote(mu, sigma, inventory, t_remaining, alpha_hat, ofi)

        # Apply forecast skew
        fc = self._last_forecast
        if fc != 0.0:
            # Convert log-return forecast to USD price adjustment
            # fc is the predicted log-return over horizon_bars bars
            # Scale by mu to get USD units, then by forecast_scale for aggressiveness
            fc_usd = fc * mu * self.forecast_scale

            # Cap adjustment to ≤ one full half-spread to avoid crossing quotes
            hs    = max((ask - bid) / 2.0, 1.0)
            fc_usd = np.clip(fc_usd, -hs, hs)

            if fc_usd > 0:
                # Bullish prediction: shift reference price UP
                # → bid more aggressive (higher), ask neutral
                # Logic: expecting higher prices, so buy at higher price and
                # demand more premium to sell (lean toward the sell side)
                bid += fc_usd * 0.70
                ask += fc_usd * 0.30
            else:
                # Bearish prediction: shift reference price DOWN
                # → ask more aggressive (lower), bid neutral
                bid += fc_usd * 0.30
                ask += fc_usd * 0.70

        return round(bid, 4), round(ask, 4)

    @property
    def name(self) -> str:
        return "Forecast-Adaptive"
