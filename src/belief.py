"""
belief.py
Bayesian belief state over the hidden true value V.

Core model: Gaussian posterior updated via a Kalman-style gain.
    P(V | trade history) ≈ Normal(μ, σ²)

Extensions beyond the classic Glosten-Milgrom belief:
  - Adverse selection detector: rolling alpha_hat (informed fraction estimate)
    derived from |OFI| — independent of the MM's own spread decisions.
    See GaussianBelief.update() for the derivation and the circular-feedback
    problem that this design avoids.
  - Trade imbalance signal: EMA of buy/sell ratio as a toxicity proxy.
  - GridBelief: exact Bayesian reference implementation on a discrete grid.
    Used in tests to validate the Gaussian approximation.

Literature:
  Glosten & Milgrom (1985): belief update derivation in the GM framework.
  Easley et al. (2012): OFI/VPIN as a signed order-flow toxicity measure.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter1d
from typing import List, Literal, Tuple

Side = Literal["buy", "sell"]


@dataclass
class GaussianBelief:
    """
    Gaussian posterior with adverse-selection awareness.

    Parameters
    ----------
    mu0           : prior mean
    sigma0        : prior std
    alpha         : believed fraction of informed traders
    noise_var     : variance of uninformed signal
    process_noise : variance added per step (tracks V movement)
    toxicity_ema  : EMA halflife for adverse selection detector (steps)
    """
    mu0:           float = 100.0
    sigma0:        float = 3.0
    alpha:         float = 0.3
    noise_var:     float = 1.0
    process_noise: float = 0.02
    toxicity_ema:  int   = 30      # halflife for order-flow toxicity EMA

    def __post_init__(self):
        self.mu    = self.mu0
        self.var   = self.sigma0 ** 2
        self.history: List[dict] = []

        # Adverse selection monitoring
        self._ema_alpha = 2.0 / (self.toxicity_ema + 1)
        self.alpha_hat   = self.alpha   # rolling estimate of informed fraction
        self._buy_ema    = 0.5          # EMA of buy indicator (0/1)
        self._sell_ema   = 0.5
        self._trade_count = 0

    @property
    def sigma(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))

    @property
    def order_flow_imbalance(self) -> float:
        """
        Rolling buy/sell imbalance in [-1, +1].
        +1 = all buys (potential informed buying pressure)
        -1 = all sells
        0  = balanced (probably uninformed)
        """
        total = self._buy_ema + self._sell_ema + 1e-9
        return (self._buy_ema - self._sell_ema) / total

    def update(self, side: Side, bid: float, ask: float, was_informed: bool = False) -> Tuple[float, float]:
        """
        Bayesian update given an observed trade.
        Also updates the adverse selection detector.
        """
        signal      = +1.0 if side == "buy" else -1.0
        half_spread = (ask - bid) / 2.0

        # Kalman gain: how much of the observed signal (half_spread) to incorporate.
        # Posterior variance drives the gain; alpha then scales the mean update
        # because only the informed fraction carries a true signal about V.
        #
        # Previous code used (alpha * var) in both numerator and denominator,
        # which conflated the informed-fraction weight with the variance ratio and
        # produced a gain systematically below the correct Kalman value.
        # Correct derivation:
        #   gain = var / (var + noise_var)          ← standard Kalman ratio
        #   mu update: gain * alpha * signal * half_spread
        #   var update: var * (1 - gain)            ← unchanged
        #
        # At alpha=0.3 and var=noise_var the old code gave gain=0.231 vs the
        # correct 0.500; the posterior was under-updated by ~2×.
        gain     = self.var / (self.var + self.noise_var)
        self.mu  = self.mu + gain * self.alpha * signal * half_spread
        self.var = self.var * (1.0 - gain)

        # Update order-flow EMAs
        is_buy = 1.0 if side == "buy" else 0.0
        self._buy_ema  = self._ema_alpha * is_buy       + (1 - self._ema_alpha) * self._buy_ema
        self._sell_ema = self._ema_alpha * (1 - is_buy) + (1 - self._ema_alpha) * self._sell_ema

        # Alpha-hat: EMA of |OFI| — a VPIN-style toxicity proxy.
        #
        # Previous approach (REMOVED): classified each trade as "informed" based on
        # how much it moved the MM's own Bayesian posterior, then EMA'd that binary.
        # Problem: the posterior move depends on the current half_spread, which depends
        # on alpha_hat → circular feedback. In volatile regimes the MM widens spreads,
        # making every posterior update look "large", pushing alpha_hat up further.
        #
        # Correct approach: estimate informed fraction from order flow imbalance.
        # |OFI| ∈ [0,1]: near 0 = balanced (probably uninformed), near 1 = one-sided
        # (potential directional informed flow). This signal is independent of the
        # MM's own quoting decisions and breaks the feedback loop.
        ofi_toxicity   = abs(self.order_flow_imbalance)   # ∈ [0, 1]
        self.alpha_hat = self._ema_alpha * ofi_toxicity + (1 - self._ema_alpha) * self.alpha_hat
        self._trade_count += 1

        self.history.append({
            "mu":         self.mu,
            "sigma":      self.sigma,
            "gain":       gain,
            "side":       side,
            "alpha_hat":  self.alpha_hat,
            "ofi":        self.order_flow_imbalance,
        })
        return self.mu, self.sigma

    def time_step(self):
        """Add process noise each period to prevent variance collapse."""
        self.var = self.var + self.process_noise

    def reset(self):
        self.mu           = self.mu0
        self.var          = self.sigma0 ** 2
        self.alpha_hat    = self.alpha
        self._buy_ema     = 0.5
        self._sell_ema    = 0.5
        self._trade_count = 0
        self.history.clear()


@dataclass
class GridBelief:
    """
    Exact Bayesian update on a discrete grid.

    Serves as the reference implementation against which GaussianBelief is
    validated. GaussianBelief uses a Kalman-style linear approximation that
    is fast but approximate; GridBelief computes the exact posterior on a
    discrete grid and can represent multi-modal or skewed distributions.

    Used in tests/test_belief.py to verify that GaussianBelief produces
    posterior means and variances close to the exact grid solution across
    a range of alpha and sigma values.

    Note: not wired into the main simulation (O(n_grid) per update vs O(1)
    for Gaussian), but kept as a correctness reference.
    """
    v_min:  float = 85.0
    v_max:  float = 115.0
    n_grid: int   = 200
    mu0:    float = 100.0
    sigma0: float = 3.0
    alpha:  float = 0.3

    def __post_init__(self):
        self.grid  = np.linspace(self.v_min, self.v_max, self.n_grid)
        prior      = np.exp(-0.5 * ((self.grid - self.mu0) / self.sigma0) ** 2)
        self.probs = prior / prior.sum()
        # compatibility shim
        self.alpha_hat        = self.alpha
        self.order_flow_imbalance = 0.0

    @property
    def mu(self) -> float:
        return float(np.dot(self.probs, self.grid))

    @property
    def sigma(self) -> float:
        return float(np.sqrt(np.dot(self.probs, (self.grid - self.mu) ** 2)))

    def update(self, side: Side, bid: float, ask: float, was_informed: bool = False) -> Tuple[float, float]:
        if side == "buy":
            likelihood = self.alpha * (self.grid > ask).astype(float) + (1 - self.alpha) * 0.5
        else:
            likelihood = self.alpha * (self.grid < bid).astype(float) + (1 - self.alpha) * 0.5
        posterior  = likelihood * self.probs
        total      = posterior.sum()
        if total > 0:
            self.probs = posterior / total
        return self.mu, self.sigma

    def time_step(self):
        self.probs = gaussian_filter1d(self.probs, sigma=0.5)
        self.probs /= self.probs.sum()

    def reset(self):
        prior      = np.exp(-0.5 * ((self.grid - self.mu0) / self.sigma0) ** 2)
        self.probs = prior / prior.sum()
