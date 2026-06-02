"""
price_process.py
Replaces synthetic true_value.py with a real-data price process.

Instead of generating V(t) from a stochastic model, we replay real
BTC/USDT mid prices from Binance. The "true value" at each step is
the actual market mid — this is what the MM is trying to estimate.

Why this is valid:
  - In a deep crypto market, the Binance mid IS the consensus value
  - Microstructure noise means a single MM can't observe it directly
  - The Bayesian belief updates are still meaningful: the MM infers
    the current mid from its own trade flow, not from peeking at the tape

ReplayProcess:
  - Iterates through real mid prices one bar at a time
  - Exposes the same .step() / .regime interface as synthetic processes
  - Supports looping (for multi-episode runs) with optional noise augmentation

HybridProcess:
  - Uses real vol/regime structure but generates synthetic paths
  - Useful when you need more episodes than data bars
  - Calibrates OU / jump params from real data, then simulates
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReplayProcess:
    """
    Replay real BTC/USDT mid prices.

    Each call to step() returns the next real mid price.
    When the sequence ends, it loops (optionally with Gaussian noise
    to break exact repetition across episodes).

    Parameters
    ----------
    prices      : np.ndarray of real mid prices (chronological)
    regimes     : np.ndarray of regime labels per bar
    noise_scale : std of Gaussian noise added on loop (fraction of price)
    seed        : random seed for noise
    """
    prices:      np.ndarray
    regimes:     np.ndarray
    noise_scale: float = 0.0002   # 0.02% noise on repeat
    seed:        Optional[int] = None

    def __post_init__(self):
        self.rng     = np.random.default_rng(self.seed)
        self._idx    = 0
        self._loop   = 0
        self.V       = float(self.prices[0])
        self.regime  = str(self.regimes[0]) if len(self.regimes) > 0 else "quiet"

    def step(self) -> float:
        price = float(self.prices[self._idx])
        if self._loop > 0:
            # Add small noise to break repetition across episodes
            price *= (1.0 + self.rng.normal(0, self.noise_scale))
        self.V      = price
        self.regime = str(self.regimes[self._idx])
        self._idx  += 1
        if self._idx >= len(self.prices):
            self._idx  = 0
            self._loop += 1
        return self.V

    def reset(self, offset: int = 0):
        """Reset to a given offset (for episode variety)."""
        self._idx   = offset % len(self.prices)
        self._loop  = 0
        self.V      = float(self.prices[self._idx])
        self.regime = str(self.regimes[self._idx])


@dataclass
class HybridProcess:
    """
    Synthetic process calibrated from real BTC/USDT data.

    Uses real calibrated parameters (vol, mean-reversion speed) but
    generates synthetic paths — giving unlimited episodes while
    matching real distributional properties.

    This is regime-switching OU + jumps, where:
      - quiet   params: calibrated from quiet-regime bars
      - volatile params: calibrated from volatile-regime bars
      - trending params: calibrated from trending-regime bars

    The regime transition matrix is estimated from real regime sequence.
    """
    # Calibrated from real data (set by from_features classmethod)
    mu0:     float = 50000.0
    dt:      float = 1 / (365 * 24 * 60)   # 1-minute bar

    # Per-regime params (set by calibration)
    # drift_sigma_mult: per-step drift as a multiple of (sigma × sqrt(dt)).
    # Storing it this way ensures drift is always on the same scale as diffusion
    # and doesn't collapse to zero when dt is small (as multiplying by dt alone does).
    # 0.0 → no trend;  0.5 → drift = 50% of the diffusion magnitude per step.
    _quiet_params:    dict = field(default_factory=lambda: {"kappa": 3.0, "sigma": 0.15, "drift_sigma_mult": 0.0})
    _volatile_params: dict = field(default_factory=lambda: {"kappa": 1.0, "sigma": 0.80, "drift_sigma_mult": 0.0})
    _trending_params: dict = field(default_factory=lambda: {"kappa": 0.5, "sigma": 0.30, "drift_sigma_mult": 0.5})
    _trans:           np.ndarray = field(default_factory=lambda: np.array([
        [0.97, 0.015, 0.015],
        [0.03, 0.94,  0.03 ],
        [0.03, 0.03,  0.94 ],
    ]))

    seed: Optional[int] = None

    def __post_init__(self):
        self.rng         = np.random.default_rng(self.seed)
        self.V           = self.mu0
        self._names      = ["quiet", "volatile", "trending"]
        self._regime_idx = 0
        self._trend_dir  = 1.0
        self.regime      = "quiet"
        self.regime_history = []

    @classmethod
    def from_features(cls, features: pd.DataFrame, seed: Optional[int] = None) -> "HybridProcess":
        """
        Calibrate regime params from real feature DataFrame.
        """
        proc = cls(mu0=float(features["mid"].iloc[-1]), seed=seed)

        # Per-regime vol calibration
        for regime_name, attr in [
            ("quiet",    "_quiet_params"),
            ("volatile", "_volatile_params"),
            ("trending", "_trending_params"),
        ]:
            mask = features["regime"] == regime_name
            if mask.sum() < 5:
                continue
            sub = features[mask]
            sigma_ann = float(sub["realised_vol"].median())
            # realised_vol is dimensionless annualised pct vol (e.g. 0.286 = 28.6%)
            # Per-step USD std = ann_vol_pct * mid_price
            # (because: per_step_pct = ann_pct * sqrt(dt), and USD = pct * price,
            #  but in the OU step we use sigma * N(0,1) not sigma * N(0,sqrt(dt)))
            sigma_step = sigma_ann * float(features["mid"].mean())

            # Mean-reversion speed from autocorrelation
            rets = sub["log_ret"].dropna()
            ac1  = float(rets.autocorr(1)) if len(rets) > 10 else -0.1
            kappa = max(0.1, -np.log(max(ac1, 0.01)))

            p = getattr(proc, attr)
            p["sigma"] = sigma_step
            p["kappa"] = kappa
            # Trending regime: add a persistent drift equal to 50% of diffusion.
            # drift_sigma_mult × sigma × sqrt(dt) gives a per-step drift in USD
            # that's on the same scale as the diffusion term, ensuring the trending
            # regime produces a visible directional move over an episode.
            if attr == "_trending_params":
                p["drift_sigma_mult"] = 0.5

        # Estimate transition matrix from actual regime sequence
        labels = features["regime"].values
        name_to_idx = {"quiet": 0, "volatile": 1, "trending": 2}
        counts = np.ones((3, 3))   # Laplace smoothing
        for i in range(len(labels) - 1):
            a = name_to_idx.get(labels[i], 0)
            b = name_to_idx.get(labels[i+1], 0)
            counts[a, b] += 1
        proc._trans = counts / counts.sum(axis=1, keepdims=True)

        return proc

    def _switch_regime(self):
        new = self.rng.choice(3, p=self._trans[self._regime_idx])
        if new != self._regime_idx:
            self._regime_idx = new
            if self._names[new] == "trending":
                self._trend_dir = self.rng.choice([-1.0, 1.0])

    def step(self) -> float:
        self._switch_regime()
        self.regime = self._names[self._regime_idx]
        self.regime_history.append(self.regime)

        params = [self._quiet_params, self._volatile_params, self._trending_params][self._regime_idx]
        eps    = self.rng.normal(0, 1)
        # Drift: expressed as a multiple of the per-step diffusion magnitude so that
        # it remains meaningful regardless of dt (multiplying by dt alone gives
        # drift ≈ 1e-6 × mu0 per step for 1-minute BTC bars — effectively zero).
        drift_mult = params.get("drift_sigma_mult", 0.0)
        drift      = drift_mult * params["sigma"] * np.sqrt(self.dt) * self._trend_dir
        self.V = (self.V
                  + params["kappa"] * (self.mu0 - self.V) * self.dt
                  + drift
                  + params["sigma"] * np.sqrt(self.dt) * eps)
        return self.V

    def reset(self):
        self.V              = self.mu0
        self._regime_idx    = 0
        self._trend_dir     = 1.0
        self.regime_history = []
