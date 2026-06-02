"""
traders.py
Trader population with realistic order-flow simulation.

Trader types:
  - InformedTrader: knows V exactly; only trades when the edge exceeds a
    minimum threshold (aggressiveness * spread), preventing trades on
    infinitesimal mispricings. Size drawn from exponential distribution.
  - UninformedTrader: trades for liquidity/hedging reasons; direction random,
    size smaller than informed.
  - MomentumTrader: reinforces recent price direction, creating autocorrelated
    flow. Not in classic GM — a realistic extension.

The heterogeneous trader mix creates the adverse selection dynamics that make
spread calibration genuinely hard.
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

Side = Literal["buy", "sell", "none"]


@dataclass
class InformedTrader:
    """
    Knows V exactly; trades when the edge exceeds a minimum threshold.

    aggressiveness: minimum required edge as a fraction of the spread.
      At 0.1, the informed trader needs V - ask > 0.1 * (ask - bid) to buy.
      This prevents spurious trades on infinitesimal V vs. quote differences
      and models the real-world friction of execution risk and opportunity cost.
      Default 0.1 ≈ 1 bps minimum edge at a 10 bps spread.
    """
    aggressiveness: float = 0.1     # min edge / spread required to trade
    lambda_size:    float = 1.5     # mean trade size (extra units above 1)
    seed:           Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def decide(self, V: float, bid: float, ask: float) -> Tuple[Side, float]:
        """Returns (side, size)."""
        threshold = self.aggressiveness * (ask - bid)
        edge_buy  = V - ask
        edge_sell = bid - V

        if edge_buy > threshold:
            # Size in BTC: 0.01–0.05 BTC per trade (realistic retail/prop size)
            size = round(0.01 + self.rng.exponential(0.02), 4)
            return "buy", size
        elif edge_sell > threshold:
            size = round(0.01 + self.rng.exponential(0.02), 4)
            return "sell", size
        return "none", 0.0


@dataclass
class UninformedTrader:
    """
    Trades for liquidity / hedging; direction random, size small.
    """
    buy_prob:    float = 0.5
    lambda_size: float = 0.5    # smaller average size than informed
    seed:        Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def decide(self, bid: float, ask: float) -> Tuple[Side, float]:
        side = "buy" if self.rng.random() < self.buy_prob else "sell"
        # Uninformed: smaller fractional BTC sizes than informed
        size = round(0.005 + self.rng.exponential(0.01), 4)
        return side, size


@dataclass
class MomentumTrader:
    """
    Reinforces recent price direction — creates autocorrelated flow.
    Buys after recent up-moves, sells after down-moves.
    Contributes to inventory accumulation and trend-following dynamics.
    This is NOT in classic GM — it's a realistic extension.

    threshold: minimum absolute USD price move over `lookback` bars to trigger.
    At BTC/USDT ≈ $50,000, a threshold of 10 USD ≈ 2 bps — activates only on
    meaningful directional moves, not every 1-minute microstructure tick.
    Calibrate to ~1–3 bps of the typical mid price.
    """
    lookback:    int   = 5
    threshold:   float = 10.0   # min USD move over lookback bars; ~2 bps at $50K BTC
    seed:        Optional[int] = None

    def __post_init__(self):
        self.rng         = np.random.default_rng(self.seed)
        self._price_hist = []

    def observe(self, mid: float):
        self._price_hist.append(mid)
        if len(self._price_hist) > self.lookback:
            self._price_hist.pop(0)

    def decide(self, bid: float, ask: float) -> Tuple[Side, float]:
        if len(self._price_hist) < 2:
            return "none", 0.0
        move = self._price_hist[-1] - self._price_hist[0]
        if move > self.threshold:
            size = round(0.01 + self.rng.exponential(0.015), 4)
            return "buy", size
        elif move < -self.threshold:
            size = round(0.01 + self.rng.exponential(0.015), 4)
            return "sell", size
        return "none", 0.0


@dataclass
class TraderPopulation:
    """
    Mixed population: informed + uninformed + momentum.

    alpha          : fraction of informed traders
    momentum_frac  : fraction of momentum traders (from uninformed pool)
    buy_prob_uninformed : directional tilt for uninformed
    """
    alpha:               float = 0.3
    momentum_frac:       float = 0.1
    buy_prob_uninformed: float = 0.5
    aggressiveness:      float = 0.0
    seed:                Optional[int] = None

    def __post_init__(self):
        self.rng      = np.random.default_rng(self.seed)
        self.informed = InformedTrader(
            self.aggressiveness, seed=self.rng.integers(1_000_000))
        self.noise    = UninformedTrader(
            self.buy_prob_uninformed, seed=self.rng.integers(1_000_000))
        self.momentum = MomentumTrader(seed=self.rng.integers(1_000_000))

    def observe_mid(self, mid: float):
        """Feed mid price to momentum trader each step."""
        self.momentum.observe(mid)

    def arrive(
        self, V: float, bid: float, ask: float
    ) -> Tuple[Side, float, str]:
        """
        Returns (side, size, trader_type).
        trader_type ∈ {'informed', 'uninformed', 'momentum'}
        """
        r = self.rng.random()
        if r < self.alpha:
            side, size = self.informed.decide(V, bid, ask)
            return side, size, "informed"
        elif r < self.alpha + self.momentum_frac:
            side, size = self.momentum.decide(bid, ask)
            return side, size, "momentum"
        else:
            side, size = self.noise.decide(bid, ask)
            return side, size, "uninformed"
