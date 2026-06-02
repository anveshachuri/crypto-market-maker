"""
rl_market_maker.py  —  v4 (fixed)
Reinforcement Learning Market Maker using PPO.

If stable-baselines3 + gymnasium are installed, trains a real PPO agent.
Otherwise, uses an inventory-aware rule-based policy that is explicitly
distinct from AdaptiveMM and Passive:

  Rule-based policy design principles:
    - Baseline spread: calibrated AS spread (like AdaptiveMM core, but simpler)
    - Inventory management: exponential skew (reduces inventory aggressively)
    - OFI response: one-sided (raise ask on buy OFI, lower bid on sell OFI)
    - Vol regime: widen spread by local_vol / sigma_calibrated ratio
    - Quote centering: reservation price shift (not symmetric like Passive)

State space (8 dims):
  inventory (normalised), spread (in σ units), realised_vol,
  OFI, 1m return, 5m return, trade intensity (normalised), time fraction

Actions (7):
  0=hold  1=tighten  2=widen  3=skew_bid_up  4=skew_ask_up
  5=skew_bid_down  6=skew_ask_down

Reward:
  + spread_capture per fill
  - inventory_risk  (quadratic in inventory)
  - drawdown_penalty
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

STATE_DIM = 8
N_ACTIONS = 7


class RLMarketMakerEnv:
    """Minimal market-making environment for PPO training."""

    def __init__(self, features_df, params: dict, n_steps: int = 500):
        self.features = features_df.reset_index(drop=True)
        self.params   = params
        self.n_steps  = n_steps
        self.reset()

    def reset(self):
        import random
        self.t           = 0
        self.inventory   = 0.0
        self.cash        = 0.0
        self.max_pnl     = 0.0
        self._offset     = random.randint(0, max(1, len(self.features) - self.n_steps - 1))
        # Start spread at 2× min_spread (not too tight, not too wide)
        self._hs         = self.params.get("sigma_v", 50.0)   # half-spread ≈ sigma_v
        return self._get_state()

    def _get_price(self):
        idx = min(self._offset + self.t, len(self.features) - 1)
        return float(self.features.at[idx, "mid"])

    def _get_row(self):
        idx = min(self._offset + self.t, len(self.features) - 1)
        return self.features.iloc[idx]

    def _get_state(self) -> np.ndarray:
        row    = self._get_row()
        sigma0 = self.params.get("sigma0", 200.0)
        state  = np.array([
            float(self.inventory) / self.params.get("inv_limit", 0.5),
            float(self._hs) / (sigma0 + 1e-6),
            float(row.get("realised_vol", 0.6)),
            float(row.get("ofi_proxy", 0.0)),
            float(row.get("ret_1m", 0.0)) * 100,    # scale to % units
            float(row.get("ret_5m", 0.0)) * 100 if "ret_5m" in row.index else 0.0,
            float(row.get("trade_intensity", 200.0)) / 500.0,
            float(self.t) / self.n_steps,
        ], dtype=np.float32)
        return np.clip(state, -5.0, 5.0)

    def step(self, action: int):
        row    = self._get_row()
        V      = float(row["mid"])
        sigma  = self.params.get("sigma_v", 50.0)

        # Apply action: modify half-spread
        delta  = sigma * 0.3
        bid_adj = ask_adj = 0.0
        if   action == 1: self._hs = max(self._hs * 0.90, sigma * 0.3)
        elif action == 2: self._hs = min(self._hs * 1.10, sigma * 5.0)
        elif action == 3: bid_adj  =  delta
        elif action == 4: ask_adj  =  delta
        elif action == 5: bid_adj  = -delta
        elif action == 6: ask_adj  = -delta

        # Inventory reservation price
        gamma = self.params.get("gamma", 0.1)
        r   = V - self.inventory * gamma * sigma * max(1.0 - self.t / self.n_steps, 0.01)
        bid = r - self._hs + bid_adj
        ask = r + self._hs + ask_adj

        # Stochastic fill (exponential model)
        fill_decay = self.params.get("fill_decay", 1.8)
        fill_prob  = np.exp(-fill_decay * self._hs / (sigma + 1e-6))
        rng        = np.random.default_rng()
        trade_size = 0.02

        reward = 0.0
        if rng.random() < fill_prob:
            if rng.random() < 0.5:
                self.cash += ask * trade_size
                self.inventory -= trade_size
                reward += (ask - V) * trade_size    # spread capture
            else:
                self.cash -= bid * trade_size
                self.inventory += trade_size
                reward += (V - bid) * trade_size

        # Inventory risk penalty (quadratic)
        inv_limit = self.params.get("inv_limit", 0.5)
        reward -= 0.001 * (self.inventory / inv_limit) ** 2 * sigma

        # Drawdown penalty
        mtm = self.cash + self.inventory * V
        self.max_pnl = max(self.max_pnl, mtm)
        dd = max(0.0, self.max_pnl - mtm)
        reward -= dd * 0.0001

        self.t += 1
        done  = self.t >= self.n_steps
        state = self._get_state()
        return state, reward, done


@dataclass
class RLMarketMaker:
    """
    RL-based market maker (PPO or rule-based fallback).

    Rule-based fallback is explicitly calibrated to be competitive with
    AdaptiveMM in fill rate and inventory management, while being
    structurally distinct (simpler response logic, discrete actions).
    """
    gamma:       float = 0.1
    inv_limit:   float = 0.5
    min_spread:  float = 1.0
    max_spread:  float = 500.0

    _policy:       object        = field(default=None,  repr=False)
    _trained:      bool          = False
    _price_history: List[float]  = field(default_factory=list, repr=False)
    _params:       dict          = field(default_factory=dict, repr=False)
    _vol_window:   int           = 20

    def train(self, features_df, params: dict, total_timesteps: int = 50_000) -> None:
        """Train PPO if SB3 available, else activate rule-based policy."""
        self._params = dict(params)
        self._params.setdefault("inv_limit", self.inv_limit)
        self._params.setdefault("gamma", self.gamma)

        if HAS_SB3 and HAS_GYM:
            print(f"  [RL] Training PPO agent ({total_timesteps:,} steps)...")
            env = RLMarketMakerEnv(features_df, self._params)
            gym_env = _GymWrapper(env)
            self._policy = PPO(
                "MlpPolicy", gym_env, verbose=0,
                n_steps=512, batch_size=64, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, ent_coef=0.01,
            )
            self._policy.learn(total_timesteps=total_timesteps)
            self._trained = True
            print("  [RL] PPO training complete.")
        else:
            self._trained = True
            print("  [RL] SB3/Gym not installed → rule-based policy active.")
            if not HAS_SB3:
                print("       Install: pip install stable-baselines3 gymnasium")

    def observe_price(self, price: float) -> None:
        self._price_history.append(price)
        if len(self._price_history) > self._vol_window + 1:
            self._price_history.pop(0)

    @property
    def local_vol(self) -> float:
        if len(self._price_history) < 3:
            return 0.0
        return float(np.std(np.diff(self._price_history)) * np.sqrt(1_440))

    def quote(
        self,
        mu:          float,
        sigma:       float,
        inventory:   float = 0.0,
        t_remaining: float = 1.0,
        alpha_hat:   float = None,
        ofi:         float = 0.0,
    ) -> Tuple[float, float]:
        if HAS_SB3 and self._trained and self._policy is not None:
            state = self._build_state(mu, sigma, inventory, t_remaining, ofi)
            action, _ = self._policy.predict(state, deterministic=True)
            action = int(action)
        else:
            action = self._rule_action(inventory, ofi, sigma, mu, t_remaining)

        return self._action_to_quote(action, mu, sigma, inventory, t_remaining, ofi)

    def _build_state(self, mu, sigma, inventory, t_remaining, ofi) -> np.ndarray:
        return np.array([
            inventory / (self.inv_limit + 1e-9),
            sigma / (self._params.get("sigma0", sigma) + 1e-9),
            self.local_vol / (mu + 1e-9),
            ofi,
            0.0, 0.0,
            1.0 / 5.0,          # normalised intensity proxy
            1.0 - t_remaining,
        ], dtype=np.float32).reshape(1, -1)

    def _rule_action(
        self, inventory: float, ofi: float, sigma: float, mu: float, t_remaining: float
    ) -> int:
        """
        Rule-based policy: inventory-prioritised with OFI secondary.

        Priority order:
          1. High inventory → widen or skew to reduce position
          2. Strong OFI signal → one-sided quote adjustment
          3. High local vol → widen
          4. Low inventory, mild conditions → tighten for more fills
        """
        inv_ratio = abs(inventory) / (self.inv_limit + 1e-9)

        # Urgency: time-to-close pressure on inventory
        urgency = 1.0 - t_remaining   # 0 at start, 1 at end
        effective_ratio = inv_ratio + urgency * inv_ratio * 0.5

        # Priority 1: large inventory → skew quotes to unwind
        if effective_ratio > 0.7:
            if inventory > 0:    # long → lower ask to encourage selling
                return 6   # skew_ask_down
            else:                # short → raise bid to encourage buying
                return 3   # skew_bid_up

        if effective_ratio > 0.4:
            # Widen to slow new position buildup
            return 2   # widen

        # Priority 2: OFI signal (only when inventory is manageable)
        if abs(ofi) > 0.25:
            if ofi > 0:
                # Buy pressure → raise ask only (protect; don't encourage sells)
                return 4   # skew_ask_up
            else:
                # Sell pressure → lower bid only
                return 5   # skew_bid_down

        # Priority 3: vol regime
        calib_vol = self._params.get("sigma_v", sigma)
        if self.local_vol > calib_vol * 1.5:
            return 2   # widen in high-vol regime

        # Priority 4: low inventory, calm conditions → tighten for more fills
        if inv_ratio < 0.2 and abs(ofi) < 0.1:
            return 1   # tighten

        return 0   # hold

    def _action_to_quote(
        self, action: int, mu: float, sigma: float,
        inventory: float, t_remaining: float, ofi: float,
    ) -> Tuple[float, float]:
        """Convert discrete action to bid/ask quotes."""
        # Base: AS-style reservation price + spread
        gamma = self._params.get("gamma", self.gamma) if self._params else self.gamma
        # Reservation price: shift to reduce inventory
        r = mu - inventory * gamma * (sigma ** 2) * t_remaining

        # Base half-spread: calibrated to be competitive (≈ AdaptiveMM without vol/toxicity layers)
        s0 = gamma * (sigma ** 2) * t_remaining
        # Ensure a minimum meaningful spread: at least 3× min_spread half
        hs = max(s0, self.min_spread * 1.5)
        hs = min(hs, self.max_spread / 2.0)

        # Action modifications
        delta = sigma * 0.2    # ≈ 10% of sigma per adjustment
        bid_adj = ask_adj = 0.0

        if   action == 1: hs = max(hs * 0.80, self.min_spread)   # tighten
        elif action == 2: hs = min(hs * 1.20, self.max_spread / 2)  # widen
        elif action == 3: bid_adj =  delta    # skew bid up
        elif action == 4: ask_adj =  delta    # skew ask up
        elif action == 5: bid_adj = -delta    # skew bid down
        elif action == 6: ask_adj = -delta    # skew ask down

        bid = r - hs + bid_adj
        ask = r + hs + ask_adj

        # Ensure minimum spread
        if ask - bid < self.min_spread:
            mid_q = (bid + ask) / 2.0
            bid = mid_q - self.min_spread / 2.0
            ask = mid_q + self.min_spread / 2.0

        return round(bid, 4), round(ask, 4)

    @property
    def name(self) -> str:
        if HAS_SB3 and self._trained and self._policy is not None:
            return "RL-PPO"
        return "RL-RuleBased"


class _GymWrapper:
    """Minimal Gym compatibility wrapper for SB3."""
    def __init__(self, env: RLMarketMakerEnv):
        self._env = env
        if HAS_GYM:
            self.observation_space = spaces.Box(-5.0, 5.0, shape=(STATE_DIM,), dtype=np.float32)
            self.action_space      = spaces.Discrete(N_ACTIONS)

    def reset(self, **kwargs):
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done = self._env.step(int(action))
        return obs, reward, done, done, {}
