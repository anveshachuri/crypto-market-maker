"""
regime_detector.py  —  v4 Priority 2
Volatility regime detection and classification.

Regimes:
  - low_vol    : rolling realised vol in bottom tercile
  - medium_vol : middle tercile
  - high_vol   : top tercile

Every simulation episode is tagged with a regime.
All performance metrics are reported separately by regime.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RegimeDetector:
    """
    Rolling volatility-based regime classifier.

    Uses expanding-window percentiles to avoid look-ahead bias.
    Smooths labels with a rolling mode to prevent flickering.
    """
    vol_window:   int   = 20     # bars for realised vol estimate
    smooth_window: int  = 5      # bars for label smoothing
    low_pct:      float = 0.33   # low/medium boundary percentile
    high_pct:     float = 0.67   # medium/high boundary percentile

    _vol_history: List[float] = field(default_factory=list)
    _regime_history: List[str] = field(default_factory=list)

    def update(self, realised_vol: float) -> str:
        """
        Add a new vol observation and return current regime label.
        Uses expanding window: no look-ahead bias.
        """
        self._vol_history.append(realised_vol)
        history = np.array(self._vol_history)

        p_low  = np.percentile(history, self.low_pct * 100)
        p_high = np.percentile(history, self.high_pct * 100)

        if realised_vol <= p_low:
            raw_label = "low_vol"
        elif realised_vol > p_high:
            raw_label = "high_vol"
        else:
            raw_label = "medium_vol"

        self._regime_history.append(raw_label)

        # Smooth: rolling mode
        recent = self._regime_history[-self.smooth_window:]
        from collections import Counter
        label = Counter(recent).most_common(1)[0][0]
        return label

    def classify_series(self, vol_series: pd.Series) -> pd.Series:
        """Classify a full series of realised vol values. Expanding window."""
        p33 = vol_series.expanding(min_periods=1).quantile(self.low_pct)
        p67 = vol_series.expanding(min_periods=1).quantile(self.high_pct)

        labels = pd.Series("medium_vol", index=vol_series.index)
        labels[vol_series <= p33] = "low_vol"
        labels[vol_series >  p67] = "high_vol"

        # Smooth
        encoded  = labels.map({"low_vol": 0, "medium_vol": 1, "high_vol": 2})
        smoothed = encoded.rolling(self.smooth_window, min_periods=1).apply(
            lambda x: pd.Series(x).mode().iloc[0], raw=False
        )
        return smoothed.map({0: "low_vol", 1: "medium_vol", 2: "high_vol"})


def compute_regime_stats(
    results: Dict[str, pd.DataFrame],
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute per-regime performance metrics for multiple strategies.

    Parameters
    ----------
    results : dict mapping strategy_name → episode summary DataFrame
              (output of run_many_episodes or compare_strategies_v4)
    metric_cols : which metrics to report (defaults to standard set)

    Returns
    -------
    DataFrame with MultiIndex (strategy, regime) and metric columns.
    """
    if metric_cols is None:
        metric_cols = [
            "final_pnl", "sharpe", "sortino", "max_drawdown",
            "fill_rate", "inv_variance", "spread_capture",
            "adverse_sel_cost",
        ]

    rows = []
    for strat_name, summary in results.items():
        if "regime" not in summary.columns:
            # Add a synthetic regime column if missing
            summary = summary.copy()
            summary["regime"] = "all"

        for regime in ["low_vol", "medium_vol", "high_vol"]:
            sub = summary[summary["regime"] == regime]
            if len(sub) == 0:
                continue
            row = {"strategy": strat_name, "regime": regime, "n_episodes": len(sub)}
            for col in metric_cols:
                if col in sub.columns:
                    row[f"{col}_mean"] = float(sub[col].mean())
                    row[f"{col}_std"]  = float(sub[col].std())
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["strategy", "regime"])


def regime_performance_table(
    results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build the summary table requested in v4 requirements:

    | Strategy | Low Vol | Medium Vol | High Vol |
    |----------|---------|------------|----------|
    | GM       |  ...    |    ...     |   ...    |
    | AS       |  ...    |    ...     |   ...    |
    | Adaptive |  ...    |    ...     |   ...    |

    Metric shown: mean final PnL (USD).
    """
    table_data = {}
    for strat_name, summary in results.items():
        row = {}
        if "regime" not in summary.columns:
            row["Low Vol"]    = f"${summary['final_pnl'].mean():,.0f}"
            row["Medium Vol"] = "—"
            row["High Vol"]   = "—"
        else:
            for col_label, regime_key in [
                ("Low Vol", "low_vol"),
                ("Medium Vol", "medium_vol"),
                ("High Vol", "high_vol"),
            ]:
                sub = summary[summary["regime"] == regime_key]
                if len(sub) > 0:
                    row[col_label] = f"${sub['final_pnl'].mean():,.0f}"
                else:
                    row[col_label] = "—"
        table_data[strat_name] = row

    return pd.DataFrame(table_data).T
