"""
ml_forecaster.py  —  v4.2 (train/inference parity fix)

What changed vs v4.1
--------------------
v4.1 had a critical train/inference feature-generation mismatch:

  * `_make_features()` z-score normalised using the mean/std *of whatever
    frame it was handed*. During batch training that frame had thousands of
    rows, so the statistics were meaningful. During online inference the
    quoting loop handed it a SINGLE row (`row.to_frame().T`), so:
        - every rolling / lag feature was NaN -> filled with 0
        - z-scoring one value gives (x - x) / 1 = 0 for every column
    The model therefore received an all-zero feature vector on every call and
    returned ONE constant prediction regardless of market state. The forecast
    signal was dead, and rebuilding a DataFrame + rolling stats per row made
    the simulation ~50x slower (the observed "slow -> KeyboardInterrupt").

v4.2 fixes this by separating the two concerns that were tangled together:

  1. RAW feature construction (`_raw_feature_frame`) is pure and vectorised.
     Identical code runs for training and inference, so feature *definitions*
     can never drift between the two paths.
  2. NORMALISATION statistics (mean, std) are fitted ONCE on the training
     split and STORED on the model. Inference reuses the stored stats; it
     never recomputes them. This is the textbook fit/transform contract and
     also removes the prior train/validation normalisation leakage.
  3. Online prediction is served from a precomputed, index-keyed cache
     (`prime_cache`), built with a single vectorised pass over the feature
     frame. Per-call cost drops from ~9 ms (DataFrame rebuild + rolling) to an
     O(1) dictionary lookup, and the served value is *identical* to the batch
     prediction for that row -> exact train/inference parity.
  When a row cannot be resolved against the cache, `predict` returns a neutral
  0.0 (ForecastAdaptiveMM degenerates to AdaptiveMM at 0) instead of feeding
  the model a garbage all-zero vector.

On real Binance data the dominant predictive signal is SHORT-TERM MEAN
REVERSION (IC ~ -0.03 to -0.06 for ret_1m/ret_5m), consistent with the
high-frequency microstructure literature (Bouchaud et al. 2018). IC, tree
count and feature-importance spread are data-dependent and must be assessed on
real data; they are honest outputs of the fit, not targets to engineer toward.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.linear_model import Ridge
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Raw feature columns — unit-scale features only.
# signed_vol and trade_intensity have std ~5-200x target; use z-scored versions.
FEATURE_COLS = [
    "ret_1m",          # 1-bar log return       (~same scale as target)
    "ret_5m",          # 5-bar log return
    "ret_15m",         # 15-bar log return
    "ofi_proxy",       # order flow imbalance in [-1, 1]
    "buy_ratio",       # buyer fraction in [0, 1]
    "vol_5m",          # short vol (log-return std over 5 bars)
    "vol_30m",         # medium vol
    "realised_vol",    # annualised vol (will be z-scored)
    "trend_signal",    # MA-crossover (small float)
]


def _raw_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the RAW (un-normalised) feature frame.

    This is a pure function of `df` and is used UNCHANGED for both training and
    inference, which guarantees the two paths share identical feature
    definitions. Rolling / lag features are computed over whatever history
    `df` contains; callers are responsible for passing enough history
    (the whole frame at train time, the whole frame at prime_cache time).

    Normalisation is intentionally NOT done here — see ReturnForecaster._transform.
    """
    X = pd.DataFrame(index=df.index)

    # Core features — direct value and 1-bar lag.
    for col in FEATURE_COLS:
        if col in df.columns:
            X[col]           = df[col]
            X[f"{col}_lag1"] = df[col].shift(1)

    # OFI rolling features (already on [-1, 1] scale).
    if "ofi_proxy" in df.columns:
        X["ofi_ma5"]      = df["ofi_proxy"].rolling(5).mean()
        X["ofi_ma10"]     = df["ofi_proxy"].rolling(10).mean()
        X["ofi_momentum"] = df["ofi_proxy"] - df["ofi_proxy"].rolling(10).mean()

    # Volatility ratio (vol-regime change signal).
    if "vol_5m" in df.columns and "vol_30m" in df.columns:
        X["vol_ratio"] = df["vol_5m"] / (df["vol_30m"].replace(0, np.nan) + 1e-9)

    # Trade-intensity z-score (rolling, so safe to include).
    if "trade_intensity" in df.columns:
        ti = df["trade_intensity"]
        X["intensity_zscore"] = (ti - ti.rolling(30).mean()) / (ti.rolling(30).std() + 1e-9)

    # Buy pressure, centred.
    if "buy_ratio" in df.columns:
        X["buy_pressure"] = df["buy_ratio"] - 0.5

    return X, list(X.columns)


def _make_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    DEPRECATED batch-only helper, retained for backward-compatible imports.

    Returns a self-normalised feature matrix using the statistics of `df`
    itself. This is ONLY valid on a large batch frame; calling it on a single
    row collapses every column to 0. Internal code no longer uses this; use
    ReturnForecaster.fit / .predict / .predict_series instead.
    """
    X, cols = _raw_feature_frame(df)
    arr = X.fillna(0).values.astype(np.float64)
    std = arr.std(axis=0)
    std[std < 1e-10] = 1.0
    arr = (arr - arr.mean(axis=0)) / std
    return arr.astype(np.float32), cols


@dataclass
class ReturnForecaster:
    """
    Short-horizon return forecaster (LightGBM or Ridge fallback).

    Predicts the sum of log returns over the next horizon_bars bars.
    IC (rank correlation on a held-out time block) is the primary quality
    metric. Typical achievable IC on 1-min crypto data:
      5-bar:  IC ~ 0.02-0.06  (mean-reversion dominant)
      15-bar: IC ~ 0.03-0.08  (momentum emerges at longer horizon)
    """
    horizon_bars:  int   = 5
    n_estimators:  int   = 500
    learning_rate: float = 0.05
    max_depth:     int   = 3
    min_samples:   int   = 300

    _model:               object          = field(default=None, repr=False)
    _feature_cols_fitted: List[str]       = field(default_factory=list, repr=False)
    _feat_mean:           Optional[np.ndarray] = field(default=None, repr=False)
    _feat_std:            Optional[np.ndarray] = field(default=None, repr=False)
    _is_fitted:           bool            = False
    _use_lgb:             bool            = True
    _ic_rank:             float           = 0.0
    _ic_pearson:          float           = 0.0
    _ic_pvalue:           float           = 1.0
    _n_train:             int             = 0
    _n_val:               int             = 0
    _model_name:          str             = ""
    _feature_importance:  Dict[str, float] = field(default_factory=dict, repr=False)
    _pred_cache:          Dict            = field(default_factory=dict, repr=False)
    _warned_no_cache:     bool            = field(default=False, repr=False)

    # ── fitting ──────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "ReturnForecaster":
        if len(df) < self.min_samples + self.horizon_bars + 20:
            print(f"  [ML] Insufficient data ({len(df)} rows). Skipped.")
            return self

        df_in = df  # keep original (already 0..N-1 indexed) for cache priming
        work = df.copy().reset_index(drop=True)
        work["target"] = (
            work["log_ret"].rolling(self.horizon_bars).sum().shift(-self.horizon_bars)
        )
        work = work.dropna(subset=["target"]).reset_index(drop=True)

        Xraw, col_names = _raw_feature_frame(work)
        self._feature_cols_fitted = col_names
        arr = Xraw.values.astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)   # warmup NaN/inf -> 0 (pre-norm)
        y = work["target"].values.astype(np.float32)

        # Time-based 80/20 split — no shuffling.
        split = int(len(arr) * 0.8)

        # Fit normalisation on the TRAIN split ONLY (no validation leakage),
        # then store the stats so inference can reuse them.
        train_block = arr[:split]
        mean = train_block.mean(axis=0)
        std  = train_block.std(axis=0)
        std[std < 1e-10] = 1.0
        self._feat_mean = mean
        self._feat_std  = std

        arr_n = ((arr - mean) / std).astype(np.float32)
        X_tr, X_val = arr_n[:split], arr_n[split:]
        y_tr, y_val = y[:split], y[split:]
        self._n_train, self._n_val = int(len(X_tr)), int(len(X_val))

        if HAS_LGB and self._use_lgb:
            self._fit_lgb(X_tr, y_tr, X_val, y_val)
        elif HAS_SKLEARN:
            self._fit_ridge(X_tr, y_tr)
        else:
            print("  [ML] No ML library available.")
            return self

        y_pred_val = self._predict_raw(X_val)
        self._compute_ic(y_val.astype(np.float64), y_pred_val.astype(np.float64))
        self._is_fitted = True

        if HAS_LGB and self._use_lgb and self._model is not None:
            imp = self._model.feature_importance(importance_type="gain")
            self._feature_importance = dict(zip(col_names, imp.tolist()))

        # Precompute online predictions for every row of the ORIGINAL frame in
        # one vectorised pass. Rolling features over the full frame match those
        # over the (full minus last horizon) training frame for every shared
        # row, so cached predictions equal batch predictions exactly.
        self.prime_cache(df_in)

        print(f"  [ML] {self._model_name} | "
              f"IC(rank)={self._ic_rank:+.4f}  IC(pearson)={self._ic_pearson:+.4f}  "
              f"p={self._ic_pvalue:.4f} | "
              f"n_train={self._n_train:,}  n_val={self._n_val:,}")
        return self

    def _fit_lgb(self, X_tr, y_tr, X_val, y_val):
        params = {
            "objective":           "regression",
            "metric":              "mse",
            "learning_rate":       self.learning_rate,
            "max_depth":           self.max_depth,
            "num_leaves":          15,
            "min_data_in_leaf":    20,
            "subsample":           0.8,
            "colsample_bytree":    0.7,
            "reg_alpha":           0.1,
            "reg_lambda":          0.5,
            "force_col_wise":      True,
            "verbosity":           -1,
        }
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)
        callbacks  = [lgb.early_stopping(50, verbose=False),
                      lgb.log_evaluation(period=-1)]
        self._model = lgb.train(
            params, train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )
        self._model_name = f"LightGBM({self._model.num_trees()} trees)"

    def _fit_ridge(self, X_tr, y_tr):
        self._model = Ridge(alpha=10.0)
        self._model.fit(X_tr, y_tr)
        self._use_lgb = False
        self._model_name = "Ridge"

    # ── transforms & raw prediction ──────────────────────────────────────
    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        """Raw features -> align to fitted schema -> fill NaN -> apply stored z-stats."""
        Xraw, _ = _raw_feature_frame(df)
        # Reindex to the exact fitted column set AND order. Missing columns
        # become NaN here and are zeroed below; extras are dropped.
        Xraw = Xraw.reindex(columns=self._feature_cols_fitted)
        arr = Xraw.values.astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = (arr - self._feat_mean) / self._feat_std
        return arr.astype(np.float32)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(X), dtype=np.float64)
        return self._model.predict(X).astype(np.float64)

    def _compute_ic(self, y_true: np.ndarray, y_pred: np.ndarray):
        from scipy.stats import rankdata
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        n = len(y_true)
        if n < 10:
            self._ic_rank = self._ic_pearson = 0.0; self._ic_pvalue = 1.0
            return
        if y_true.std() > 1e-10 and y_pred.std() > 1e-10:
            self._ic_pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            self._ic_pearson = 0.0
        r_t = rankdata(y_true); r_p = rankdata(y_pred)
        if r_t.std() > 1e-10 and r_p.std() > 1e-10:
            self._ic_rank = float(np.corrcoef(r_t, r_p)[0, 1])
            t_stat = self._ic_rank * np.sqrt((n - 2) / max(1 - self._ic_rank**2, 1e-10))
            from scipy.stats import t as t_dist
            self._ic_pvalue = float(2 * t_dist.sf(abs(t_stat), df=n - 2))
        else:
            self._ic_rank = 0.0; self._ic_pvalue = 1.0

    # ── prediction API ───────────────────────────────────────────────────
    def prime_cache(self, df: pd.DataFrame) -> "ReturnForecaster":
        """
        Precompute predictions for every row of `df` in one vectorised pass and
        store them keyed by the row's index label. The online quoting loop hands
        `predict` a row whose `.name` is its position in this frame, so lookups
        are O(1) and return exactly the batch prediction for that row.
        """
        if not self._is_fitted or self._model is None:
            return self
        preds = self._predict_raw(self._transform(df))
        preds = np.where(np.isfinite(preds), preds, 0.0)
        self._pred_cache = dict(zip(df.index.tolist(), preds.tolist()))
        return self

    def predict(self, row) -> float:
        """
        Online single-row prediction. Served from the precomputed cache by row
        index (exact train/inference parity, O(1)). If the row cannot be
        resolved against the cache, returns a neutral 0.0 rather than feeding
        the model an ill-defined single-row feature vector.
        """
        if not self._is_fitted or self._model is None:
            return 0.0
        name = getattr(row, "name", None)
        if name is not None and name in self._pred_cache:
            v = self._pred_cache[name]
            return float(v) if np.isfinite(v) else 0.0
        if not self._warned_no_cache:
            warnings.warn(
                "ReturnForecaster.predict: row not found in precomputed cache; "
                "returning neutral 0.0. Call prime_cache(features) on the frame "
                "used by the simulation before quoting.",
                RuntimeWarning, stacklevel=2,
            )
            self._warned_no_cache = True
        return 0.0

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """Vectorised predictions for a whole frame (correct rolling history)."""
        if not self._is_fitted or self._model is None:
            return pd.Series(0.0, index=df.index)
        preds = self._predict_raw(self._transform(df))
        preds = np.where(np.isfinite(preds), preds, 0.0)
        return pd.Series(preds, index=df.index)

    def feature_importance_df(self) -> pd.DataFrame:
        if not self._feature_importance:
            return pd.DataFrame(columns=["feature", "importance", "importance_pct"])
        items = sorted(self._feature_importance.items(), key=lambda x: -x[1])
        df = pd.DataFrame(items, columns=["feature", "importance"])
        total = df["importance"].sum()
        df["importance_pct"] = df["importance"] / total * 100 if total > 0 else 0.0
        return df

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def information_coefficient(self) -> float:
        return self._ic_rank

    @property
    def pearson_ic(self) -> float:
        return self._ic_pearson

    @property
    def ic_pvalue(self) -> float:
        return self._ic_pvalue


def train_forecaster(
    features: pd.DataFrame,
    horizon_bars: int = 5,
    verbose: bool = True,
) -> ReturnForecaster:
    if verbose:
        print(f"  [ML] Training LightGBM forecaster (horizon={horizon_bars} bars)...")
        print(f"  [ML] n_rows={len(features):,} | features z-score normalised (stored stats)")
    model = ReturnForecaster(horizon_bars=horizon_bars)
    model.fit(features)
    return model
