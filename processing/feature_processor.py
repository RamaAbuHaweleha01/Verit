#!/usr/bin/env python3
"""
Verit NIDS - Stage 2c: Preprocessing, Encoding & Scaling
------------------------------------------------------------------
Takes the raw flow-feature DataFrame produced by flow_extractor.py and
makes it model-ready:

    1. NaN / Infinity handling        (replace inf, impute or drop NaN)
    2. Drop identity features         (IPs, ports, flow_id, timestamp --
                                        useful for tracing, useless/harmful
                                        as model input)
    3. Drop zero-variance / constant  features
    4. Drop duplicate rows
    5. Categorical encoding           (one-hot for low-cardinality like
                                        `protocol`; optional target/mean
                                        encoding for high-cardinality columns)
    6. Label encoding                 for the target column (attack type)
    7. Feature scaling                (StandardScaler, fit on train only)

`FeatureProcessor` is fit once on training data and then saved with
joblib; at inference time you load it back and call `.transform()` so
new traffic goes through *exactly* the same encoding/scaling as training
data -- critical for the autoencoder + XGBoost models to see consistent
input distributions.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_IDENTITY_COLUMNS = [
    "flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "timestamp",
]

DEFAULT_ONEHOT_COLUMNS = ["protocol"]


class TargetEncoder:
    """Simple smoothed mean-target encoder for high-cardinality categorical
    columns. Not used by default -- kept available for future feature engineering."""

    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self.global_mean_ = None
        self.mapping_ = {}

    def fit(self, series: pd.Series, target: pd.Series):
        self.global_mean_ = target.mean()
        stats = target.groupby(series).agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
            stats["count"] + self.smoothing
        )
        self.mapping_ = smoothed.to_dict()
        return self

    def transform(self, series: pd.Series):
        return series.map(self.mapping_).fillna(self.global_mean_)

    def fit_transform(self, series, target):
        return self.fit(series, target).transform(series)


class FeatureProcessor:
    def __init__(
        self,
        identity_columns=None,
        onehot_columns=None,
        nan_strategy="median",
        drop_duplicate_rows=True,
        drop_zero_variance=True,
        label_column=None,
    ):
        self.identity_columns = identity_columns or list(DEFAULT_IDENTITY_COLUMNS)
        self.onehot_columns = onehot_columns or list(DEFAULT_ONEHOT_COLUMNS)
        self.nan_strategy = nan_strategy
        self.drop_duplicate_rows = drop_duplicate_rows
        self.drop_zero_variance = drop_zero_variance
        self.label_column = label_column

        self.impute_values_ = {}
        self.zero_variance_columns_ = []
        self.onehot_categories_ = {}
        self.label_encoder_ = None
        self.scaler_ = None
        self.feature_columns_ = []
        self.is_fitted_ = False

    def fit_transform(self, df: pd.DataFrame):
        identity_df = df[[c for c in self.identity_columns if c in df.columns]].copy()

        y = None
        if self.label_column and self.label_column in df.columns:
            y_raw = df[self.label_column]
            self.label_encoder_ = LabelEncoder()
            y = pd.Series(self.label_encoder_.fit_transform(y_raw.astype(str)), index=df.index)

        # NOTE: no df.copy() here. `work` below is already a fresh object
        # (drop() never returns a view), and we never mutate the caller's
        # original `df`. On multi-million-row datasets an extra full copy
        # of the input frame was the single biggest avoidable memory spike
        # in this pipeline -- skip it.
        work = df.drop(columns=[c for c in self.identity_columns if c in df.columns], errors="ignore")
        if self.label_column and self.label_column in work.columns:
            work = work.drop(columns=[self.label_column])

        work = self._replace_inf_with_nan(work)
        work = self._handle_nan_fit(work)
        work = self._onehot_encode_fit(work)

        if self.drop_duplicate_rows:
            before = len(work)
            work = work.drop_duplicates()
            dropped = before - len(work)
            if dropped:
                print(f"[processing] Dropped {dropped} duplicate rows")

        if self.drop_zero_variance:
            self.zero_variance_columns_ = [
                col for col in work.columns if work[col].nunique(dropna=False) <= 1
            ]
            if self.zero_variance_columns_:
                print(f"[processing] Dropping {len(self.zero_variance_columns_)} zero-variance columns: "
                      f"{self.zero_variance_columns_}")
                work = work.drop(columns=self.zero_variance_columns_)

        self.feature_columns_ = list(work.columns)

        self.scaler_ = StandardScaler()
        # float32 throughout -- float64 would silently double the memory
        # footprint of the largest array in this whole pipeline right at
        # its peak. Precision loss is negligible for these feature scales.
        scaled = self.scaler_.fit_transform(work.values.astype(np.float32))
        scaled_df = pd.DataFrame(scaled, columns=self.feature_columns_, index=work.index)
        del work, scaled

        self.is_fitted_ = True

        result = {"X": scaled_df, "identity": identity_df.loc[scaled_df.index]}
        if y is not None:
            result["y"] = y.loc[scaled_df.index]
            result["label_classes"] = list(self.label_encoder_.classes_)
        return result

    def transform(self, df: pd.DataFrame):
        if not self.is_fitted_:
            raise RuntimeError("FeatureProcessor must be fit (fit_transform) or loaded before transform().")

        identity_df = df[[c for c in self.identity_columns if c in df.columns]].copy()

        work = df.drop(columns=[c for c in self.identity_columns if c in df.columns], errors="ignore")
        if self.label_column and self.label_column in work.columns:
            work = work.drop(columns=[self.label_column])

        work = self._replace_inf_with_nan(work)
        work = self._handle_nan_transform(work)
        work = self._onehot_encode_transform(work)

        for col in self.feature_columns_:
            if col not in work.columns:
                work[col] = 0.0
        work = work[self.feature_columns_]

        scaled = self.scaler_.transform(work.values.astype(np.float32))
        scaled_df = pd.DataFrame(scaled, columns=self.feature_columns_, index=work.index)
        del work
        return {"X": scaled_df, "identity": identity_df.loc[scaled_df.index]}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[processing] Saved fitted FeatureProcessor -> {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)

    @staticmethod
    def _replace_inf_with_nan(df):
        return df.replace([np.inf, -np.inf], np.nan)

    def _handle_nan_fit(self, df):
        if self.nan_strategy == "drop":
            before = len(df)
            df = df.dropna()
            dropped = before - len(df)
            if dropped:
                print(f"[processing] Dropped {dropped} rows containing NaN/Inf")
            return df

        for col in df.columns:
            if df[col].isna().any():
                if self.nan_strategy == "median":
                    fill = df[col].median()
                elif self.nan_strategy == "mean":
                    fill = df[col].mean()
                else:
                    fill = 0.0
                fill = 0.0 if pd.isna(fill) else fill
                self.impute_values_[col] = fill
                df[col] = df[col].fillna(fill)
        return df

    def _handle_nan_transform(self, df):
        if self.nan_strategy == "drop":
            return df.dropna()
        for col, fill in self.impute_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill)
        return df.fillna(0.0)

    def _onehot_encode_fit(self, df):
        for col in self.onehot_columns:
            if col not in df.columns:
                continue
            categories = sorted(df[col].dropna().unique().tolist())
            self.onehot_categories_[col] = categories
            for cat in categories:
                df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            df = df.drop(columns=[col])
        return df

    def _onehot_encode_transform(self, df):
        for col, categories in self.onehot_categories_.items():
            if col in df.columns:
                for cat in categories:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                df = df.drop(columns=[col])
            else:
                for cat in categories:
                    if f"{col}_{cat}" not in df.columns:
                        df[f"{col}_{cat}"] = 0
        return df

    def summary(self):
        info = {
            "fitted": self.is_fitted_,
            "n_features": len(self.feature_columns_),
            "feature_columns": self.feature_columns_,
            "zero_variance_dropped": self.zero_variance_columns_,
            "onehot_columns": list(self.onehot_categories_.keys()),
            "nan_strategy": self.nan_strategy,
            "label_classes": list(self.label_encoder_.classes_) if self.label_encoder_ is not None else None,
        }
        return json.dumps(info, indent=2, default=str)
