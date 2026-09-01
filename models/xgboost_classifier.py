#!/usr/bin/env python3
"""
Verit NIDS - Known-Attack Classifier (XGBoost)
------------------------------------------------------------------
Supervised multi-class classifier trained on labeled flow features
(BENIGN + every known attack type in your training data, e.g. the 15
CICIDS2017 classes). Confident on attack types it has seen before;
by design it has nothing useful to say about traffic patterns it has
never seen -- that's what the autoencoder half of the hybrid system
is for.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class XGBoostAttackClassifier:
    def __init__(self, label_classes, params=None):
        """
        label_classes: ordered list of class name strings, where the
            position in the list is the integer label used during
            training (i.e. LabelEncoder.classes_ from FeatureProcessor).
        """
        self.label_classes = list(label_classes)
        self.num_class = len(self.label_classes)

        default_params = dict(
            objective="multi:softprob",
            num_class=self.num_class,
            eval_metric="mlogloss",
            n_estimators=400,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = XGBClassifier(**self.params)
        self.feature_columns_ = None
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train, X_val=None, y_val=None,
            test_size=0.15, early_stopping_rounds=25, verbose=False):
        """If X_val/y_val are provided, they're used directly for early
        stopping (the proper approach when you already have a dedicated
        validation split). Otherwise, `test_size` of X_train is carved off
        internally -- kept for convenience/backward compatibility, but
        prefer passing an explicit validation set from your train/val/test
        split so the same rows are never used for both fitting and
        validation in ways that leak across your evaluation splits."""
        self.feature_columns_ = list(X_train.columns)

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=test_size, stratify=y_train, random_state=42
            )

        try:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
        except TypeError:
            # older xgboost versions take early_stopping_rounds in .fit() directly
            self.model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds, verbose=verbose,
            )

        self.is_fitted_ = True

        y_pred = self.model.predict(X_val)
        report = classification_report(
            y_val, y_pred, target_names=self.label_classes, zero_division=0
        )
        cm = confusion_matrix(y_val, y_pred, labels=list(range(self.num_class)))
        print("[xgboost] Validation classification report:")
        print(report)
        return {"classification_report": report, "confusion_matrix": cm}

    def evaluate(self, X: pd.DataFrame, y, dataset_name="test"):
        """Full evaluation on an arbitrary held-out split (e.g. your final
        test set). Returns predictions, probabilities, per-class metrics,
        confusion matrix, and per-class ROC/AUC (one-vs-rest)."""
        self._check_fitted()
        X = self._align_columns(X)
        y = np.asarray(y)

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        report_dict = classification_report(
            y, y_pred, target_names=self.label_classes, output_dict=True, zero_division=0
        )
        report_str = classification_report(
            y, y_pred, target_names=self.label_classes, zero_division=0
        )
        cm = confusion_matrix(y, y_pred, labels=list(range(self.num_class)))

        print(f"[xgboost] {dataset_name} classification report:")
        print(report_str)

        return {
            "y_true": y,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "classification_report_dict": report_dict,
            "classification_report_str": report_str,
            "confusion_matrix": cm,
        }

    def predict_top(self, X: pd.DataFrame):
        """Returns (predicted_label_names, confidence) arrays for each row --
        the single most likely class and its softmax probability."""
        self._check_fitted()
        X = self._align_columns(X)
        proba = self.model.predict_proba(X)
        top_idx = np.argmax(proba, axis=1)
        top_conf = proba[np.arange(len(proba)), top_idx]
        top_labels = np.array(self.label_classes)[top_idx]
        return top_labels, top_conf

    def predict_proba(self, X: pd.DataFrame):
        self._check_fitted()
        X = self._align_columns(X)
        return self.model.predict_proba(X)

    def _align_columns(self, X: pd.DataFrame):
        missing = [c for c in self.feature_columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} expected feature columns, e.g. {missing[:5]}")
        return X[self.feature_columns_]

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("XGBoostAttackClassifier must be fit or loaded before predicting.")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[xgboost] Saved model -> {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)
