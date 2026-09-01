#!/usr/bin/env python3
"""
Verit NIDS - Zero-Day Anomaly Detector (Autoencoder)
------------------------------------------------------------------
Unsupervised reconstruction-based anomaly detector. Trained ONLY on
BENIGN flows, so it learns what "normal" traffic looks like in the
scaled feature space. At inference time, a flow that reconstructs
poorly (high MSE between input and the autoencoder's output) doesn't
resemble anything the model saw as normal -- which includes attack
types never seen during training (the whole point of pairing this
with the supervised XGBoost classifier, which can only recognize
attacks it was explicitly trained on).

The anomaly threshold is learned from the reconstruction-error
distribution on held-out BENIGN validation data (not from any attack
samples), so it stays a well-calibrated "this doesn't look normal"
signal rather than something quietly tuned to the mix of attacks in
one particular training set.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AutoencoderAnomalyDetector:
    def __init__(self, input_dim=None, encoding_dim=None, hidden_layers=None,
                 threshold_percentile=99.0):
        """
        input_dim: number of input features (set automatically on fit() if omitted)
        encoding_dim: size of the bottleneck layer (default: max(4, input_dim // 8))
        hidden_layers: list of hidden layer sizes for the encoder (decoder mirrors it),
            e.g. [64, 32] for a 3-hop-deep encoder down to the bottleneck
        threshold_percentile: percentile of the BENIGN validation reconstruction-error
            distribution used as the anomaly cutoff (99.0 = flag the worst 1% of
            normal-looking traffic as the boundary; raise this to reduce false positives,
            lower it to catch subtler anomalies at the cost of more noise)
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.threshold_percentile = threshold_percentile

        self.model = None
        self.threshold_ = None
        self.feature_columns_ = None
        self.is_fitted_ = False
        self.training_history_ = None

    def _build(self, input_dim):
        hidden = self.hidden_layers or self._default_hidden_layers(input_dim)
        encoding_dim = self.encoding_dim or max(4, input_dim // 8)

        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for units in hidden:
            x = layers.Dense(units, activation="relu")(x)
        bottleneck = layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)
        x = bottleneck
        for units in reversed(hidden):
            x = layers.Dense(units, activation="relu")(x)
        # linear output -- scaled features (StandardScaler) can be negative,
        # so a sigmoid/relu output activation would clip the target range.
        outputs = layers.Dense(input_dim, activation="linear")(x)

        model = keras.Model(inputs, outputs, name="verit_autoencoder")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
        return model

    @staticmethod
    def _default_hidden_layers(input_dim):
        # a modest 2-hop encoder is plenty for ~40-80 flow features;
        # deeper nets tend to just overfit "normal" too tightly on this
        # kind of tabular data.
        h1 = max(16, input_dim // 2)
        h2 = max(8, input_dim // 4)
        return [h1, h2]

    def fit(self, X_benign: pd.DataFrame, X_val_benign=None, validation_split=0.15, epochs=100,
            batch_size=256, patience=8, verbose=1):
        """If X_val_benign is provided (recommended: benign rows from your
        dedicated validation split), it's used both for Keras's validation
        monitoring and for threshold calibration. Otherwise a slice of
        X_benign is carved off internally via `validation_split`."""
        self.feature_columns_ = list(X_benign.columns)
        input_dim = X_benign.shape[1]
        self.input_dim = input_dim
        self.model = self._build(input_dim)

        X_arr = X_benign.values.astype(np.float32)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        if X_val_benign is not None:
            X_val_arr = self._align_columns(X_val_benign).values.astype(np.float32)
            history = self.model.fit(
                X_arr, X_arr,
                validation_data=(X_val_arr, X_val_arr),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[early_stop],
                verbose=verbose,
            )
        else:
            history = self.model.fit(
                X_arr, X_arr,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[early_stop],
                verbose=verbose,
            )
            n_val = int(len(X_arr) * validation_split)
            X_val_arr = X_arr[-n_val:] if n_val > 0 else X_arr

        self.training_history_ = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        val_errors = self._reconstruction_error_array(X_val_arr)
        self.threshold_ = float(np.percentile(val_errors, self.threshold_percentile))

        self.is_fitted_ = True
        print(f"[autoencoder] Trained. Anomaly threshold (p{self.threshold_percentile} of benign "
              f"validation reconstruction error) = {self.threshold_:.6f}")
        return {
            "history": self.training_history_,
            "threshold": self.threshold_,
            "val_error_mean": float(val_errors.mean()),
            "val_error_std": float(val_errors.std()),
        }

    def _reconstruction_error_array(self, X_arr):
        reconstructed = self.model.predict(X_arr, verbose=0)
        return np.mean(np.square(X_arr - reconstructed), axis=1)

    def reconstruction_error(self, X: pd.DataFrame):
        self._check_fitted()
        X = self._align_columns(X)
        return self._reconstruction_error_array(X.values.astype(np.float32))

    def predict_anomaly(self, X: pd.DataFrame, threshold=None):
        threshold = threshold if threshold is not None else self.threshold_
        errors = self.reconstruction_error(X)
        return errors > threshold, errors

    def evaluate_separation(self, X_benign: pd.DataFrame, X_attack: pd.DataFrame):
        """Optional sanity check: how well does reconstruction error separate
        held-out benign traffic from KNOWN attack traffic? (Not a substitute
        for true zero-day validation, since these attack types may overlap
        with training data elsewhere in the pipeline -- but a large gap here
        is a good sign the autoencoder learned a meaningful notion of 'normal'.)"""
        benign_errors = self.reconstruction_error(X_benign)
        attack_errors = self.reconstruction_error(X_attack)
        detected = (attack_errors > self.threshold_).mean()
        false_positive_rate = (benign_errors > self.threshold_).mean()
        print(f"[autoencoder] Separation check -- benign mean error: {benign_errors.mean():.6f}, "
              f"attack mean error: {attack_errors.mean():.6f}")
        print(f"[autoencoder] At current threshold: {detected*100:.1f}% of attack samples flagged, "
              f"{false_positive_rate*100:.1f}% of benign samples false-flagged")
        return {
            "benign_error_mean": float(benign_errors.mean()),
            "attack_error_mean": float(attack_errors.mean()),
            "attack_detection_rate": float(detected),
            "benign_false_positive_rate": float(false_positive_rate),
        }

    def _align_columns(self, X: pd.DataFrame):
        missing = [c for c in self.feature_columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} expected feature columns, e.g. {missing[:5]}")
        return X[self.feature_columns_]

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("AutoencoderAnomalyDetector must be fit or loaded before predicting.")

    def save(self, dir_path):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.model.save(dir_path / "autoencoder.keras")
        meta = {
            "threshold_": self.threshold_,
            "feature_columns_": self.feature_columns_,
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_layers": self.hidden_layers,
            "threshold_percentile": self.threshold_percentile,
            "training_history_": self.training_history_,
        }
        joblib.dump(meta, dir_path / "autoencoder_meta.joblib")
        with open(dir_path / "autoencoder_meta.json", "w") as f:
            json.dump({k: v for k, v in meta.items() if k != "training_history_"}, f, indent=2, default=str)
        print(f"[autoencoder] Saved model + metadata -> {dir_path}")

    @staticmethod
    def load(dir_path):
        dir_path = Path(dir_path)
        meta = joblib.load(dir_path / "autoencoder_meta.joblib")
        obj = AutoencoderAnomalyDetector(
            input_dim=meta["input_dim"],
            encoding_dim=meta["encoding_dim"],
            hidden_layers=meta["hidden_layers"],
            threshold_percentile=meta["threshold_percentile"],
        )
        obj.model = keras.models.load_model(dir_path / "autoencoder.keras")
        obj.threshold_ = meta["threshold_"]
        obj.feature_columns_ = meta["feature_columns_"]
        obj.training_history_ = meta.get("training_history_")
        obj.is_fitted_ = True
        return obj
