#!/usr/bin/env python3
"""
Verit NIDS - Hybrid Detection Engine
------------------------------------------------------------------
Combines the two trained models into one verdict per flow:

    1. XGBoost gets first say. If it predicts a known attack class with
       confidence >= xgb_confidence_threshold, that's the verdict --
       it's a known, previously-seen attack pattern.
    2. Otherwise (XGBoost says BENIGN, or isn't confident), the
       autoencoder gets a look. If the flow reconstructs poorly (error
       above its threshold, calibrated on benign validation data), it's
       flagged ZERO_DAY_SUSPECTED -- something that doesn't look like
       normal traffic, but doesn't match any known attack signature
       either.
    3. Otherwise: BENIGN.

This is exactly the point of pairing a supervised and an unsupervised
model -- XGBoost is precise but blind to anything it wasn't trained on;
the autoencoder is comprehensive (trained only on "normal") but noisier
and unable to name what it's seeing.
"""

import time

import numpy as np
import pandas as pd
from scapy.all import sniff

from processing import PacketCleaner, FlowExtractor, FeatureProcessor
from models.xgboost_classifier import XGBoostAttackClassifier
from models.autoencoder import AutoencoderAnomalyDetector

# columns carried through from the raw (pre-scaling) flow row into alerts,
# when present -- gives the admin actual context, not just a verdict
_ALERT_CONTEXT_COLUMNS = [
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes", "total_bwd_bytes",
]


class HybridNIDS:
    def __init__(self, processor: FeatureProcessor, xgb_model: XGBoostAttackClassifier,
                 autoencoder: AutoencoderAnomalyDetector = None, benign_label="BENIGN",
                 xgb_confidence_threshold=0.6, high_confidence_threshold=0.9,
                 alert_manager=None):
        self.processor = processor
        self.xgb = xgb_model
        self.ae = autoencoder
        self.benign_label = benign_label
        self.xgb_confidence_threshold = xgb_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.alerts = alert_manager
        self._warned_missing_columns = False

    # -- core combiner -------------------------------------------------------

    def predict_batch(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """raw_df: UNSCALED flow rows (identity columns + raw feature values),
        exactly what FlowExtractor.to_dataframe()/drain_completed() produces.
        Applies the fitted FeatureProcessor internally, so callers never
        need to scale/encode data themselves."""
        if raw_df.empty:
            return pd.DataFrame()

        self._warn_if_schema_mismatch(raw_df)

        result = self.processor.transform(raw_df)
        X, identity = result["X"], result["identity"]
        if X.empty:
            return pd.DataFrame()

        xgb_labels, xgb_conf = self.xgb.predict_top(X)

        if self.ae is not None:
            ae_errors = self.ae.reconstruction_error(X)
            ae_anomaly = ae_errors > self.ae.threshold_
        else:
            ae_errors = np.full(len(X), np.nan)
            ae_anomaly = np.zeros(len(X), dtype=bool)

        verdicts, sources, scores, severities = [], [], [], []
        for label, conf, anomaly, err in zip(xgb_labels, xgb_conf, ae_anomaly, ae_errors):
            if label != self.benign_label and conf >= self.xgb_confidence_threshold:
                verdicts.append(label)
                sources.append("xgboost")
                scores.append(float(conf))
                severities.append("CRITICAL" if conf >= self.high_confidence_threshold else "WARNING")
            elif anomaly:
                verdicts.append("ZERO_DAY_SUSPECTED")
                sources.append("autoencoder")
                scores.append(float(err))
                severities.append("WARNING")
            else:
                verdicts.append(self.benign_label)
                sources.append("xgboost")
                scores.append(float(conf))
                severities.append("INFO")

        out = identity.reset_index(drop=True).copy()
        out["verdict"] = verdicts
        out["source"] = sources
        out["score"] = scores
        out["severity"] = severities
        out["xgb_predicted_label"] = xgb_labels
        out["xgb_confidence"] = xgb_conf
        out["ae_reconstruction_error"] = ae_errors

        context_cols = [c for c in _ALERT_CONTEXT_COLUMNS if c in raw_df.columns]
        if context_cols:
            context_df = raw_df.loc[identity.index, context_cols].reset_index(drop=True)
            out = pd.concat([out, context_df], axis=1)

        return out

    def process_and_alert(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        results = self.predict_batch(raw_df)
        if results.empty or self.alerts is None:
            return results

        for _, row in results.iterrows():
            if row["verdict"] == self.benign_label:
                continue  # INFO-level benign rows are noisy; skip unless you want full audit logging of every flow
            flow = row.to_dict()
            self.alerts.alert(
                flow=flow, verdict=row["verdict"], source=row["source"],
                severity=row["severity"], score=row["score"],
            )
        return results

    def _warn_if_schema_mismatch(self, raw_df):
        if self._warned_missing_columns:
            return
        expected = set(self.processor.feature_columns_)
        # account for the onehot-expanded protocol_* columns, which won't
        # exist in raw_df under those exact names
        raw_cols = set(raw_df.columns)
        onehot_bases = set(self.processor.onehot_categories_.keys())
        missing = [
            c for c in expected
            if c not in raw_cols and not any(c.startswith(f"{b}_") for b in onehot_bases)
        ]
        if missing:
            print(f"[hybrid_detect] WARNING: {len(missing)}/{len(expected)} features the model "
                  f"was trained on are missing from this input and will be zero-filled "
                  f"(e.g. {missing[:5]}). This will degrade detection accuracy -- the live "
                  f"capture pipeline's feature set doesn't yet match what the model was "
                  f"trained on. Resolve this before trusting live results.")
            self._warned_missing_columns = True

    # -- live capture loop -------------------------------------------------

    def run_live(self, interface, local_ips=None, bpf_filter=None, idle_timeout=120,
                 flush_interval=10, validate_checksums=True, checksum_direction="inbound",
                 verbose=True):
        """Continuously captures on `interface`, builds flows, and runs them
        through the hybrid detector + alert manager every `flush_interval`
        seconds. Blocks until Ctrl+C."""
        cleaner = PacketCleaner(
            validate_checksums=validate_checksums,
            validate_checksum_direction=checksum_direction,
            local_ips=local_ips,
        )
        extractor = FlowExtractor(idle_timeout=idle_timeout)

        print(f"[*] Live detection starting on interface '{interface}' "
              f"(flushing flows every {flush_interval}s, idle_timeout={idle_timeout}s)")
        print("[*] Press Ctrl+C to stop.\n")

        try:
            while True:
                packets = sniff(iface=interface, timeout=flush_interval, filter=bpf_filter, store=True)
                if packets:
                    cleaned = list(cleaner.clean(packets))
                    extractor.process(cleaned)

                extractor.sweep_idle_flows()
                new_flows = extractor.drain_completed()

                if not new_flows.empty:
                    results = self.process_and_alert(new_flows)
                    if verbose:
                        self._print_live_summary(results)
                elif verbose:
                    print(f"[{time.strftime('%H:%M:%S')}] No completed flows this interval "
                          f"({cleaner.stats.kept} packets kept so far).")

        except KeyboardInterrupt:
            print("\n[*] Stopping -- flushing remaining active flows...")
            extractor.flush_all()
            remaining = extractor.drain_completed()
            if not remaining.empty:
                results = self.process_and_alert(remaining)
                self._print_live_summary(results)
            print(cleaner.stats.summary())
            print("[*] Live detection stopped.")

    @staticmethod
    def _print_live_summary(results: pd.DataFrame):
        counts = results["verdict"].value_counts()
        ts = time.strftime("%H:%M:%S")
        summary = ", ".join(f"{v}: {c}" for v, c in counts.items())
        print(f"[{ts}] {len(results)} flow(s) completed -- {summary}")
        alerts = results[results["severity"] != "INFO"]
        for _, row in alerts.iterrows():
            print(f"    -> {row['severity']:8s} {row['verdict']:20s} "
                  f"{row.get('src_ip', '?')}:{row.get('src_port', '?')} -> "
                  f"{row.get('dst_ip', '?')}:{row.get('dst_port', '?')}  "
                  f"(source={row['source']}, score={row['score']:.4f})")
