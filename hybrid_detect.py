#!/usr/bin/env python3
"""
Verit NIDS - Hybrid Detection Engine (Main Entry Point)
------------------------------------------------------------------
Loads the trained processor + XGBoost + autoencoder artifacts and runs
detection either against a batch of already-extracted flow features, or
continuously against live traffic on a network interface -- alerting
via the local log + Telegram in either case.

Usage (batch mode -- e.g. re-scoring a captured pcap you already ran
through extract_features.py --raw-out):
    python3 hybrid_detect.py --mode batch --input flows_raw.csv \
        --artifacts-dir models/artifacts --out detections.csv

Usage (live mode -- continuous detection on a NIC):
    sudo python3 hybrid_detect.py --mode live --interface enp0s3 \
        --artifacts-dir models/artifacts

    # or let it auto-detect the first active interface:
    sudo python3 hybrid_detect.py --mode live --artifacts-dir models/artifacts
"""

import argparse
import sys
from pathlib import Path

from models.dependency_manager import ensure_dependencies
ensure_dependencies()

import pandas as pd

from processing import FeatureProcessor
from models.xgboost_classifier import XGBoostAttackClassifier
from models.autoencoder import AutoencoderAnomalyDetector
from models.hybrid_detector import HybridNIDS
from alerting.alert_manager import AlertManager
from alerting.config import AlertConfig
from nic_capture import get_active_interfaces, require_root_or_cap
from extract_features import _normalize_cicids_columns


def load_artifacts(artifacts_dir):
    artifacts_dir = Path(artifacts_dir)
    print(f"[*] Loading artifacts from {artifacts_dir} ...")

    processor = FeatureProcessor.load(artifacts_dir / "processor.joblib")
    xgb_model = XGBoostAttackClassifier.load(artifacts_dir / "xgboost_model.joblib")

    ae_dir = artifacts_dir / "autoencoder"
    autoencoder = AutoencoderAnomalyDetector.load(ae_dir) if ae_dir.exists() else None
    if autoencoder is None:
        print("[!] No autoencoder found -- running in XGBoost-only mode (no zero-day detection).")

    print(f"[*] Loaded. XGBoost classes: {xgb_model.label_classes}")
    if autoencoder:
        print(f"[*] Autoencoder threshold: {autoencoder.threshold_:.6f}")

    return processor, xgb_model, autoencoder


def parse_args():
    p = argparse.ArgumentParser(description="Verit NIDS - hybrid detection engine")
    p.add_argument("--mode", choices=["batch", "live"], required=True)
    p.add_argument("--artifacts-dir", type=str, default="models/artifacts")

    # batch mode
    p.add_argument("--input", type=str, help="[batch] Path to a raw (unscaled) flow feature CSV")
    p.add_argument("--out", type=str, default=None, help="[batch] Where to write the detection results CSV")

    # live mode
    p.add_argument("--interface", type=str, default=None,
                    help="[live] NIC to capture on (default: auto-detect first active NIC)")
    p.add_argument("--local-ips", type=str, default=None,
                    help="[live] Comma-separated local IPs for this host (default: auto-detected from --interface)")
    p.add_argument("--bpf", type=str, default=None, help="[live] BPF filter, e.g. 'tcp or udp'")
    p.add_argument("--flush-interval", type=float, default=10.0,
                    help="[live] Seconds between flow-completion checks + detection passes")
    p.add_argument("--idle-timeout", type=float, default=120.0)
    p.add_argument("--no-checksum-validation", action="store_true")
    p.add_argument("--checksum-direction", choices=["inbound", "both", "none"], default="inbound")

    # detection thresholds
    p.add_argument("--benign-label", type=str, default="BENIGN")
    p.add_argument("--xgb-confidence-threshold", type=float, default=0.6,
                    help="Minimum XGBoost confidence to trust a known-attack verdict; below this, "
                         "the autoencoder gets the final say instead")
    p.add_argument("--high-confidence-threshold", type=float, default=0.9,
                    help="XGBoost confidence above which a known-attack alert is CRITICAL rather than WARNING")

    # alerting
    p.add_argument("--no-alerts", action="store_true", help="Disable Telegram alerting entirely (local log only)")

    return p.parse_args()


def main():
    args = parse_args()

    processor, xgb_model, autoencoder = load_artifacts(args.artifacts_dir)

    alert_manager = None
    if not args.no_alerts:
        alert_manager = AlertManager(AlertConfig())

    detector = HybridNIDS(
        processor=processor, xgb_model=xgb_model, autoencoder=autoencoder,
        benign_label=args.benign_label,
        xgb_confidence_threshold=args.xgb_confidence_threshold,
        high_confidence_threshold=args.high_confidence_threshold,
        alert_manager=alert_manager,
    )

    try:
        if args.mode == "batch":
            run_batch(detector, args)
        else:
            run_live(detector, args)
    finally:
        if alert_manager:
            alert_manager.close()


def run_batch(detector, args):
    if not args.input:
        print("[!] --input is required for --mode batch", file=sys.stderr)
        sys.exit(1)

    raw_df = pd.read_csv(args.input)
    # Column names go through the same whitespace-stripping / identity-
    # column normalization used at training time (extract_features.py's
    # CSV loader), so a CICIDS-formatted input lines up with what the
    # processor was actually fit on. Harmless no-op for already-clean
    # column names (e.g. from your own pcap pipeline).
    raw_df = _normalize_cicids_columns(raw_df)
    print(f"[*] Loaded {len(raw_df)} raw flow rows from {args.input}")

    results = detector.process_and_alert(raw_df)
    if results.empty:
        print("[!] No flows survived processing -- nothing to report.")
        return

    counts = results["verdict"].value_counts()
    print("\n[*] Detection summary:")
    for verdict, count in counts.items():
        print(f"    {verdict}: {count}")

    out_path = args.out or (Path(args.input).with_suffix("").as_posix() + "_detections.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\n[*] Full results -> {out_path}")


def run_live(detector, args):
    require_root_or_cap()

    interface = args.interface
    if not interface:
        active = get_active_interfaces()
        if not active:
            print("[!] No active interfaces found and none specified with --interface.", file=sys.stderr)
            sys.exit(1)
        interface = active[0]
        print(f"[*] No --interface given, auto-selected: {interface}")

    local_ips = args.local_ips.split(",") if args.local_ips else _infer_local_ips(interface)

    detector.run_live(
        interface=interface,
        local_ips=local_ips,
        bpf_filter=args.bpf,
        idle_timeout=args.idle_timeout,
        flush_interval=args.flush_interval,
        validate_checksums=not args.no_checksum_validation,
        checksum_direction=args.checksum_direction,
    )


def _infer_local_ips(interface):
    import psutil
    addrs = psutil.net_if_addrs().get(interface, [])
    ips = [a.address for a in addrs if a.family.name == "AF_INET"]
    if ips:
        print(f"[*] Auto-detected local IP(s) for {interface}: {ips}")
    return ips or None


if __name__ == "__main__":
    main()
