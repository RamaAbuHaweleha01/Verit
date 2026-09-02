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

Usage (live mode -- continuous detection on a NIC, with the live dashboard
at http://localhost:8080 by default):
    sudo python3 hybrid_detect.py --mode live --interface enp0s3,enp0s8 \
        --artifacts-dir models/artifacts

    # or let it auto-detect and monitor ALL active interfaces:
    sudo python3 hybrid_detect.py --mode live --artifacts-dir models/artifacts

    # disable the dashboard entirely:
    sudo python3 hybrid_detect.py --mode live --no-dashboard --artifacts-dir models/artifacts
"""

import argparse
import sys
import threading
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
    p.add_argument("--csv-chunksize", type=int, default=200_000,
                    help="[batch] Rows per chunk when streaming large input files (lower if you hit memory pressure)")

    # live mode
    p.add_argument("--interface", type=str, default=None,
                    help="[live] Comma-separated NIC(s) to capture on, e.g. 'enp0s3,enp0s8' "
                         "(default: auto-detect and monitor ALL active NICs)")
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

    # dashboard
    p.add_argument("--no-dashboard", action="store_true",
                    help="[live] Disable the live web dashboard (on by default in live mode)")
    p.add_argument("--dashboard-host", type=str, default="0.0.0.0",
                    help="[live] Dashboard bind address (default: all interfaces)")
    p.add_argument("--dashboard-port", type=int, default=8080, help="[live] Dashboard port")

    return p.parse_args()


def main():
    args = parse_args()

    processor, xgb_model, autoencoder = load_artifacts(args.artifacts_dir)

    alert_manager = None
    if not args.no_alerts:
        alert_manager = AlertManager(AlertConfig())

    event_bus = None
    if args.mode == "live" and not args.no_dashboard:
        from dashboard.event_bus import EventBus
        from dashboard.app import run_dashboard
        event_bus = EventBus()
        dashboard_thread = threading.Thread(
            target=run_dashboard,
            kwargs={"event_bus": event_bus, "host": args.dashboard_host, "port": args.dashboard_port},
            daemon=True,
        )
        dashboard_thread.start()

    detector = HybridNIDS(
        processor=processor, xgb_model=xgb_model, autoencoder=autoencoder,
        benign_label=args.benign_label,
        xgb_confidence_threshold=args.xgb_confidence_threshold,
        high_confidence_threshold=args.high_confidence_threshold,
        alert_manager=alert_manager,
        event_bus=event_bus,
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

    out_path = args.out or (Path(args.input).with_suffix("").as_posix() + "_detections.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    total_counts = {}
    n_total = 0
    first_chunk = True

    # Stream the file in bounded chunks rather than loading it whole --
    # on a multi-million-row file, holding the raw data AND the scaled
    # features AND both models' outputs in memory simultaneously (as a
    # single big pd.read_csv() forces) is what caused the earlier OOM
    # kill during training too. Peak memory here stays O(chunksize)
    # regardless of total file size.
    for raw_chunk in pd.read_csv(args.input, chunksize=args.csv_chunksize, low_memory=False):
        raw_chunk = _normalize_cicids_columns(raw_chunk)
        n_total += len(raw_chunk)

        results = detector.process_and_alert(raw_chunk)
        if results.empty:
            print(f"[*] Processed {n_total} rows so far (0 survived processing in this chunk)")
            continue

        for verdict, count in results["verdict"].value_counts().items():
            total_counts[verdict] = total_counts.get(verdict, 0) + int(count)

        results.to_csv(out_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        print(f"[*] Processed {n_total} rows so far -- {dict(total_counts)}")

    if not total_counts:
        print("[!] No flows survived processing -- nothing to report.")
        return

    print("\n[*] Detection summary:")
    for verdict, count in total_counts.items():
        print(f"    {verdict}: {count}")
    print(f"\n[*] Full results -> {out_path}")


def run_live(detector, args):
    require_root_or_cap()

    if args.interface:
        interfaces = [i.strip() for i in args.interface.split(",") if i.strip()]
    else:
        interfaces = get_active_interfaces()
        if not interfaces:
            print("[!] No active interfaces found and none specified with --interface.", file=sys.stderr)
            sys.exit(1)
        print(f"[*] No --interface given, auto-selected ALL active NICs: {interfaces}")

    local_ips = args.local_ips.split(",") if args.local_ips else _infer_local_ips(interfaces)

    detector.run_live(
        interface=interfaces,
        local_ips=local_ips,
        bpf_filter=args.bpf,
        idle_timeout=args.idle_timeout,
        flush_interval=args.flush_interval,
        validate_checksums=not args.no_checksum_validation,
        checksum_direction=args.checksum_direction,
    )


def _infer_local_ips(interfaces):
    import psutil
    all_addrs = psutil.net_if_addrs()
    ips = []
    for iface in interfaces:
        for a in all_addrs.get(iface, []):
            if a.family.name == "AF_INET":
                ips.append(a.address)
    if ips:
        print(f"[*] Auto-detected local IP(s) across {interfaces}: {ips}")
    return ips or None


if __name__ == "__main__":
    main()
