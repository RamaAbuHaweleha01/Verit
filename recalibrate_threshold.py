#!/usr/bin/env python3
"""
Verit NIDS - Autoencoder Threshold Recalibration
------------------------------------------------------------------
Your dashboard just showed a 79% ZERO_DAY_SUSPECTED rate over a 4+ hour
run. That is NOT a healthy detection rate -- it means the autoencoder's
anomaly threshold, calibrated against BENIGN traffic from CICIDS2017
(captured in 2017), is now firing on almost everything. This is a real
and well-documented form of model drift, not a bug: TLS 1.3, HTTP/2
connection multiplexing, modern TCP window scaling, background service
discovery (mDNS/SSDP), and CDN behavior all look statistically different
today than they did when that dataset was captured.

The fix is NOT to retrain the whole autoencoder network from scratch --
the network itself still encodes a reasonable notion of "normal traffic
shape". The fix is to recalibrate just the ANOMALY THRESHOLD against a
window of TODAY's actual benign traffic, using the existing trained
network to score it.

IMPORTANT: run this only during a period you're confident has NO attacks
running -- the recalibration assumes whatever it captures is
representative of normal traffic. Don't run this while testing hydra/
hping3/slowhttptest/etc. against the box.

Usage (live capture -- recommended, run during a quiet 10-15 minute window):
    sudo venv/bin/python3 recalibrate_threshold.py \
        --interface enp0s3,enp0s8 --duration 600 \
        --artifacts-dir models/artifacts --percentile 99.0

Usage (from an existing raw flow CSV you already know is clean benign traffic):
    python3 recalibrate_threshold.py --csv features/clean_capture_raw.csv \
        --artifacts-dir models/artifacts --percentile 99.0
"""

import argparse
import sys
import time
from pathlib import Path

from models.dependency_manager import ensure_dependencies
ensure_dependencies()

import numpy as np
import pandas as pd

from processing import PacketCleaner, FlowExtractor, FeatureProcessor
from models.autoencoder import AutoencoderAnomalyDetector
from nic_capture import get_active_interfaces, require_root_or_cap


def capture_flows(interfaces, duration, idle_timeout=120, local_ips=None, bpf_filter=None):
    from scapy.all import sniff

    cleaner = PacketCleaner(validate_checksums=True, validate_checksum_direction="inbound", local_ips=local_ips)
    extractor = FlowExtractor(idle_timeout=idle_timeout)

    print(f"[*] Capturing on {interfaces} for {duration}s to build a fresh benign baseline "
          f"(make sure nothing but normal traffic is happening right now)...")
    end_time = time.time() + duration
    while time.time() < end_time:
        window = max(1, min(15, end_time - time.time()))
        packets = sniff(iface=interfaces, timeout=window, filter=bpf_filter, store=True)
        if packets:
            extractor.process(cleaner.clean(packets))
        extractor.sweep_idle_flows()
        remaining = int(end_time - time.time())
        if remaining > 0 and remaining % 60 < window:
            print(f"    ... {remaining}s remaining, {cleaner.stats.kept} packets kept so far")

    extractor.flush_all()
    df = extractor.to_dataframe()
    print(cleaner.stats.summary())
    print(f"[*] Captured {len(df)} flows for recalibration.")
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Recalibrate the autoencoder's anomaly threshold against fresh traffic")
    p.add_argument("--interface", type=str, default=None,
                    help="Comma-separated NICs to capture on (default: all active NICs)")
    p.add_argument("--duration", type=int, default=600, help="Seconds to capture live (default: 600 = 10 min)")
    p.add_argument("--csv", type=str, default=None,
                    help="Alternative to live capture: a raw (unscaled) flow CSV already known to be "
                         "clean benign traffic (e.g. from extract_features.py --raw-out)")
    p.add_argument("--artifacts-dir", type=str, default="models/artifacts")
    p.add_argument("--percentile", type=float, default=99.0,
                    help="Percentile of the fresh reconstruction-error distribution to use as the "
                         "new threshold. Higher = fewer false alarms but less sensitive; lower = "
                         "the opposite. 99.0 matches how the original threshold was calibrated.")
    p.add_argument("--local-ips", type=str, default=None)
    p.add_argument("--bpf", type=str, default=None)
    p.add_argument("--dry-run", action="store_true",
                    help="Compute and print the new threshold but don't overwrite the saved autoencoder")
    return p.parse_args()


def main():
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    ae_dir = artifacts_dir / "autoencoder"
    processor = FeatureProcessor.load(artifacts_dir / "processor.joblib")
    autoencoder = AutoencoderAnomalyDetector.load(ae_dir)
    print(f"[*] Current threshold: {autoencoder.threshold_:.6f} (calibrated on 2017 CICIDS2017 benign validation data)")

    if args.csv:
        raw_df = pd.read_csv(args.csv)
        from extract_features import _normalize_cicids_columns
        raw_df = _normalize_cicids_columns(raw_df)
    else:
        require_root_or_cap()
        interfaces = args.interface.split(",") if args.interface else get_active_interfaces()
        if not interfaces:
            print("[!] No active interfaces found and none specified with --interface.", file=sys.stderr)
            sys.exit(1)
        local_ips = args.local_ips.split(",") if args.local_ips else None
        raw_df = capture_flows(interfaces, args.duration, local_ips=local_ips, bpf_filter=args.bpf)

    if raw_df.empty:
        print("[!] No flows captured/loaded -- nothing to recalibrate against.", file=sys.stderr)
        sys.exit(1)

    result = processor.transform(raw_df)
    X = result["X"]
    errors = autoencoder.reconstruction_error(X)

    print(f"\n[*] Fresh-traffic reconstruction error distribution ({len(errors)} flows):")
    print(f"    mean: {errors.mean():.6f}   median: {np.percentile(errors, 50):.6f}   "
          f"p95: {np.percentile(errors, 95):.6f}   p99: {np.percentile(errors, 99):.6f}   "
          f"max: {errors.max():.6f}")

    old_threshold = autoencoder.threshold_
    new_threshold = float(np.percentile(errors, args.percentile))
    flagged_by_old = (errors > old_threshold).mean() * 100
    flagged_by_new = (errors > new_threshold).mean() * 100

    print(f"\n[*] Old threshold ({old_threshold:.6f}) would flag {flagged_by_old:.1f}% of THIS fresh, "
          f"presumed-benign batch as anomalous.")
    print(f"[*] New threshold ({new_threshold:.6f}, p{args.percentile} of fresh data) flags "
          f"{flagged_by_new:.1f}% of this batch by construction.")

    if flagged_by_old < 5.0:
        print("\n[*] The old threshold actually looks reasonable against this batch (<5% flagged). "
              "If your earlier long-running test showed ~79% flagged, that traffic mix may have "
              "included something genuinely unusual, or this capture window was too different from "
              "that one. Consider a longer/more representative capture before trusting this result.")

    if args.dry_run:
        print("\n[*] --dry-run set: NOT overwriting the saved autoencoder. Re-run without --dry-run to apply.")
        return

    autoencoder.threshold_ = new_threshold
    autoencoder.save(ae_dir)
    print(f"\n[*] Recalibrated and saved -> {ae_dir}")
    print("[*] Restart hybrid_detect.py (or `sudo systemctl restart verit-nids`) to pick up the new threshold.")


if __name__ == "__main__":
    main()
