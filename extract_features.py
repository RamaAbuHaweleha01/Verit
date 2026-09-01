#!/usr/bin/env python3
"""
Verit NIDS - Stage 2: Full Feature Extraction Pipeline
------------------------------------------------------------------
Two input modes:

  A) RAW PCAP mode  (your own live/offline captures)
     pcap -> pre-clean -> bidirectional flows -> flow features -> FeatureProcessor

  B) PRE-EXTRACTED CSV mode  (e.g. the official CICIDS2017 *_ISCX.csv files,
     already produced by CICFlowMeter -- packet cleaning/flow-building is
     skipped since that tool already did it)
     csv -> column normalization -> FeatureProcessor

Both modes end at the same FeatureProcessor stage: NaN/Inf handling,
identity drop, zero-variance drop, dedup, categorical + label encoding,
scaling -> model-ready CSV + saved FeatureProcessor.

Usage (pcap mode):
    python3 extract_features.py --pcap capture.pcap --out flows_train.csv \
        --label-column Label --fit-processor --processor-out models/processor.joblib

    python3 extract_features.py --pcap new_traffic.pcap --out flows_live.csv \
        --processor-in models/processor.joblib

    python3 extract_features.py --pcap-dir captures/ --out flows_all.csv --fit-processor

Usage (CICIDS2017 / pre-extracted CSV mode):
    # single combined file
    python3 extract_features.py --csv Database/all_data_combined.csv \
        --out features/train.csv --label-column Label \
        --fit-processor --processor-out models/processor.joblib

    # or a directory of the per-day ISCX files (reads + concatenates all of them)
    python3 extract_features.py --csv-glob "Database/*_ISCX.csv" \
        --out features/train.csv --label-column Label \
        --fit-processor --processor-out models/processor.joblib
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.utils import PcapReader

from processing import PacketCleaner, FlowExtractor, FeatureProcessor

_CICIDS_COLUMN_ALIASES = {
    "flow id": "flow_id",
    "source ip": "src_ip",
    "src ip": "src_ip",
    "source port": "src_port",
    "src port": "src_port",
    "destination ip": "dst_ip",
    "dst ip": "dst_ip",
    "destination port": "dst_port",
    "dst port": "dst_port",
    "protocol": "protocol",
    "timestamp": "timestamp",
    "label": "Label",
}


def _normalize_cicids_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in _CICIDS_COLUMN_ALIASES:
            rename_map[col] = _CICIDS_COLUMN_ALIASES[key]
    df = df.rename(columns=rename_map)
    return df


def load_precomputed_csv(paths, label_column="Label", chunksize=200_000):
    chunks = []
    for path in paths:
        print(f"[*] Reading {path} (chunked, {chunksize} rows/chunk) ...")
        n_rows = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = _normalize_cicids_columns(chunk)

            for col in chunk.columns:
                if col == label_column or col in ("flow_id", "src_ip", "dst_ip", "timestamp"):
                    continue
                if chunk[col].dtype == object:
                    chunk[col] = chunk[col].replace(
                        {"Infinity": np.inf, "-Infinity": -np.inf, "NaN": np.nan}
                    )
                    try:
                        chunk[col] = pd.to_numeric(chunk[col])
                    except (ValueError, TypeError):
                        pass
                if pd.api.types.is_numeric_dtype(chunk[col]):
                    chunk[col] = chunk[col].astype("float32")

            n_rows += len(chunk)
            chunks.append(chunk)
        print(f"    -> {n_rows} rows")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"[*] Combined CSV shape: {df.shape}")
    return df


def iter_pcap_files(pcap, pcap_dir):
    if pcap:
        yield Path(pcap)
    if pcap_dir:
        for p in sorted(Path(pcap_dir).glob("*.pcap*")):
            yield p


def build_flows_dataframe(pcap_paths, idle_timeout, local_ips, validate_checksums,
                           checksum_direction, drop_retransmissions, drop_out_of_order,
                           drop_l2_noise):
    extractor = FlowExtractor(idle_timeout=idle_timeout)

    for pcap_path in pcap_paths:
        print(f"[*] Reading {pcap_path} ...")
        cleaner = PacketCleaner(
            validate_checksums=validate_checksums,
            validate_checksum_direction=checksum_direction,
            local_ips=local_ips,
            drop_retransmissions=drop_retransmissions,
            drop_out_of_order=drop_out_of_order,
            drop_l2_noise=drop_l2_noise,
        )
        with PcapReader(str(pcap_path)) as reader:
            extractor.process(cleaner.clean(reader))
        print(cleaner.stats.summary())

    extractor.flush_all()
    return extractor.to_dataframe()


def parse_args():
    p = argparse.ArgumentParser(description="Verit NIDS - flow feature extraction pipeline")
    p.add_argument("--pcap", type=str, default=None, help="Path to a single pcap file")
    p.add_argument("--pcap-dir", type=str, default=None, help="Directory of pcap files")
    p.add_argument("--csv", type=str, default=None,
                    help="Path to a single pre-extracted flow CSV (e.g. CICIDS2017 all_data_combined.csv)")
    p.add_argument("--csv-glob", type=str, default=None,
                    help="Glob pattern matching multiple pre-extracted flow CSVs, e.g. 'Database/*_ISCX.csv'")
    p.add_argument("--csv-chunksize", type=int, default=200_000,
                    help="Rows per chunk when streaming large CSVs (lower this if you hit memory pressure)")
    p.add_argument("--out", type=str, required=True, help="Output CSV path for the model-ready feature table")
    p.add_argument("--raw-out", type=str, default=None,
                    help="Optional: also dump the raw (pre-scaling) flow feature table here")

    p.add_argument("--idle-timeout", type=float, default=120.0, help="Flow idle timeout in seconds")
    p.add_argument("--local-ips", type=str, default=None,
                    help="Comma-separated local host IPs, used to skip checksum validation on outbound packets")
    p.add_argument("--no-checksum-validation", action="store_true", help="Disable checksum validation entirely")
    p.add_argument("--checksum-direction", choices=["inbound", "both", "none"], default="inbound")
    p.add_argument("--keep-retransmissions", action="store_true")
    p.add_argument("--keep-out-of-order", action="store_true")
    p.add_argument("--keep-l2-noise", action="store_true")

    p.add_argument("--label-column", type=str, default=None,
                    help="Name of a label column to attach/encode (e.g. 'Label' with values like BENIGN/DoS/PortScan). "
                         "If your pcap has no labels, omit this.")
    p.add_argument("--fit-processor", action="store_true",
                    help="Fit a new FeatureProcessor on this data (use for training data)")
    p.add_argument("--processor-out", type=str, default=None,
                    help="Where to save the fitted FeatureProcessor (joblib)")
    p.add_argument("--processor-in", type=str, default=None,
                    help="Path to a previously fitted FeatureProcessor to reuse (use for inference/live data)")
    p.add_argument("--nan-strategy", choices=["median", "mean", "zero", "drop"], default="median")

    return p.parse_args()


def main():
    args = parse_args()

    input_modes = [bool(args.pcap or args.pcap_dir), bool(args.csv or args.csv_glob)]
    if sum(input_modes) != 1:
        print("[!] Provide exactly one input mode: (--pcap / --pcap-dir) OR (--csv / --csv-glob)",
              file=sys.stderr)
        sys.exit(1)

    if args.csv or args.csv_glob:
        csv_paths = [args.csv] if args.csv else sorted(glob.glob(args.csv_glob))
        if not csv_paths:
            print("[!] No CSV files matched.", file=sys.stderr)
            sys.exit(1)
        if not args.label_column:
            args.label_column = "Label"
        flows_df = load_precomputed_csv(csv_paths, label_column=args.label_column,
                                         chunksize=args.csv_chunksize)
    else:
        local_ips = args.local_ips.split(",") if args.local_ips else None
        pcap_paths = list(iter_pcap_files(args.pcap, args.pcap_dir))
        if not pcap_paths:
            print("[!] No pcap files found.", file=sys.stderr)
            sys.exit(1)

        flows_df = build_flows_dataframe(
            pcap_paths,
            idle_timeout=args.idle_timeout,
            local_ips=local_ips,
            validate_checksums=not args.no_checksum_validation,
            checksum_direction=args.checksum_direction,
            drop_retransmissions=not args.keep_retransmissions,
            drop_out_of_order=not args.keep_out_of_order,
            drop_l2_noise=not args.keep_l2_noise,
        )

    print(f"[*] Built {len(flows_df)} flows total.")
    if flows_df.empty:
        print("[!] No flows extracted -- nothing to write.")
        return

    if args.raw_out:
        Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
        flows_df.to_csv(args.raw_out, index=False)
        print(f"[*] Raw (pre-scaling) flow features -> {args.raw_out}")

    if args.processor_in:
        processor = FeatureProcessor.load(args.processor_in)
        result = processor.transform(flows_df)
    elif args.fit_processor:
        processor = FeatureProcessor(
            nan_strategy=args.nan_strategy,
            label_column=args.label_column,
        )
        result = processor.fit_transform(flows_df)
        if args.processor_out:
            processor.save(args.processor_out)
    else:
        print("[!] Specify either --fit-processor (training data) or --processor-in (inference data).",
              file=sys.stderr)
        sys.exit(1)

    X = result["X"]
    out_df = X.copy()
    if "y" in result:
        out_df["label_encoded"] = result["y"].values
        print(f"[*] Label classes: {result['label_classes']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[*] Model-ready feature table ({out_df.shape[0]} rows x {out_df.shape[1]} cols) -> {args.out}")

    if args.fit_processor:
        print("\n[*] Processor summary:")
        print(processor.summary())


if __name__ == "__main__":
    main()
