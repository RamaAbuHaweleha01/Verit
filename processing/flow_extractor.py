#!/usr/bin/env python3
"""
Verit NIDS - Bidirectional Flow Builder + Feature Extraction
------------------------------------------------------------------
Groups cleaned packets into bidirectional flows (5-tuple: src_ip, dst_ip,
src_port, dst_port, protocol) the same way CICFlowMeter does, then computes
the exact CICFlowMeter / CICIDS2017 feature set per flow.

IMPORTANT -- zero-duration flow handling:
A flow with only one packet (or where every packet arrives at the exact
same timestamp) has a duration of 0. CICFlowMeter's real behavior for
rate features (Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s) in that
case is to report literal Infinity -- NOT a huge-but-finite number from
dividing by a tiny epsilon. This matters a great deal: FeatureProcessor
converts that Infinity to NaN and imputes it with the TRAINING median (a
normal value), exactly matching how these rare edge cases were handled
when the model was trained. Flooring the duration at a tiny epsilon
instead produces a real, finite, but astronomically large number (e.g.
54 bytes / 0.000001s = 54,000,000 bytes/sec) that skips inf-handling
entirely and gets scaled as if it were a normal value -- corrupting the
whole feature vector's scale for exactly the short, bursty, single-packet
flows a port scan produces. This file reproduces the Infinity convention
deliberately, for that reason.
"""

import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scapy.all import IP, IPv6, TCP, UDP


PROTO_TCP = 6
PROTO_UDP = 17


@dataclass
class _Flow:
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    start_time: float
    last_seen: float = 0.0

    fwd_lengths: list = field(default_factory=list)
    bwd_lengths: list = field(default_factory=list)
    fwd_timestamps: list = field(default_factory=list)
    bwd_timestamps: list = field(default_factory=list)
    all_timestamps: list = field(default_factory=list)

    flag_counts: dict = field(default_factory=lambda: {
        "FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0, "ECE": 0, "CWR": 0,
    })
    fwd_psh, bwd_psh = 0, 0
    fwd_urg, bwd_urg = 0, 0

    fin_seen_fwd: bool = False
    fin_seen_bwd: bool = False
    rst_seen: bool = False

    fwd_header_bytes: int = 0
    bwd_header_bytes: int = 0

    init_win_bytes_fwd: int = -1
    init_win_bytes_bwd: int = -1
    act_data_pkt_fwd: int = 0
    min_seg_size_fwd: int = None


_TCP_FLAG_BITS = {
    "FIN": 0x01, "SYN": 0x02, "RST": 0x04, "PSH": 0x08,
    "ACK": 0x10, "URG": 0x20, "ECE": 0x40, "CWR": 0x80,
}


class FlowExtractor:
    def __init__(self, idle_timeout=120.0, activity_timeout=5.0):
        """
        idle_timeout: flow-level timeout (seconds) -- no packets on this
            flow for this long closes it entirely. CICFlowMeter default: 120s.
        activity_timeout: WITHIN an open flow, a gap between consecutive
            packets longer than this ends the current "active" burst and
            starts counting an "idle" period. CICFlowMeter default: 5s.
        """
        self.idle_timeout = idle_timeout
        self.activity_timeout = activity_timeout
        self._active_flows = {}
        self._completed_rows = []
        self._flow_counter = 0

    # -- public API ------------------------------------------------------

    def process(self, packets):
        for pkt in packets:
            self._ingest(pkt)
        return self

    def flush_all(self):
        for key in list(self._active_flows.keys()):
            self._close_flow(key)
        return self

    def sweep_idle_flows(self, now=None):
        """Close any active flow that's gone quiet longer than idle_timeout,
        WITHOUT requiring a new packet on that flow to trigger it."""
        now = now if now is not None else time.time()
        for key in list(self._active_flows.keys()):
            flow = self._active_flows[key]
            if now - flow.last_seen > self.idle_timeout:
                self._close_flow(key)
        return self

    def drain_completed(self):
        """Pop and return every flow completed since the last drain, as a
        DataFrame, clearing internal storage."""
        rows, self._completed_rows = self._completed_rows, []
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def to_dataframe(self):
        if not self._completed_rows:
            return pd.DataFrame()
        return pd.DataFrame(self._completed_rows)

    # -- internals ---------------------------------------------------------

    def _packet_key_and_direction(self, pkt):
        if IP in pkt:
            ip_layer = pkt[IP]
            src_ip, dst_ip = ip_layer.src, ip_layer.dst
            proto = ip_layer.proto
        elif IPv6 in pkt:
            ip_layer = pkt[IPv6]
            src_ip, dst_ip = ip_layer.src, ip_layer.dst
            proto = ip_layer.nh
        else:
            return None, None, None

        if TCP in pkt:
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
        else:
            sport, dport = 0, 0

        fwd_tuple = (src_ip, dst_ip, sport, dport, proto)
        bwd_tuple = (dst_ip, src_ip, dport, sport, proto)

        if fwd_tuple in self._active_flows:
            return fwd_tuple, "fwd", fwd_tuple
        if bwd_tuple in self._active_flows:
            return bwd_tuple, "bwd", fwd_tuple

        return fwd_tuple, "fwd", fwd_tuple

    def _ingest(self, pkt):
        if not hasattr(pkt, "time"):
            return
        ts = float(pkt.time)

        key, direction, canonical_fwd_tuple = self._packet_key_and_direction(pkt)
        if key is None:
            return

        if key not in self._active_flows:
            src_ip, dst_ip, sport, dport, proto = canonical_fwd_tuple
            self._flow_counter += 1
            self._active_flows[key] = _Flow(
                flow_id=f"flow_{self._flow_counter}",
                src_ip=src_ip, dst_ip=dst_ip,
                src_port=sport, dst_port=dport, protocol=proto,
                start_time=ts, last_seen=ts,
            )

        flow = self._active_flows[key]

        if ts - flow.last_seen > self.idle_timeout:
            self._close_flow(key)
            src_ip, dst_ip, sport, dport, proto = canonical_fwd_tuple
            self._flow_counter += 1
            self._active_flows[key] = _Flow(
                flow_id=f"flow_{self._flow_counter}",
                src_ip=src_ip, dst_ip=dst_ip,
                src_port=sport, dst_port=dport, protocol=proto,
                start_time=ts, last_seen=ts,
            )
            flow = self._active_flows[key]

        plen = len(bytes(pkt))
        flow.all_timestamps.append(ts)
        flow.last_seen = ts

        if direction == "fwd":
            flow.fwd_lengths.append(plen)
            flow.fwd_timestamps.append(ts)
        else:
            flow.bwd_lengths.append(plen)
            flow.bwd_timestamps.append(ts)

        if TCP in pkt:
            tcp = pkt[TCP]
            for name, bit in _TCP_FLAG_BITS.items():
                if tcp.flags & bit:
                    flow.flag_counts[name] += 1
            if tcp.flags & _TCP_FLAG_BITS["PSH"]:
                if direction == "fwd":
                    flow.fwd_psh += 1
                else:
                    flow.bwd_psh += 1
            if tcp.flags & _TCP_FLAG_BITS["URG"]:
                if direction == "fwd":
                    flow.fwd_urg += 1
                else:
                    flow.bwd_urg += 1
            if tcp.flags & _TCP_FLAG_BITS["FIN"]:
                if direction == "fwd":
                    flow.fin_seen_fwd = True
                else:
                    flow.fin_seen_bwd = True
            if tcp.flags & _TCP_FLAG_BITS["RST"]:
                flow.rst_seen = True

            tcp_header_len = (tcp.dataofs or 5) * 4
            payload_len = len(bytes(tcp.payload))
            if direction == "fwd":
                if flow.init_win_bytes_fwd == -1:
                    flow.init_win_bytes_fwd = int(tcp.window)
                if payload_len > 0:
                    flow.act_data_pkt_fwd += 1
                if flow.min_seg_size_fwd is None or tcp_header_len < flow.min_seg_size_fwd:
                    flow.min_seg_size_fwd = tcp_header_len
            else:
                if flow.init_win_bytes_bwd == -1:
                    flow.init_win_bytes_bwd = int(tcp.window)

        if IP in pkt:
            header_len = pkt[IP].ihl * 4 if pkt[IP].ihl else 20
        else:
            header_len = 40
        if TCP in pkt:
            header_len += (pkt[TCP].dataofs or 5) * 4
        elif UDP in pkt:
            header_len += 8

        if direction == "fwd":
            flow.fwd_header_bytes += header_len
        else:
            flow.bwd_header_bytes += header_len

        if flow.rst_seen or (flow.fin_seen_fwd and flow.fin_seen_bwd):
            self._close_flow(key)

    def _close_flow(self, key):
        flow = self._active_flows.pop(key, None)
        if flow is None:
            return
        row = self._compute_features(flow)
        self._completed_rows.append(row)

    # -- feature computation ------------------------------------------------

    @staticmethod
    def _stats(values):
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        if len(values) == 1:
            return {"min": values[0], "max": values[0], "mean": values[0], "std": 0.0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
        }

    @staticmethod
    def _iat(timestamps):
        if len(timestamps) < 2:
            return []
        ordered = sorted(timestamps)
        return [b - a for a, b in zip(ordered, ordered[1:])]

    def _active_idle_periods(self, all_timestamps):
        if not all_timestamps:
            return [], []
        ordered = sorted(all_timestamps)
        active_periods, idle_periods = [], []
        active_start = ordered[0]
        last_time = ordered[0]
        for t in ordered[1:]:
            gap = t - last_time
            if gap > self.activity_timeout:
                active_periods.append(last_time - active_start)
                idle_periods.append(gap)
                active_start = t
            last_time = t
        active_periods.append(last_time - active_start)
        return active_periods, idle_periods

    def _compute_features(self, flow: _Flow):
        # NOTE: raw_duration is intentionally allowed to be exactly 0.0 --
        # no epsilon floor here. See module docstring for why.
        raw_duration = flow.last_seen - flow.start_time

        fwd_len_stats = self._stats(flow.fwd_lengths)
        bwd_len_stats = self._stats(flow.bwd_lengths)
        all_lengths = flow.fwd_lengths + flow.bwd_lengths
        all_len_stats = self._stats(all_lengths)
        all_len_var = statistics.variance(all_lengths) if len(all_lengths) > 1 else 0.0

        flow_iat = self._iat(flow.all_timestamps)
        fwd_iat = self._iat(flow.fwd_timestamps)
        bwd_iat = self._iat(flow.bwd_timestamps)

        flow_iat_stats = self._stats(flow_iat)
        fwd_iat_stats = self._stats(fwd_iat)
        bwd_iat_stats = self._stats(bwd_iat)

        active_periods, idle_periods = self._active_idle_periods(flow.all_timestamps)
        active_stats = self._stats(active_periods)
        idle_stats = self._stats(idle_periods)

        total_fwd_packets = len(flow.fwd_lengths)
        total_bwd_packets = len(flow.bwd_lengths)
        total_packets = total_fwd_packets + total_bwd_packets
        total_fwd_bytes = sum(flow.fwd_lengths)
        total_bwd_bytes = sum(flow.bwd_lengths)
        total_bytes = total_fwd_bytes + total_bwd_bytes

        # Rate features: literal Infinity for a zero-duration flow (single
        # packet, or all packets at the same timestamp) -- matches
        # CICFlowMeter's actual output, which FeatureProcessor already
        # knows how to handle (inf -> NaN -> median-imputed from training).
        # This is the fix: previously these divided by a tiny epsilon
        # floor instead, producing huge-but-finite numbers that skipped
        # that handling entirely and corrupted the scaled feature vector
        # for exactly the short single-packet flows a port scan produces.
        if raw_duration > 0:
            flow_bytes_per_s = total_bytes / raw_duration
            flow_packets_per_s = total_packets / raw_duration
            fwd_packets_per_s = total_fwd_packets / raw_duration
            bwd_packets_per_s = total_bwd_packets / raw_duration
        else:
            flow_bytes_per_s = np.inf
            flow_packets_per_s = np.inf
            fwd_packets_per_s = np.inf
            bwd_packets_per_s = np.inf

        MICROS = 1_000_000  # CICFlowMeter reports duration/IAT/Active/Idle in microseconds

        row = {
            "flow_id": flow.flow_id,
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            "timestamp": flow.start_time,

            "Flow Duration": raw_duration * MICROS,
            "Total Fwd Packets": total_fwd_packets,
            "Total Backward Packets": total_bwd_packets,
            "Total Length of Fwd Packets": total_fwd_bytes,
            "Total Length of Bwd Packets": total_bwd_bytes,

            "Fwd Packet Length Max": fwd_len_stats["max"],
            "Fwd Packet Length Min": fwd_len_stats["min"],
            "Fwd Packet Length Mean": fwd_len_stats["mean"],
            "Fwd Packet Length Std": fwd_len_stats["std"],
            "Bwd Packet Length Max": bwd_len_stats["max"],
            "Bwd Packet Length Min": bwd_len_stats["min"],
            "Bwd Packet Length Mean": bwd_len_stats["mean"],
            "Bwd Packet Length Std": bwd_len_stats["std"],

            "Flow Bytes/s": flow_bytes_per_s,
            "Flow Packets/s": flow_packets_per_s,

            "Flow IAT Mean": flow_iat_stats["mean"] * MICROS,
            "Flow IAT Std": flow_iat_stats["std"] * MICROS,
            "Flow IAT Max": flow_iat_stats["max"] * MICROS,
            "Flow IAT Min": flow_iat_stats["min"] * MICROS,
            "Fwd IAT Total": sum(fwd_iat) * MICROS,
            "Fwd IAT Mean": fwd_iat_stats["mean"] * MICROS,
            "Fwd IAT Std": fwd_iat_stats["std"] * MICROS,
            "Fwd IAT Max": fwd_iat_stats["max"] * MICROS,
            "Fwd IAT Min": fwd_iat_stats["min"] * MICROS,
            "Bwd IAT Total": sum(bwd_iat) * MICROS,
            "Bwd IAT Mean": bwd_iat_stats["mean"] * MICROS,
            "Bwd IAT Std": bwd_iat_stats["std"] * MICROS,
            "Bwd IAT Max": bwd_iat_stats["max"] * MICROS,
            "Bwd IAT Min": bwd_iat_stats["min"] * MICROS,

            "Fwd PSH Flags": flow.fwd_psh,
            "Fwd URG Flags": flow.fwd_urg,
            "Fwd Header Length": flow.fwd_header_bytes,
            "Bwd Header Length": flow.bwd_header_bytes,
            "Fwd Packets/s": fwd_packets_per_s,
            "Bwd Packets/s": bwd_packets_per_s,

            "Min Packet Length": all_len_stats["min"],
            "Max Packet Length": all_len_stats["max"],
            "Packet Length Mean": all_len_stats["mean"],
            "Packet Length Std": all_len_stats["std"],
            "Packet Length Variance": all_len_var,

            "FIN Flag Count": flow.flag_counts["FIN"],
            "SYN Flag Count": flow.flag_counts["SYN"],
            "RST Flag Count": flow.flag_counts["RST"],
            "PSH Flag Count": flow.flag_counts["PSH"],
            "ACK Flag Count": flow.flag_counts["ACK"],
            "URG Flag Count": flow.flag_counts["URG"],
            "CWE Flag Count": flow.flag_counts["CWR"],
            "ECE Flag Count": flow.flag_counts["ECE"],

            "Down/Up Ratio": (total_bwd_packets / total_fwd_packets) if total_fwd_packets else 0.0,
            "Average Packet Size": (total_bytes / total_packets) if total_packets else 0.0,
            "Avg Fwd Segment Size": (total_fwd_bytes / total_fwd_packets) if total_fwd_packets else 0.0,
            "Avg Bwd Segment Size": (total_bwd_bytes / total_bwd_packets) if total_bwd_packets else 0.0,
            "Fwd Header Length.1": flow.fwd_header_bytes,

            "Subflow Fwd Packets": total_fwd_packets,
            "Subflow Fwd Bytes": total_fwd_bytes,
            "Subflow Bwd Packets": total_bwd_packets,
            "Subflow Bwd Bytes": total_bwd_bytes,

            "Init_Win_bytes_forward": flow.init_win_bytes_fwd,
            "Init_Win_bytes_backward": flow.init_win_bytes_bwd,
            "act_data_pkt_fwd": flow.act_data_pkt_fwd,
            "min_seg_size_forward": flow.min_seg_size_fwd if flow.min_seg_size_fwd is not None else 0,

            "Active Mean": active_stats["mean"] * MICROS,
            "Active Std": active_stats["std"] * MICROS,
            "Active Max": active_stats["max"] * MICROS,
            "Active Min": active_stats["min"] * MICROS,
            "Idle Mean": idle_stats["mean"] * MICROS,
            "Idle Std": idle_stats["std"] * MICROS,
            "Idle Max": idle_stats["max"] * MICROS,
            "Idle Min": idle_stats["min"] * MICROS,
        }
        return row
