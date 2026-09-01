#!/usr/bin/env python3
"""
Verit NIDS - Stage 2a: Packet-Level Pre-Cleaning
------------------------------------------------------------------
Takes raw captured packets (from a pcap file or a live scapy sniff) and
strips out everything that would poison flow-level feature extraction:

    1. Corrupted / truncated packets      (malformed headers, snaplen cuts)
    2. Checksum mismatches                (IP / TCP / UDP)
    3. Exact duplicate packets            (capture-level dupes, e.g. from
                                            promiscuous mode or multi-tap)
    4. TCP retransmissions & out-of-order (same-flow sequence tracking)
    5. L2 / control-plane noise           (ARP, IGMP, LLDP, STP, DHCP, ...)

Output is a clean list of scapy packets (or a generator, for streaming)
ready to be handed to the flow extractor.

IMPORTANT CAVEAT ABOUT CHECKSUM VALIDATION ON LIVE CAPTURE:
Most modern NICs do TCP/UDP checksum offloading in hardware. For traffic
*originating* from the capture host, the OS often hands the packet to the
NIC with a placeholder/zero checksum and lets the NIC fill it in during
transmission -- but tcpdump/scapy captures the packet *before* that happens,
so outbound packets can show a checksum that looks "invalid" even though
the wire packet was fine. By default this module only checksum-validates
*inbound* packets for that reason (see `validate_checksum_direction`).
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

from scapy.all import IP, IPv6, TCP, UDP, ICMP
from scapy.layers.l2 import ARP, STP
from scapy.layers.inet import IP as IPv4
from scapy.packet import Raw

# L2 / control-plane protocols to drop outright -- they carry no useful
# host-to-host traffic signal for a flow-based NIDS and just add noise.
_NOISE_LAYERS = (ARP, STP)

# Matched by well-known port/protocol combos that scapy doesn't always
# expose as a distinct layer without extra imports (LLDP, IGMP, DHCP).
_LLDP_ETHERTYPE = 0x88CC
_IGMP_PROTO_NUM = 2       # IP protocol number for IGMP
_DHCP_PORTS = {67, 68}    # BOOTP/DHCP


@dataclass
class CleaningStats:
    total_seen: int = 0
    dropped_corrupted: int = 0
    dropped_checksum: int = 0
    dropped_duplicate: int = 0
    dropped_retransmission: int = 0
    dropped_out_of_order: int = 0
    dropped_l2_noise: int = 0
    kept: int = 0

    def as_dict(self):
        return {
            "total_seen": self.total_seen,
            "dropped_corrupted": self.dropped_corrupted,
            "dropped_checksum": self.dropped_checksum,
            "dropped_duplicate": self.dropped_duplicate,
            "dropped_retransmission": self.dropped_retransmission,
            "dropped_out_of_order": self.dropped_out_of_order,
            "dropped_l2_noise": self.dropped_l2_noise,
            "kept": self.kept,
        }

    def summary(self):
        lines = [f"[cleaning] Packets seen:            {self.total_seen}"]
        for label, val in [
            ("corrupted/truncated", self.dropped_corrupted),
            ("checksum mismatch", self.dropped_checksum),
            ("duplicate", self.dropped_duplicate),
            ("retransmission", self.dropped_retransmission),
            ("out-of-order", self.dropped_out_of_order),
            ("L2/control noise", self.dropped_l2_noise),
        ]:
            pct = (val / self.total_seen * 100) if self.total_seen else 0
            lines.append(f"[cleaning] Dropped ({label}):{' ' * (10 - len(label) if len(label) < 10 else 1)} {val} ({pct:.1f}%)")
        kept_pct = (self.kept / self.total_seen * 100) if self.total_seen else 0
        lines.append(f"[cleaning] Kept:                    {self.kept} ({kept_pct:.1f}%)")
        return "\n".join(lines)


@dataclass
class _TcpFlowState:
    """Tracks per-direction sequence numbers within one TCP flow to detect
    retransmissions and out-of-order segments."""
    next_expected_seq: dict = field(default_factory=dict)  # direction -> seq
    seen_seqs: dict = field(default_factory=lambda: defaultdict(set))  # direction -> set(seq)


class PacketCleaner:
    def __init__(
        self,
        validate_checksums=True,
        validate_checksum_direction="inbound",   # "inbound" | "both" | "none"
        local_ips=None,                          # needed to determine inbound/outbound
        drop_retransmissions=True,
        drop_out_of_order=True,
        drop_l2_noise=True,
        drop_dhcp=True,
        dedup_window=4096,                       # how many recent packet hashes to remember
    ):
        self.validate_checksums = validate_checksums
        self.validate_checksum_direction = validate_checksum_direction
        self.local_ips = set(local_ips or [])
        self.drop_retransmissions = drop_retransmissions
        self.drop_out_of_order = drop_out_of_order
        self.drop_l2_noise = drop_l2_noise
        self.drop_dhcp = drop_dhcp

        self.stats = CleaningStats()
        self._dedup_hashes = set()
        self._dedup_order = []
        self._dedup_window = dedup_window
        self._tcp_flows = {}  # 5-tuple (normalized) -> _TcpFlowState

    # -- public API ---------------------------------------------------

    def clean(self, packets):
        """Generator: yields only the packets that pass every filter."""
        for pkt in packets:
            self.stats.total_seen += 1
            verdict = self._evaluate(pkt)
            if verdict:
                self.stats.kept += 1
                yield pkt

    # -- individual filters --------------------------------------------

    def _evaluate(self, pkt):
        if self._is_corrupted_or_truncated(pkt):
            self.stats.dropped_corrupted += 1
            return False

        if self.drop_l2_noise and self._is_l2_noise(pkt):
            self.stats.dropped_l2_noise += 1
            return False

        if self._is_duplicate(pkt):
            self.stats.dropped_duplicate += 1
            return False

        if self.validate_checksums and not self._checksum_ok(pkt):
            self.stats.dropped_checksum += 1
            return False

        if TCP in pkt:
            tcp_verdict = self._check_tcp_sequence(pkt)
            if tcp_verdict == "retransmission" and self.drop_retransmissions:
                self.stats.dropped_retransmission += 1
                return False
            if tcp_verdict == "out_of_order" and self.drop_out_of_order:
                self.stats.dropped_out_of_order += 1
                return False

        return True

    def _is_corrupted_or_truncated(self, pkt):
        try:
            raw_len = len(bytes(pkt))
            if raw_len == 0:
                return True

            # scapy sets .original / wirelen when it reads from a pcap;
            # if the capture snapshot length cut the packet short, the
            # declared header length won't match the bytes actually present.
            if IP in pkt:
                ip_layer = pkt[IP]
                declared_total_len = ip_layer.len
                # bytes available from the IP header onward
                available = len(bytes(ip_layer))
                if declared_total_len and declared_total_len > available:
                    return True
                if ip_layer.ihl and ip_layer.ihl < 5:
                    return True  # invalid IP header length

            if IPv6 in pkt:
                ip6_layer = pkt[IPv6]
                declared_payload_len = ip6_layer.plen
                available_payload = len(bytes(ip6_layer.payload))
                if declared_payload_len and declared_payload_len > available_payload:
                    return True

            if TCP in pkt:
                tcp_layer = pkt[TCP]
                if tcp_layer.dataofs and tcp_layer.dataofs < 5:
                    return True  # invalid TCP header length

            return False
        except Exception:
            # scapy raised while dissecting -> treat as corrupted
            return True

    def _is_l2_noise(self, pkt):
        if any(layer in pkt for layer in _NOISE_LAYERS):
            return True
        if pkt.haslayer("Dot3") or (hasattr(pkt, "type") and pkt.type == _LLDP_ETHERTYPE):
            return True
        if IP in pkt and pkt[IP].proto == _IGMP_PROTO_NUM:
            return True
        if self.drop_dhcp and UDP in pkt:
            if pkt[UDP].sport in _DHCP_PORTS or pkt[UDP].dport in _DHCP_PORTS:
                return True
        return False

    def _is_duplicate(self, pkt):
        try:
            digest = hashlib.blake2b(bytes(pkt), digest_size=16).digest()
        except Exception:
            return False

        if digest in self._dedup_hashes:
            return True

        self._dedup_hashes.add(digest)
        self._dedup_order.append(digest)
        if len(self._dedup_order) > self._dedup_window:
            oldest = self._dedup_order.pop(0)
            self._dedup_hashes.discard(oldest)
        return False

    def _checksum_ok(self, pkt):
        direction = self._direction(pkt)
        if self.validate_checksum_direction == "none":
            return True
        if self.validate_checksum_direction == "inbound" and direction != "inbound":
            return True  # skip validation for outbound (offload artifacts)

        try:
            if TCP in pkt and IP in pkt:
                orig_chk = pkt[TCP].chksum
                recomputed = pkt[IP].copy()
                del recomputed[TCP].chksum
                recomputed = recomputed.__class__(bytes(recomputed))
                return orig_chk == recomputed[TCP].chksum

            if UDP in pkt and IP in pkt:
                orig_chk = pkt[UDP].chksum
                if orig_chk == 0:
                    return True  # UDP checksum optional over IPv4
                recomputed = pkt[IP].copy()
                del recomputed[UDP].chksum
                recomputed = recomputed.__class__(bytes(recomputed))
                return orig_chk == recomputed[UDP].chksum

            if IP in pkt:
                orig_chk = pkt[IP].chksum
                recomputed = pkt[IP].copy()
                del recomputed.chksum
                recomputed = IPv4(bytes(recomputed))
                return orig_chk == recomputed.chksum

        except Exception:
            return True  # if we can't validate, don't punish the packet

        return True

    def _direction(self, pkt):
        if not self.local_ips or IP not in pkt:
            return "unknown"
        if pkt[IP].src in self.local_ips:
            return "outbound"
        if pkt[IP].dst in self.local_ips:
            return "inbound"
        return "unknown"

    def _flow_key(self, pkt):
        ip = pkt[IP]
        tcp = pkt[TCP]
        a = (ip.src, tcp.sport)
        b = (ip.dst, tcp.dport)
        # normalize so both directions of the same connection map to one key
        endpoints = tuple(sorted([a, b]))
        return endpoints

    def _check_tcp_sequence(self, pkt):
        key = self._flow_key(pkt)
        ip, tcp = pkt[IP], pkt[TCP]
        direction = (ip.src, tcp.sport)
        payload_len = len(bytes(tcp.payload))

        state = self._tcp_flows.setdefault(key, _TcpFlowState())

        # SYN (without ACK) always starts a fresh sequence space for this direction
        if tcp.flags & 0x02 and not (tcp.flags & 0x10):
            state.next_expected_seq[direction] = tcp.seq + 1
            state.seen_seqs[direction] = {tcp.seq}
            return "ok"

        seq = tcp.seq
        # Sequence space actually consumed by this segment: payload bytes,
        # plus 1 if FIN is set (FIN consumes a sequence number like SYN
        # does). A pure ACK with no payload and no FIN consumes ZERO
        # sequence space -- TCP allows it to legitimately repeat the same
        # seq as the previous segment any number of times (e.g. duplicate
        # ACKs, keep-alives). Treating that as a retransmission/out-of-
        # order event was a bug: it caused the very next real data
        # packet -- which correctly starts at that same seq, since the
        # preceding pure ACK never advanced it -- to be wrongly flagged
        # and dropped.
        consumed = payload_len + (1 if tcp.flags & 0x01 else 0)  # + FIN

        if consumed == 0:
            return "ok"

        seen = state.seen_seqs[direction]
        if seq in seen:
            return "retransmission"
        seen.add(seq)

        expected = state.next_expected_seq.get(direction)
        if expected is not None:
            if seq < expected:
                return "retransmission"
            if seq > expected:
                state.next_expected_seq[direction] = seq + consumed
                return "out_of_order"

        state.next_expected_seq[direction] = seq + consumed
        return "ok"
