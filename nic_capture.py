#!/usr/bin/env python3
"""
Verit NIDS - Stage 1: NIC Discovery + Real-Time Traffic Capture
------------------------------------------------------------------
Dynamically enumerates active (UP, non-loopback) network interfaces on the
host, then spawns one `tcpdump` process per interface to capture live
traffic and stream it to the screen in real time.

This is the first stage of the Verit pipeline:
    [capture: this script] -> [flow processing/cleaning] -> [AI models]

Usage:
    sudo python3 nic_capture.py                      # auto-detect + capture all active NICs
    sudo python3 nic_capture.py -i eth0,wlan0         # capture only specific NICs
    sudo python3 nic_capture.py --rescan 10           # re-check for new/removed NICs every 10s
    sudo python3 nic_capture.py --pcap-dir ./captures # also dump raw pcap per interface
    sudo python3 nic_capture.py --bpf "tcp or udp"    # apply a BPF filter

Requires:
    - tcpdump installed and on PATH
    - root privileges (or CAP_NET_RAW/CAP_NET_ADMIN on the tcpdump binary)
    - psutil  (pip install psutil)
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("[!] Missing dependency 'psutil'. Install it with: pip install psutil", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Interface discovery
# --------------------------------------------------------------------------

def get_active_interfaces(exclude_prefixes=("lo", "docker", "veth", "br-")):
    """
    Return a sorted list of interface names that are currently UP and have
    a non-loopback link. Uses psutil for cross-checked stats, falling back
    to /sys/class/net if psutil data looks incomplete.
    """
    active = []
    try:
        stats = psutil.net_if_stats()
    except Exception as e:
        print(f"[!] Failed to read interface stats via psutil: {e}", file=sys.stderr)
        stats = {}

    for name, st in stats.items():
        if name.startswith(exclude_prefixes):
            continue
        if st.isup:
            active.append(name)

    # Fallback / cross-check against /sys/class/net in case psutil misses something
    sys_net = Path("/sys/class/net")
    if sys_net.exists():
        for iface_path in sys_net.iterdir():
            name = iface_path.name
            if name.startswith(exclude_prefixes) or name in active:
                continue
            operstate_file = iface_path / "operstate"
            if operstate_file.exists():
                try:
                    state = operstate_file.read_text().strip()
                    if state == "up":
                        active.append(name)
                except OSError:
                    pass

    return sorted(set(active))


def describe_interfaces(interfaces):
    """Print a short table of interface name, addresses, speed."""
    stats = psutil.net_if_stats()
    addrs = psutil.net_if_addrs()
    print(f"{'Interface':<12}{'Speed(Mb)':<12}{'MTU':<8}{'Addresses'}")
    print("-" * 70)
    for name in interfaces:
        st = stats.get(name)
        speed = st.speed if st else "?"
        mtu = st.mtu if st else "?"
        ip_list = []
        for a in addrs.get(name, []):
            if a.family.name in ("AF_INET", "AF_INET6"):
                ip_list.append(a.address)
        print(f"{name:<12}{speed:<12}{mtu:<8}{', '.join(ip_list) if ip_list else '-'}")
    print()


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

class InterfaceCapture:
    """Wraps a single tcpdump subprocess for one interface."""

    def __init__(self, iface, bpf_filter=None, pcap_dir=None, verbose_level=1):
        self.iface = iface
        self.bpf_filter = bpf_filter
        self.pcap_dir = pcap_dir
        self.verbose_level = verbose_level
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()

    def _build_cmd(self):
        # -l  : line-buffered stdout (so we see packets as they arrive)
        # -n  : don't resolve hostnames (faster, avoids DNS noise)
        # -tttt: human-readable timestamps
        cmd = ["tcpdump", "-i", self.iface, "-l", "-n", "-tttt"]

        if self.verbose_level > 0:
            cmd.append("-" + "v" * self.verbose_level)

        if self.pcap_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pcap_path = Path(self.pcap_dir) / f"{self.iface}_{ts}.pcap"
            cmd += ["-w", str(pcap_path)]
            # When writing to a pcap file, tcpdump won't also print decoded
            # packets to stdout unless we add a second reader; simplest is
            # to still allow -l printing by NOT using -w exclusively.
            # We'll instead tee: capture to file AND print summary via -U.
            cmd += ["-U"]  # flush packet-by-packet when writing to file too

        if self.bpf_filter:
            cmd.append(self.bpf_filter)

        return cmd

    def start(self):
        cmd = self._build_cmd()
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError:
            print("[!] 'tcpdump' not found on PATH. Install it with: sudo apt install tcpdump",
                  file=sys.stderr)
            sys.exit(1)

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if self.stop_event.is_set():
                break
            line = line.rstrip("\n")
            if not line:
                continue
            # tcpdump prints a couple of setup lines ("tcpdump: verbose...",
            # "listening on ...") — pass those through too, just tagged.
            print(f"[{self.iface}] {line}")

    def stop(self):
        self.stop_event.set()
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=2)


class CaptureManager:
    """Owns one InterfaceCapture per active NIC, with optional periodic rescan."""

    def __init__(self, interfaces=None, bpf_filter=None, pcap_dir=None,
                 rescan_interval=None, verbose_level=1, exclude_prefixes=("lo",)):
        self.explicit_interfaces = interfaces  # None => auto-detect
        self.bpf_filter = bpf_filter
        self.pcap_dir = pcap_dir
        self.rescan_interval = rescan_interval
        self.verbose_level = verbose_level
        self.exclude_prefixes = exclude_prefixes
        self.captures = {}  # iface -> InterfaceCapture
        self.stop_event = threading.Event()

        if self.pcap_dir:
            Path(self.pcap_dir).mkdir(parents=True, exist_ok=True)

    def _target_interfaces(self):
        if self.explicit_interfaces:
            return sorted(self.explicit_interfaces)
        return get_active_interfaces(exclude_prefixes=self.exclude_prefixes)

    def _sync_captures(self):
        wanted = set(self._target_interfaces())
        current = set(self.captures.keys())

        # Start new interfaces
        for iface in wanted - current:
            print(f"[+] Starting capture on new interface: {iface}")
            cap = InterfaceCapture(iface, self.bpf_filter, self.pcap_dir, self.verbose_level)
            cap.start()
            self.captures[iface] = cap

        # Stop interfaces that disappeared / went down
        for iface in current - wanted:
            print(f"[-] Interface no longer active, stopping capture: {iface}")
            self.captures[iface].stop()
            del self.captures[iface]

    def run(self):
        interfaces = self._target_interfaces()
        if not interfaces:
            print("[!] No active interfaces found (or none matched your filter). Exiting.")
            return

        print("[*] Active interfaces detected:")
        describe_interfaces(interfaces)

        self._sync_captures()

        print(f"[*] Capturing live traffic on: {', '.join(sorted(self.captures.keys()))}")
        if self.bpf_filter:
            print(f"[*] BPF filter: {self.bpf_filter}")
        if self.pcap_dir:
            print(f"[*] Writing pcap files to: {self.pcap_dir}")
        print("[*] Press Ctrl+C to stop.\n")

        try:
            while not self.stop_event.is_set():
                if self.rescan_interval:
                    time.sleep(self.rescan_interval)
                    self._sync_captures()
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n[*] Stopping all captures...")
        for iface, cap in self.captures.items():
            cap.stop()
        print("[*] Done.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def require_root_or_cap():
    if os.geteuid() != 0:
        # Check if tcpdump has the raw-capture capability set (cap_net_raw)
        tcpdump_path = shutil.which("tcpdump")
        has_cap = False
        if tcpdump_path:
            try:
                out = subprocess.run(["getcap", tcpdump_path], capture_output=True, text=True)
                has_cap = "cap_net_raw" in out.stdout
            except FileNotFoundError:
                pass
        if not has_cap:
            print("[!] This script needs root privileges (or tcpdump must have "
                  "cap_net_raw/cap_net_admin capabilities) to capture packets.")
            print("    Run with: sudo python3 nic_capture.py")
            print("    Or grant tcpdump capabilities once with:")
            print(f"        sudo setcap cap_net_raw,cap_net_admin=eip {tcpdump_path or '/usr/sbin/tcpdump'}")
            sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Verit NIDS - Stage 1: NIC discovery + live capture")
    p.add_argument("-i", "--interfaces", type=str, default=None,
                    help="Comma-separated list of interfaces to capture (default: auto-detect all active NICs)")
    p.add_argument("--bpf", type=str, default=None,
                    help="BPF filter expression passed to tcpdump, e.g. 'tcp or udp'")
    p.add_argument("--pcap-dir", type=str, default=None,
                    help="If set, also write raw pcap files per interface into this directory")
    p.add_argument("--rescan", type=int, default=None, metavar="SECONDS",
                    help="Re-check for new/removed active interfaces every N seconds (hot-plug support)")
    p.add_argument("-v", "--verbosity", type=int, default=1, choices=[0, 1, 2, 3],
                    help="tcpdump verbosity level (0=none, up to 3 for -vvv)")
    p.add_argument("--exclude", type=str, default="lo",
                    help="Comma-separated interface name prefixes to exclude (default: 'lo')")
    p.add_argument("--list-only", action="store_true",
                    help="Only list active interfaces and exit, don't capture")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_only:
        ifaces = get_active_interfaces(exclude_prefixes=tuple(args.exclude.split(",")))
        if not ifaces:
            print("[!] No active interfaces found.")
        else:
            describe_interfaces(ifaces)
        return

    require_root_or_cap()

    interfaces = args.interfaces.split(",") if args.interfaces else None

    manager = CaptureManager(
        interfaces=interfaces,
        bpf_filter=args.bpf,
        pcap_dir=args.pcap_dir,
        rescan_interval=args.rescan,
        verbose_level=args.verbosity,
        exclude_prefixes=tuple(args.exclude.split(",")),
    )
    manager.run()


if __name__ == "__main__":
    main()
