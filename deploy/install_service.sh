#!/usr/bin/env bash
# Verit NIDS - systemd service installer
# ------------------------------------------------------------------
# Run with sudo from the Verit project root:
#     sudo bash deploy/install_service.sh
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Run this with sudo: sudo bash deploy/install_service.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_FILE="$SCRIPT_DIR/verit-nids.service"

if [ ! -f "$UNIT_FILE" ]; then
    echo "Can't find verit-nids.service next to this script."
    exit 1
fi

echo "[*] Installing $UNIT_FILE -> /etc/systemd/system/verit-nids.service"
cp "$UNIT_FILE" /etc/systemd/system/verit-nids.service

echo "[*] Reloading systemd unit files..."
systemctl daemon-reload

echo "[*] Done."
echo ""
echo "Before starting the service, double-check the unit file's User=,"
echo "WorkingDirectory=, and --interface list match your setup:"
echo "    sudo nano /etc/systemd/system/verit-nids.service"
echo "    sudo systemctl daemon-reload   # after any edits"
echo ""
echo "Then:"
echo "    sudo systemctl enable verit-nids     # start automatically on boot"
echo "    sudo systemctl start verit-nids      # start it now"
echo "    sudo systemctl status verit-nids     # check it's running"
echo "    sudo journalctl -u verit-nids -f     # follow live logs"
echo "    sudo systemctl stop verit-nids       # graceful stop (flushes open flows)"
