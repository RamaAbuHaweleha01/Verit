#!/usr/bin/env python3
"""
Verit NIDS - Alerting System Test/Demo
------------------------------------------------------------------
Run this after setting up your Telegram bot token + chat_id (see
alerting/config.py) to confirm alerts actually arrive before wiring
this into the main detection engine.

    python3 test_telegram_alert.py
"""

import time

from alerting.alert_manager import AlertManager
from alerting.config import AlertConfig


def main():
    config = AlertConfig()
    print(config.summary())

    if not config.telegram_configured:
        print("\n[!] No Telegram credentials found. Set VERIT_TELEGRAM_BOT_TOKEN and "
              "VERIT_TELEGRAM_CHAT_ID (env vars) or fill in config/alerting_config.json "
              "(copy from config/alerting_config.example.json).")
        print("    Continuing anyway -- alerts will still be written to the local log file.")

    alerts = AlertManager(config)

    print("\n[*] Sending a known-attack (XGBoost) test alert ...")
    alerts.alert(
        flow={
            "src_ip": "203.0.113.42", "dst_ip": "10.0.0.5",
            "src_port": 51000, "dst_port": 80, "protocol": 6,
            "flow_duration": 0.842, "total_fwd_packets": 15000,
            "total_bwd_packets": 3, "total_fwd_bytes": 780000, "total_bwd_bytes": 180,
        },
        verdict="DDoS", source="xgboost", severity="CRITICAL", score=0.9987,
    )

    print("[*] Sending a zero-day (autoencoder) test alert ...")
    alerts.alert(
        flow={
            "src_ip": "198.51.100.7", "dst_ip": "10.0.0.5",
            "src_port": 33221, "dst_port": 8080, "protocol": 6,
        },
        verdict="ZERO_DAY_SUSPECTED", source="autoencoder", severity="WARNING", score=0.5123,
    )

    print("[*] Sending 3 more identical alerts from the same source (should trigger cooldown "
          "suppression -- only the first should have gone to Telegram) ...")
    for _ in range(3):
        alerts.alert(
            flow={"src_ip": "203.0.113.42", "dst_ip": "10.0.0.5",
                  "src_port": 51001, "dst_port": 80, "protocol": 6},
            verdict="DDoS", source="xgboost", severity="CRITICAL", score=0.995,
        )

    print("[*] Sending a low-severity INFO alert (should NOT reach Telegram at default "
          "min_severity=WARNING, but WILL be in the local log) ...")
    alerts.alert(
        flow={"src_ip": "10.0.0.20", "dst_ip": "10.0.0.5", "src_port": 443, "dst_port": 51500,
              "protocol": 6},
        verdict="BENIGN", source="xgboost", severity="INFO", score=0.999,
    )

    print("\n[*] Waiting a few seconds for the Telegram delivery queue to flush ...")
    time.sleep(4)
    alerts.close()

    print(f"\n[*] Done. Check your Telegram chat, and inspect the local log at: {config.log_file}")


if __name__ == "__main__":
    main()
