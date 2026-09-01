#!/usr/bin/env python3
"""
Verit NIDS - Alerting Configuration
------------------------------------------------------------------
Loads Telegram bot credentials and alerting settings. Checks environment
variables FIRST (the recommended way to supply the bot token -- never
commit it to git), falling back to a local config/alerting_config.json
file if present.

Setup:
    1. Message @BotFather on Telegram, /newbot, follow the prompts ->
       you get a bot token like "123456789:AAExampleTokenTextHere".
    2. Send any message to your new bot, then run:
           python3 get_telegram_chat_id.py <your_bot_token>
       to find your chat_id.
    3. Either:
       export VERIT_TELEGRAM_BOT_TOKEN="123456789:AA..."
       export VERIT_TELEGRAM_CHAT_ID="987654321"
       OR copy config/alerting_config.example.json -> config/alerting_config.json
       and fill in the values there.

config/alerting_config.json is gitignored on purpose -- it holds a secret.
"""

import json
import os
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "alerting_config.json"

DEFAULTS = {
    "enabled": True,
    "min_severity": "WARNING",       # INFO | WARNING | CRITICAL -- floor for Telegram pushes
    "cooldown_seconds": 300,         # suppress repeat Telegram alerts for the same (src_ip, verdict) within this window
    "log_file": "logs/verit_alerts.log",
    "log_max_bytes": 10 * 1024 * 1024,
    "log_backup_count": 5,
    "telegram_min_interval_seconds": 1.0,  # floor between consecutive Telegram sends (API courtesy limit)
    "telegram_max_retries": 3,
}

SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


class AlertConfig:
    def __init__(self, **overrides):
        cfg = dict(DEFAULTS)
        cfg.update(self._load_file_config())
        cfg.update({k: v for k, v in overrides.items() if v is not None})

        self.bot_token = os.environ.get("VERIT_TELEGRAM_BOT_TOKEN") or cfg.get("telegram_bot_token")
        self.chat_id = os.environ.get("VERIT_TELEGRAM_CHAT_ID") or cfg.get("telegram_chat_id")

        self.enabled = bool(cfg["enabled"])
        self.min_severity = cfg["min_severity"]
        self.cooldown_seconds = float(cfg["cooldown_seconds"])
        self.log_file = cfg["log_file"]
        self.log_max_bytes = int(cfg["log_max_bytes"])
        self.log_backup_count = int(cfg["log_backup_count"])
        self.telegram_min_interval_seconds = float(cfg["telegram_min_interval_seconds"])
        self.telegram_max_retries = int(cfg["telegram_max_retries"])

        self.telegram_configured = bool(self.bot_token and self.chat_id)

    @staticmethod
    def _load_file_config():
        if DEFAULT_CONFIG_PATH.exists():
            try:
                with open(DEFAULT_CONFIG_PATH) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[alerting] Warning: couldn't read {DEFAULT_CONFIG_PATH}: {e}")
        return {}

    def severity_meets_floor(self, severity):
        return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(self.min_severity, 0)

    def summary(self):
        return (
            f"Telegram configured: {self.telegram_configured} | "
            f"enabled: {self.enabled} | min_severity: {self.min_severity} | "
            f"cooldown: {self.cooldown_seconds}s | log_file: {self.log_file}"
        )
