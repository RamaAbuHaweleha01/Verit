#!/usr/bin/env python3
"""
Verit NIDS - Alert Manager
------------------------------------------------------------------
The administrator-facing logging/alerting layer sitting on top of
detection output. Every alert is ALWAYS written to a local structured
log file (JSON lines -- one detection event per line, easy to grep or
load into pandas later for reporting). A subset of those also goes to
Telegram, gated by:

    - severity floor (config.min_severity)
    - a per-(source-IP, verdict) cooldown window, so a sustained DDoS
      generating thousands of matching flows per second sends ONE
      Telegram alert, then stays quiet for `cooldown_seconds`, then
      sends a follow-up that reports how many were suppressed in
      between -- rather than flooding your phone.

Usage:
    from alerting.alert_manager import AlertManager
    from alerting.config import AlertConfig

    alerts = AlertManager(AlertConfig())
    alerts.alert(
        flow={"src_ip": "10.0.0.5", "dst_ip": "10.0.0.1", "src_port": 51000,
              "dst_port": 80, "protocol": 6},
        verdict="DDoS",
        source="xgboost",
        severity="CRITICAL",
        score=0.998,
    )
    ...
    alerts.close()
"""

import json
import logging
import logging.handlers
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from alerting.config import AlertConfig
from alerting.telegram_notifier import TelegramNotifier

_SEVERITY_TO_LOGLEVEL = {
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "CRITICAL": logging.CRITICAL,
}
_SEVERITY_EMOJI = {"INFO": "\u2139\ufe0f", "WARNING": "\u26a0\ufe0f", "CRITICAL": "\U0001f6a8"}


def _setup_logger(log_file, max_bytes, backup_count):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("verit.alerts")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))  # message is already a JSON line
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
        logger.addHandler(console_handler)

    return logger


class AlertManager:
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.logger = _setup_logger(
            self.config.log_file, self.config.log_max_bytes, self.config.log_backup_count
        )

        self.notifier = None
        if self.config.enabled and self.config.telegram_configured:
            self.notifier = TelegramNotifier(
                bot_token=self.config.bot_token,
                chat_id=self.config.chat_id,
                min_interval_seconds=self.config.telegram_min_interval_seconds,
                max_retries=self.config.telegram_max_retries,
            )
        elif self.config.enabled and not self.config.telegram_configured:
            print("[alerting] Telegram not configured (no bot token/chat_id found) -- "
                  "alerts will be logged locally only. See alerting/config.py for setup.")

        self._cooldowns = {}  # (src_ip, verdict) -> {"last_sent": ts, "suppressed": int}
        self._lock = threading.Lock()

        print(f"[alerting] AlertManager ready. {self.config.summary()}")

    # -- public API -------------------------------------------------------

    def alert(self, flow: dict, verdict: str, source: str, severity: str,
              score: float = None, extra: dict = None):
        """
        flow: dict with at least src_ip/dst_ip; src_port/dst_port/protocol
              and any flow stats you have (duration, packet counts, etc.)
              are included in the log/message if present.
        verdict: e.g. "DDoS", "PortScan", "ZERO_DAY_SUSPECTED", "BENIGN"
        source: "xgboost" | "autoencoder" | "system"
        severity: "INFO" | "WARNING" | "CRITICAL"
        score: classifier confidence or autoencoder reconstruction error, if applicable
        extra: any additional fields to attach to the log record
        """
        record = self._build_record(flow, verdict, source, severity, score, extra)
        self._write_local_log(record, severity)

        if self.notifier and self.config.severity_meets_floor(severity):
            self._maybe_send_telegram(record, flow, verdict, severity)

        return record

    def close(self):
        if self.notifier:
            self.notifier.stop()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _build_record(flow, verdict, source, severity, score, extra):
        base_keys = {"src_ip", "dst_ip", "src_port", "dst_port", "protocol"}
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "verdict": verdict,
            "source": source,
            "score": score,
            "src_ip": flow.get("src_ip"),
            "dst_ip": flow.get("dst_ip"),
            "src_port": flow.get("src_port"),
            "dst_port": flow.get("dst_port"),
            "protocol": flow.get("protocol"),
        }
        # Carry through EVERY other field the caller supplied in `flow`
        # (flow stats, flag counts, whatever) rather than a hardcoded
        # column-name list -- that list previously went stale silently
        # when the upstream feature schema changed names, and every
        # alert quietly lost its context fields without erroring.
        # Internal bookkeeping columns (verdict/source/severity/score,
        # already recorded above) are skipped to avoid duplicating them.
        skip_keys = base_keys | {"verdict", "source", "severity", "score"}
        for key, value in flow.items():
            if key not in skip_keys and key not in record:
                record[key] = value
        if extra:
            record.update(extra)
        return record

    def _write_local_log(self, record, severity):
        level = _SEVERITY_TO_LOGLEVEL.get(severity, logging.INFO)
        self.logger.log(level, json.dumps(record, default=str))

    def _maybe_send_telegram(self, record, flow, verdict, severity):
        key = (flow.get("src_ip"), verdict)
        now = time.monotonic()

        with self._lock:
            entry = self._cooldowns.get(key)
            if entry is not None and (now - entry["last_sent"]) < self.config.cooldown_seconds:
                entry["suppressed"] += 1
                return  # within cooldown window -- logged locally already, skip Telegram

            suppressed_count = entry["suppressed"] if entry else 0
            self._cooldowns[key] = {"last_sent": now, "suppressed": 0}

        message = self._format_telegram_message(record, suppressed_count)
        self.notifier.send(message)

    def _format_telegram_message(self, record, suppressed_count):
        esc = TelegramNotifier.escape_html
        emoji = _SEVERITY_EMOJI.get(record["severity"], "")
        lines = [
            f"{emoji} <b>{esc(record['severity'])} - {esc(record['verdict'])}</b>",
            f"<b>Source:</b> {esc(record['source'])}",
        ]
        if record.get("score") is not None:
            label = "Confidence" if record["source"] == "xgboost" else "Reconstruction error"
            lines.append(f"<b>{label}:</b> {record['score']:.4f}")

        lines.append("")
        lines.append(f"<b>From:</b> {esc(record.get('src_ip'))}:{esc(record.get('src_port'))}")
        lines.append(f"<b>To:</b> {esc(record.get('dst_ip'))}:{esc(record.get('dst_port'))}")
        if record.get("protocol") is not None:
            lines.append(f"<b>Protocol:</b> {esc(record['protocol'])}")

        # Show whatever extra context fields came through with this
        # record (flow stats, flag counts, etc.) -- generic rather than a
        # hardcoded column list, so this can't go silently stale again if
        # the upstream feature schema's column names change.
        skip = {"timestamp", "severity", "verdict", "source", "score",
                "src_ip", "dst_ip", "src_port", "dst_port", "protocol"}
        for key, value in record.items():
            if key in skip or value is None:
                continue
            lines.append(f"<b>{esc(key)}:</b> {esc(value)}")

        lines.append("")
        lines.append(f"<i>{esc(record['timestamp'])}</i>")

        if suppressed_count > 0:
            lines.append(f"\n<i>({suppressed_count} similar alert(s) from this source were "
                          f"suppressed during the last {int(self.config.cooldown_seconds)}s "
                          f"cooldown window)</i>")

        return "\n".join(lines)
