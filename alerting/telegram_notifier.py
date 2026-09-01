#!/usr/bin/env python3
"""
Verit NIDS - Telegram Notifier
------------------------------------------------------------------
Sends messages to a Telegram bot via the HTTP Bot API. Delivery happens
on a background thread through a queue, so a slow/unreachable network
call never blocks the detection loop that's calling into this.

Enforces a minimum interval between sends (Telegram's own guidance is
to avoid bursting more than ~1 message/second to the same chat) and
retries transient failures with backoff.
"""

import html
import queue
import threading
import time

import requests

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramNotifier:
    def __init__(self, bot_token, chat_id, min_interval_seconds=1.0, max_retries=3, timeout=10):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.min_interval_seconds = min_interval_seconds
        self.max_retries = max_retries
        self.timeout = timeout

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def send(self, text, parse_mode="HTML"):
        """Non-blocking: enqueues the message and returns immediately."""
        self._queue.put((text, parse_mode))

    def send_sync(self, text, parse_mode="HTML"):
        """Blocking variant -- sends immediately in the calling thread.
        Useful for the one-off startup/test message where you want to
        know right away whether credentials are valid."""
        return self._do_send(text, parse_mode)

    def _run(self):
        last_sent = 0.0
        while not self._stop_event.is_set():
            try:
                text, parse_mode = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            elapsed = time.monotonic() - last_sent
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)

            self._do_send(text, parse_mode)
            last_sent = time.monotonic()

    def _do_send(self, text, parse_mode):
        url = f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return True
                if resp.status_code == 429:
                    # rate limited -- Telegram tells us how long to back off
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 2 * attempt)
                    print(f"[telegram] Rate limited, retrying after {retry_after}s")
                    time.sleep(retry_after)
                    continue
                print(f"[telegram] Send failed ({resp.status_code}): {resp.text[:300]}")
                if resp.status_code in (400, 401, 403):
                    return False  # bad token/chat_id/message -- retrying won't help
            except requests.RequestException as e:
                print(f"[telegram] Network error (attempt {attempt}/{self.max_retries}): {e}")
                time.sleep(min(2 ** attempt, 30))
        print(f"[telegram] Giving up after {self.max_retries} attempts")
        return False

    def stop(self, wait=True):
        self._stop_event.set()
        if wait:
            self._worker.join(timeout=5)

    @staticmethod
    def escape_html(text):
        return html.escape(str(text))
