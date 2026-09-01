#!/usr/bin/env python3
"""
Verit NIDS - Find Your Telegram chat_id
------------------------------------------------------------------
1. Message @BotFather on Telegram -> /newbot -> follow prompts -> copy the token.
2. Send ANY message to your new bot (search for it by the username you gave it).
3. Run this script with your token:
       python3 get_telegram_chat_id.py 123456789:AAExampleTokenTextHere

It'll print the chat_id(s) of anyone who has messaged the bot.
"""

import sys
import requests


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 get_telegram_chat_id.py <bot_token>")
        sys.exit(1)

    token = sys.argv[1]
    url = f"https://api.telegram.org/bot{token}/getUpdates"

    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        print(f"[!] Request failed ({resp.status_code}): {resp.text}")
        print("    Double-check the token is correct (copy it exactly from @BotFather).")
        sys.exit(1)

    data = resp.json()
    results = data.get("result", [])
    if not results:
        print("[!] No messages found yet. Send a message to your bot on Telegram first, "
              "then run this script again.")
        sys.exit(1)

    seen = {}
    for update in results:
        msg = update.get("message") or update.get("channel_post")
        if not msg:
            continue
        chat = msg["chat"]
        seen[chat["id"]] = chat

    print("Found chat(s):")
    for chat_id, chat in seen.items():
        name = chat.get("username") or chat.get("title") or chat.get("first_name") or "?"
        print(f"  chat_id = {chat_id}   ({chat.get('type')}, {name})")

    print("\nSet this in your environment or config/alerting_config.json:")
    print(f"  export VERIT_TELEGRAM_CHAT_ID=\"{list(seen.keys())[0]}\"")


if __name__ == "__main__":
    main()
