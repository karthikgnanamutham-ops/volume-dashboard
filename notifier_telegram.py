import os
import requests
import pandas as pd
from datetime import datetime, time as dt_time
import pytz

from volume_board_fast import (
    load_universe_from_symbols,
    run_volume_filter_scan,
    run_indicator_analysis,
    SYMBOLS_CSV,
)

# ===== CONFIG =====
IST = pytz.timezone("Asia/Kolkata")

MARKET_START = dt_time(9, 30)
MARKET_END   = dt_time(15, 15)

MAX_SYMBOLS = 500
MIN_LTP = 10
MAX_LTP = 500
MIN_CLOSE = 10
MIN_VOLUME = 200_000

# Secrets from GitHub
DHAN_TOKEN = os.environ.get("DHAN_TOKEN")
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

# ==================

def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text}
    requests.post(url, json=payload, timeout=10)

def within_market_hours():
    now = datetime.now(IST).time()
    return MARKET_START <= now <= MARKET_END

def main():
    # TEMP for testing:
    # if not within_market_hours():
    #     return

    if not DHAN_TOKEN:
        send_telegram("⚠️ DHAN_TOKEN missing in GitHub Secrets")
        return

    universe = load_universe_from_symbols(
        SYMBOLS_CSV,
        max_symbols=MAX_SYMBOLS,
        min_ltp=MIN_LTP,
        max_ltp=MAX_LTP,
    )

    df_volume = run_volume_filter_scan(
        dhan_token=DHAN_TOKEN,
        universe=universe,
        min_last_close=MIN_CLOSE,
        min_last_5m_volume=MIN_VOLUME,
        max_workers=8,
    )

    df_ok = df_volume[df_volume["Status"] == "OK"].copy()
    if df_ok.empty:
        return

    df_ind = run_indicator_analysis(df_ok)

    alerts = df_ind[df_ind["Final Mode"].isin([
        "STRONG BUY", "BUY MODE", "SELL MODE", "STRONG SELL"
    ])]

    if alerts.empty:
        return

    now_str = datetime.now(IST).strftime("%H:%M")

    for _, r in alerts.head(3).iterrows():
        msg = (
            f"⏱ 5m Signal {now_str}\n\n"
            f"Stock: {r['Company name']} ({r['Symbol']})\n"
            f"Volume: {int(r['Last_5m_Volume']):,}\n"
            f"Action: {r['Final Mode']}\n"
            f"Trend: {r['Supertrend']}\n"
            f"RSI: {r['RSI(14)']}"
        )
        send_telegram(msg)

if __name__ == "__main__":
    main()
