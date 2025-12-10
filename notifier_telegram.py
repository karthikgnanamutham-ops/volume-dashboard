# notifier_telegram.py

import time
import os
import requests
import pandas as pd
from datetime import datetime, time as dt_time
import pytz

from volume_board_fast import (
    run_volume_filter_scan,
    run_indicator_analysis,
    SYMBOLS_CSV,
)

# ======== CONFIG ========

IST = pytz.timezone("Asia/Kolkata")

# Trading window (IST)
MARKET_START = dt_time(9, 30)   # 9:30 am
MARKET_END   = dt_time(15, 15)  # 3:15 pm

SCAN_INTERVAL_SECONDS = 5 * 60  # 5 minutes

# Filters (you can tune these)
MAX_SYMBOLS   = 500
MIN_LTP       = 10
MAX_LTP       = 500
MIN_CLOSE     = 10
MIN_VOLUME    = 200_000

# ---- TOKENS FROM ENV ----
DHAN_TOKEN = os.environ.get("DHAN_TOKEN")       # set this in Railway
TELEGRAM_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")  # set this in Railway
CHAT_ID = os.environ.get("TG_CHAT_ID")         # set this in Railway

# ==========================

def send_telegram(text: str):
    """Send a message to your Telegram chat via bot."""
    if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
        print("Telegram config missing, cannot send:", text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Telegram send error:", e)


def load_universe_for_bot(csv_path: str, max_symbols: int, min_ltp: float, max_ltp: float):
    """
    Lightweight version of your CSV loader (no Streamlit).
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Symbols file not found at {csv_path}")
        return []

    df.columns = [c.strip() for c in df.columns]

    symbol_candidates = ["SYMBOL", "Symbol", "symbol", "SM_SYMBOL", "SM_SYMBOL_NAME"]
    name_candidates = [
        "NAME OF COMPANY", "Company name", "COMPANY", "NAME", "Name", "COMPANY NAME",
    ]
    series_candidates = ["SERIES", "Series", "series"]
    secid_candidates = [
        "Security", "Security ID", "SecurityId", "security_id", "SECURITYID",
        "SecurityID", "SEM_SMST_SECURITY_ID", "SM_ST_SECURITY_ID", "SM_ID",
    ]
    ltp_candidates = ["LTP", "ltp", "Last Price", "Last", "Close", "close", "LTP (₹)"]

    def _first_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_symbol = _first_col(symbol_candidates)
    col_name = _first_col(name_candidates) or col_symbol
    col_series = _first_col(series_candidates)
    col_secid = _first_col(secid_candidates)
    col_ltp = _first_col(ltp_candidates)

    if col_symbol is None:
        print("No symbol column found in Symbols.csv")
        return []

    if col_series and col_series in df.columns:
        df = df[df[col_series].astype(str).str.upper() == "EQ"]

    if col_ltp and col_ltp in df.columns:
        df[col_ltp] = pd.to_numeric(df[col_ltp], errors="coerce")
        df = df[df[col_ltp].between(min_ltp, max_ltp, inclusive="both")]
    else:
        df["__LTP_missing__"] = None
        col_ltp = "__LTP_missing__"

    if max_symbols is not None:
        df = df.head(max_symbols)

    universe = []
    for _, row in df.iterrows():
        secid_val = None
        if col_secid and col_secid in df.columns:
            try:
                secid_raw = row[col_secid]
                if pd.notna(secid_raw):
                    secid_val = int(secid_raw)
            except Exception:
                secid_val = None

        universe.append(
            {
                "symbol": row[col_symbol],
                "name": row[col_name] if col_name in df.columns else row[col_symbol],
                "securityId": secid_val,
                "LTP": float(row[col_ltp]) if pd.notna(row[col_ltp]) else None,
            }
        )
    return universe


def within_market_hours(now_ist: datetime) -> bool:
    t = now_ist.time()
    return MARKET_START <= t <= MARKET_END


def run_scan_once():
    if not DHAN_TOKEN:
        send_telegram("⚠️ DHAN_TOKEN missing in environment. Please set it in Railway.")
        return

    # 1) Load universe
    universe = load_universe_for_bot(
        SYMBOLS_CSV,
        max_symbols=MAX_SYMBOLS,
        min_ltp=MIN_LTP,
        max_ltp=MAX_LTP
    )

    if not universe:
        print("Universe empty, skipping scan.")
        return

    # 2) Phase 1: volume filter
    df_volume = run_volume_filter_scan(
        dhan_token=DHAN_TOKEN,
        universe=universe,
        min_last_close=MIN_CLOSE,
        min_last_5m_volume=MIN_VOLUME,
        max_workers=10,
    )

    df_ok = df_volume[df_volume["Status"] == "OK"].copy()
    if df_ok.empty:
        return

    # 3) Phase 2: indicator analysis
    df_ind = run_indicator_analysis(df_ok, max_workers=6)
    if df_ind.empty:
        return

    # 4) Filter interesting signals
    alerts = df_ind[df_ind["Final Mode"].isin(["STRONG BUY", "BUY MODE", "SELL MODE", "STRONG SELL"])]
    if alerts.empty:
        return

    # 5) Build & send top alerts
    now_ist_str = datetime.now(IST).strftime("%H:%M")
    for _, r in alerts.head(5).iterrows():
        volume = int(r["Last_5m_Volume"]) if not pd.isna(r["Last_5m_Volume"]) else 0
        rsi_val = r["RSI(14)"] if not pd.isna(r["RSI(14)"]) else "N/A"
        trend = r["Supertrend"] if isinstance(r["Supertrend"], str) else "N/A"

        msg = (
            f"⏱ *5m Signal {now_ist_str}*\n\n"
            f"*Stock:* {r['Company name']} ({r['Symbol']})\n"
            f"*Volume:* {volume:,}\n"
            f"*Action:* {r['Final Mode']}\n"
            f"*Trend:* {trend}\n"
            f"*RSI:* {rsi_val}"
        )
        send_telegram(msg)


if __name__ == "__main__":
    send_telegram("🤖 Scanner started (Railway worker online).")

    while True:
        now = datetime.now(IST)
        if within_market_hours(now):
            try:
                run_scan_once()
            except Exception as e:
                send_telegram(f"⚠️ Scanner Error: {e}")
            time.sleep(SCAN_INTERVAL_SECONDS)  # wait 5 min
        else:
            # Outside market hours, just sleep 60s and check again
            time.sleep(60)
