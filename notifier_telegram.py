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

MARKET_START = dt_time(7, 30)
MARKET_END   = dt_time(15, 15)

MAX_SYMBOLS = 1000
MIN_LTP = 100
MAX_LTP = 500
MIN_CLOSE = 100
MIN_VOLUME = 100_000

# Secrets from GitHub
DHAN_TOKEN = os.environ.get("DHAN_TOKEN")
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

# ==================

def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Telegram config missing, cannot send:", text)
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        # debug logging for GitHub Actions output:
        print("Telegram API URL:", url)
        print("Payload:", payload)
        print("Response status:", resp.status_code)
        try:
            print("Response body:", resp.json())
        except Exception:
            print("Response text:", resp.text)
        if resp.status_code != 200:
            print("Telegram send failed with status", resp.status_code)
    except Exception as e:
        print("Telegram send exception:", type(e).__name__, str(e))

def build_signals_table(df_alerts):
    """
    Build a single text message with multiple signals in a table-style format.
    Columns: Symbol | Price | RSI | Volume | Mode
    """
    if df_alerts.empty:
        return None

    # limit signals per message (optional)
    df = df_alerts.copy().head(10)

    lines = []
    now_str = datetime.now(IST).strftime("%H:%M")
    lines.append(f"⏱ 5m Signals {now_str}\n")

    # header
    header = f"{'SYMBOL':<8} {'PRICE':>8} {'RSI':>6} {'VOLUME':>12} {'MODE':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for _, r in df.iterrows():
        sym = str(r.get('Symbol', ''))[:8]
        price_val = r.get('Last close')
        rsi_val = r.get('RSI(14)')
        vol_val = r.get('Last_5m_Volume')
        mode = str(r.get('Final Mode', ''))

        price = f"{price_val:.2f}" if pd.notna(price_val) else "NA"
        rsi = f"{rsi_val:.1f}" if pd.notna(rsi_val) else "NA"
        vol = f"{int(vol_val):,}" if pd.notna(vol_val) else "0"

        line = f"{sym:<8} {price:>8} {rsi:>6} {vol:>12} {mode:>12}"
        lines.append(line)

    return "\n".join(lines)



def within_market_hours():
    now = datetime.now(IST).time()
    return MARKET_START <= now <= MARKET_END

def main():
    # --- TEMP: always send a ping so we can test Telegram ---
    from datetime import datetime
    now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    send_telegram(f"✅ Test ping from Karthik_Robot at {now_str}")

    # If you commented out within_market_hours earlier, leave it commented for now
    # After testing, you can re-enable the time filter.
    #
    if not within_market_hours():
         return

    if not DHAN_TOKEN:
        send_telegram("⚠️ Password missing in GitHub Secrets")
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

    msg = build_signals_table(alerts)
    if msg:
        send_telegram(msg)

if __name__ == "__main__":
    main()
