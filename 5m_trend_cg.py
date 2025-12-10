# volume_board_fast.py

import time
import base64
from datetime import datetime, time as dt_time, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import streamlit as st
import pytz

# ========= CONFIG (UPDATED) =========
SYMBOLS_CSV = "Symbols.csv"
DHAN_BASE_URL = "https://api.dhan.co/v2"
DEFAULT_EXCHANGE_SEGMENT = "NSE"
DEFAULT_INSTRUMENT = "EQUITY"
DEFAULT_INTRADAY_INTERVAL = 5        # minutes

# --- Indicator Constants ---
RSI_PERIOD = 14
EMA_9 = 9
EMA_26 = 26
EMA_50 = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SR_LOOKBACK_CANDLES = 12 
ATR_PERIOD = 10 
ATR_MULTIPLIER = 3 
REFRESH_INTERVAL_SECONDS = 60  # Auto-refresh rate (logic via last_run)

# ========= GLOBAL TIME CONSTANTS =========
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN_TIME = dt_time(9, 15, 0)
MARKET_CLOSE_TIME = dt_time(15, 30, 0)
SWITCH_TO_TODAY_TIME = dt_time(9, 0, 0) 

# ========= HELPER: LOAD UNIVERSE (NO CHANGE) =========
@st.cache_data(show_spinner=False)
def load_universe_from_symbols(
    csv_path: str,
    max_symbols: int | None = None,
    min_ltp: float | None = None,
    max_ltp: float | None = None,
) -> list[dict]:
    """ Load Symbols.csv robustly. """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: Symbols file not found at '{csv_path}'. Please check the path and filename.")
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
        raise ValueError("No symbol column found.")

    if col_series and col_series in df.columns:
        df = df[df[col_series].astype(str).str.upper() == "EQ"]

    if col_ltp and col_ltp in df.columns:
        df[col_ltp] = pd.to_numeric(df[col_ltp], errors="coerce")
        if min_ltp is not None:
            df = df[df[col_ltp] >= min_ltp]
        if max_ltp is not None:
            df = df[df[col_ltp] <= max_ltp]
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

# ========= HELPER: FETCH CANDLE DATA =========
def fetch_candle_data(
    dhan_token: str,
    security_id: int,
) -> dict | None:
    """Fetches intraday candles for the target day and returns last candle + history."""
    now_ist = datetime.now(IST)
    today = now_ist.date()
    target_date = today

    # Before 9:00 use previous trading day
    if now_ist.time() < SWITCH_TO_TODAY_TIME:
        lookback_date = today - timedelta(days=1)
        while lookback_date.weekday() > 4:  # skip weekends
            lookback_date -= timedelta(days=1)
        target_date = lookback_date

    from_dt = IST.localize(datetime.combine(target_date, MARKET_OPEN_TIME))
    to_dt = IST.localize(datetime.combine(target_date, MARKET_CLOSE_TIME))
    if from_dt >= to_dt:
        return None

    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")

    url = f"{DHAN_BASE_URL}/charts/intraday"
    headers = {
        "access-token": dhan_token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "securityId": str(security_id),
        "exchangeSegment": "NSE_EQ",               # <— important
        "instrument": DEFAULT_INSTRUMENT,          # "EQUITY"
        "interval": str(DEFAULT_INTRADAY_INTERVAL),# "5"
        "oi": False,
        "fromDate": from_str,
        "toDate": to_str,
    }

    try:
        r = requests.post(url, headers=headers, json=body, timeout=10)
        r.raise_for_status()
    except Exception:
        return None

    data = r.json() or {}

    # New v2 format: arrays
    if all(k in data for k in ("open", "high", "low", "close", "volume", "timestamp")):
        df = pd.DataFrame({
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "timestamp": data["timestamp"],
        })
        if df.empty:
            return None

        # convert timestamp (epoch) to datetime IST
        df["Datetime"] = df["timestamp"].apply(
            lambda ts: datetime.fromtimestamp(ts, tz=IST)
        )
        last = df.iloc[-1]
        dt_ist = last["Datetime"]

        return {
            "Datetime": dt_ist,
            "Open": float(last["open"]),
            "High": float(last["high"]),
            "Low": float(last["low"]),
            "Close": float(last["close"]),
            "Volume": int(last["volume"]),
            "df": df,  # full history for indicators
        }

    # Old format fallback: list of candles (if Dhan still supports it)
    candles = data.get("data") or data.get("candles") or []
    if not candles:
        return None

    df = pd.DataFrame(candles)
    if df.empty:
        return None

    last = df.iloc[-1]
    ts = last.get("startTime") or last.get("time") or last.get("datetime")
    if isinstance(ts, str):
        try:
            ts = ts.replace("T", " ")
            dt_ist = IST.localize(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        except Exception:
            dt_ist = now_ist
    else:
        dt_ist = now_ist

    return {
        "Datetime": dt_ist,
        "Open": float(last.get("open")),
        "High": float(last.get("high")),
        "Low": float(last.get("low")),
        "Close": float(last.get("close")),
        "Volume": int(last.get("volume")),
        "df": df,
    }



# ========= HELPER: S&R ESTIMATION & CPR =========
def get_pivot_levels(df: pd.DataFrame) -> dict:
    """Estimates short-term S&R and calculates CPR from history."""
    
    df['High'] = pd.to_numeric(df['high'], errors='coerce')
    df['Low'] = pd.to_numeric(df['low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['close'], errors='coerce')
    
    recent_df = df.tail(SR_LOOKBACK_CANDLES)
    resistance = recent_df['High'].max()
    support = recent_df['Low'].min()
    
    high_full = df['High'].max()
    low_full = df['Low'].min()
    close_full = df['Close'].iloc[-1]
    
    if pd.isna(high_full) or pd.isna(low_full) or pd.isna(close_full):
        return {"Resistance": None, "Support": None, "CPR_C": None, "CPR_B": None, "CPR_T": None}
    
    # Classical Pivot Calculation
    Pivot = (high_full + low_full + close_full) / 3
    R1 = (2 * Pivot) - low_full
    S1 = (2 * Pivot) - high_full
    
    # CPR Calculation
    CPR_C = Pivot
    CPR_B = (high_full + low_full) / 2
    CPR_T = (Pivot - CPR_B) + Pivot
    
    return {
        "Resistance": round(R1, 2), 
        "Support": round(S1, 2), 
        "CPR_C": round(CPR_C, 2),
        "CPR_B": round(CPR_B, 2),
        "CPR_T": round(CPR_T, 2)
    }

# ========= HELPER: DIRECTIONAL ANALYSIS =========
def analyze_direction(df: pd.DataFrame) -> dict:
    """EMA stacking + last 2-candle HH/HL / LL/LH detection."""
    
    df['Close'] = pd.to_numeric(df['close'], errors='coerce')
    df['High'] = pd.to_numeric(df['high'], errors='coerce')
    df['Low'] = pd.to_numeric(df['low'], errors='coerce')
    
    # --- 1. EMA Stacking ---
    df[f'EMA{EMA_9}'] = df['Close'].ewm(span=EMA_9, adjust=False).mean()
    df[f'EMA{EMA_26}'] = df['Close'].ewm(span=EMA_26, adjust=False).mean()
    df[f'EMA{EMA_50}'] = df['Close'].ewm(span=EMA_50, adjust=False).mean()
    
    if len(df) < EMA_50:
        return {"EMA_Direction": "N/A", "HLC_Direction": "N/A"}

    e9 = df[f'EMA{EMA_9}'].iloc[-1]
    e26 = df[f'EMA{EMA_26}'].iloc[-1]
    e50 = df[f'EMA{EMA_50}'].iloc[-1]
    
    if e9 > e26 and e26 > e50:
        ema_dir = "BULLISH STACK"
    elif e9 < e26 and e26 < e50:
        ema_dir = "BEARISH STACK"
    else:
        ema_dir = "MIXED"

    # --- 2. HH/HL/LL/LH (Simplified to last 10 candles) ---
    recent_df = df.tail(10).copy()
    
    if len(recent_df) >= 2:
        hh = recent_df['High'].iloc[-1] > recent_df['High'].iloc[-2]
        hl = recent_df['Low'].iloc[-1] > recent_df['Low'].iloc[-2]
        ll = recent_df['Low'].iloc[-1] < recent_df['Low'].iloc[-2]
        lh = recent_df['High'].iloc[-1] < recent_df['High'].iloc[-2]
        
        if (hh and hl):
            hlc_dir = "HH/HL (UP)"
        elif (ll and lh):
            hlc_dir = "LL/LH (DOWN)"
        else:
            hlc_dir = "SIDEWAYS/CHOP"
    else:
        hlc_dir = "N/A"
        
    return {
        "EMA_Direction": ema_dir,
        "HLC_Direction": hlc_dir
    }

# ========= HELPER: SUPER TREND =========
def calculate_supertrend(df: pd.DataFrame) -> str:
    """Calculates SuperTrend direction (requires ATR)."""
    
    df['High'] = pd.to_numeric(df['high'], errors='coerce')
    df['Low'] = pd.to_numeric(df['low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['close'], errors='coerce')
    
    if len(df) < ATR_PERIOD:
        return "N/A"

    df['HL'] = df['High'] - df['Low']
    df['HC'] = abs(df['High'] - df['Close'].shift(1))
    df['LC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)

    df['ATR'] = df['TR'].ewm(span=ATR_PERIOD, adjust=False).mean()

    df['Basic Upper Band'] = (df['High'] + df['Low']) / 2 + ATR_MULTIPLIER * df['ATR']
    df['Basic Lower Band'] = (df['High'] + df['Low']) / 2 - ATR_MULTIPLIER * df['ATR']

    last_close = df['Close'].iloc[-1]
    last_upper = df['Basic Upper Band'].iloc[-1]
    last_lower = df['Basic Lower Band'].iloc[-1]
    
    if last_close > last_lower:
        return "UPTREND"
    elif last_close < last_upper:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

# ========= HELPER: INDICATOR CALCULATION =========
def calculate_advanced_indicators(df: pd.DataFrame) -> dict:
    """
    Calculates required indicators (RSI, VWAP, MACD)
    """
    
    df['Close'] = pd.to_numeric(df['close'], errors='coerce')
    df['High'] = pd.to_numeric(df['high'], errors='coerce')
    df['Low'] = pd.to_numeric(df['low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    # --- 1. VWAP Proxy ---
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['TP'] * df['Volume']
    df['VWAP'] = df['TPV'].cumsum() / df['Volume'].cumsum()
    last_vwap = df['VWAP'].iloc[-1]

    # --- 2. MACD Calculation ---
    if len(df) >= MACD_SLOW + MACD_SIGNAL:
        df['EMA_Fast'] = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
        df['MACD_Signal_Line'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()

        last_macd = df['MACD'].iloc[-1]
        last_signal_line = df['MACD_Signal_Line'].iloc[-1]
        macd_signal_status = last_macd > last_signal_line
    else:
        last_macd = None
        last_signal_line = None
        macd_signal_status = None

    # --- 3. RSI Calculation (14) ---
    if len(df) >= RSI_PERIOD + 1:
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        last_rsi = df['RSI'].iloc[-1]
    else:
        last_rsi = None
    
    return {
        "VWAP": round(last_vwap, 2) if last_vwap is not None else None,
        "MACD": round(last_macd, 3) if last_macd is not None else None,
        "MACD_Signal_Line": round(last_signal_line, 3) if last_signal_line is not None else None,
        "MACD_Signal": "CROSS UP" if macd_signal_status is True else "CROSS DOWN" if macd_signal_status is False else "WAIT",
        "RSI": round(last_rsi, 2) if last_rsi is not None else None
    }


# ========= CORE SCANNER PHASE 1: VOLUME FILTERING =========
def run_volume_filter_scan(
    dhan_token: str,
    universe: list[dict],
    min_last_close: float,
    min_last_5m_volume: int,
    max_workers: int,
) -> pd.DataFrame:
    """ Phase 1: Scan universe for last completed 5m candle and apply initial filters. """
    results = []

    def worker(item: dict) -> dict:
        symbol = item["symbol"]
        name = item["name"]
        secid = item.get("securityId", None)
        ltp = item.get("LTP", None)

        if secid is None:
            return {
                "Company name": name, "Symbol": symbol, "Status": "Missing security id",
                "Last close": None, "Time": None, "Last_5m_Volume": None, "LTP": ltp, 
                "SecurityId": secid, "df": None
            }

        try:
            candle_data = fetch_candle_data(dhan_token=dhan_token, security_id=secid)
        except Exception:
            return {
                "Company name": name, "Symbol": symbol, "Status": "Error: Dhan API issue",
                "Last close": None, "Time": None, "Last_5m_Volume": None, "LTP": ltp,
                "SecurityId": secid, "df": None
            }

        if candle_data is None:
            return {
                "Company name": name, "Symbol": symbol, "Status": "No intraday data",
                "Last close": None, "Time": None, "Last_5m_Volume": None, "LTP": ltp,
                "SecurityId": secid, "df": None
            }

        last_close = float(candle_data["Close"])
        vol = int(candle_data["Volume"])
        t = candle_data["Datetime"].strftime("%H:%M")
        
        status = "OK"
        if last_close < min_last_close or vol < min_last_5m_volume:
            status = "Filtered"

        return {
            "Company name": name, "Symbol": symbol, "Status": status,
            "Last close": round(last_close, 2), "Time": t,
            "Last_5m_Volume": vol, "LTP": ltp,
            "SecurityId": secid,
            "df": candle_data["df"]
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, u): u for u in universe}
        for fut in as_completed(futures):
            results.append(fut.result())

    df = pd.DataFrame(results)
    df['sort_order'] = df['Status'].apply(lambda x: 0 if x == 'OK' else 1 if x == 'Filtered' else 2)
    df = df.sort_values(by=['sort_order', 'Last_5m_Volume'], ascending=[True, False]).drop(columns=['sort_order'])
    
    return df


# ========= CORE SCANNER PHASE 2: INDICATOR ANALYSIS =========
def run_indicator_analysis(
    df_filtered: pd.DataFrame,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Phase 2: Takes the required symbols and calculates indicators."""
    indicator_results = []
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    def indicator_worker(row):
        symbol = row["Symbol"]
        name = row["Company name"]
        history_df = row["df"]
        last_close = row["Last close"]
        
        # --- Handle missing data ---
        if history_df is None or history_df.empty or last_close is None:
            return {
                "Company name": name, "Symbol": symbol, "Time": row["Time"], "Last close": last_close,
                "Last_5m_Volume": row["Last_5m_Volume"], "Final Mode": "NO DATA", "Score": -99,
                "HLC Direction": "N/A", "EMA Direction": "N/A", "VWAP": None, "RSI(14)": None, 
                "MACD Signal": "N/A", "Supertrend": "N/A", "CPR": None, 
                "Resistance": None, "Support": None, "df": history_df 
            }
        
        try:
            # Calculate all indicators
            advanced = calculate_advanced_indicators(history_df)
            pivot_levels = get_pivot_levels(history_df)
            directionals = analyze_direction(history_df)
            supertrend_dir = calculate_supertrend(history_df)
        except Exception as e:
            return {
                "Company name": name, "Symbol": symbol, "Time": row["Time"], "Last close": last_close,
                "Last_5m_Volume": row["Last_5m_Volume"], "Final Mode": f"CALC ERROR: {e.__class__.__name__}", "Score": -99,
                "HLC Direction": "N/A", "EMA Direction": "N/A", "VWAP": None, "RSI(14)": None, 
                "MACD Signal": "N/A", "Supertrend": "N/A", "CPR": None, 
                "Resistance": None, "Support": None, "df": history_df 
            }
        
        # --- Decision Logic (Scoring) ---
        score = 0
        
        # 1. Price vs VWAP (Strongest)
        if advanced['VWAP'] is not None and last_close is not None:
            if last_close > advanced['VWAP']:
                score += 2
            elif last_close < advanced['VWAP']:
                score -= 2

        # 2. EMA Direction
        if directionals['EMA_Direction'] == "BULLISH STACK":
            score += 2
        elif directionals['EMA_Direction'] == "BEARISH STACK":
            score -= 2
        
        # 3. MACD Signal
        if advanced['MACD_Signal'] == "CROSS UP":
            score += 1
        elif advanced['MACD_Signal'] == "CROSS DOWN":
            score -= 1

        # 4. RSI (Oversold/Overbought)
        if advanced['RSI'] is not None:
            if advanced['RSI'] > 60:
                score += 1
            elif advanced['RSI'] < 40:
                score -= 1

        # 5. Supertrend
        if supertrend_dir == "UPTREND":
            score += 1
        elif supertrend_dir == "DOWNTREND":
            score -= 1

        # 6. HLC (Micro-trend)
        if directionals['HLC_Direction'] == "HH/HL (UP)":
            score += 1
        elif directionals['HLC_Direction'] == "LL/LH (DOWN)":
            score -= 1
        
        # Final Mode Decision
        if score >= 4:
            final_mode = "STRONG BUY"
        elif score >= 1:
            final_mode = "BUY MODE"
        elif score <= -4:
            final_mode = "STRONG SELL"
        elif score <= -1:
            final_mode = "SELL MODE"
        else:
            final_mode = "NEUTRAL/WAIT"
            
        return {
            "Company name": name,
            "Symbol": symbol,
            "Time": row["Time"],
            "Last close": last_close,
            "Last_5m_Volume": row["Last_5m_Volume"],
            "Final Mode": final_mode,
            "Score": score,
            "VWAP": advanced['VWAP'],
            "RSI(14)": advanced['RSI'],
            "MACD Signal": advanced['MACD_Signal'],
            "Supertrend": supertrend_dir,
            "HLC Direction": directionals['HLC_Direction'],
            "EMA Direction": directionals['EMA_Direction'],
            "CPR": pivot_levels['CPR_C'],
            "Resistance": pivot_levels['Resistance'],
            "Support": pivot_levels['Support'],
            "df": history_df 
        }

    ok_list = df_filtered.to_dict('records')
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(indicator_worker, item) for item in ok_list]
        for fut in as_completed(futures):
            indicator_results.append(fut.result())

    df_indicators = pd.DataFrame(indicator_results)
    df_indicators = df_indicators.sort_values(by='Score', ascending=False)
    
    return df_indicators


# ========= SIMPLE BEEP (FIXED) =========
BEEP_WAV_BASE64 = (
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA="
)
BEEP_WAV_BYTES = base64.b64decode(BEEP_WAV_BASE64)

def play_alert_if_needed(df: pd.DataFrame, volume_threshold: int):
    if df.empty:
        return
    mask = (df["Status"] == "OK") & (df["Last_5m_Volume"].fillna(0) >= volume_threshold)
    if mask.any():
        st.audio(BEEP_WAV_BYTES, format="audio/wav", start_time=0)


# ========= STREAMLIT APP (MAIN FUNCTION) =========
def main():
    # Set up auto-run timing (no infinite rerun, just logic gating)
    if 'last_run' not in st.session_state:
        st.session_state.last_run = time.time()
    
    time_since_last_run = time.time() - st.session_state.last_run
    remaining = max(0, REFRESH_INTERVAL_SECONDS - int(time_since_last_run))

    st.set_page_config(
        page_title="Fast 5-min Volume & Trend Board",
        layout="wide",
    )

    st.title("⚡ Fast 5-minute Volume Spike & Advanced Decision Board")
    st.subheader(f"Current Time (IST): {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"***Auto-refresh trigger in ~{remaining} seconds (based on last run).***")

    # --- Sidebar: Filter and Control Settings ---
    st.sidebar.header("Welcome")
    dhan_token = st.sidebar.text_input("Password", type="password")

    st.sidebar.header("Universe / Price filter")
    max_symbols = st.sidebar.number_input(
        "Max symbols to load from Symbols.csv", min_value=10, max_value=5000, value=1000, step=50,
    )
    min_ltp = st.sidebar.number_input("Min LTP (₹)", min_value=0.0, value=10.0, step=1.0)
    max_ltp = st.sidebar.number_input("Max LTP (₹)", min_value=0.0, value=500.0, step=10.0)

    st.sidebar.header("Candle filters")
    min_last_close = st.sidebar.number_input(
        "Min last-close price (₹)", min_value=0.0, value=10.0, step=1.0
    )
    min_last_5m_volume = st.sidebar.number_input(
        "Min last 5-min volume", min_value=0, value=200_000, step=10_000
    )

    st.sidebar.header("Advanced")
    max_workers = st.sidebar.number_input(
        "Parallel workers", min_value=1, max_value=64, value=16, step=1
    )
    alert_threshold = st.sidebar.number_input(
        "Sound alert: volume ≥", min_value=0, value=200_000, step=10_000
    )
    st.sidebar.markdown(f"**S&R Lookback:** {SR_LOOKBACK_CANDLES * DEFAULT_INTRADAY_INTERVAL} minutes")

    run_btn = st.sidebar.button("🚀 Run Advanced Decision Scan (Manual Refresh)")

    # Decide whether to run scan this loop
    should_run_scan = False
    if run_btn:
        st.session_state.last_run = time.time()
        should_run_scan = True
    elif time_since_last_run >= REFRESH_INTERVAL_SECONDS:
        st.session_state.last_run = time.time()
        should_run_scan = True

    # --- Decision Board Reference (always visible) ---
    st.header("📚 Decision Board Reference")
    
    col_ref_1, col_ref_2 = st.columns(2)
    
    with col_ref_1:
        st.markdown("##### Table A: Indicator Scoring (Phase 2)")
        scoring_data = {
            "Indicator Signal": [
                "Price vs VWAP (Bullish/Bearish)", 
                "EMA Direction (Stack Bullish/Bearish)", 
                "MACD (Cross Up/Cross Down)", 
                "RSI (>60 / <40)", 
                "Supertrend (Uptrend/Downtrend)",
                "HLC Direction (HH/HL or LL/LH)"
            ],
            "Points Added": [
                "+2 / -2", 
                "+2 / -2", 
                "+1 / -1", 
                "+1 / -1", 
                "+1 / -1",
                "+1 / -1"
            ]
        }
        st.table(pd.DataFrame(scoring_data))

    with col_ref_2:
        st.markdown("##### Table B: Final Mode Decision")
        mode_data = {
            "Total Score Range": [
                "≥ 4",           # STRONG BUY
                "1 to 3",        # BUY MODE
                "0",             # NEUTRAL/WAIT
                "-1 to -3",      # SELL MODE
                "≤ -4",          # STRONG SELL
            ],
            "Final Mode": [
                "STRONG BUY (High Conviction)", 
                "BUY MODE (Bullish Bias)", 
                "NEUTRAL/WAIT", 
                "SELL MODE (Bearish Bias)", 
                "STRONG SELL (High Conviction)"
            ],
            "Row Highlight": ["Green", "Green", "Gray", "Red", "Red"]
        }
        st.table(pd.DataFrame(mode_data))

    st.markdown("---")

    # If we are NOT running the scan now, show guidance and exit
    if not should_run_scan:
        now_ist = datetime.now(IST)
        st.info("Set your filters on the left and click **Run Advanced Decision Scan**, "
                "or wait for the next auto-trigger window.")
        if now_ist.time() < SWITCH_TO_TODAY_TIME:
            st.warning(
                f"Market is closed now ({now_ist.strftime('%H:%M')}). "
                "The scan will fetch data from the **previous trading day** when run."
            )
        else:
            st.info(
                "Market is open or in pre-open phase. "
                "The scan will fetch data for **today**."
            )
        if not dhan_token:
            st.error("Please enter your Dhan access token on the left to enable scanning.")
        return

    # If we *should* run, but no token, stop here
    if not dhan_token:
        st.error("Please enter your Dhan access token on the left.")
        return

    # --- Phase 1: Volume Filter Scan ---
    st.header("1️⃣ Phase 1: Volume Filtering (Table 1)")
    t0 = time.perf_counter()

    with st.spinner("Loading universe from Symbols.csv..."):
        universe = load_universe_from_symbols(
            SYMBOLS_CSV, max_symbols=max_symbols, min_ltp=min_ltp, max_ltp=max_ltp,
        )
    if not universe:
        st.error("Universe is empty after filtering.")
        return

    with st.spinner(
        f"Scanning {len(universe)} symbols for last completed 5-minute volume..."
    ):
        df_volume_filter = run_volume_filter_scan(
            dhan_token=dhan_token,
            universe=universe,
            min_last_close=min_last_close,
            min_last_5m_volume=min_last_5m_volume,
            max_workers=max_workers,
        )

    elapsed_p1 = time.perf_counter() - t0
    st.success(
        f"Phase 1 complete in **{elapsed_p1:0.1f} seconds**. "
        f"Rows: **{len(df_volume_filter)}**."
    )

    # Display Table 1
    df_table1_display = df_volume_filter.drop(columns=['SecurityId', 'df'], errors='ignore')
    st.dataframe(df_table1_display, height=300, use_container_width=True)
    
    # --- Prepare Data for Phase 2 ---
    df_ok = df_volume_filter[df_volume_filter["Status"] == "OK"].copy()
    
    ok_symbols_count = len(df_ok)
    
    # FALLBACK LOGIC
    df_to_analyze = df_ok
    fallback_mode = False
    
    if ok_symbols_count == 0:
        if not df_volume_filter.empty:
            df_to_analyze = df_volume_filter.head(1).copy()
            fallback_mode = True
            st.warning(
                "No symbols passed the volume/price filters. **FALLBACK MODE ACTIVATED:** "
                f"Analyzing the first symbol, **{df_to_analyze.iloc[0]['Symbol']}**, for testing."
            )
        else:
            st.info("The entire scan is empty. Cannot proceed to trend analysis.")
            return
            
    # --- Phase 2: Indicator Analysis ---
    st.header("2️⃣ Phase 2: Advanced Decision Board (Table 2)")
    t1 = time.perf_counter()

    with st.spinner(f"Analyzing trends and signals for {len(df_to_analyze)} symbol(s)..."):
        df_indicators = run_indicator_analysis(df_to_analyze, max_workers=max_workers)

    elapsed_p2 = time.perf_counter() - t1
    elapsed_total = time.perf_counter() - t0
    
    st.success(
        f"Phase 2 complete in **{elapsed_p2:0.1f} seconds** (Total: **{elapsed_total:0.1f}s**)."
    )
    
    if not fallback_mode:
        try:
            play_alert_if_needed(df_volume_filter, alert_threshold)
        except Exception:
            pass
        
    if df_indicators.empty:
        st.warning("Trend analysis resulted in an empty table (likely due to errors or insufficient data history).")
        return
    
    # ----------------------------------------------------
    # HIGHLIGHT TOP SYMBOL DETAILS
    # ----------------------------------------------------
    top_symbol = df_indicators.iloc[0]
    
    st.subheader(f"🥇 TOP PRIORITY DECISION: {top_symbol['Company name']} ({top_symbol['Symbol']})")
    
    def safe_format(value, prefix="₹", fmt=".2f"):
        if pd.isna(value) or value is None:
            return "N/A"
        if prefix == "₹":
            return f"₹{value:{fmt}}"
        else:
            return f"{value:{fmt}}"

    col1, col2, col3, col4, col5 = st.columns(5) 
    
    with col1:
        st.metric("DECISION", top_symbol['Final Mode'])
        st.metric("Score", top_symbol['Score'])
    
    with col2:
        st.metric("Last Close Price", safe_format(top_symbol['Last close']))
        st.metric("Last 5m Volume", safe_format(top_symbol['Last_5m_Volume'], prefix="", fmt=","))
        
    with col3:
        st.metric("VWAP Level", safe_format(top_symbol['VWAP']))
        st.metric("RSI(14)", safe_format(top_symbol['RSI(14)'], prefix="", fmt=".2f"))
        
    with col4:
        st.metric("CPR Level (Center)", safe_format(top_symbol['CPR']))
        st.metric("Supertrend Dir", top_symbol['Supertrend'])

    with col5:
        st.metric("Resistance (R1)", safe_format(top_symbol['Resistance']))
        st.metric("Support (S1)", safe_format(top_symbol['Support']))
        
    st.markdown("---")
    
    # Display the full Indicator Table 2
    st.markdown("#### Full List of Confirmed Symbols")

    # Drop the history dataframe column for final table display
    df_table2_display = df_indicators.drop(columns=['df'], errors='ignore')
    
    # Explicitly define all requested columns for display in the order you want
    display_cols = [
        "Company name", "Symbol", 
        "Final Mode", "Score", 
        "Last close", "Last_5m_Volume", 
        "VWAP", "RSI(14)", 
        "MACD Signal", "Supertrend", 
        "EMA Direction", "HLC Direction", 
        "CPR", "Resistance", "Support"
    ]
    df_table2_display = df_table2_display[[c for c in display_cols if c in df_table2_display.columns]]

    # Conditional Highlighting function for the entire ROW
    def highlight_mode_row(row):
        color = ''
        if 'BUY' in str(row['Final Mode']):
            color = 'background-color: #d4edda'  # Light Green
        elif 'SELL' in str(row['Final Mode']):
            color = 'background-color: #f8d7da'  # Light Red
        return [color] * len(row)

    st.dataframe(
        df_table2_display.style.apply(highlight_mode_row, axis=1), 
        height=400,
        use_container_width=True,
        column_config={
            "Last_5m_Volume": st.column_config.NumberColumn("Last 5m Vol", format="%d"),
            "Last close": st.column_config.NumberColumn("Close (₹)", format="%.2f"),
            "VWAP": st.column_config.NumberColumn("VWAP (₹)", format="%.2f"),
            "RSI(14)": st.column_config.NumberColumn("RSI", format="%.2f"),
            "CPR": st.column_config.NumberColumn("CPR (₹)", format="%.2f"),
            "Resistance": st.column_config.NumberColumn("R (₹)", format="%.2f"),
            "Support": st.column_config.NumberColumn("S (₹)", format="%.2f"),
            "Final Mode": st.column_config.TextColumn("Decision Mode", width="small"),
            "EMA Direction": st.column_config.TextColumn("EMA Dir", width="small"),
            "HLC Direction": st.column_config.TextColumn("HH/HL Dir", width="small"),
            "Supertrend": st.column_config.TextColumn("ST Dir", width="small"),
        }
    )
    
    # Download button for the final analysis
    csv = df_table2_display.to_csv(index=False)
    st.download_button(
        "⬇️ Download Advanced Decision Analysis (Table 2) as CSV",
        data=csv,
        file_name="volume_decision_board.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
