"""
app_test.py - Streamlit app for High Yield Trading Bot

- This app allows you to monitor and trade high-yield assets.
- It uses the Alpaca API for trading and data.
- By default, it runs in paper trading mode.
"""
import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Try to import TA-Lib; if not available, fallback to simple pandas implementations for SMA, MACD, RSI
try:
    import talib
    HAS_TALIB = True
except Exception:
    HAS_TALIB = False

# === CONFIG ===
st.set_page_config(page_title="High Yield Trading Bot", layout="wide", initial_sidebar_state="expanded")

# Secrets (use .streamlit/secrets.toml or env vars)
ALPACA_KEY = st.secrets.get("APCA_API_KEY_ID", os.getenv("APCA_API_KEY_ID"))
ALPACA_SECRET = st.secrets.get("APCA_SECRET_KEY", os.getenv("APCA_SECRET_KEY"))
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN"))
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID"))

SYMBOLS = ['EARN', 'CRF', 'CLM', 'ECC', 'GAIN']
INITIAL_CAPITAL = 5000
DIV_HOLD_PCT = 0.50
TRADE_PCT = 0.50
MAX_TRADE_PCT = 0.10
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10

# ------------------------------
# Data helpers
# ------------------------------
def fetch_bars(symbol, days=210):
    data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    end = datetime.now() - timedelta(minutes=16)
    start = end - timedelta(days=days+10)
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start,
        end=end
    )
    bars = data_client.get_stock_bars(request_params)
    df = bars.df
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={'timestamp': 'date'})
    df.columns = [c.lower() for c in df.columns]
    return df

# ------------------------------
# Indicator functions
# ------------------------------
def sma(series, n):
    return series.rolling(window=n, min_periods=1).mean()

def macd(series, fast=7, slow=49, signal=7):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    r = 100 - (100 / (1 + rs))
    return r.fillna(50)

def ttm_squeeze(close, high, low, n=7, bb_mult=2.0, kc_mult=1.5):
    if HAS_TALIB:
        try:
            sma_n = talib.SMA(close, n)[-1]
            atr = talib.ATR(high, low, close, n)[-1]
            bb_upper, _, bb_lower = talib.BBANDS(close, timeperiod=n, nbdevup=bb_mult, nbdevdn=bb_mult)
            kc_upper = sma_n + (kc_mult * atr)
            kc_lower = sma_n - (kc_mult * atr)
            return (bb_upper[-1] - bb_lower[-1]) < (kc_upper - kc_lower)
        except Exception:
            pass
    sma_n = sma(close, n)
    atr_n = (high - low).ewm(span=n, adjust=False).mean()
    bb_upper = sma_n + (close.rolling(n).std() * bb_mult)
    bb_lower = sma_n - (close.rolling(n).std() * bb_mult)
    kc_upper = sma_n + (kc_mult * atr_n)
    kc_lower = sma_n - (kc_mult * atr_n)
    return (bb_upper - bb_lower).iloc[-1] < (kc_upper - kc_lower).iloc[-1]

def generate_signal(df):
    closes = df['close']
    highs = df['high']
    lows = df['low']
    if len(closes) < 203:
        return 0.0
    if HAS_TALIB:
        try:
            ma14 = talib.SMA(closes.values, 14)[-1]
            ma49 = talib.SMA(closes.values, 49)[-1]
            ma105 = talib.SMA(closes.values, 105)[-1]
            ma203 = talib.SMA(closes.values, 203)[-1]
            macd_line, macd_signal, _ = talib.MACD(closes.values, fastperiod=7, slowperiod=49, signalperiod=7)
            rsi10 = talib.RSI(closes.values, 10)[-1]
        except Exception:
            ma14 = sma(closes, 14).iloc[-1]
            ma49 = sma(closes, 49).iloc[-1]
            ma105 = sma(closes, 105).iloc[-1]
            ma203 = sma(closes, 203).iloc[-1]
            macd_line, macd_signal, _ = macd(closes, 7, 49, 7)
            rsi10 = rsi(closes, 10).iloc[-1]
    else:
        ma14 = sma(closes, 14).iloc[-1]
        ma49 = sma(closes, 49).iloc[-1]
        ma105 = sma(closes, 105).iloc[-1]
        ma203 = sma(closes, 203).iloc[-1]
        macd_line, macd_signal, _ = macd(closes, 7, 49, 7)
        rsi10 = rsi(closes, 10).iloc[-1]

    ma_aligned = (ma14 > ma49) and (ma49 > ma105) and (ma105 > ma203)
    macd_bull = macd_line.iloc[-1] > macd_signal.iloc[-1]
    squeeze_on = ttm_squeeze(closes, highs, lows)
    rsi_safe = rsi10 < 70
    advanced_buy = ma_aligned and macd_bull and squeeze_on and rsi_safe
    sma20 = sma(closes, 20).iloc[-1]
    rsi14 = rsi(closes, 14).iloc[-1]
    old_buy = closes.iloc[-1] > sma20 and rsi14 < 70
    old_sell = closes.iloc[-1] < sma20 and rsi14 > 30

    if advanced_buy:
        if old_sell or not old_buy:
            return 0.5
        return 1.0
    return 0.0

# ------------------------------
# App code (UI)
# ------------------------------
st.title("High Yield Trading Bot")

st.markdown("**EARN • CRF • CLM • ECC • GAIN** | $5K Allocated: 50% Hold (Dividends) + 50% Trade")
st.sidebar.markdown("## Controls")

paper_trading = st.sidebar.checkbox("Paper Trading", value=True)

trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=paper_trading)

if st.sidebar.button("Allocate Initial $5K"):
    st.info("Allocating initial capital...")
    for sym in SYMBOLS:
        df = fetch_bars(sym, days=210)
        if df.empty:
            st.warning(f"No data for {sym}")
            continue
        price = df['close'].iloc[-1]
        per_stock = INITIAL_CAPITAL / len(SYMBOLS)
        hold_per = per_stock * DIV_HOLD_PCT
        hold_qty = int(hold_per / price)
        if hold_qty > 0:
            try:
                market_order_data = MarketOrderRequest(
                    symbol=sym,
                    qty=hold_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                trading_client.submit_order(order_data=market_order_data)
                st.success(f"HOLD: Bought {hold_qty} {sym} at ~{price:.2f}")
            except Exception as e:
                st.error(f"Error buying {sym}: {e}")

if st.sidebar.button("Run Backtest (quick)"):
    st.info("Run backtest from the separate backtest.py for more detailed results. Use run-backtest.ps1 to run it now.")

# Metrics
try:
    account = trading_client.get_account()
    col1, col2, col3 = st.columns(3)
    col1.metric("Equity", f"${float(account.equity):,.2f}")
    col2.metric("Cash", f"${float(account.cash):,.2f}")
    col3.metric("Positions", len(trading_client.get_all_positions()))
except Exception as e:
    st.warning("Account info not available: " + str(e))

st.markdown("---")

for symbol in SYMBOLS:
    col1, col2 = st.columns([1, 2])
    df = fetch_bars(symbol, days=400)
    if df.empty:
        col1.markdown(f"### {symbol}")
        col1.write("No data")
        continue

    price = df['close'].iloc[-1]
    qty = 0
    try:
        pos = trading_client.get_position(symbol)
        qty = int(pos.qty) if pos else 0
    except Exception:
        qty = 0

    is_hold_portion = qty >= (INITIAL_CAPITAL * DIV_HOLD_PCT / len(SYMBOLS) / price)

    sig_value = generate_signal(df)
    signal = "HOLD (Div)" if is_hold_portion else ("BUY" if sig_value > 0 else "SELL/FLAT")
    col1.markdown(f"### {symbol}")
    col1.metric("Price", f"${price:.2f}")
    col1.metric("Qty", qty)
    col1.write(f"**Action:** {signal}")

    # Chart
    fig = go.Figure([go.Scatter(x=df['date'], y=df['close'], name='Close')])
    if len(df) >= 14:
        fig.add_trace(go.Scatter(x=df.index, y=sma(df['close'], 14), name='SMA14'))
    if len(df) >= 20:
        fig.add_trace(go.Scatter(x=df.index, y=sma(df['close'], 20), name='SMA20'))
    if len(df) >= 49:
        fig.add_trace(go.Scatter(x=df.index, y=sma(df['close'], 49), name='SMA49'))
    if len(df) >= 105:
        fig.add_trace(go.Scatter(x=df.index, y=sma(df['close'], 105), name='SMA105'))
    if len(df) >= 203:
        fig.add_trace(go.Scatter(x=df.index, y=sma(df['close'], 203), name='SMA203'))
    col2.plotly_chart(fig, use_container_width=True)

    if col1.button(f"Execute {symbol}", key=f"exec_{symbol}"):
        if is_hold_portion:
            try:
                df = fetch_bars(symbol, days=10)
                price = df['close'].iloc[-1]
                hold_per = INITIAL_CAPITAL * DIV_HOLD_PCT / len(SYMBOLS)
                target_qty = int(hold_per / price)
                if target_qty > 0:
                    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=target_qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    trading_client.submit_order(order_data=market_order_data)
                    st.success(f"HOLD BUY {target_qty} {symbol}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            df = fetch_bars(symbol, days=400)
            sig = generate_signal(df)
            pos_exists = False
            try:
                pos = trading_client.get_position(symbol)
                pos_exists = True
            except Exception:
                pos_exists = False

            if sig == 0 and pos_exists:
                try:
                    trading_client.close_position(symbol)
                    st.success(f"SELL {symbol}")
                except Exception as e:
                    st.error(f"Error closing pos: {e}")
            elif sig > 0 and not pos_exists:
                try:
                    price = df['close'].iloc[-1]
                    trade_equity = INITIAL_CAPITAL * TRADE_PCT
                    target_pct = MAX_TRADE_PCT * (1 if sig == 1 else 0.5)
                    per_sym = trade_equity / len(SYMBOLS)
                    amount = per_sym * target_pct
                    qty = int(amount / price)
                    if qty > 0:
                        market_order_data = MarketOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                        trading_client.submit_order(order_data=market_order_data)
                        st.success(f"TRADE BUY {qty} {symbol}")
                    else:
                        st.info("calculated qty = 0; no order placed")
                except Exception as e:
                    st.error(f"Error placing trade: {e}")
            else:
                st.info("No action")
