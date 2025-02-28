#!/usr/bin/env python3
"""
Fully Featured Trading Bot using True Reinforcement Learning, a Web Dashboard,
market-hour stopping, and end-of-day watchlist discovery.

This script:
  - Uses yfinance to scan a watchlist for trading opportunities based on a composite
    of RSI, MACD, and SMA.
  - Trades via Alpaca's API based on the composite signal but only during US market hours.
  - Records trade performance and automatically tunes parameters over time using a true
    reinforcement learning approach (via Q-learning).
  - At market close, it "discovers" stocks by retaining only
    the best (up to 100 total) while removing the worst-performers.
  - Launches a Flask web portal so you can view the current settings, trade log, and asset signals.

Before running:
  - Install dependencies: 
        pip install alpaca-trade-api pandas numpy flask python-dotenv yfinance
  - Create a .env file or set your Alpaca API credentials as environment variables:
        ALPACA_KEY=your_api_key_here
        ALPACA_SECRET=your_api_secret_here
        ENDPOINT=https://paper-api.alpaca.markets
  - Make a file called "stocks.json" with a list of stock tickers (read-only full list).
  - Make a file called "watchlist.json" with a list of stock tickers you wish to trade.
  - Set ORDER_AMOUNT to the dollar amount you want to trade per trade.
  - Use Alpaca's paper trading endpoint for testing.
  - This example uses Python 3.11+.
"""

import os
import time
import datetime
import threading
import random
import numpy as np
import pandas as pd
from collections import deque
from flask import Flask, render_template, jsonify
import alpaca_trade_api as tradeapi
import json
from dotenv import load_dotenv
from zoneinfo import ZoneInfo  # For handling Eastern Time
import argparse
import logging
import yfinance as yf  # yfinance for free near-real-time data

# ==============================
# ARGUMENT PARSING & LOGGING SETUP
# ==============================
parser = argparse.ArgumentParser(description="Fully Featured Trading Bot with Debug Logging")
parser.add_argument('--debug', action='store_true', help='Enable debug logging')
args, unknown = parser.parse_known_args()
DEBUG = args.debug

WELCOME_MESSAGE = """
  _____                                                      _                   _____                   ___    _            
 |  __ \                                                    (_)                 |  __ \                 |__ \  ( )           
 | |__) |  _ __    ___     __ _   _ __    __ _   _ __ ___    _   _ __     __ _  | |__) |  _ __    ___      ) | |/   ___      
 |  ___/  | '__|  / _ \   / _` | | '__|  / _` | | '_ ` _ \  | | | '_ \   / _` | |  ___/  | '__|  / _ \    / /      / __|     
 | |      | |    | (_) | | (_| | | |    | (_| | | | | | | | | | | | | | | (_| | | |      | |    | (_) |  / /_      \__ \     
 |_|      |_|     \___/   \__, | |_|     \__,_| |_| |_| |_| |_| |_| |_|  \__, | |_|      |_|     \___/  |____|     |___/     
                           __/ |                                          __/ |                                              
   _____   _              |___/ _        _______                      _  |___/                             _                 
  / ____| | |                  | |      |__   __|                    | | (_)                       /\     | |                
 | (___   | |_    ___     ___  | | __      | |     _ __    __ _    __| |  _   _ __     __ _       /  \    | |   __ _    ___  
  \___ \  | __|  / _ \   / __| | |/ /      | |    | '__|  / _` |  / _` | | | | '_ \   / _` |     / /\ \   | |  / _` |  / _ \ 
  ____) | | |_  | (_) | | (__  |   <       | |    | |    | (_| | | (_| | | | | | | | | (_| |    / ____ \  | | | (_| | | (_) |
 |_____/   \__|  \___/   \___| |_|\_\      |_|    |_|     \__,_|  \__,_| |_| |_| |_|  \__, |   /_/    \_\ |_|  \__, |  \___/ 
                                                                                       __/ |                    __/ |        
                                                                                      |___/                    |___/         

                        Please star on GitHub: https://github.com/ProgramingPro2/Reinforcement-Stocks
"""

if DEBUG:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("debug.log", mode="w"),
            logging.StreamHandler()
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log", mode="w"),
            logging.StreamHandler()
        ]
    )

load_dotenv()

# ==============================
# CONFIGURATION & GLOBAL STATE
# ==============================

# Alpaca API credentials loaded from environment variables.
API_KEY = os.getenv('ALPACA_KEY', '')
API_SECRET = os.getenv('ALPACA_SECRET', '')
BASE_URL = os.getenv('ENDPOINT', 'https://paper-api.alpaca.markets')

# Initial strategy parameters (tunable)
default_params = {
    'RSI_PERIOD': 14,
    'RSI_OVERSOLD': 30,
    'RSI_OVERBOUGHT': 70,
    'ORDER_AMOUNT': 1000,  # Dollar amount per trade.
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'SMA_PERIOD': 30,
    'STOCH_K_PERIOD': 14,
    'STOCH_D_PERIOD': 3,
    'STOCH_OVERSOLD': 20,
    'STOCH_OVERBOUGHT': 80,
    'BOLLINGER_PERIOD': 20,
    'BOLLINGER_STD_DEV': 2,
    'OBV_SLOPE_PERIOD': 5,
    'ADX_PERIOD': 14,
    'ADX_THRESHOLD': 25,
    'CCI_PERIOD': 20,
    'CCI_OVERSOLD': -100,
    'CCI_OVERBOUGHT': 100,
    # Weights for the composite signal (roughly summing to 1)
    'RSI_WEIGHT': 0.125,
    'MACD_WEIGHT': 0.125,
    'SMA_WEIGHT': 0.125,
    'STOCH_WEIGHT': 0.125,
    'BOLLINGER_WEIGHT': 0.125,
    'OBV_WEIGHT': 0.125,
    'ADX_WEIGHT': 0.125,
    'CCI_WEIGHT': 0.125,
    # Composite signal decision thresholds:
    'BUY_THRESHOLD': 0.5,
    'SELL_THRESHOLD': -0.5,
    'WATCHLIST_SIZE': 100,
    'CHECK_INTERVAL_MINUTES': 5,
    'TUNE_INTERVAL_MINUTES': 10,
    'EMERGENCY_LOSS_THRESHOLD': 0.02
}
strategy_params = default_params.copy()
params_lock = threading.Lock()  # Protect access to strategy_params

working_dir = os.path.dirname(os.path.realpath(__file__))

print(f"Working directory: {working_dir}")

DATA_TIMEFRAME = '1d'
HIST_DAYS = 100          # Number of historical days used for indicator calculations
CHECK_INTERVAL_MINUTES = strategy_params['CHECK_INTERVAL_MINUTES']
CHECK_INTERVAL = 60 * CHECK_INTERVAL_MINUTES  # Check every x minutes

# Trade performance log (store recent trades in memory)
trade_log = deque(maxlen=100)
trade_log_lock = threading.Lock()

# Latest signals for dashboard (per symbol)
latest_signals = {}
signals_lock = threading.Lock()

# Files for persistence
WATCHLIST_FILE = os.path.join(working_dir, "watchlist.json")
DATA_FILE = os.path.join(working_dir, "data.json")

portfolio_history = deque(maxlen=1000)  # Store daily portfolio and index values
portfolio_history_lock = threading.Lock()

# ------------------------------
# Load Watchlist from watchlist.json
# ------------------------------
if os.path.exists(WATCHLIST_FILE):
    with open(WATCHLIST_FILE, "r") as f:
        try:
            WATCHLIST = json.load(f)
        except json.JSONDecodeError:
            logging.error("Error loading JSON from watchlist.json, initializing empty watchlist.")
            WATCHLIST = []
else:
    WATCHLIST = []

# ------------------------------
# Load Full Stocks (Candidate Pool) from stocks.json (read-only)
# ------------------------------
DEFAULT_CANDIDATE_POOL = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "DIS", "HD", "UNH", "BAC", "XOM", "VZ", "ADBE",
    "NFLX", "PFE", "KO", "CSCO", "CMCSA", "INTC", "T", "PEP", "ABT", "CRM",
    "ABBV", "ACN", "AVGO", "QCOM", "TXN", "COST", "NEE", "NKE", "MRK", "WFC",
    "LLY", "MDT", "MCD", "PM", "ORCL", "BA", "IBM", "HON", "AMGN"
]
STOCK_FILE = os.path.join(working_dir, "stocks.json")
def load_watchlist():
    if os.path.exists(STOCK_FILE):
        try:
            with open(STOCK_FILE, "r") as f:
                watchlist = json.load(f)
                if isinstance(watchlist, list) and all(isinstance(item, str) for item in watchlist):
                    logging.info(f"Loaded {len(watchlist)} stocks from stocks.json")
                    return watchlist
                else:
                    logging.error("Invalid JSON format in stocks.json. Expected a list of stock symbols.")
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing stocks.json: {e}")
    else:
        logging.warning("stocks.json not found. Creating a default watchlist.")
        with open(STOCK_FILE, "w") as f:
            json.dump(DEFAULT_CANDIDATE_POOL[:int(strategy_params['WATCHLIST_SIZE'])], f, indent=4)  # Save 50 default stocks
        return DEFAULT_CANDIDATE_POOL[:int(strategy_params['WATCHLIST_SIZE'])]  # Default to the top 50 candidates

WATCHLIST = load_watchlist()


# Last date that discovery was run.
last_discovery_date = None

# ==============================
# DATA PERSISTENCE FUNCTIONS
# ==============================

def load_data():
    global strategy_params, latest_signals, trade_log, portfolio_history
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            try:
                data = json.load(f)
                loaded_params = data.get("params", {})
                for key, default_value in default_params.items():
                    if key not in loaded_params:
                        loaded_params[key] = default_value
                strategy_params = loaded_params or strategy_params
                latest_signals = data.get("signals", latest_signals)
                trade_log = deque(data.get("trade_log", []), maxlen=100)
                portfolio_history = deque(data.get("portfolio_history", []), maxlen=1000)
                logging.debug("Data loaded successfully from data.json.")
            except json.JSONDecodeError:
                logging.error(f"Error loading JSON {DATA_FILE}, initializing default data.")
    else:
        logging.error(f"Error loading JSON {DATA_FILE}, initializing default data.")
        save_data()  # Create the file if it doesn't exist

def save_data():
    data = {
        "params": strategy_params,
        "signals": latest_signals,
        "trade_log": list(trade_log),
        "portfolio_history": list(portfolio_history)
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)
    logging.debug("Data saved to data.json.")

def update_strategy_params(new_params):
    with params_lock:
        strategy_params.update(new_params)
    save_data()

load_data()

# ==============================
# INDICATORS CALCULATIONS
# ==============================

def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    logging.debug("Calculating RSI...")
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int, slow: int, signal: int):
    logging.debug("Calculating MACD...")
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_sma(prices: pd.Series, period: int):
    logging.debug("Calculating SMA...")
    return prices.rolling(window=period).mean()

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_period: int, d_period: int) -> tuple:
    """Calculate Stochastic Oscillator %K and %D"""
    logging.debug("Calculating Stoch...")
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    d = k.rolling(d_period).mean()
    return k, d

def calculate_bollinger_bands(price: pd.Series, period: int, 
                             std_dev: float) -> tuple:
    """Calculate Bollinger Bands"""
    logging.debug("Calculating Bollinger Bands...")
    sma = price.rolling(period).mean()
    std = price.rolling(period).std()
    return (sma + std_dev * std, sma - std_dev * std)

def calculate_obv(close: pd.Series, volume) -> pd.Series:
    logging.debug("Calculating OBV...")
    # Ensure close is a Series (in case it's a one-column DataFrame)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    diff = close.diff()
    # Now diff is a Series and the lambda will get a scalar at each element.
    direction = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    obv = pd.Series(0, index=close.index, dtype=float)
    obv.iloc[1:] = np.where(direction.iloc[1:] > 0, volume.iloc[1:],
                     np.where(direction.iloc[1:] < 0, -volume.iloc[1:], 0))
    return obv

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Calculate Average Directional Index"""
    logging.debug("Calculating ADX...")

    # Ensure inputs are 1D Series
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    up = high.diff()
    down = -low.diff()
    
    # Convert np.where results into a 1D Series
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0).flatten(), index=up.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0).flatten(), index=up.index)
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, 
                 period: int) -> pd.Series:
    """Calculate Commodity Channel Index"""
    logging.debug("Calculating CCI...")
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - sma) / (0.015 * mad + 1e-10)


# ==============================
# HELPER FUNCTIONS
# ==============================

def extract_scalar(val):
    """
    Convert a pandas or numpy scalar (or one‐element array/Series) to a float.
    """
    # If it's a pandas Series or a numpy array with a single element, use .item()
    if isinstance(val, (pd.Series, np.ndarray)):
        try:
            return float(val.item())
        except Exception as e:
            logging.error(f"Error extracting scalar using .item(): {e}")
    return float(val)


def record_portfolio_value(api: tradeapi.REST):
    """Records portfolio and index values every 5 minutes"""
    try:
        # Get portfolio value
        account = api.get_account()
        cash = float(account.cash)
        positions = api.list_positions()
        positions_value = sum(float(pos.market_value) for pos in positions)
        total_value = cash + positions_value

        now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
        timestamp = now_et.isoformat()

        # Get latest available index values (intraday or previous close)
        def get_index_close(ticker_symbol):
            try:
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(period="1d", interval="1m")
                if hist.empty or 'Close' not in hist.columns:
                    logging.error(f"No data available for {ticker_symbol}")
                    return None
                return hist['Close'].iloc[-1]
            except Exception as e:
                logging.error(f"Error getting {ticker_symbol} data: {e}")
                return None

        nasdaq_close = get_index_close("^IXIC")
        sp500_close = get_index_close("^GSPC")

        entry = {
            'date': timestamp,
            'portfolio_value': total_value,
            'nasdaq_close': nasdaq_close,
            'sp500_close': sp500_close
        }

        with portfolio_history_lock:
            # Keep only one entry per 5-minute interval
            if len(portfolio_history) == 0 or \
               (now_et - datetime.datetime.fromisoformat(portfolio_history[-1]['date'])).seconds >= 300:
                portfolio_history.append(entry)
                logging.info(f"Recorded portfolio value: {entry} at time {now_et}")
                save_data()
    except Exception as e:
        logging.error(f"Error recording portfolio value: {e}")

def get_historical_data(api: tradeapi.REST, symbol: str, days: int, timeframe: str) -> pd.DataFrame:
    """
    Retrieves historical data using yfinance (free and near-real-time) instead of Alpaca's API.
    """
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        data = yf.download(symbol,
                           start=start_date.strftime("%Y-%m-%d"),
                           end=end_date.strftime("%Y-%m-%d"),
                           interval="1d")
        if data.empty:
            logging.debug(f"No data returned for {symbol} from yfinance.")
            return pd.DataFrame()
        data = data.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
        data = data.sort_index()
        logging.debug(f"Historical data retrieved for {symbol} from yfinance.")
        return data
    except Exception as e:
        logging.error(f"Error retrieving data for {symbol} from yfinance: {e}")
        return pd.DataFrame()

def get_current_position(api: tradeapi.REST, symbol: str):
    try:
        position = api.get_position(symbol)
        logging.debug(f"Position for {symbol} found.")
        return position
    except Exception:
        logging.debug(f"No current position for {symbol}.")
        return None

def submit_order(api: tradeapi.REST, symbol: str, qty: int, side: str, order_type: str = 'market', time_in_force: str = 'day'):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side,
                                 type=order_type, time_in_force=time_in_force)
        logging.info(f"Order submitted: {side.upper()} {qty} shares of {symbol}")
        return order
    except Exception as e:
        logging.error(f"Order submission failed for {symbol}: {e}")
        return None
    
def liquidate_all_positions(api: tradeapi.REST):
    try:
        positions = api.list_positions()
        i = 0
        for pos in positions:
            submit_order(api, pos.symbol, pos.qty, 'sell')
            i += 1
        logging.info(f"End of the day liquidated: {i} positions")
    except Exception as e:
        logging.error(f"Liquidation failed: {e}")

def calculate_order_quantity(price: float, order_amount: float) -> int:
    qty = max(1, round(float(order_amount / price), 0))
    logging.debug(f"Calculated order quantity: {qty} shares at price {price} for order amount {order_amount}")
    return qty

# ==============================
# MARKET HOURS HELPER
# ==============================

def is_market_open() -> bool:
    """
    Returns True if the current Eastern Time is within US market hours (9:30-16:00 ET) on weekdays.
    """
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    open_status = market_open <= now_et < market_close
    logging.debug(f"Market open status: {open_status}")
    return open_status

# ==============================
# COMPUTE COMPOSITE SIGNAL FOR A SYMBOL
# ==============================

def compute_composite_signal(symbol: str, api: tradeapi.REST) -> float:
    logging.debug(f"Computing composite signal for {symbol}...")
    bars = get_historical_data(api, symbol, HIST_DAYS, DATA_TIMEFRAME)
    if bars.empty or 'close' not in bars:
        logging.debug(f"Insufficient data for {symbol}. Returning composite signal 0.0.")
        return 0.0
    
    close = bars['close']
    high = bars['high']
    low = bars['low']
    volume = bars['volume'] if 'volume' in bars else pd.Series(0, index=bars.index)
    current_price = float(close.iloc[-1].squeeze())

    with params_lock:
        #RSI
        rsi_period     = strategy_params['RSI_PERIOD']
        rsi_oversold   = strategy_params['RSI_OVERSOLD']
        rsi_overbought = strategy_params['RSI_OVERBOUGHT']

        #MACD
        macd_fast      = strategy_params['MACD_FAST']
        macd_slow      = strategy_params['MACD_SLOW']
        macd_signal_period = strategy_params['MACD_SIGNAL']

        #SMA
        sma_period     = strategy_params['SMA_PERIOD']

        #STOCH
        stoch_k_period = strategy_params['STOCH_K_PERIOD']
        stoch_d_period = strategy_params['STOCH_D_PERIOD']
        stoch_oversold = strategy_params['STOCH_OVERSOLD']
        stoch_overbought = strategy_params['STOCH_OVERBOUGHT']

        #BOLLINGER
        bollinger_period = strategy_params['BOLLINGER_PERIOD']
        bollinger_std_dev = strategy_params['BOLLINGER_STD_DEV']

        #OBV
        obv_slope_period = strategy_params['OBV_SLOPE_PERIOD']

        #ADX
        adx_period = strategy_params['ADX_PERIOD']
        adx_threshold = strategy_params['ADX_THRESHOLD']

        #CCI
        cci_period = strategy_params['CCI_PERIOD']
        cci_oversold = strategy_params['CCI_OVERSOLD']
        cci_overbought = strategy_params['CCI_OVERBOUGHT']

        #WEIGHTS
        rsi_weight     = strategy_params['RSI_WEIGHT']
        macd_weight    = strategy_params['MACD_WEIGHT']
        sma_weight     = strategy_params['SMA_WEIGHT']
        stoch_weight   = strategy_params['STOCH_WEIGHT']
        bollinger_weight = strategy_params['BOLLINGER_WEIGHT']
        obv_weight = strategy_params['OBV_WEIGHT']
        adx_weight = strategy_params['ADX_WEIGHT']
        cci_weight = strategy_params['CCI_WEIGHT']

    #RSI
    rsi_series = calculate_rsi(close, rsi_period)
    if rsi_series.empty or pd.isna(rsi_series.iloc[-1].squeeze()):
        rsi_signal = 0
    else:
        current_rsi = float(rsi_series.iloc[-1].squeeze())
        rsi_signal = 1 if current_rsi < rsi_oversold else (-1 if current_rsi > rsi_overbought else 0)
    
    #MACD
    macd_line, signal_line = calculate_macd(close, macd_fast, macd_slow, macd_signal_period)
    macd_signal = 1 if float(macd_line.iloc[-1].squeeze()) > float(signal_line.iloc[-1].squeeze()) else -1
    
    #SMA
    sma_series = calculate_sma(close, sma_period)
    if sma_series.empty or pd.isna(sma_series.iloc[-1].squeeze()):
        sma_signal = 0
    else:
        sma_value = float(sma_series.iloc[-1].squeeze())
        sma_signal = 1 if current_price > sma_value else -1

    #STOCH
    stoch_k, stoch_d = calculate_stochastic(high, low, close, stoch_k_period, stoch_d_period)
    if stoch_k.empty or pd.isna(stoch_k.iloc[-1].squeeze()) or stoch_d.empty or pd.isna(stoch_d.iloc[-1].squeeze()):
        stoch_signal = 0
    else:
        stoch_k_value = float(stoch_k.iloc[-1].squeeze())
        stoch_d_value = float(stoch_d.iloc[-1].squeeze())
        stoch_signal = 1 if stoch_k_value > stoch_d_value else -1

    #BOLLINGER
    upper_band, lower_band = calculate_bollinger_bands(close, bollinger_period, bollinger_std_dev)
    if upper_band.empty or pd.isna(upper_band.iloc[-1].squeeze()) or lower_band.empty or pd.isna(lower_band.iloc[-1].squeeze()):
        bollinger_signal = 0
    else:
        bollinger_signal = 1 if current_price > upper_band.iloc[-1].squeeze() else -1

    #OBV
    obv = calculate_obv(close, volume)
    if obv.empty or pd.isna(obv.iloc[-1].squeeze()):
        obv_signal = 0
    else:
        obv_signal = 1 if float(obv.iloc[-1].squeeze()) > 0 else -1

    #ADX
    adx = calculate_adx(high, low, close, adx_period)
    if adx.empty or pd.isna(adx.iloc[-1].squeeze()):
        adx_signal = 0
    else:
        adx_signal = 1 if float(adx.iloc[-1].squeeze()) > adx_threshold else -1

    #CCI
    cci = calculate_cci(high, low, close, cci_period)
    if cci.empty or pd.isna(cci.iloc[-1].squeeze()):
        cci_signal = 0
    else:
        cci_signal = 1 if float(cci.iloc[-1].squeeze()) < cci_oversold else (-1 if float(cci.iloc[-1].squeeze()) > cci_overbought else 0)


    #COMPOSITE
    composite_signal = (rsi_weight * rsi_signal +
                        macd_weight * macd_signal +
                        sma_weight * sma_signal +
                        stoch_weight * stoch_signal +
                        bollinger_weight * bollinger_signal +
                        obv_weight * obv_signal +
                        adx_weight * adx_signal +
                        cci_weight * cci_signal)
    logging.debug(f"{symbol} composite signal computed as {composite_signal:.2f}")
    return composite_signal

# ==============================
# TRADING & SIGNAL LOGIC
# ==============================

def scan_and_trade(api: tradeapi.REST):
    """
    For each symbol in the WATCHLIST, fetch recent data, compute RSI, MACD, and SMA signals,
    combine them with weights, and decide whether to buy or sell.
    Only runs when the market is open.
    """
    global strategy_params
    for symbol in WATCHLIST:
        logging.debug(f"Processing {symbol}...")
        bars = get_historical_data(api, symbol, HIST_DAYS, DATA_TIMEFRAME)
        if bars.empty or 'close' not in bars:
            logging.debug(f"Skipping {symbol} (no valid data).")
            continue

        close_prices = bars['close']
        close = bars['close']
        high = bars['high']
        low = bars['low']
        volume = bars['volume'] if 'volume' in bars else pd.Series(0, index=bars.index)
        current_price = float(close_prices.iloc[-1].squeeze())
        with params_lock:
            #RSI
            rsi_period     = strategy_params['RSI_PERIOD']
            rsi_oversold   = strategy_params['RSI_OVERSOLD']
            rsi_overbought = strategy_params['RSI_OVERBOUGHT']

            #MACD
            macd_fast      = strategy_params['MACD_FAST']
            macd_slow      = strategy_params['MACD_SLOW']
            macd_signal_period = strategy_params['MACD_SIGNAL']

            #SMA
            sma_period     = strategy_params['SMA_PERIOD']

            #STOCH
            stoch_k_period = strategy_params['STOCH_K_PERIOD']
            stoch_d_period = strategy_params['STOCH_D_PERIOD']
            stoch_oversold = strategy_params['STOCH_OVERSOLD']
            stoch_overbought = strategy_params['STOCH_OVERBOUGHT']

            #BOLLINGER
            bollinger_period = strategy_params['BOLLINGER_PERIOD']
            bollinger_std_dev = strategy_params['BOLLINGER_STD_DEV']

            #OBV
            obv_slope_period = strategy_params['OBV_SLOPE_PERIOD']

            #ADX
            adx_period = strategy_params['ADX_PERIOD']
            adx_threshold = strategy_params['ADX_THRESHOLD']

            #CCI
            cci_period = strategy_params['CCI_PERIOD']
            cci_oversold = strategy_params['CCI_OVERSOLD']
            cci_overbought = strategy_params['CCI_OVERBOUGHT']

            #WEIGHTS
            rsi_weight     = strategy_params['RSI_WEIGHT']
            macd_weight    = strategy_params['MACD_WEIGHT']
            sma_weight     = strategy_params['SMA_WEIGHT']
            stoch_weight   = strategy_params['STOCH_WEIGHT']
            bollinger_weight = strategy_params['BOLLINGER_WEIGHT']
            obv_weight = strategy_params['OBV_WEIGHT']
            adx_weight = strategy_params['ADX_WEIGHT']
            cci_weight = strategy_params['CCI_WEIGHT']

            buy_threshold  = strategy_params['BUY_THRESHOLD']
            sell_threshold = strategy_params['SELL_THRESHOLD']
            order_amount  = strategy_params['ORDER_AMOUNT']

        # Calculate RSI signal.
        rsi_series = calculate_rsi(close_prices, rsi_period)
        if rsi_series.empty or pd.isna(rsi_series.iloc[-1].squeeze()):
            logging.debug(f"Insufficient data for RSI calculation for {symbol}.")
            continue
        current_rsi = float(rsi_series.iloc[-1].squeeze())
        rsi_signal = 1 if current_rsi < rsi_oversold else (-1 if current_rsi > rsi_overbought else 0)
        
        # Calculate MACD signal.
        macd_line, signal_line = calculate_macd(close_prices, macd_fast, macd_slow, macd_signal_period)
        macd_signal = 1 if float(macd_line.iloc[-1].squeeze()) > float(signal_line.iloc[-1].squeeze()) else -1
        
        # Calculate SMA signal.
        sma_series = calculate_sma(close_prices, sma_period)
        if sma_series.empty or pd.isna(sma_series.iloc[-1].squeeze()):
            logging.debug(f"Insufficient data for SMA calculation for {symbol}.")
            continue
        sma_value = float(sma_series.iloc[-1].squeeze())
        sma_signal = 1 if current_price > sma_value else -1

        # Calculate STOCH signal.
        stoch_k_series, stoch_d_series = calculate_stochastic(close_prices, high, low, stoch_k_period, stoch_d_period)
        if (stoch_k_series.empty or pd.isna(stoch_k_series.iloc[-1].squeeze()) or
            stoch_d_series.empty or pd.isna(stoch_d_series.iloc[-1].squeeze())):
            logging.debug(f"Insufficient data for STOCH calculation for {symbol}.")
            continue

        stoch_k_value = extract_scalar(stoch_k_series.iloc[-1])
        stoch_d_value = extract_scalar(stoch_d_series.iloc[-1])
        stoch_signal = 1 if stoch_k_value > stoch_d_value else -1

        # Calculate BOLLINGER signal.
        upper_band, lower_band = calculate_bollinger_bands(close_prices, bollinger_period, bollinger_std_dev)
        if upper_band.empty or pd.isna(upper_band.iloc[-1].squeeze()) or lower_band.empty or pd.isna(lower_band.iloc[-1].squeeze()):
            logging.debug(f"Insufficient data for BOLLINGER calculation for {symbol}.")
            continue
        bollinger_signal = 1 if current_price > upper_band.iloc[-1].squeeze() else -1

        # Calculate OBV signal.
        obv_series = calculate_obv(close_prices, volume)
        if obv_series.empty or pd.isna(obv_series.iloc[-1].squeeze()):
            logging.debug(f"Insufficient data for OBV calculation for {symbol}.")
            continue
        obv_value = float(obv_series.iloc[-1].squeeze())
        obv_signal = 1 if obv_value > 0 else -1

        # Calculate ADX signal.
        adx_series = calculate_adx(high, low, close, adx_period)
        if adx_series.empty or pd.isna(adx_series.iloc[-1].squeeze()):
            logging.debug(f"Insufficient data for ADX calculation for {symbol}.")
            continue
        adx_value = float(adx_series.iloc[-1].squeeze())
        adx_signal = 1 if adx_value > adx_threshold else -1

        # Calculate CCI signal.
        cci_series = calculate_cci(high, low, close, cci_period)
        if cci_series.empty or pd.isna(cci_series.iloc[-1].squeeze()):
            logging.debug(f"Insufficient data for CCI calculation for {symbol}.")
            continue
        cci_value = float(cci_series.iloc[-1].squeeze())
        cci_signal = 1 if cci_value < cci_oversold else (-1 if cci_value > cci_overbought else 0)

        composite_signal = (rsi_weight * rsi_signal +
                            macd_weight * macd_signal +
                            sma_weight * sma_signal +
                            stoch_weight * stoch_signal +
                            bollinger_weight * bollinger_signal +
                            obv_weight * obv_signal +
                            adx_weight * adx_signal +
                            cci_weight * cci_signal)
        logging.debug(f"{symbol}: RSI={current_rsi:.2f} (sig {rsi_signal}), MACD signal={macd_signal}, SMA value={sma_value:.2f} (sig {sma_signal}), STOCH signal={stoch_signal}, BOLLINGER signal={bollinger_signal}, OBV signal={obv_signal}, ADX signal={adx_signal}, CCI signal={cci_signal}")
        logging.debug(f"{symbol}: Composite signal (weighted): {composite_signal:.2f}")

        position = get_current_position(api, symbol)
        decision = 'neutral'
        if composite_signal > buy_threshold:
            if position is None:
                decision = 'buy'
                logging.info(f"{symbol}: Composite signal {composite_signal:.2f} > BUY_THRESHOLD {buy_threshold} => BUY")
                order_qty = calculate_order_quantity(current_price, order_amount)
                order = submit_order(api, symbol, order_qty, side='buy')
                if order:
                    record_trade(symbol, 'buy', order_qty, current_price)
            else:
                decision = 'hold (already long)'
                logging.debug(f"{symbol}: Bullish composite signal but already holding position.")
        elif composite_signal < sell_threshold:
            if position is not None:
                decision = 'sell'
                logging.info(f"{symbol}: Composite signal {composite_signal:.2f} < SELL_THRESHOLD {sell_threshold} => SELL")
                order = submit_order(api, symbol, float(position.qty), side='sell')
                if order:
                    record_trade(symbol, 'sell', float(position.qty), current_price)
            else:
                decision = 'hold (no position)'
                logging.debug(f"{symbol}: Bearish composite signal but no position held.")
        elif position is not None and symbol in open_positions_rl:
            emergency_loss_threshold = strategy_params.get('EMERGENCY_LOSS_THRESHOLD', 0.02)
            purchase_price = open_positions_rl[symbol][0]
            threshold_price = purchase_price * (1 - emergency_loss_threshold)
            if current_price < threshold_price:
                decision = 'sell (emergency)'
                logging.info(f"{symbol}: Emergency sell triggered: current price {current_price:.2f} below threshold {threshold_price:.2f} (purchase price {purchase_price:.2f})")
                order = submit_order(api, symbol, float(position.qty), side='sell')
                if order:
                    record_trade(symbol, 'sell', float(position.qty), current_price)
            else:
                decision = 'neutral'
                logging.debug(f"{symbol}: No emergency sell triggered; current price {current_price:.2f} stays above threshold {threshold_price:.2f}")
        else:
            decision = 'neutral'
            logging.debug(f"{symbol}: Composite signal in neutral zone.")

        with signals_lock:
            latest_signals[symbol] = {
                'RSI': round(current_rsi, 2),
                'MACD': macd_signal,
                'SMA': sma_signal,
                'STOCH': stoch_signal,
                'BOLLINGER': bollinger_signal,
                'OBV': obv_signal,
                'ADX': adx_signal,
                'CCI': cci_signal,
                'composite': round(composite_signal, 2),
                'signal': decision,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

def record_trade(symbol: str, side: str, qty: int, price: float):
    """
    Records a trade. For a buy, the trade is recorded and the open position stored.
    For a sell, if a matching open position exists, compute PnL and update the cumulative reward.
    """
    global cumulative_reward
    with trade_log_lock:
        trade_entry = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'pnl': 0.0
        }
        if side == 'buy':
            open_positions_rl[symbol] = (price, qty)
        elif side == 'sell':
            if symbol in open_positions_rl:
                buy_price, buy_qty = open_positions_rl[symbol]
                pnl = (price - buy_price) * qty
                trade_entry['pnl'] = pnl
                cumulative_reward += pnl
                del open_positions_rl[symbol]
                logging.debug(f"Updated cumulative reward: {cumulative_reward}")
            else:
                trade_entry['pnl'] = 0.0
        trade_log.append(trade_entry)
        logging.debug(f"Recorded trade: {trade_entry}")

def update_held_positions(api: tradeapi.REST):
    try:
        positions = api.list_positions()
        with signals_lock:
            for pos in positions:
                try:
                    last_price = float(pos.current_price)
                except Exception:
                    last_price = float(pos.avg_entry_price)
                latest_signals[pos.symbol] = {
                    'RSI': None,
                    'MACD': None,
                    'SMA': None,
                    'STOCH': None,
                    'BOLLINGER': None,
                    'OBV': None,
                    'ADX': None,
                    'CCI': None,
                    'composite': None,
                    'signal': f'held ({pos.qty} shares)',
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        logging.info("Updated held positions from account.")
    except Exception as e:
        logging.error(f"Error updating held positions: {e}")

# ==============================
# TRUE REINFORCEMENT LEARNING AGENT FOR PARAMETER TUNING
# ==============================

# Global variables for RL-based parameter tuning.
# cumulative_reward tracks the total realized PnL from closed trades.
cumulative_reward = 0.0
open_positions_rl = {}  # For matching buys and sells for RL reward calculation.
previous_cumulative_reward = 0.0  # Used to compute reward over tuning intervals.
previous_state = 0  # State can be -1, 0, or 1 representing performance trend.

NUM_ACTIONS = 56  # Define number of discrete actions.

class ParameterTuningAgent:
    def __init__(self, alpha=0.4, gamma=0.7, epsilon=0.5):
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}       # Q-table mapping state -> list of Q-values for each action

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(NUM_ACTIONS)]
        return self.q_table[state]

    def choose_action(self, state):
        q_values = self.get_q_values(state)
        if random.random() < self.epsilon:
            action = random.randint(0, NUM_ACTIONS - 1)
            logging.debug(f"RL Agent: Randomly chosen action {action} for state {state}")
            return action
        else:
            max_q = max(q_values)
            candidates = [i for i, q in enumerate(q_values) if q == max_q]
            action = random.choice(candidates)
            logging.debug(f"RL Agent: Chosen best action {action} for state {state} with Q-values {q_values}")
            return action

    def update(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        best_next = max(next_q_values)
        td_target = reward + self.gamma * best_next
        td_error = td_target - q_values[action]
        q_values[action] += self.alpha * td_error
        logging.debug(f"RL Agent: Updated Q-value for state {state}, action {action}: {q_values[action]:.4f} (TD error: {td_error:.4f})")

# Instantiate the RL agent.
rl_agent = ParameterTuningAgent()

def apply_rl_action(action):
    """
    Applies a discrete action to adjust strategy parameters.
    
    Actions:
      0: Increase RSI_OVERSOLD by 0.5 (max 40)
      1: Decrease RSI_OVERSOLD by 0.5 (min 20)
      2: Increase RSI_OVERBOUGHT by 0.5 (max 80)
      3: Decrease RSI_OVERBOUGHT by 0.5 (min 60)
      
      4: Increase RSI_WEIGHT by 0.01
      5: Decrease RSI_WEIGHT by 0.01 (min 0.1)
      6: Increase MACD_WEIGHT by 0.01
      7: Decrease MACD_WEIGHT by 0.01 (min 0.1)
      8: Increase SMA_WEIGHT by 0.01
      9: Decrease SMA_WEIGHT by 0.01 (min 0.1)
      10: Increase STOCH_WEIGHT by 0.01
      11: Decrease STOCH_WEIGHT by 0.01 (min 0.1)
      12: Increase BOLLINGER_WEIGHT by 0.01
      13: Decrease BOLLINGER_WEIGHT by 0.01 (min 0.1)
      14: Increase OBV_WEIGHT by 0.01
      15: Decrease OBV_WEIGHT by 0.01 (min 0.1)
      16: Increase ADX_WEIGHT by 0.01
      17: Decrease ADX_WEIGHT by 0.01 (min 0.1)
      18: Increase CCI_WEIGHT by 0.01
      19: Decrease CCI_WEIGHT by 0.01 (min 0.1)
      
      20: Increase STOCH_K_PERIOD by 1 (max 30)
      21: Decrease STOCH_K_PERIOD by 1 (min 5)
      22: Increase STOCH_D_PERIOD by 1 (max 10)
      23: Decrease STOCH_D_PERIOD by 1 (min 1)
      24: Increase BOLLINGER_PERIOD by 1 (max 50)
      25: Decrease BOLLINGER_PERIOD by 1 (min 10)
      26: Increase BOLLINGER_STD_DEV by 0.1 (max 5)
      27: Decrease BOLLINGER_STD_DEV by 0.1 (min 1)
      28: Increase OBV_SLOPE_PERIOD by 1 (max 20)
      29: Decrease OBV_SLOPE_PERIOD by 1 (min 1)
      30: Increase ADX_PERIOD by 1 (max 30)
      31: Decrease ADX_PERIOD by 1 (min 5)
      32: Increase ADX_THRESHOLD by 1 (max 35)
      33: Decrease ADX_THRESHOLD by 1 (min 15)
      34: Increase CCI_PERIOD by 1 (max 40)
      35: Decrease CCI_PERIOD by 1 (min 10)
      36: Increase CCI_OVERSOLD by 5 (cap at -50)
      37: Decrease CCI_OVERSOLD by 5 (min -200)
      38: Increase CCI_OVERBOUGHT by 5 (max 200)
      39: Decrease CCI_OVERBOUGHT by 5 (min 50)
      40: Increase RSI_PERIOD by 1 (max 30)
      41: Decrease RSI_PERIOD by 1 (min 5)
      42: Increase ORDER_AMOUNT by 100 (max 5000)
      43: Decrease ORDER_AMOUNT by 100 (min 500)
      44: Increase MACD_FAST by 1 (max 20)
      45: Decrease MACD_FAST by 1 (min 5)
      46: Increase MACD_SLOW by 1 (max 40)
      47: Decrease MACD_SLOW by 1 (min 20)
      48: Increase MACD_SIGNAL by 1 (max 20)
      49: Decrease MACD_SIGNAL by 1 (min 5)
      50: Increase SMA_PERIOD by 1 (max 100)
      51: Decrease SMA_PERIOD by 1 (min 10)
      52: Increase STOCH_OVERSOLD by 1 (max 50)
      53: Decrease STOCH_OVERSOLD by 1 (min 10)
      54: Increase STOCH_OVERBOUGHT by 1 (max 90)
      55: Decrease STOCH_OVERBOUGHT by 1 (min 50)
      Else: No change
    """
    with params_lock:
        # Adjust RSI oversold/overbought thresholds
        if action == 0:
            strategy_params['RSI_OVERSOLD'] = min(40, strategy_params['RSI_OVERSOLD'] + 0.5)
            logging.debug("RL Action: Increased RSI_OVERSOLD")
        elif action == 1:
            strategy_params['RSI_OVERSOLD'] = max(20, strategy_params['RSI_OVERSOLD'] - 0.5)
            logging.debug("RL Action: Decreased RSI_OVERSOLD")
        elif action == 2:
            strategy_params['RSI_OVERBOUGHT'] = min(80, strategy_params['RSI_OVERBOUGHT'] + 0.5)
            logging.debug("RL Action: Increased RSI_OVERBOUGHT")
        elif action == 3:
            strategy_params['RSI_OVERBOUGHT'] = max(60, strategy_params['RSI_OVERBOUGHT'] - 0.5)
            logging.debug("RL Action: Decreased RSI_OVERBOUGHT")
        
        # Adjust indicator weights
        elif action == 4:
            strategy_params['RSI_WEIGHT'] += 0.01
            logging.debug("RL Action: Increased RSI_WEIGHT")
        elif action == 5:
            strategy_params['RSI_WEIGHT'] = max(0.1, strategy_params['RSI_WEIGHT'] - 0.01)
            logging.debug("RL Action: Decreased RSI_WEIGHT")
        elif action == 6:
            strategy_params['MACD_WEIGHT'] += 0.01
            logging.debug("RL Action: Increased MACD_WEIGHT")
        elif action == 7:
            strategy_params['MACD_WEIGHT'] = max(0.1, strategy_params['MACD_WEIGHT'] - 0.01)
            logging.debug("RL Action: Decreased MACD_WEIGHT")
        elif action == 8:
            strategy_params['SMA_WEIGHT'] += 0.01
            logging.debug("RL Action: Increased SMA_WEIGHT")
        elif action == 9:
            strategy_params['SMA_WEIGHT'] = max(0.1, strategy_params['SMA_WEIGHT'] - 0.01)
            logging.debug("RL Action: Decreased SMA_WEIGHT")
        elif action == 10:
            strategy_params['STOCH_WEIGHT'] += 0.01
            logging.debug("RL Action: Increased STOCH_WEIGHT")
        elif action == 11:
            strategy_params['STOCH_WEIGHT'] = max(0.1, strategy_params['STOCH_WEIGHT'] - 0.01)
            logging.debug("RL Action: Decreased STOCH_WEIGHT")
        elif action == 12:
            strategy_params['BOLLINGER_WEIGHT'] += 0.01
            logging.debug("RL Action: Increased BOLLINGER_WEIGHT")
        elif action == 13:
            strategy_params['BOLLINGER_WEIGHT'] = max(0.1, strategy_params['BOLLINGER_WEIGHT'] - 0.01)
            logging.debug("RL Action: Decreased BOLLINGER_WEIGHT")
        elif action == 14:
            strategy_params['OBV_WEIGHT'] += 0.01
            logging.debug("RL Action: Increased OBV_WEIGHT")
        elif action == 15:
            strategy_params['OBV_WEIGHT'] = max(0.1, strategy_params['OBV_WEIGHT'] - 0.01)
            logging.debug("RL Action: Decreased OBV_WEIGHT")
        elif action == 16:
            strategy_params['ADX_WEIGHT'] += 0.01
            logging.debug("RL Action: Increased ADX_WEIGHT")
        elif action == 17:
            strategy_params['ADX_WEIGHT'] = max(0.1, strategy_params['ADX_WEIGHT'] - 0.01)
            logging.debug("RL Action: Decreased ADX_WEIGHT")
        elif action == 18:
            strategy_params['CCI_WEIGHT'] += 0.01
            logging.debug("RL Action: Increased CCI_WEIGHT")
        elif action == 19:
            strategy_params['CCI_WEIGHT'] = max(0.1, strategy_params['CCI_WEIGHT'] - 0.01)
            logging.debug("RL Action: Decreased CCI_WEIGHT")
        
        # After weight adjustments, re-normalize the weights so they sum roughly to 1.
        if action in {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
            total_weight = (
                strategy_params['RSI_WEIGHT'] +
                strategy_params['MACD_WEIGHT'] +
                strategy_params['SMA_WEIGHT'] +
                strategy_params['STOCH_WEIGHT'] +
                strategy_params['BOLLINGER_WEIGHT'] +
                strategy_params['OBV_WEIGHT'] +
                strategy_params['ADX_WEIGHT'] +
                strategy_params['CCI_WEIGHT']
            )
            strategy_params['RSI_WEIGHT'] /= total_weight
            strategy_params['MACD_WEIGHT'] /= total_weight
            strategy_params['SMA_WEIGHT'] /= total_weight
            strategy_params['STOCH_WEIGHT'] /= total_weight
            strategy_params['BOLLINGER_WEIGHT'] /= total_weight
            strategy_params['OBV_WEIGHT'] /= total_weight
            strategy_params['ADX_WEIGHT'] /= total_weight
            strategy_params['CCI_WEIGHT'] /= total_weight

        # Adjust other parameters using default_params as guidance:
        elif action == 20:
            strategy_params['STOCH_K_PERIOD'] = min(30, strategy_params['STOCH_K_PERIOD'] + 1)
            logging.debug("RL Action: Increased STOCH_K_PERIOD")
        elif action == 21:
            strategy_params['STOCH_K_PERIOD'] = max(5, strategy_params['STOCH_K_PERIOD'] - 1)
            logging.debug("RL Action: Decreased STOCH_K_PERIOD")
        elif action == 22:
            strategy_params['STOCH_D_PERIOD'] = min(10, strategy_params['STOCH_D_PERIOD'] + 1)
            logging.debug("RL Action: Increased STOCH_D_PERIOD")
        elif action == 23:
            strategy_params['STOCH_D_PERIOD'] = max(1, strategy_params['STOCH_D_PERIOD'] - 1)
            logging.debug("RL Action: Decreased STOCH_D_PERIOD")
        elif action == 24:
            strategy_params['BOLLINGER_PERIOD'] = min(50, strategy_params['BOLLINGER_PERIOD'] + 1)
            logging.debug("RL Action: Increased BOLLINGER_PERIOD")
        elif action == 25:
            strategy_params['BOLLINGER_PERIOD'] = max(10, strategy_params['BOLLINGER_PERIOD'] - 1)
            logging.debug("RL Action: Decreased BOLLINGER_PERIOD")
        elif action == 26:
            # Increase by 0.1, rounded to 1 decimal place
            strategy_params['BOLLINGER_STD_DEV'] = min(5, round(strategy_params['BOLLINGER_STD_DEV'] + 0.1, 1))
            logging.debug("RL Action: Increased BOLLINGER_STD_DEV")
        elif action == 27:
            strategy_params['BOLLINGER_STD_DEV'] = max(1, round(strategy_params['BOLLINGER_STD_DEV'] - 0.1, 1))
            logging.debug("RL Action: Decreased BOLLINGER_STD_DEV")
        elif action == 28:
            strategy_params['OBV_SLOPE_PERIOD'] = min(20, strategy_params['OBV_SLOPE_PERIOD'] + 1)
            logging.debug("RL Action: Increased OBV_SLOPE_PERIOD")
        elif action == 29:
            strategy_params['OBV_SLOPE_PERIOD'] = max(1, strategy_params['OBV_SLOPE_PERIOD'] - 1)
            logging.debug("RL Action: Decreased OBV_SLOPE_PERIOD")
        elif action == 30:
            strategy_params['ADX_PERIOD'] = min(30, strategy_params['ADX_PERIOD'] + 1)
            logging.debug("RL Action: Increased ADX_PERIOD")
        elif action == 31:
            strategy_params['ADX_PERIOD'] = max(5, strategy_params['ADX_PERIOD'] - 1)
            logging.debug("RL Action: Decreased ADX_PERIOD")
        elif action == 32:
            strategy_params['ADX_THRESHOLD'] = min(35, strategy_params['ADX_THRESHOLD'] + 1)
            logging.debug("RL Action: Increased ADX_THRESHOLD")
        elif action == 33:
            strategy_params['ADX_THRESHOLD'] = max(15, strategy_params['ADX_THRESHOLD'] - 1)
            logging.debug("RL Action: Decreased ADX_THRESHOLD")
        elif action == 34:
            strategy_params['CCI_PERIOD'] = min(40, strategy_params['CCI_PERIOD'] + 1)
            logging.debug("RL Action: Increased CCI_PERIOD")
        elif action == 35:
            strategy_params['CCI_PERIOD'] = max(10, strategy_params['CCI_PERIOD'] - 1)
            logging.debug("RL Action: Decreased CCI_PERIOD")
        elif action == 36:
            # For CCI_OVERSOLD, “increasing” makes the value less negative but cap at -50
            strategy_params['CCI_OVERSOLD'] = min(strategy_params['CCI_OVERSOLD'] + 5, -50)
            logging.debug("RL Action: Increased CCI_OVERSOLD")
        elif action == 37:
            strategy_params['CCI_OVERSOLD'] = max(strategy_params['CCI_OVERSOLD'] - 5, -200)
            logging.debug("RL Action: Decreased CCI_OVERSOLD")
        elif action == 38:
            strategy_params['CCI_OVERBOUGHT'] = min(strategy_params['CCI_OVERBOUGHT'] + 5, 200)
            logging.debug("RL Action: Increased CCI_OVERBOUGHT")
        elif action == 39:
            strategy_params['CCI_OVERBOUGHT'] = max(strategy_params['CCI_OVERBOUGHT'] - 5, 50)
            logging.debug("RL Action: Decreased CCI_OVERBOUGHT")
        elif action == 40:
            strategy_params['RSI_PERIOD'] = min(30, strategy_params['RSI_PERIOD'] + 1)
            logging.debug("RL Action: Increased RSI_PERIOD")
        elif action == 41:
            strategy_params['RSI_PERIOD'] = max(5, strategy_params['RSI_PERIOD'] - 1)
            logging.debug("RL Action: Decreased RSI_PERIOD")
        elif action == 42:
            strategy_params['ORDER_AMOUNT'] = min(5000, strategy_params['ORDER_AMOUNT'] + 10)
            logging.debug("RL Action: Increased ORDER_AMOUNT")
        elif action == 43:
            strategy_params['ORDER_AMOUNT'] = max(500, strategy_params['ORDER_AMOUNT'] - 10)
            logging.debug("RL Action: Decreased ORDER_AMOUNT")
        elif action == 44:
            strategy_params['MACD_FAST'] = min(20, strategy_params['MACD_FAST'] + 1)
            logging.debug("RL Action: Increased MACD_FAST")
        elif action == 45:
            strategy_params['MACD_FAST'] = max(5, strategy_params['MACD_FAST'] - 1)
            logging.debug("RL Action: Decreased MACD_FAST")
        elif action == 46:
            strategy_params['MACD_SLOW'] = min(40, strategy_params['MACD_SLOW'] + 1)
            logging.debug("RL Action: Increased MACD_SLOW")
        elif action == 47:
            strategy_params['MACD_SLOW'] = max(20, strategy_params['MACD_SLOW'] - 1)
            logging.debug("RL Action: Decreased MACD_SLOW")
        elif action == 48:
            strategy_params['MACD_SIGNAL'] = min(20, strategy_params['MACD_SIGNAL'] + 1)
            logging.debug("RL Action: Increased MACD_SIGNAL")
        elif action == 49:
            strategy_params['MACD_SIGNAL'] = max(5, strategy_params['MACD_SIGNAL'] - 1)
            logging.debug("RL Action: Decreased MACD_SIGNAL")
        elif action == 50:
            strategy_params['SMA_PERIOD'] = min(100, strategy_params['SMA_PERIOD'] + 1)
            logging.debug("RL Action: Increased SMA_PERIOD")
        elif action == 51:
            strategy_params['SMA_PERIOD'] = max(10, strategy_params['SMA_PERIOD'] - 1)
            logging.debug("RL Action: Decreased SMA_PERIOD")
        elif action == 52:
            strategy_params['STOCH_OVERSOLD'] = min(50, strategy_params['STOCH_OVERSOLD'] + 1)
            logging.debug("RL Action: Increased STOCH_OVERSOLD")
        elif action == 53:
            strategy_params['STOCH_OVERSOLD'] = max(10, strategy_params['STOCH_OVERSOLD'] - 1)
            logging.debug("RL Action: Decreased STOCH_OVERSOLD")
        elif action == 54:
            strategy_params['STOCH_OVERBOUGHT'] = min(90, strategy_params['STOCH_OVERBOUGHT'] + 1)
            logging.debug("RL Action: Increased STOCH_OVERBOUGHT")
        elif action == 55:
            strategy_params['STOCH_OVERBOUGHT'] = max(50, strategy_params['STOCH_OVERBOUGHT'] - 1)
            logging.debug("RL Action: Decreased STOCH_OVERBOUGHT")
        else:
            logging.debug("RL Action: No change")
        # Re-normalize weights so that they sum to 1.
        total = strategy_params['RSI_WEIGHT'] + strategy_params['MACD_WEIGHT'] + strategy_params['SMA_WEIGHT']
        strategy_params['RSI_WEIGHT'] /= total
        strategy_params['MACD_WEIGHT'] /= total
        strategy_params['SMA_WEIGHT'] /= total

def tuning_loop():
    """
    RL-based tuning loop:
      - Every TUNE_INTERVAL, calculate the reward as the change in cumulative trading profit.
      - The RL agent observes the current state (performance trend), chooses an action,
        applies that action to adjust strategy parameters, and updates its Q-values.
    """
    TUNE_INTERVAL_MINUTES = strategy_params['TUNE_INTERVAL_MINUTES']
    TUNE_INTERVAL = 60 * TUNE_INTERVAL_MINUTES  # Tune every 10 minutes
    logging.info("Starting RL parameter tuning loop...")
    global previous_cumulative_reward, previous_state
    previous_cumulative_reward = cumulative_reward
    previous_state = 0  # initial state: neutral
    while True:
        try:
            time.sleep(TUNE_INTERVAL)
            with trade_log_lock:
                current_reward = cumulative_reward
                logging.debug(f"Current cumulative reward: {current_reward:.2f}")
            # Compute reward as the change in cumulative reward over the interval.
            reward = (current_reward - previous_cumulative_reward)*3
            # Determine new state: 1 if positive reward, -1 if negative, 0 if no change.
            new_state = 1 if reward > 0 else (-1 if reward < 0 else 0)
            # RL agent chooses an action based on the previous state.
            action = rl_agent.choose_action(previous_state)
            apply_rl_action(action)
            update_strategy_params(strategy_params)
            # Update Q-values for the RL agent.
            rl_agent.update(previous_state, action, reward, new_state)
            logging.info("RL tuning: prev_state=%s, action=%s, reward=%.2f, new_state=%s", previous_state, action, reward, new_state)
            previous_state = new_state
            previous_cumulative_reward = current_reward
        except Exception as e:
            logging.error("Error in tuning loop: %s", e)

# ==============================
# END-OF-DAY WATCHLIST DISCOVERY
# ==============================

def update_watchlist(api: tradeapi.REST):
    """
    At the end of the day, update the watchlist:
      - Evaluate the composite signal for stocks already in the watchlist.
      - Retain only the top 100 (by composite signal).
      - If there are fewer than 100 stocks, scan a candidate pool for additional stocks.
    """
    global WATCHLIST
    logging.debug("Starting end-of-day watchlist discovery...")

    with signals_lock:
        current_scores = {symbol: info.get('composite', 0) for symbol, info in latest_signals.items()}
    watchlist_scores = []
    for symbol in WATCHLIST:
        score = current_scores.get(symbol)
        if score is None:
            score = compute_composite_signal(symbol, api)
        watchlist_scores.append((symbol, score))
    watchlist_scores.sort(key=lambda x: x[1], reverse=True)
    new_watchlist = [symbol for symbol, score in watchlist_scores][:100]

    # If fewer than 100 stocks, add candidates (only if they are not already in the watchlist)
    if len(new_watchlist) < 100:
        candidate_scores = []
        for symbol in DEFAULT_CANDIDATE_POOL:
            if symbol in new_watchlist:
                continue
            score = compute_composite_signal(symbol, api)
            candidate_scores.append((symbol, score))
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        for symbol, score in candidate_scores:
            if len(new_watchlist) < 100:
                new_watchlist.append(symbol)
            else:
                break

    # Remove duplicates and limit to 100.
    new_watchlist = list(dict.fromkeys(new_watchlist))[:100]
    WATCHLIST = new_watchlist
    # Save the updated watchlist to watchlist.json
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(WATCHLIST, f, indent=4)
    logging.info("Updated watchlist at end of day. Total stocks in watchlist: %d", len(WATCHLIST))

# ==============================
# THREAD FUNCTIONS
# ==============================

def trading_loop():
    """
    Main trading loop:
      - Checks if the market is open before scanning and trading.
      - After 4:30 PM ET (once per day), runs watchlist discovery.
    """
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    global last_discovery_date
    logging.info("Starting trading loop...")
    scan_and_trade(api)  # Initial scan and trade
    now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    liquidation_start_time = now_et.replace(hour=15, minute=50, second=0)
    liquidation_end_time = now_et.replace(hour=16, minute=0, second=0)

    while True:
        try:
            now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
            liquidation_start_time = now_et.replace(hour=15, minute=50, second=0)
            liquidation_end_time = now_et.replace(hour=16, minute=0, second=0)
            if is_market_open() and now_et <= liquidation_start_time:
                scan_and_trade(api)
                record_portfolio_value(api)
            elif now_et >= liquidation_start_time and now_et < liquidation_end_time:
                logging.info("Market is closing soon. Liquidating all positions.")
                liquidate_all_positions(api)
            else:
                # Handle end-of-day discovery once per day
                now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
                discovery_time = now_et.replace(hour=16, minute=30, second=0)
                if now_et >= discovery_time:
                    today_str = now_et.strftime("%Y-%m-%d")
                    if last_discovery_date != today_str:
                        update_watchlist(api)
                        last_discovery_date = today_str
            
            time.sleep(60 * 5)  # Run every 5 minutes

        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
            time.sleep(60)

# ==============================
# FLASK WEB DASHBOARD
# ==============================

app = Flask(__name__)


@app.route('/')
def dashboard():
    with params_lock:
        current_params = strategy_params.copy()
    with signals_lock:
        current_signals = latest_signals.copy()
    with trade_log_lock:
        trades = list(trade_log)
    with portfolio_history_lock:
        history = list(portfolio_history)

    performance_data = []
    if history:
        baseline = next((
            item for item in history 
            if item.get('nasdaq_close', 0) and 
               item.get('sp500_close', 0)
        ), None)

        if baseline:
            for entry in history:
                # Use 'timestamp' if available, otherwise fall back to 'date'
                ts = entry.get('timestamp', entry.get('date'))
                try:
                    dt = datetime.datetime.fromisoformat(ts)
                    display_time = dt.strftime("%m-%d %H:%M")
                except Exception as e:
                    logging.error(f"Error parsing timestamp {ts}: {e}")
                    display_time = "Invalid Time"

                # Normalization logic
                norm_portfolio = (entry['portfolio_value'] / baseline['portfolio_value']) * 100
                norm_nasdaq = (entry['nasdaq_close'] / baseline['nasdaq_close']) * 100
                norm_sp500 = (entry['sp500_close'] / baseline['sp500_close']) * 100
                
                performance_data.append({
                    'time': display_time,
                    'portfolio': round(norm_portfolio, 2),
                    'nasdaq': round(norm_nasdaq, 2),
                    'sp500': round(norm_sp500, 2)
                })

    return render_template('index.html',
                           params=current_params,
                           signals=current_signals,
                           trade_log=trades,
                           performance_data=performance_data,
                           check_interval=CHECK_INTERVAL)



@app.route('/api/status')
def api_status():
    with params_lock:
        current_params = strategy_params.copy()
    with signals_lock:
        current_signals = latest_signals.copy()
    with trade_log_lock:
        trades = list(trade_log)
    return jsonify({
        'params': current_params,
        'signals': current_signals,
        'trade_log': trades
    })

# ==============================
# MAIN EXECUTION
# ==============================

def main():
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    update_held_positions(api)
    trading_thread = threading.Thread(target=trading_loop, daemon=True)
    tuning_thread = threading.Thread(target=tuning_loop, daemon=True)
    trading_thread.start()
    tuning_thread.start()
    logging.info(WELCOME_MESSAGE)
    logging.info("Starting Flask dashboard on http://0.0.0.0:5000")
    # Disable Flask's auto-reloader to prevent duplicate threads in debug mode.
    app.run(host='0.0.0.0', port=5000, debug=DEBUG, use_reloader=False)

if __name__ == '__main__':
    main()
