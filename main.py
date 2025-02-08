#!/usr/bin/env python3
"""
Fully Featured Trading Bot using Reinforcement Learning, a Web Dashboard,
market-hour stopping, and end-of-day watchlist discovery.

This script:
  - Uses yfinance  to scan a watchlist for trading opportunities based on a composite
    of RSI, MACD, and SMA.
  - Trades via Alpaca's API based on the composite signal but only during US market hours.
  - Records trade performance and automatically tunes parameters over time using a simple
    reinforcement learning approach.
  - At market close, it “discovers” stocks by retaining only
    the best (up to 100 total) while removing the worst-performers.
  - Launches a Flask web portal so you can view the current settings, trade log, and asset signals.

Before running:
  - Install dependencies: 
        pip install alpaca-trade-api pandas numpy flask python-dotenv yfinance
  - Create a .env file or set your Alpaca API credentials as environment variables:
        ALPACA_KEY=your_api_key_here
        ALPACA_SECRET=your_api_secret_here
        ENDPOINT=https://paper-api.alpaca.markets
  - Make a file called "stocks.json" with a list of stock tickers in the watchlist
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

if DEBUG:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug("Debug flag enabled. Logging set to DEBUG.")
else:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

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
    # Weights for the composite signal (roughly summing to 1)
    'RSI_WEIGHT': 0.34,
    'MACD_WEIGHT': 0.33,
    'SMA_WEIGHT': 0.33,
    # Composite signal decision thresholds:
    'BUY_THRESHOLD': 0.5,
    'SELL_THRESHOLD': -0.5,
}
strategy_params = default_params.copy()
params_lock = threading.Lock()  # Protect access to strategy_params

rl_epsilon = 0.2    # 20% chance for a random parameter adjustment

# Load watchlist
STOCK_FILE = "stocks.json"
if os.path.exists(STOCK_FILE):
    with open(STOCK_FILE, "r") as f:
        try:
            data = json.load(f)
            WATCHLIST = data
        except json.JSONDecodeError:
            logging.error("Error loading JSON, initializing empty watchlist.")
            WATCHLIST = []
else:
    WATCHLIST = []

DATA_TIMEFRAME = '1d'
HIST_DAYS = 100          # Number of historical days used for indicator calculations
CHECK_INTERVAL = 60 * 5  # Check every 5 minutes

# Trade performance log (store recent trades in memory)
trade_log = deque(maxlen=100)
trade_log_lock = threading.Lock()

# Latest signals for dashboard (per symbol)
latest_signals = {}
signals_lock = threading.Lock()

DATA_FILE = "data.json"

DEFAULT_CANDIDATE_POOL = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "DIS", "HD", "UNH", "BAC", "XOM", "VZ", "ADBE",
    "NFLX", "PFE", "KO", "CSCO", "CMCSA", "INTC", "T", "PEP", "ABT", "CRM",
    "ABBV", "ACN", "AVGO", "QCOM", "TXN", "COST", "NEE", "NKE", "MRK", "WFC",
    "LLY", "MDT", "MCD", "PM", "ORCL", "BA", "IBM", "HON", "AMGN"
]
STOCK_FILE = "stocks.json"
if os.path.exists(STOCK_FILE):
    with open(STOCK_FILE, "r") as f:
        try:
            data = json.load(f)
            DEFAULT_CANDIDATE_POOL = data
        except json.JSONDecodeError:
            logging.error(f"Error loading JSON {STOCK_FILE}, initializing default data.")
else:
    logging.error(f"Error loading JSON {STOCK_FILE}, initializing default data.")

# Last date that discovery was run.
last_discovery_date = None

# ==============================
# DATA PERSISTENCE FUNCTIONS
# ==============================

def load_data():
    global strategy_params, latest_signals, trade_log
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
        "trade_log": list(trade_log)
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
# HELPER FUNCTIONS & INDICATORS
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
        data = data.rename(columns={'Close': 'close'})
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

def submit_order(api: tradeapi.REST, symbol: str, qty: int, side: str, order_type: str = 'market', time_in_force: str = 'gtc'):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side,
                                 type=order_type, time_in_force=time_in_force)
        logging.info(f"Order submitted: {side.upper()} {qty} shares of {symbol}")
        return order
    except Exception as e:
        logging.error(f"Order submission failed for {symbol}: {e}")
        return None

def calculate_order_quantity(price: float, order_amount: float) -> int:
    qty = max(1, int(order_amount / price))
    logging.debug(f"Calculated order quantity: {qty} shares at price {price} for order amount {order_amount}")
    return qty

# ==============================
# Market Hours Helper
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
# Compute composite signal for a symbol.
# ==============================

def compute_composite_signal(symbol: str, api: tradeapi.REST) -> float:
    logging.debug(f"Computing composite signal for {symbol}...")
    bars = get_historical_data(api, symbol, HIST_DAYS, DATA_TIMEFRAME)
    if bars.empty or 'close' not in bars:
        logging.debug(f"Insufficient data for {symbol}. Returning composite signal 0.0.")
        return 0.0
    close_prices = bars['close']
    current_price = float(close_prices.iloc[-1].squeeze())
    with params_lock:
        rsi_period     = strategy_params['RSI_PERIOD']
        rsi_oversold   = strategy_params['RSI_OVERSOLD']
        rsi_overbought = strategy_params['RSI_OVERBOUGHT']
        macd_fast      = strategy_params['MACD_FAST']
        macd_slow      = strategy_params['MACD_SLOW']
        macd_signal_period = strategy_params['MACD_SIGNAL']
        sma_period     = strategy_params['SMA_PERIOD']
        rsi_weight     = strategy_params['RSI_WEIGHT']
        macd_weight    = strategy_params['MACD_WEIGHT']
        sma_weight     = strategy_params['SMA_WEIGHT']
    rsi_series = calculate_rsi(close_prices, rsi_period)
    if rsi_series.empty or pd.isna(rsi_series.iloc[-1].squeeze()):
        rsi_signal = 0
    else:
        current_rsi = float(rsi_series.iloc[-1].squeeze())
        rsi_signal = 1 if current_rsi < rsi_oversold else (-1 if current_rsi > rsi_overbought else 0)
    macd_line, signal_line = calculate_macd(close_prices, macd_fast, macd_slow, macd_signal_period)
    macd_signal = 1 if float(macd_line.iloc[-1].squeeze()) > float(signal_line.iloc[-1].squeeze()) else -1
    sma_series = calculate_sma(close_prices, sma_period)
    if sma_series.empty or pd.isna(sma_series.iloc[-1].squeeze()):
        sma_signal = 0
    else:
        sma_value = float(sma_series.iloc[-1].squeeze())
        sma_signal = 1 if current_price > sma_value else -1
    composite_signal = (rsi_weight * rsi_signal +
                        macd_weight * macd_signal +
                        sma_weight * sma_signal)
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

        # Convert the 'close' column value into a scalar float.
        close_prices = bars['close']
        current_price = float(close_prices.iloc[-1].squeeze())
        with params_lock:
            rsi_period     = strategy_params['RSI_PERIOD']
            rsi_oversold   = strategy_params['RSI_OVERSOLD']
            rsi_overbought = strategy_params['RSI_OVERBOUGHT']
            order_amount   = strategy_params['ORDER_AMOUNT']
            macd_fast      = strategy_params['MACD_FAST']
            macd_slow      = strategy_params['MACD_SLOW']
            macd_signal_period = strategy_params['MACD_SIGNAL']
            sma_period     = strategy_params['SMA_PERIOD']
            rsi_weight     = strategy_params['RSI_WEIGHT']
            macd_weight    = strategy_params['MACD_WEIGHT']
            sma_weight     = strategy_params['SMA_WEIGHT']
            buy_threshold  = strategy_params['BUY_THRESHOLD']
            sell_threshold = strategy_params['SELL_THRESHOLD']

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

        composite_signal = (rsi_weight * rsi_signal +
                            macd_weight * macd_signal +
                            sma_weight * sma_signal)
        logging.debug(f"{symbol}: RSI={current_rsi:.2f} (sig {rsi_signal}), MACD signal={macd_signal}, SMA value={sma_value:.2f} (sig {sma_signal})")
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
                order = submit_order(api, symbol, int(float(position.qty)), side='sell')
                if order:
                    record_trade(symbol, 'sell', int(float(position.qty)), current_price)
            else:
                decision = 'hold (no position)'
                logging.debug(f"{symbol}: Bearish composite signal but no position held.")
        else:
            decision = 'neutral'
            logging.debug(f"{symbol}: Composite signal in neutral zone.")

        with signals_lock:
            latest_signals[symbol] = {
                'RSI': round(current_rsi, 2),
                'MACD': macd_signal,
                'SMA': sma_signal,
                'composite': round(composite_signal, 2),
                'signal': decision,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

def record_trade(symbol: str, side: str, qty: int, price: float):
    with trade_log_lock:
        trade_entry = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'pnl': 0.0  # In a production system, compute PnL appropriately.
        }
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
                    'composite': None,
                    'signal': f'held ({pos.qty} shares)',
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        logging.info("Updated held positions from account.")
    except Exception as e:
        logging.error(f"Error updating held positions: {e}")

# ==============================
# PARAMETER TUNING LOGIC (RL-Style)
# ==============================

def tune_parameters():
    global strategy_params, rl_epsilon
    with trade_log_lock:
        if not trade_log:
            logging.debug("No trades to base tuning on. Skipping parameter tuning.")
            return
        buys = sum(1 for t in trade_log if t['side'] == 'buy')
        sells = sum(1 for t in trade_log if t['side'] == 'sell')
    with params_lock:
        if random.random() < rl_epsilon:
            logging.debug("RL Exploration: Randomly adjusting RSI thresholds.")
            strategy_params['RSI_OVERSOLD'] = random.uniform(20, 40)
            strategy_params['RSI_OVERBOUGHT'] = random.uniform(60, 80)
        else:
            if sells > buys:
                logging.debug("RL Exploitation: More sell trades detected; making RSI thresholds more conservative.")
                strategy_params['RSI_OVERSOLD'] = min(40, strategy_params['RSI_OVERSOLD'] + 0.5)
                strategy_params['RSI_OVERBOUGHT'] = max(60, strategy_params['RSI_OVERBOUGHT'] - 0.5)
            else:
                if strategy_params['RSI_OVERSOLD'] > 30:
                    strategy_params['RSI_OVERSOLD'] -= 0.5
                if strategy_params['RSI_OVERBOUGHT'] < 70:
                    strategy_params['RSI_OVERBOUGHT'] += 0.5

        if random.random() < rl_epsilon:
            logging.debug("RL Exploration: Randomly adjusting indicator weights.")
            strategy_params['RSI_WEIGHT']  = random.uniform(0.2, 0.5)
            strategy_params['MACD_WEIGHT'] = random.uniform(0.2, 0.5)
            strategy_params['SMA_WEIGHT']  = random.uniform(0.2, 0.5)
        else:
            if sells > buys:
                strategy_params['RSI_WEIGHT']  = max(0.1, strategy_params['RSI_WEIGHT'] - 0.01)
                strategy_params['MACD_WEIGHT'] = max(0.1, strategy_params['MACD_WEIGHT'] - 0.01)
                strategy_params['SMA_WEIGHT']  = max(0.1, strategy_params['SMA_WEIGHT'] - 0.01)
            else:
                strategy_params['RSI_WEIGHT']  = min(0.9, strategy_params['RSI_WEIGHT'] + 0.01)
                strategy_params['MACD_WEIGHT'] = min(0.9, strategy_params['MACD_WEIGHT'] + 0.01)
                strategy_params['SMA_WEIGHT']  = min(0.9, strategy_params['SMA_WEIGHT'] + 0.01)
        total = strategy_params['RSI_WEIGHT'] + strategy_params['MACD_WEIGHT'] + strategy_params['SMA_WEIGHT']
        strategy_params['RSI_WEIGHT']  /= total
        strategy_params['MACD_WEIGHT'] /= total
        strategy_params['SMA_WEIGHT']  /= total

        logging.debug(f"Updated strategy parameters: {strategy_params}")

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
    # Sort descending (best composite signal first)
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
    with open(STOCK_FILE, "w") as f:
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
    while True:
        if is_market_open():
            scan_and_trade(api)
        else:
            logging.debug("Market is closed. Skipping trading scan.")
        
        # Check if it is time for end-of-day watchlist discovery.
        now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
        discovery_time = now_et.replace(hour=16, minute=30, second=0, microsecond=0)
        if now_et >= discovery_time:
            today_str = now_et.strftime("%Y-%m-%d")
            if last_discovery_date != today_str:
                logging.info("Running end-of-day watchlist discovery...")
                update_watchlist(api)
                last_discovery_date = today_str
        
        time.sleep(CHECK_INTERVAL)

def tuning_loop():
    TUNE_INTERVAL = 60 * 10  # Tune every 10 minutes
    logging.info("Starting parameter tuning loop...")
    while True:
        tune_parameters()
        update_strategy_params(strategy_params)
        time.sleep(TUNE_INTERVAL)

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
    return render_template('index.html',
                           params=current_params,
                           signals=current_signals,
                           trade_log=trades,
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
    logging.info("Starting Flask dashboard on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=DEBUG)

if __name__ == '__main__':
    main()
 