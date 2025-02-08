# Reinforcement Trading Bot

A fully featured trading bot that uses reinforcement learning–inspired parameter tuning, a Flask web dashboard, market-hour awareness, and end-of-day watchlist discovery. This bot uses [yfinance](https://github.com/ranaroussi/yfinance) to retrieve near real-time historical data and executes trades via [Alpaca's API](https://alpaca.markets/). It calculates a composite signal based on technical indicators (RSI, MACD, and SMA) to determine buy/sell decisions during US market hours and automatically adjusts its strategy parameters over time.

## Features

- **Near Real-Time Data:** Uses yfinance for up-to-date historical data.
- **Automated Trading:** Executes trades via Alpaca’s API (paper trading for testing).
- **Composite Signal Calculation:** Combines RSI, MACD, and SMA signals using configurable weights.
- **Reinforcement Learning–Style Parameter Tuning:** Automatically adjusts trading parameters based on recent trade performance.
- **End-of-Day Watchlist Discovery:** Updates the watchlist at market close, retaining only the top-performing stocks.
- **Web Dashboard:** Hosts a Flask-based dashboard to view current strategy settings, trade logs, and asset signals.

## Requirements

- Python 3.11+
- [alpaca-trade-api](https://pypi.org/project/alpaca-trade-api/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [Flask](https://flask.palletsprojects.com/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [yfinance](https://pypi.org/project/yfinance/)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ProgramingPro2/Reinforcement-Stocks.git
   cd Reinforcement-Stocks
