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

   ```
   git clone https://github.com/ProgramingPro2/Reinforcement-Stocks.git
   cd Reinforcement-Stocks
   ```
   
2. **Create a virtual environment and install packages:**

   Linux:
   ```shell
   python3 -m venv venv
   source venv/bin/activate
   pip install alpaca_trade_api pandas numpy flask python-dotenv yfinance
   ```
   Windows (Not Tested):
   ```bash
   python3 -m venv venv
   venv\Scripts\activate
   pip install alpaca_trade_api pandas numpy flask python-dotenv yfinance
   ```

3. **Start script:**
   
   ```shell
   python main.py
   ```
   or in debug mode:
   ```shell
   python main.py --debug
   ```
   
## Configuration

   Environment Variables:
   Create a .env file in the project root and add your Alpaca API credentials:
   ```.env
   ALPACA_KEY=your_api_key_here
   ALPACA_SECRET=your_api_secret_here
   ENDPOINT=https://paper-api.alpaca.markets
   ```
   Adjust the ORDER_AMOUNT (dollar amount per trade) and other strategy parameters in the source code as needed.

## Watchlist
Create a stocks.json file in the working directory containing a list of stock tickers. For example:

   ```json
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
   ```

## Usage

Run the trading bot with the debug flag (optional):

```shell
python main.py --debug
```

When you run the script, it will:

    Retrieve near real-time historical data via yfinance.
    Compute composite signals based on RSI, MACD, and SMA.
    Execute trades on Alpaca’s paper trading endpoint (ensure your API credentials are set).
    Automatically adjust strategy parameters over time.
    Launch a flask dashboard you can access http://0.0.0.0:5000.


## Disclaimer

This project is for educational purposes only. Trading involves risk, and you should always perform thorough testing using paper trading before using any strategy with real capital. The author is not responsible for any losses incurred.

## Contributing

Contributions, issues, and feature requests are welcome! Please check the issues page for open problems or to suggest enhancements.
