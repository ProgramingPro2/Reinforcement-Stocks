import json
import yfinance as yf

def is_valid_stock(symbol):
    """
    Checks whether a stock symbol returns valid price data.
    Returns True if data exists, False otherwise.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Attempt to download 1-day historical data.
        hist = ticker.history(period="1d")
        if hist.empty:
            print(f"Stock '{symbol}' returned no data (possibly delisted).")
            return False
        return True
    except Exception as e:
        print(f"Error downloading data for '{symbol}': {e}")
        return False

def main():
    with open("stocks.json", "r") as f:
        stocks = json.load(f)

    valid_stocks = []
    for symbol in stocks:
        if is_valid_stock(symbol):
            valid_stocks.append(symbol)
        else:
            print(f"Removing '{symbol}' from list.")

    with open("stocks.json", "w") as f:
        json.dump(valid_stocks, f, indent=2)
    
    print("Updated stocks.json. Valid stocks:")
    print(valid_stocks)

if __name__ == "__main__":
    main()