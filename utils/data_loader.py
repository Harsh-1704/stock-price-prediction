"""
data_loader.py
----------------
Handles downloading historical stock market data using the yfinance API.

Functions:
    fetch_stock_data(symbol, start_date, end_date) -> pd.DataFrame
    validate_symbol(symbol) -> bool
"""

import pandas as pd
import yfinance as yf


def validate_symbol(symbol: str) -> bool:
    """
    Quickly checks whether a stock symbol looks valid and has data available.

    Args:
        symbol (str): Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'RELIANCE.NS'

    Returns:
        bool: True if the symbol returns data, False otherwise.
    """
    if not symbol or not isinstance(symbol, str):
        return False

    try:
        ticker = yf.Ticker(symbol.strip().upper())
        # Try to pull a tiny slice of history to confirm the symbol is real
        hist = ticker.history(period="5d")
        return not hist.empty
    except Exception:
        return False


def fetch_stock_data(symbol: str, start_date, end_date) -> pd.DataFrame:
    """
    Fetches historical OHLCV stock data for a given symbol and date range.

    Args:
        symbol (str): Stock ticker symbol (e.g. 'AAPL', 'TCS.NS')
        start_date (str or datetime.date): Start date for historical data
        end_date (str or datetime.date): End date for historical data

    Returns:
        pd.DataFrame: DataFrame with columns
            ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    Raises:
        ValueError: If the symbol is invalid or no data could be retrieved.
    """
    symbol = symbol.strip().upper()

    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        raise ValueError(f"Error fetching data for '{symbol}': {e}")

    if df is None or df.empty:
        raise ValueError(
            f"No data found for symbol '{symbol}'. "
            "Please check the ticker (e.g. use 'RELIANCE.NS' for NSE stocks) "
            "and try a valid date range."
        )

    # yfinance sometimes returns a MultiIndex column structure (esp. for
    # single-ticker downloads with newer versions). Flatten it if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()

    # Ensure the expected columns exist; some exchanges/symbols may not
    # return 'Adj Close' depending on yfinance version/config.
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in expected_cols if c in df.columns]]

    return df
