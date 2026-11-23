"""
indicators.py
---------------
Calculates common technical indicators used in stock trend analysis:
    - Simple Moving Averages (20-day, 50-day)
    - Relative Strength Index (RSI)
    - MACD (Moving Average Convergence Divergence)
"""

import pandas as pd


def add_moving_averages(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """
    Adds short-term and long-term simple moving averages to the dataframe.

    Args:
        df (pd.DataFrame): Stock data with a 'Close' column.
        short_window (int): Window size for short MA (default 20).
        long_window (int): Window size for long MA (default 50).

    Returns:
        pd.DataFrame: Dataframe with 'MA20' and 'MA50' columns added.
    """
    df = df.copy()
    df[f"MA{short_window}"] = df["Close"].rolling(window=short_window).mean()
    df[f"MA{long_window}"] = df["Close"].rolling(window=long_window).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI), a momentum oscillator
    that measures the speed and magnitude of price changes (0-100 scale).

    Args:
        df (pd.DataFrame): Stock data with a 'Close' column.
        period (int): Look-back period for RSI (default 14).

    Returns:
        pd.DataFrame: Dataframe with an 'RSI' column added.
    """
    df = df.copy()
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculates MACD (Moving Average Convergence Divergence) and its
    signal line, used to identify trend direction and momentum shifts.

    Args:
        df (pd.DataFrame): Stock data with a 'Close' column.
        fast (int): Fast EMA period (default 12).
        slow (int): Slow EMA period (default 26).
        signal (int): Signal line EMA period (default 9).

    Returns:
        pd.DataFrame: Dataframe with 'MACD', 'MACD_Signal', and
                       'MACD_Histogram' columns added.
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Histogram"] = histogram
    return df


def get_trend_signal(df: pd.DataFrame) -> dict:
    """
    Derives a simple rule-based Buy / Sell / Hold signal from the most
    recent RSI, MACD, and moving-average values. This is a lightweight
    heuristic (not a guarantee) meant to summarize current momentum.

    Rules (each contributes one vote):
        - RSI < 30  -> bullish (oversold, may bounce back up)
          RSI > 70  -> bearish (overbought, may pull back)
        - MACD > MACD_Signal -> bullish crossover
          MACD < MACD_Signal -> bearish crossover
        - MA20 > MA50 -> bullish (short-term uptrend)
          MA20 < MA50 -> bearish (short-term downtrend)

    Args:
        df (pd.DataFrame): Dataframe that already has RSI, MACD, MACD_Signal,
                            MA20, and MA50 columns (from calculate_all_indicators).

    Returns:
        dict: {
            'signal': 'BUY' | 'SELL' | 'HOLD',
            'bullish_votes': int,
            'bearish_votes': int,
            'rsi': float,
            'macd': float,
            'macd_signal': float,
        }
    """
    clean = df.dropna(subset=["RSI", "MACD", "MACD_Signal", "MA20", "MA50"])
    if clean.empty:
        return {
            "signal": "HOLD",
            "bullish_votes": 0,
            "bearish_votes": 0,
            "rsi": float("nan"),
            "macd": float("nan"),
            "macd_signal": float("nan"),
        }
    latest = clean.iloc[-1]

    bullish_votes = 0
    bearish_votes = 0

    if latest["RSI"] < 30:
        bullish_votes += 1
    elif latest["RSI"] > 70:
        bearish_votes += 1

    if latest["MACD"] > latest["MACD_Signal"]:
        bullish_votes += 1
    else:
        bearish_votes += 1

    if latest["MA20"] > latest["MA50"]:
        bullish_votes += 1
    else:
        bearish_votes += 1

    if bullish_votes >= 2 and bullish_votes > bearish_votes:
        signal = "BUY"
    elif bearish_votes >= 2 and bearish_votes > bullish_votes:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "bullish_votes": bullish_votes,
        "bearish_votes": bearish_votes,
        "rsi": round(float(latest["RSI"]), 2),
        "macd": round(float(latest["MACD"]), 4),
        "macd_signal": round(float(latest["MACD_Signal"]), 4),
    }


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function that adds all technical indicators at once.

    Args:
        df (pd.DataFrame): Cleaned stock data with a 'Close' column.

    Returns:
        pd.DataFrame: Dataframe enriched with MA20, MA50, RSI, and MACD.
    """
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    return df
