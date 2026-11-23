"""
preprocessing.py
------------------
Cleans and prepares stock data for machine learning:
    - handles missing values
    - sorts by date
    - normalizes the Close price
    - builds sliding-window sequences for LSTM
    - splits data into train/test sets
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values and sorts the dataframe chronologically.

    Args:
        df (pd.DataFrame): Raw stock data.

    Returns:
        pd.DataFrame: Cleaned, sorted dataframe with no missing values.
    """
    df = df.copy()

    # Sort by date to make sure the time series is in correct order
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Forward-fill then back-fill any missing values (common with holidays,
    # corporate actions, or partial data from the API)
    df = df.ffill().bfill()

    # Drop any remaining rows that still contain NaNs
    df = df.dropna().reset_index(drop=True)

    return df


def scale_close_prices(df: pd.DataFrame, feature_col: str = "Close"):
    """
    Normalizes the closing price column using MinMaxScaler (0-1 range).

    Args:
        df (pd.DataFrame): Cleaned stock data.
        feature_col (str): Column to scale (default 'Close').

    Returns:
        tuple: (scaled_values (np.ndarray), fitted MinMaxScaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = df[[feature_col]].values
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def create_sequences(scaled_data: np.ndarray, window_size: int = 60):
    """
    Builds sliding-window sequences suitable for LSTM training.
    Each input sample X consists of `window_size` previous days,
    and the target y is the next day's scaled closing price.

    Args:
        scaled_data (np.ndarray): Normalized closing prices, shape (n, 1).
        window_size (int): Number of previous days to use per sequence.

    Returns:
        tuple: (X, y) as numpy arrays.
            X shape: (samples, window_size, 1)
            y shape: (samples,)
    """
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


def train_test_split_sequence(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
    """
    Splits sequential data into training and testing sets while
    preserving chronological order (no shuffling, since this is time series).

    Args:
        X (np.ndarray): Input sequences.
        y (np.ndarray): Targets.
        train_ratio (float): Proportion of data used for training.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def prepare_tabular_features(df: pd.DataFrame, window_size: int = 60):
    """
    Builds a tabular (non-sequence) feature set for classic ML models
    (Linear Regression, Random Forest). Each row's features are the
    previous `window_size` closing prices, target is the next close.

    Args:
        df (pd.DataFrame): Cleaned stock data (must contain 'Close').
        window_size (int): Number of lagged days to use as features.

    Returns:
        tuple: (X (np.ndarray), y (np.ndarray))
    """
    close_prices = df["Close"].values
    X, y = [], []
    for i in range(window_size, len(close_prices)):
        X.append(close_prices[i - window_size:i])
        y.append(close_prices[i])
    return np.array(X), np.array(y)
