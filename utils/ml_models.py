"""
ml_models.py
--------------
Defines and trains the three prediction models used in this project:
    1. Linear Regression (baseline)
    2. Random Forest Regressor
    3. LSTM Deep Learning model (TensorFlow/Keras)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ---------------------------------------------------------------------------
# Classic ML Models
# ---------------------------------------------------------------------------

def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Trains a simple Linear Regression model as a baseline predictor.

    Args:
        X_train (np.ndarray): Training features (lagged closing prices).
        y_train (np.ndarray): Training targets (next closing price).

    Returns:
        LinearRegression: Trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                         n_estimators: int = 200, random_state: int = 42) -> RandomForestRegressor:
    """
    Trains a Random Forest Regressor for stock price prediction.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Seed for reproducibility.

    Returns:
        RandomForestRegressor: Trained model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# LSTM Deep Learning Model
# ---------------------------------------------------------------------------

def build_lstm_model(input_shape: tuple, units: int = 50, dropout_rate: float = 0.2) -> Sequential:
    """
    Builds an LSTM neural network for time-series stock price prediction.

    Architecture:
        Input -> LSTM(units, return_sequences=True) -> Dropout
              -> LSTM(units) -> Dropout
              -> Dense(25) -> Dense(1)

    Args:
        input_shape (tuple): Shape of a single input sample, e.g. (60, 1).
        units (int): Number of LSTM units per layer.
        dropout_rate (float): Dropout rate to reduce overfitting.

    Returns:
        Sequential: Compiled (but untrained) Keras LSTM model.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


def train_lstm_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
                      epochs: int = 20, batch_size: int = 32, validation_split: float = 0.1):
    """
    Trains the LSTM model with early stopping to avoid overfitting.

    Args:
        model (Sequential): Compiled Keras LSTM model.
        X_train (np.ndarray): Training sequences, shape (samples, window, 1).
        y_train (np.ndarray): Training targets.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Training batch size.
        validation_split (float): Fraction of training data used for validation.

    Returns:
        History: Keras training history object.
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0,
    )
    return history
