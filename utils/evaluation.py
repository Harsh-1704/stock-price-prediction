"""
evaluation.py
---------------
Computes standard regression evaluation metrics used to compare
model performance: MAE, MSE, RMSE, and R^2 Score.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates key regression evaluation metrics.

    Args:
        y_true (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        dict: {'MAE': ..., 'MSE': ..., 'RMSE': ..., 'R2': ...}
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def format_metrics_table(metrics_by_model: dict):
    """
    Converts a dictionary of {model_name: metrics_dict} into a tidy
    pandas DataFrame for easy display in Streamlit.

    Args:
        metrics_by_model (dict): e.g.
            {
                "Linear Regression": {"MAE": .., "MSE": .., "RMSE": .., "R2": ..},
                "Random Forest": {...},
                "LSTM": {...},
            }

    Returns:
        pd.DataFrame: Rows = models, Columns = metrics.
    """
    import pandas as pd
    df = pd.DataFrame(metrics_by_model).T
    df.index.name = "Model"
    return df.reset_index()
