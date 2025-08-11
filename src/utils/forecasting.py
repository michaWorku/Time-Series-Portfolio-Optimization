"""
This script contains the core forecasting and evaluation functions for both
ARIMA and LSTM models. It is designed to be imported by other scripts
that need to perform predictions or assess model performance.
"""

from typing import Tuple, Optional, Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# statsmodels / pmdarima imports are optional here, we handle potential ImportError
try:
    from pmdarima import ARIMA as pmdARIMA
    # This import is not necessary for the core functions but good for context
    from statsmodels.tsa.arima.model import ARIMAResults
except ImportError:
    pmdARIMA = None
    ARIMAResults = None

# keras / tensorflow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    tf = None
    load_model = None
    MinMaxScaler = None


# --- Evaluation metrics ---
def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates the Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    and Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics.
    """
    # Align the arrays and drop NaNs if necessary
    y_true_clean = np.nan_to_num(y_true, nan=0.0)
    y_pred_clean = np.nan_to_num(y_pred, nan=0.0)

    if len(y_true_clean) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

    mae_val = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse_val = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

    # MAPE calculation, handling division by zero
    mape_val = np.mean(np.abs((y_true_clean - y_pred_clean) / np.where(y_true_clean == 0, 1, y_true_clean))) * 100

    return {'MAE': mae_val, 'RMSE': rmse_val, 'MAPE': mape_val}


# --- Forecasting functions ---
def forecast_arima(model: pmdARIMA, n_periods: int) -> pd.Series:
    """
    Generates a forecast for the specified number of periods using an ARIMA model.

    Args:
        model (pmdARIMA): The fitted pmdarima.ARIMA model.
        n_periods (int): The number of steps to forecast.

    Returns:
        pd.Series: A series of forecasted values.
    """
    if pmdARIMA is None:
        raise ImportError("pmdarima library is not installed.")
    
    forecasts = model.predict(n_periods=n_periods)
    return pd.Series(forecasts)


def predict_lstm_chain(model: Any, scaler: MinMaxScaler, initial_window: np.ndarray, n_steps: int, window_size: int) -> np.ndarray:
    """
    Performs multi-step forecasting with an LSTM model by iteratively predicting
    the next step and feeding it back into the model.

    Args:
        model (Any): The trained Keras LSTM model.
        scaler (MinMaxScaler): The scaler used for data normalization.
        initial_window (np.ndarray): The initial window of historical data to start the forecast.
        n_steps (int): The number of future steps to predict.
        window_size (int): The size of the sliding window used for training.

    Returns:
        np.ndarray: An array of unscaled, forecasted values.
    """
    if tf is None:
        raise ImportError("Tensorflow and Keras are not installed.")

    current_window = initial_window.reshape((1, window_size, 1))
    forecasts = []

    for _ in range(n_steps):
        # Predict the next step
        next_step = model.predict(current_window, verbose=0)[0]
        forecasts.append(next_step)
        
        # Shift the window to include the new prediction
        current_window = np.append(current_window[:, 1:, :], [[next_step]], axis=1)

    # Inverse transform the scaled predictions
    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
    return forecasts.flatten()

