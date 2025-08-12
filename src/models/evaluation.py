"""
This script provides functions to evaluate the performance of time series forecasting
models. It includes standard metrics like Mean Absolute Error (MAE), Root Mean
Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

This script is designed to be a central utility for comparing different models
on a consistent set of metrics.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict
from pathlib import Path
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
# Used for loading the Keras model
from tensorflow.keras.models import load_model as keras_load_model
import sys

# Add the project root to the path for module imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.arima_model import load_series, forecast_arima
from src.models.lstm_model import load_processed_data, predict_lstm_chain
from src.utils.save_load import load_pickle, load_keras_model

# --- Path and Configuration Constants ---
PROCESSED_DATA_PATH = project_root / 'data' / 'processed' / 'TSLA_processed.csv'
MODELS_PATH = project_root / 'models'
ARIMA_MODEL_PATH = MODELS_PATH / 'arima_model.pkl'
LSTM_MODEL_PATH = MODELS_PATH / 'lstm_model.keras'
LSTM_SCALER_PATH = MODELS_PATH / 'lstm_scaler.pkl'
TRAIN_END_DATE = '2023-12-31'
LSTM_WINDOW_SIZE = 60

# --- Evaluation Metrics ---
def evaluate_forecasts(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Calculates the Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
    It handles potential NaN values by aligning the series indices and dropping NaNs.

    Args:
        y_true (Union[pd.Series, np.ndarray]): The true values.
        y_pred (Union[pd.Series, np.ndarray]): The predicted values.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics.
    """
    # Ensure both inputs are pandas Series for easy alignment
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)

    # Align the two series based on their index and drop any NaN values
    y_true_aligned, y_pred_aligned = y_true.align(y_pred, join='inner')

    # Drop NaNs from both aligned series
    combined_df = pd.DataFrame({'y_true': y_true_aligned, 'y_pred': y_pred_aligned}).dropna()
    y_true_clean = combined_df['y_true']
    y_pred_clean = combined_df['y_pred']

    if y_true_clean.empty:
        print("Warning: No common, non-NaN values found for evaluation.")
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

    # Calculate metrics
    mae_val = np.mean(np.abs(y_true_clean - y_pred_clean))
    rmse_val = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    
    # MAPE calculation, handling division by zero
    mape_val = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean.replace(0, np.nan).dropna())) * 100
    
    return {'MAE': mae_val, 'RMSE': rmse_val, 'MAPE': mape_val}


def evaluate_arima_model() -> Dict[str, float]:
    """
    Loads a saved ARIMA model, generates a forecast, and evaluates its performance on the test data.
    """
    print("--- Evaluating ARIMA Model ---")
    try:
        # Load the full series and split it chronologically
        ts = load_series(PROCESSED_DATA_PATH)
        train_end_date = pd.to_datetime(TRAIN_END_DATE)
        train_data = ts.loc[ts.index <= train_end_date]
        test_data = ts.loc[ts.index > train_end_date]
        
        # Load the saved ARIMA model
        arima_model = load_pickle(ARIMA_MODEL_PATH)
        if arima_model is None:
            raise FileNotFoundError("ARIMA model not found.")
        print("ARIMA model loaded successfully.")

        # Generate a new forecast for the test period using the loaded model
        n_periods_test = len(test_data)
        forecasts = forecast_arima(arima_model, n_periods=n_periods_test)
        forecasts.index = test_data.index
        
        # Evaluate performance
        metrics = evaluate_forecasts(test_data, forecasts)

        print("ARIMA Model Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return metrics
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}
    except Exception as e:
        print(f"An error occurred during ARIMA evaluation: {e}")
        return {}


def evaluate_lstm_model() -> Dict[str, float]:
    """
    Loads a saved LSTM model and scaler, generates a forecast, and evaluates its performance.
    """
    print("--- Evaluating LSTM Model ---")
    try:
        # Load the full series and split it chronologically
        df_full = load_processed_data(PROCESSED_DATA_PATH)
        train_end_date = pd.to_datetime(TRAIN_END_DATE)
        train_data = df_full.loc[df_full.index <= train_end_date]
        test_data = df_full.loc[df_full.index > train_end_date]

        # Load the saved LSTM model and scaler
        lstm_model = load_keras_model(LSTM_MODEL_PATH)
        scaler = load_pickle(LSTM_SCALER_PATH)

        if lstm_model is None or scaler is None:
            raise FileNotFoundError("LSTM model or scaler not found.")
        print("LSTM model and scaler loaded successfully.")

        # Prepare for forecasting
        n_forecast_steps = len(test_data)
        train_close_series = train_data['Close'].values.reshape(-1, 1)
        scaler.fit(train_close_series)
        initial_window_unscaled = train_data['Close'].values[-LSTM_WINDOW_SIZE:]
        initial_window_scaled = scaler.transform(initial_window_unscaled.reshape(-1, 1)).flatten()
        
        # Forecast the entire test period
        predicted_prices_unscaled = predict_lstm_chain(
            model=lstm_model,
            scaler=scaler,
            initial_window=initial_window_scaled,
            n_steps=n_forecast_steps,
            window_size=LSTM_WINDOW_SIZE
        )
        
        # Create a pandas Series with the correct date index
        preds = pd.Series(predicted_prices_unscaled, index=test_data.index)
        
        # Evaluate performance
        metrics = evaluate_forecasts(test_data['Close'], preds)
        
        print("LSTM Model Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
        return metrics
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}
    except Exception as e:
        print(f"An error occurred during LSTM evaluation: {e}")
        return {}


def main():
    """
    Main function to run the complete model evaluation.
    """
    print("--- Starting Model Evaluation Pipeline ---")
    
    # Evaluate ARIMA model
    arima_metrics = evaluate_arima_model()
    
    print("\n" + "="*40 + "\n")
    
    # Evaluate LSTM model
    lstm_metrics = evaluate_lstm_model()
    
    print("\n" + "="*40 + "\n")
    
    print("--- Performance Comparison ---")
    if arima_metrics and lstm_metrics:
        print(f"ARIMA Metrics: {arima_metrics}")
        print(f"LSTM Metrics: {lstm_metrics}")
        
        arima_mae = arima_metrics.get('MAE', np.inf)
        lstm_mae = lstm_metrics.get('MAE', np.inf)
        
        if lstm_mae < arima_mae:
            print("\nBased on Mean Absolute Error, the LSTM model performed better.")
        else:
            print("\nBased on Mean Absolute Error, the ARIMA model performed better.")
    else:
        print("Unable to compare models due to errors in evaluation.")
    
    print("\nModel evaluation pipeline finished.")

if __name__ == '__main__':
    main()
