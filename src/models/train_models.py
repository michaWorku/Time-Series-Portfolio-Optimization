"""
This script serves as the main pipeline for training both the ARIMA and LSTM
forecasting models. It loads preprocessed data, splits it into training and
testing sets, trains each model, and saves the trained models and their
forecasts. It then evaluates and compares the performance of both models.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import pmdarima as pm
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import callbacks
from typing import Dict, Tuple

# Add the project root to the path for module imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.arima_model import load_series, train_auto_arima, forecast_arima
from src.models.lstm_model import load_processed_data, create_windowed_dataset, predict_lstm_chain, build_lstm_model, train_lstm_model
from src.models.evaluation import evaluate_forecasts

# --- Path and Configuration Constants ---
PROCESSED_DATA_PATH = project_root / 'data' / 'processed' / 'TSLA_processed.csv'
OUTPUTS_PATH = project_root / 'outputs'
MODELS_PATH = project_root / 'models'
TRAIN_END_DATE = '2023-12-31'
LSTM_WINDOW_SIZE = 60
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Ensure output and model directories exist
OUTPUTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

# --- Pipelines for each model ---
def run_arima_pipeline(
    data_path: Path, 
    train_end_date: str
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Runs the complete ARIMA training and forecasting pipeline.
    """
    print("--- Running ARIMA Pipeline ---")
    try:
        # Load data and split
        ts = load_series(data_path)
        train_data = ts.loc[ts.index <= train_end_date]
        test_data = ts.loc[ts.index > train_end_date]
        
        # Train model
        arima_model_path = MODELS_PATH / 'arima_model.pkl'
        model = train_auto_arima(train_data, save_path=arima_model_path)
        
        # Forecast
        n_forecast_steps = len(test_data)
        forecasts = forecast_arima(model, n_forecast_steps)
        forecasts.index = test_data.index
        
        # Evaluate performance
        metrics = evaluate_forecasts(test_data, forecasts)
        
        # Save forecast to CSV
        forecasts.to_csv(OUTPUTS_PATH / 'tsla_arima_forecast.csv', header=['forecast'])
        print(f"ARIMA forecast saved to: {OUTPUTS_PATH / 'tsla_arima_forecast.csv'}")

        print("ARIMA Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics, forecasts
    except Exception as e:
        print(f"An error occurred during the ARIMA pipeline: {e}")
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}, pd.Series()

def run_lstm_pipeline(
    data_path: Path, 
    train_end_date: str,
    window_size: int,
    epochs: int,
    batch_size: int
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Runs the complete LSTM training and forecasting pipeline.
    """
    print("\n" + "="*40 + "\n")
    print("--- Running LSTM Pipeline ---")
    try:
        # Load and split data
        df = load_processed_data(data_path)
        train_data = df.loc[df.index <= train_end_date]
        test_data = df.loc[df.index > train_end_date]
        
        # Create validation set from the end of the training data
        train_size = int(len(train_data) * 0.8)
        train_set = train_data['Close'].values[:train_size].reshape(-1, 1)
        val_set = train_data['Close'].values[train_size:].reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_set)
        val_scaled = scaler.transform(val_set)
        
        # Train model
        print("Building and training LSTM model...")
        model = train_lstm_model(train_scaled, val_scaled, window_size, epochs, batch_size)
        
        # Save the model and scaler
        lstm_model_path = MODELS_PATH / 'lstm_model.keras'
        lstm_scaler_path = MODELS_PATH / 'lstm_scaler.pkl'
        model.save(lstm_model_path)
        joblib.dump(scaler, lstm_scaler_path)
        print(f"LSTM model saved to: {lstm_model_path}")
        print(f"LSTM scaler saved to: {lstm_scaler_path}")
        
        # Forecast
        n_forecast_steps = len(test_data)
        initial_window = train_scaled[-window_size:].flatten()
        predicted_prices_unscaled = predict_lstm_chain(
            model=model,
            scaler=scaler,
            initial_window=initial_window,
            n_steps=n_forecast_steps,
            window_size=window_size
        )
        
        # Convert to pandas Series with correct index
        forecasts = pd.Series(predicted_prices_unscaled, index=test_data.index)
        
        # Evaluate performance
        metrics = evaluate_forecasts(test_data['Close'], forecasts)
        
        # Save forecast to CSV
        forecasts.to_csv(OUTPUTS_PATH / 'tsla_lstm_forecast.csv', header=['forecast'])
        print(f"LSTM forecast saved to: {OUTPUTS_PATH / 'tsla_lstm_forecast.csv'}")

        print("\nLSTM Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
        return metrics, forecasts
    except Exception as e:
        print(f"An error occurred during the LSTM pipeline: {e}")
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}, pd.Series()

def main():
    """
    Main function to run the combined training and evaluation pipeline.
    """
    print("--- Starting Combined Model Training and Evaluation Pipeline ---")
    print(f"Using data file: {PROCESSED_DATA_PATH}")

    # Run ARIMA pipeline
    arima_metrics, arima_forecast = run_arima_pipeline(PROCESSED_DATA_PATH, TRAIN_END_DATE)
    
    # Run LSTM pipeline
    lstm_metrics, lstm_forecast = run_lstm_pipeline(PROCESSED_DATA_PATH, TRAIN_END_DATE, LSTM_WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE)
    
    print("\n" + "="*40 + "\n")
    
    print("--- Training and Evaluation Summary ---")
    print("ARIMA Metrics:", arima_metrics)
    print("LSTM Metrics:", lstm_metrics)
    
    # Compare models based on MAE, handling potential NaNs
    arima_mae = arima_metrics.get('MAE', float('inf'))
    lstm_mae = lstm_metrics.get('MAE', float('inf'))
    
    if lstm_mae < arima_mae:
        print("\nBased on MAE, the LSTM model performed better during the training phase.")
    else:
        print("\nBased on MAE, the ARIMA model performed better during the training phase.")
    
    print("\nPipeline finished successfully.")


if __name__ == '__main__':
    main()
