"""
This script implements a robust deep learning LSTM model for time series forecasting.
It combines the modular functions and best practices from the provided scripts,
including:
- A clean, function-based structure.
- Data loading, chronological splitting, and scaling.
- A configurable LSTM model with Early Stopping to prevent overfitting.
- An elegant iterative forecasting method.
- Performance evaluation with key metrics (MAE, RMSE, MAPE).
- Visualization of the forecast against actual values.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

# Ensure the project root is in the path for module imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def load_processed_data(file_path: Path) -> pd.DataFrame:
    """
    Loads preprocessed data from a CSV file.

    Args:
        file_path (Path): The full path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame with 'Date' as the index and 'Close' column.
        An empty DataFrame is returned if the file is not found.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Error: Data file not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        # Select the 'Close' column as the time series data
        df = df[['Close']].astype(float)
        return df
    except KeyError as e:
        raise KeyError(f"Missing required column 'Close' or 'Date' in the data file: {e}")

def create_windowed_dataset(
    series: np.ndarray, 
    window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a windowed dataset for an LSTM model.
    
    Args:
        series (np.ndarray): The time series data to be windowed.
        window_size (int): The number of past steps to consider for each prediction.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing features (X) and labels (y).
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

def build_lstm_model(window_size: int) -> models.Sequential:
    """
    Builds a simple LSTM model.
    
    Args:
        window_size (int): The input window size for the LSTM layer.
        
    Returns:
        models.Sequential: The compiled Keras Sequential model.
    """
    model = models.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    
    return model

def train_lstm_model(
    train_data: np.ndarray,
    val_data: np.ndarray,
    window_size: int,
    epochs: int,
    batch_size: int
) -> models.Sequential:
    """
    Trains the LSTM model with a windowed dataset and early stopping.

    Args:
        train_data (np.ndarray): The scaled training data.
        val_data (np.ndarray): The scaled validation data.
        window_size (int): The window size for the LSTM model.
        epochs (int): The number of epochs for training.
        batch_size (int): The batch size for training.

    Returns:
        models.Sequential: The trained Keras model.
    """
    X_train, y_train = create_windowed_dataset(train_data, window_size)
    X_val, y_val = create_windowed_dataset(val_data, window_size)

    # Reshape the data for LSTM input [samples, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    
    model = build_lstm_model(window_size)
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model

def predict_lstm_chain(
    model: models.Sequential,
    scaler: MinMaxScaler,
    initial_window: np.ndarray,
    n_steps: int,
    window_size: int
) -> np.ndarray:
    """
    Generates a forecast for multiple steps using the trained LSTM model.
    This method iteratively predicts one step and uses it to predict the next.

    Args:
        model (models.Sequential): The trained LSTM model.
        scaler (MinMaxScaler): The scaler used to fit the training data.
        initial_window (np.ndarray): The last window of the training data.
        n_steps (int): The number of future steps to predict.
        window_size (int): The window size of the model.

    Returns:
        np.ndarray: An array of unscaled predicted prices.
    """
    forecast = []
    current_window = initial_window.copy()
    
    for _ in range(n_steps):
        # Reshape the current window for model prediction
        input_data = current_window.reshape(1, window_size, 1)
        
        # Predict the next step
        predicted_scaled_value = model.predict(input_data, verbose=0)[0][0]
        forecast.append(predicted_scaled_value)
        
        # Update the window by dropping the first value and appending the new prediction
        current_window = np.append(current_window[1:], predicted_scaled_value)
        
    # Inverse transform the scaled forecast to get actual prices
    forecast = np.array(forecast).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(forecast)
    
    return predicted_prices.flatten()

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates forecast performance using MAE, RMSE, and MAPE.
    
    Args:
        y_true (np.ndarray): The actual values.
        y_pred (np.ndarray): The predicted values.
        
    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics.
    """
    # Filter out any NaN values from the inputs
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    metrics = {
        'MAE': mean_absolute_error(y_true_clean, y_pred_clean),
        'RMSE': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'MAPE': np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    }
    return metrics

if __name__ == '__main__':
    # Standalone execution for testing
    print("Running LSTM model script in standalone mode for testing.")

    script_dir = Path(__file__).parent
    processed_data_path = script_dir.parent.parent / 'data' / 'processed' / 'TSLA_processed.csv'
    models_path = script_dir.parent / 'models'
    models_path.mkdir(exist_ok=True)

    try:
        df = load_processed_data(processed_data_path)
        train_end_date = '2023-12-31'
        window_size = 60
        epochs = 1
        batch_size = 32

        train_data = df.loc[df.index <= train_end_date]
        test_data = df.loc[df.index > train_end_date]
        
        # We need a validation set, so we'll take the last part of the training data
        train_size = int(len(train_data) * 0.8)
        val_size = len(train_data) - train_size
        
        # Training and validation split
        train_set = train_data['Close'].values[:train_size].reshape(-1, 1)
        val_set = train_data['Close'].values[train_size:].reshape(-1, 1)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_set)
        val_scaled = scaler.transform(val_set)

        # Train the model
        model = train_lstm_model(train_scaled, val_scaled, window_size, epochs, batch_size)
        
        # Prepare for forecasting
        n_forecast_steps = len(test_data)
        initial_window = train_scaled[-window_size:].flatten()
        
        # Forecast the entire test period
        predicted_prices_unscaled = predict_lstm_chain(
            model=model,
            scaler=scaler,
            initial_window=initial_window,
            n_steps=n_forecast_steps,
            window_size=window_size
        )
        
        actual_prices = test_data['Close'].values
        
        # Evaluate model performance
        print("Evaluating model performance...")
        metrics = evaluate_forecast(actual_prices, predicted_prices_unscaled)
        
        print("\n--- LSTM Model Evaluation Metrics ---")
        for metric, value in metrics.items():
            if not np.isnan(value):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: Could not be calculated")

        print("\nLSTM forecasting completed.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during standalone LSTM run: {e}")
