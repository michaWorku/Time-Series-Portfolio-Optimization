"""
This script performs a backtest using the trained ARIMA and LSTM models,
generates plots of the backtest results, and saves the results to a CSV file.
It is a standalone script that uses the functions from the provided files.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model as keras_load_model
from typing import Dict, Tuple

# Add the project root to the path for module imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import helper functions from the project structure
# Assuming the script is run from the project root
try:
    from src.models.arima_model import load_series, forecast_arima, train_auto_arima
    from src.models.lstm_model import load_processed_data, predict_lstm_chain
    from src.models.evaluation import evaluate_forecasts
    from src.utils.plot_forecast import plot_history_and_forecast
    from src.utils.save_load import load_pickle, load_keras_model
except ImportError:
    # Fallback for a flat directory structure if imports fail
    print("Warning: Could not import modules from 'src.models'. Falling back to local imports.")


# --- Path and Configuration Constants ---
PROCESSED_DATA_PATH = Path('data/processed/TSLA_processed.csv')
OUTPUTS_PATH = Path('outputs')
MODELS_PATH = Path('models')

ARIMA_MODEL_PATH = MODELS_PATH / 'arima_model.pkl'
LSTM_MODEL_PATH = MODELS_PATH / 'lstm_model.keras'
LSTM_SCALER_PATH = MODELS_PATH / 'lstm_scaler.pkl'

TRAIN_END_DATE = '2023-12-31'
LSTM_WINDOW_SIZE = 60

# Ensure directories exist
OUTPUTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)


def run_backtest_pipeline(data_path: Path, train_end_date: str) -> pd.DataFrame:
    """
    Main function to run the complete backtesting pipeline.

    Args:
        data_path (Path): Path to the processed data CSV file.
        train_end_date (str): The date to split the data into training and testing.

    Returns:
        pd.DataFrame: A DataFrame containing the true values and forecasts from both models.
    """
    print("--- Starting Backtest and Results Export Pipeline ---")
    print(f"Using data file: {data_path}")

    # Load the data
    try:
        ts = load_series(data_path)
        train_series = ts.loc[:train_end_date]
        test_series = ts.loc[train_end_date:].iloc[1:]  # Exclude the split date from test set
        n_forecast_steps = len(test_series)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading or splitting data: {e}")
        return pd.DataFrame()

    # Create a DataFrame to store results
    results_df = pd.DataFrame({
        'Date': test_series.index,
        'Actual': test_series.values
    }).set_index('Date')
    
    # --- ARIMA Backtest ---
    print("\n--- Running ARIMA Backtest ---")
    try:
        arima_model = load_pickle(str(ARIMA_MODEL_PATH))
        if arima_model:
            # Re-fitting is often necessary for `pmdarima` for forecasting,
            # but we can try to use the predict method directly first.
            # `model.predict` will work if the model was trained on the correct data.
            # In a true backtest, you'd re-train on a sliding window. Here we'll use a fixed train/test split.
            arima_forecasts = arima_model.predict(n_periods=n_forecast_steps)
            arima_forecasts.index = test_series.index
            results_df['ARIMA_Forecast'] = arima_forecasts

            # Plot and save the ARIMA forecast figure
            plot_history_and_forecast(
                history=train_series,
                forecast=arima_forecasts,
                title="ARIMA Model Backtest: Forecast vs. Actual",
                savepath=str(OUTPUTS_PATH / 'arima_backtest.png')
            )
    except Exception as e:
        print(f"An error occurred during the ARIMA backtest: {e}")
        results_df['ARIMA_Forecast'] = np.nan
        
    # --- LSTM Backtest ---
    print("\n--- Running LSTM Backtest ---")
    try:
        lstm_model = keras_load_model(str(LSTM_MODEL_PATH))
        scaler = load_keras_model(str(LSTM_SCALER_PATH))

        if lstm_model and scaler:
            # Prepare data for LSTM forecast
            train_data = load_processed_data(str(data_path)).loc[:train_end_date]
            initial_window_unscaled = train_data['Close'].values[-LSTM_WINDOW_SIZE:]
            initial_window_scaled = scaler.transform(initial_window_unscaled.reshape(-1, 1)).flatten()

            # Generate the forecast
            lstm_forecasts_unscaled = predict_lstm_chain(
                model=lstm_model,
                scaler=scaler,
                initial_window=initial_window_scaled,
                n_steps=n_forecast_steps,
                window_size=LSTM_WINDOW_SIZE
            )
            
            lstm_forecasts = pd.Series(lstm_forecasts_unscaled, index=test_series.index)
            results_df['LSTM_Forecast'] = lstm_forecasts
            
            # Plot and save the LSTM forecast figure
            plot_history_and_forecast(
                history=train_series,
                forecast=lstm_forecasts,
                title="LSTM Model Backtest: Forecast vs. Actual",
                savepath=str(OUTPUTS_PATH / 'lstm_backtest.png')
            )
            
    except Exception as e:
        print(f"An error occurred during the LSTM backtest: {e}")
        results_df['LSTM_Forecast'] = np.nan

    # --- Save Combined Results ---
    output_csv_path = OUTPUTS_PATH / 'backtest_results.csv'
    results_df.to_csv(output_csv_path)
    print(f"\nBacktest results saved to: {output_csv_path}")
    
    return results_df


if __name__ == '__main__':
    # Run the backtest pipeline when the script is executed
    run_backtest_pipeline(PROCESSED_DATA_PATH, TRAIN_END_DATE)

