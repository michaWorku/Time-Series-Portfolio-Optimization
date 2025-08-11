"""
This is the main script to run a new forecast using the pre-trained models.

It loads the models, performs the forecast for a specified number of periods,
and saves plots of the results.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple

# Add the project root to the path for module imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import helper functions from the project's sub-modules
from src.utils.save_load import load_pickle, load_keras_model
from src.models.arima_model import load_series
from src.utils.forecasting import forecast_arima, predict_lstm_chain
from src.utils.plot_forecast import plot_history_and_forecast

# --- Path and Configuration Constants ---
project_root = Path(__file__).resolve().parents[1]
OUTPUTS_PATH = project_root / 'outputs'
OUTPUTS_PATH.mkdir(exist_ok=True)

def run_full_forecast_pipeline(
    data_path: Path,
    arima_model_path: Path,
    lstm_model_path: Path,
    lstm_scaler_path: Path,
    n_forecast_periods: int = 30,
    history_lookback_years: int = 3
):
    """
    Main function to run the forecasting pipeline.

    Args:
        data_path (Path): Path to processed CSV data file.
        arima_model_path (Path): Path to saved ARIMA model pickle.
        lstm_model_path (Path): Path to saved LSTM Keras model.
        lstm_scaler_path (Path): Path to saved LSTM scaler pickle.
        n_forecast_periods (int): Number of future days to forecast.
        history_lookback_years (int): Years of history to include in plots.
    """
    print("--- Starting Full Forecasting Pipeline ---")

    # 1. Load data
    try:
        full_series = load_series(data_path)
        history_start_date = full_series.index.max() - pd.DateOffset(years=history_lookback_years)
        plot_history = full_series[history_start_date:]
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading data: {e}")
        return

    # 2. Load models & scaler
    print("\n--- Loading Models ---")
    arima_model = load_pickle(arima_model_path)
    lstm_model = load_keras_model(lstm_model_path)
    lstm_scaler = load_pickle(lstm_scaler_path)

    if arima_model is None or lstm_model is None or lstm_scaler is None:
        print("\nCould not load all necessary models. Please ensure they have been trained and saved.")
        return

    # 3. ARIMA forecast
    print(f"\n--- Forecasting with ARIMA model for {n_forecast_periods} periods ---")
    arima_forecast = forecast_arima(arima_model, n_periods=n_forecast_periods)
    arima_forecast_index = pd.date_range(start=full_series.index[-1], periods=n_forecast_periods + 1, freq='B')[1:]
    arima_forecast.index = arima_forecast_index

    # 4. LSTM forecast
    print(f"\n--- Forecasting with LSTM model for {n_forecast_periods} periods ---")
    window_size = 60
    initial_window = full_series.values[-window_size:]
    lstm_forecast_vals = predict_lstm_chain(
        model=lstm_model,
        scaler=lstm_scaler,
        initial_window=initial_window,
        n_steps=n_forecast_periods,
        window_size=window_size
    )
    lstm_forecast = pd.Series(lstm_forecast_vals, index=arima_forecast_index)

    # 5. Plot results
    print("\n--- Plotting Results ---")
    plot_history_and_forecast(
        history=plot_history,
        forecast=arima_forecast,
        title=f"ARIMA Forecast for the next {n_forecast_periods} days",
        savepath=str(OUTPUTS_PATH / 'arima_full_forecast.png')
    )
    plot_history_and_forecast(
        history=plot_history,
        forecast=lstm_forecast,
        title=f"LSTM Forecast for the next {n_forecast_periods} days",
        savepath=str(OUTPUTS_PATH / 'lstm_full_forecast.png')
    )

    print("\n--- Forecasting pipeline finished successfully ---")
    print(f"ARIMA forecast saved to: {OUTPUTS_PATH / 'arima_full_forecast.png'}")
    print(f"LSTM forecast saved to: {OUTPUTS_PATH / 'lstm_full_forecast.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full forecasting pipeline with ARIMA and LSTM.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to processed TSLA CSV file.")
    parser.add_argument("--arima_model_path", type=str, required=True, help="Path to saved ARIMA model.")
    parser.add_argument("--lstm_model_path", type=str, required=True, help="Path to saved LSTM model.")
    parser.add_argument("--lstm_scaler_path", type=str, required=True, help="Path to saved LSTM scaler.")
    parser.add_argument("--n_forecast_periods", type=int, default=30, help="Number of forecast days.")
    parser.add_argument("--history_lookback_years", type=int, default=3, help="Years of history for plots.")

    args = parser.parse_args()

    run_full_forecast_pipeline(
        data_path=Path(args.data_path),
        arima_model_path=Path(args.arima_model_path),
        lstm_model_path=Path(args.lstm_model_path),
        lstm_scaler_path=Path(args.lstm_scaler_path),
        n_forecast_periods=args.n_forecast_periods,
        history_lookback_years=args.history_lookback_years
    )
