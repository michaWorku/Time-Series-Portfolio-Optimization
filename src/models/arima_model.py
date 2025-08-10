"""
This script provides functions for loading data, training, and forecasting
with an ARIMA model. It is designed to be a modular component of a larger
forecasting pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pmdarima as pm
import joblib
from typing import Tuple, Union

def load_series(file_path: Path) -> pd.Series:
    """
    Loads time series data from a CSV file, setting 'Date' as the index.
    
    Args:
        file_path (Path): The path to the CSV data file.
        
    Returns:
        pd.Series: A time series with a DatetimeIndex and the 'Close' prices.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If the 'Date' or 'Close' columns are not found.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Error: Data file not found at: {file_path}")
    
    try:
        # Load the CSV, ensuring 'Date' is parsed as a datetime object
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        # Select the 'Close' column, which is the time series we're interested in
        ts = df['Close'].astype(float)
        return ts
    except KeyError as e:
        raise KeyError(f"Missing required column in the data file: {e}")

def train_auto_arima(
    train_series: pd.Series, 
    save_path: Union[Path, None] = None
) -> pm.ARIMA:
    """
    Trains an ARIMA model using pmdarima's auto_arima to find optimal parameters.

    Args:
        train_series (pd.Series): The training data for the model.
        save_path (Path, optional): Path to save the fitted model. Defaults to None.
        
    Returns:
        pm.ARIMA: The fitted auto-ARIMA model.
    """
    print("Finding optimal ARIMA parameters with pmdarima.auto_arima...")
    # The 'D' seasonal period is daily, which is appropriate for daily stock data
    # The 'm' parameter is set to 1 for non-seasonal data.
    model = pm.auto_arima(
        train_series,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        m=1, # Non-seasonal
        seasonal=False,
        d=None, # Let auto_arima determine 'd'
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    if save_path:
        joblib.dump(model, save_path)
        print(f"Fitted ARIMA model saved to: {save_path}")
    return model

def forecast_arima(
    model: pm.ARIMA, 
    n_periods: int
) -> pd.Series:
    """
    Generates a forecast for the specified number of periods.
    
    Args:
        model (pm.ARIMA): The fitted ARIMA model.
        n_periods (int): The number of steps to forecast.
        
    Returns:
        pd.Series: A series of forecasted values.
    """
    forecast = model.predict(n_periods=n_periods)
    return pd.Series(forecast)

if __name__ == '__main__':
    # This block is for testing the functions in this script.
    # It will not run when imported by another script.
    print("Running ARIMA model script in standalone mode for testing.")
    
    # Example usage:
    # Set up paths relative to the script location
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / 'data' / 'processed' / 'TSLA_processed.csv'
    models_dir = script_dir.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    try:
        ts = load_series(data_path)
        train, test = ts.iloc[:-100], ts.iloc[-100:]
        
        # Train with auto_arima
        auto_model = train_auto_arima(train, save_path=models_dir / 'arima_auto_test.pkl')
        
        # Forecast and print metrics
        forecasts = forecast_arima(auto_model, len(test))
        forecasts.index = test.index
        
        # Print a snippet of the forecast
        print("\nARIMA Model Forecast:")
        print(forecasts.head())
        
    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(e)
