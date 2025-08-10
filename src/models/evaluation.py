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
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    mae_val = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse_val = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    # MAPE calculation, handling division by zero
    mape_val = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean.replace(0, np.nan))) * 100
    
    return {'MAE': mae_val, 'RMSE': rmse_val, 'MAPE': mape_val}

if __name__ == '__main__':
    # Example usage for testing the evaluation script
    print("Running evaluation script in standalone mode for testing.")
    
    # Sample data
    y_actual = pd.Series([100, 110, 105, 120, 125])
    y_predicted = pd.Series([101, 112, 107, 118, 128])
    
    metrics = evaluate_forecasts(y_actual, y_predicted)
    
    print("Example Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

