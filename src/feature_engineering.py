"""
This script provides functions for feature engineering on financial time series data.
It includes methods to calculate daily returns, rolling volatility, and other metrics
necessary for time series forecasting and portfolio optimization.
"""
import pandas as pd
import numpy as np

def calculate_daily_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Calculates the daily percentage change (daily return) of a specified price column.

    Args:
        df (pd.DataFrame): The input DataFrame containing financial data.
        price_col (str): The name of the column to calculate returns on (e.g., 'Close').

    Returns:
        pd.DataFrame: The DataFrame with a new 'daily_return' column.
    """
    if price_col not in df.columns:
        print(f"Error: The specified price column '{price_col}' does not exist in the DataFrame.")
        return df

    # Calculate daily returns as percentage change.
    df['daily_return'] = df[price_col].pct_change()
    return df

def calculate_rolling_metrics(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Calculates rolling mean and standard deviation for daily returns.

    Args:
        df (pd.DataFrame): The input DataFrame, expected to have a 'daily_return' column.
        window (int): The rolling window size in days.

    Returns:
        pd.DataFrame: The DataFrame with new columns for rolling mean and rolling standard deviation.
    """
    if 'daily_return' not in df.columns:
        print("Error: 'daily_return' column not found. Please run calculate_daily_returns first.")
        return df

    # Calculate rolling mean and standard deviation on daily returns
    df['rolling_mean'] = df['daily_return'].rolling(window=window).mean()
    df['rolling_std'] = df['daily_return'].rolling(window=window).std()
    return df

def calculate_risk_metrics(returns: pd.Series, risk_free_rate: float = 0) -> dict:
    """
    Calculates key risk metrics: Value at Risk (VaR) and Sharpe Ratio.

    Args:
        returns (pd.Series): A pandas Series of daily returns.
        risk_free_rate (float): The risk-free rate for Sharpe Ratio calculation (default is 0 for simplicity).

    Returns:
        dict: A dictionary containing the calculated VaR and Sharpe Ratio.
    """
    if returns.empty:
        return {'VaR_95': np.nan, 'Sharpe_Ratio': np.nan}

    # Calculate Value at Risk (VaR) at 95% confidence level
    # We use the 5th percentile of the returns distribution
    var_95 = returns.quantile(0.05)

    # Calculate Sharpe Ratio
    # We assume 252 trading days for annualization
    annualized_returns = returns.mean() * 252
    annualized_std = returns.std() * np.sqrt(252)
    
    if annualized_std == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (annualized_returns - risk_free_rate) / annualized_std
        
    return {
        'VaR_95': var_95,
        'Sharpe_Ratio': sharpe_ratio
    }

def normalize_price_data(df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
    """
    Normalizes a specified price column using min-max scaling to a range of 0 to 1.

    Args:
        df (pd.DataFrame): The input DataFrame.
        price_col (str): The name of the column to normalize.

    Returns:
        pd.DataFrame: The DataFrame with a new normalized column.
    """
    if price_col not in df.columns:
        print(f"Error: The specified price column '{price_col}' does not exist in the DataFrame.")
        return df

    min_val = df[price_col].min()
    max_val = df[price_col].max()
    df[f'{price_col}_scaled'] = (df[price_col] - min_val) / (max_val - min_val)
    return df
