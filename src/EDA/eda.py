# src/EDA/eda.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

def load_processed_data(input_dir: str = "data/processed") -> dict:
    """
    Loads all processed CSV files from a specified directory.

    Args:
        input_dir (str): The directory containing the processed CSV files.

    Returns:
        dict: A dictionary where keys are ticker symbols and values are DataFrames.
    """
    data = {}
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {input_dir}. Please run data_preprocessor.py first.")
        return data

    for file_name in csv_files:
        ticker = file_name.split('_')[0]
        file_path = os.path.join(input_dir, file_name)
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            data[ticker] = df
            print(f"Loaded processed data for {ticker}. Shape: {df.shape}")
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
    return data

def visualize_prices(data: dict) -> None:
    """
    Plots the Adjusted Close prices over time for each asset.

    Args:
        data (dict): A dictionary of processed pandas DataFrames.
    """
    print("\n--- Visualizing Adjusted Close Prices ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for ticker, df in data.items():
        if 'Adj Close' in df.columns:
            ax.plot(df.index, df['Adj Close'], label=ticker, linewidth=1.5)
        elif 'Close' in df.columns: # Fallback if Adj Close is not present
            ax.plot(df.index, df['Close'], label=ticker, linewidth=1.5)
    
    ax.set_title('Adjusted Close Prices Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_volatility(data: dict, window: int = 30) -> None:
    """
    Analyzes volatility by plotting daily returns and rolling statistics.

    Args:
        data (dict): A dictionary of processed pandas DataFrames.
        window (int): The window size for rolling calculations (e.g., 30 days).
    """
    print(f"\n--- Analyzing Volatility with a {window}-day Rolling Window ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    for ticker, df in data.items():
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Volatility Analysis for {ticker}', fontsize=16, fontweight='bold')

        # Plot Daily Returns
        axes[0].plot(df['daily_return'], color='blue', alpha=0.7, linewidth=0.8)
        axes[0].set_title('Daily Percentage Change (Returns)')
        axes[0].set_ylabel('Daily Return')
        axes[0].grid(True)

        # Plot Rolling Mean and Standard Deviation
        rolling_mean = df['daily_return'].rolling(window=window).mean()
        rolling_std = df['daily_return'].rolling(window=window).std()
        
        axes[1].plot(rolling_mean, label='Rolling Mean', color='green', linewidth=2)
        axes[1].plot(rolling_std, label='Rolling Std Dev', color='red', linewidth=2)
        axes[1].set_title(f'{window}-Day Rolling Mean and Standard Deviation of Returns')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True)
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def outlier_detection(data: dict, std_dev_threshold: int = 3) -> None:
    """
    Identifies and reports outliers in daily returns.

    Args:
        data (dict): A dictionary of processed pandas DataFrames.
        std_dev_threshold (int): The number of standard deviations from the mean to consider an outlier.
    """
    print(f"\n--- Detecting Outliers (>{std_dev_threshold} Std Deviations from Mean) ---")
    for ticker, df in data.items():
        mean_return = df['daily_return'].mean()
        std_return = df['daily_return'].std()
        
        outliers = df[(df['daily_return'] - mean_return).abs() > std_dev_threshold * std_return]
        
        print(f"\nOutliers for {ticker}:")
        if outliers.empty:
            print("No significant outliers detected.")
        else:
            print(outliers[['daily_return']].to_string())

def check_stationarity(data: dict) -> None:
    """
    Performs the Augmented Dickey-Fuller (ADF) test on the Adjusted Close prices.

    Args:
        data (dict): A dictionary of processed pandas DataFrames.
    """
    print("\n--- Checking for Stationarity (ADF Test) ---")
    for ticker, df in data.items():
        print(f"\nADF Test for {ticker} (Adj Close Price):")
        # Use a price column that exists
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

        if df[price_col].isnull().any():
            print(f"Warning: Data for {ticker} contains NaNs. The ADF test may not be reliable.")
            
        result = adfuller(df[price_col].dropna())
        adf_statistic = result[0]
        p_value = result[1]
        
        print(f'ADF Statistic: {adf_statistic:.4f}')
        print(f'p-value: {p_value:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')

        if p_value <= 0.05:
            print(f"Conclusion: The series for {ticker} is likely stationary.")
        else:
            print(f"Conclusion: The series for {ticker} is likely non-stationary and will require differencing.")
            
def calculate_risk_metrics(data: dict) -> None:
    """
    Calculates Value at Risk (VaR) and the Sharpe Ratio.

    Args:
        data (dict): A dictionary of processed pandas DataFrames.
    """
    print("\n--- Calculating Risk Metrics (VaR and Sharpe Ratio) ---")
    for ticker, df in data.items():
        print(f"\nMetrics for {ticker}:")
        returns = df['daily_return'].dropna()
        
        # Calculate Value at Risk (VaR) at 95% confidence level
        # This is the maximum loss expected over a given time frame with a certain confidence level.
        # We calculate the 5th percentile of returns.
        var_95 = np.percentile(returns, 5)
        print(f"95% VaR: {var_95:.4f} (Max expected loss of {abs(var_95):.2%} on a single day)")

        # Calculate Sharpe Ratio
        # Assuming a risk-free rate of 0 for simplicity.
        # Sharpe Ratio = (Average Return - Risk-Free Rate) / Standard Deviation of Return
        annualized_returns = returns.mean() * 252 # 252 trading days in a year
        annualized_std_dev = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_returns / annualized_std_dev
        print(f"Annualized Sharpe Ratio: {sharpe_ratio:.4f}")

if __name__ == '__main__':
    # Load the processed data
    processed_dfs = load_processed_data()

    if processed_dfs:
        # Step 1: Visualize Prices
        visualize_prices(processed_dfs)
        
        # Step 2: Analyze Volatility
        analyze_volatility(processed_dfs)

        # Step 3: Outlier Detection
        outlier_detection(processed_dfs)

        # Step 4: Check for Stationarity
        check_stationarity(processed_dfs)

        # Step 5: Calculate Risk Metrics
        calculate_risk_metrics(processed_dfs)

