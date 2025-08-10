# src/data_preprocessor.py

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def load_data(input_dir: str = "data/raw") -> dict:
    """
    Loads all CSV files from a specified directory, handling a non-standard header
    by manually assigning column names.

    Args:
        input_dir (str): The directory containing the raw CSV files.

    Returns:
        dict: A dictionary where keys are ticker symbols and values are DataFrames.
    """
    data = {}
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {input_dir}. Please run data_fetch.py first.")
        return data

    # Define the expected columns from the yfinance download
    # Note: 'Adj Close' is not in the user's provided raw data snippet, so we will use 'Close' as a proxy.
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    for file_name in csv_files:
        ticker = file_name.split('_')[0]
        file_path = os.path.join(input_dir, file_name)
        try:
            # Read the CSV file, skipping the first three lines of metadata and
            # explicitly setting the columns.
            df = pd.read_csv(file_path, skiprows=3, header=None, names=['Date'] + columns)
            
            # Set 'Date' column as the index and parse it as datetime
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
            
            data[ticker] = df
            print(f"Loaded raw data for {ticker}. Shape: {df.shape}")
            print(f"Columns for {ticker}: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")

    return data

def preprocess_data(data: dict) -> dict:
    """
    Cleans and preprocesses a dictionary of raw DataFrames by handling missing values,
    calculating returns, and scaling the data.

    Args:
        data (dict): A dictionary of raw pandas DataFrames.

    Returns:
        dict: A dictionary of processed pandas DataFrames.
    """
    processed_data = {}
    for ticker, df in data.items():
        try:
            print(f"\nPreprocessing data for {ticker}...")

            # --- Data Cleaning ---
            # Ensure the index is sorted
            df.sort_index(inplace=True)

            # Handle missing values using modern fillna syntax to avoid warnings
            df = df.ffill().bfill()
            print(f"Handled missing values for {ticker}. Shape: {df.shape}")

            # Define the key price column. We will use 'Close' as 'Adj Close' is not present.
            price_col = 'Close'
            if price_col not in df.columns:
                raise KeyError(f"'{price_col}' not found in data for {ticker}.")

            # Ensure numeric columns are correct dtype
            numeric_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop any rows that may have become all NaN after coercion
            df.dropna(inplace=True)
            print(f"After type coercion and dropping NaNs, shape is: {df.shape}")

            # --- Feature Engineering and Scaling ---
            # Calculate daily percentage change (returns)
            df['daily_return'] = df[price_col].pct_change()

            # Use MinMaxScaler on the price column for deep learning models
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[f'{price_col}_scaled'] = scaler.fit_transform(df[[price_col]])

            # Drop the first row with the NaN return value
            df.dropna(inplace=True)
            print(f"After calculating returns and dropping NaN, shape is: {df.shape}")

            processed_data[ticker] = df
        except Exception as e:
            print(f"Error preprocessing data for {ticker}: {e}")

    return processed_data

def save_processed(processed_data: dict, output_dir: str = 'data/processed') -> None:
    """
    Saves a dictionary of processed DataFrames to CSV files.

    Args:
        processed_data (dict): A dictionary of processed pandas DataFrames.
        output_dir (str): The directory to save the processed files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ticker, df in processed_data.items():
        output_path = os.path.join(output_dir, f"{ticker}_processed.csv")
        df.to_csv(output_path)
        print(f"Processed data for {ticker} saved to {output_path}")

if __name__ == '__main__':
    # Full preprocessing workflow
    raw_data = load_data()
    if raw_data:
        processed_dfs = preprocess_data(raw_data)
        if processed_dfs:
            save_processed(processed_dfs)
            print("\nSuccessfully processed and saved all data.")
            if 'TSLA' in processed_dfs:
                print("\nExample of processed TSLA data:")
                print(processed_dfs['TSLA'].head())
