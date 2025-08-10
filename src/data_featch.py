import yfinance as yf
import pandas as pd
import os

def save_data(df: pd.DataFrame, filename: str, folder: str = "data/raw") -> None:
    """
    Saves a pandas DataFrame to a CSV file in the specified folder.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the file (e.g., 'TSLA_raw.csv').
        folder (str): The directory to save the file.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, filename)
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")

def fetch_data(tickers: list, start_date: str, end_date: str) -> None:
    """
    Fetches historical stock data from Yahoo Finance and saves it as CSV files.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['TSLA', 'BND', 'SPY']).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
    """
    print(f"Fetching data from {start_date} to {end_date} for tickers: {tickers}...")
    for ticker in tickers:
        try:
            # Download the data
            df = yf.download(ticker, start=start_date, end=end_date)
            
            # Use the new save_data function to save the fetched data
            save_data(df, f"{ticker}_raw.csv", "data/raw")
            print(f"Successfully fetched and saved data for {ticker}")

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

if __name__ == '__main__':
    # Define the assets and timeframe
    TICKERS = ['TSLA', 'BND', 'SPY']
    START_DATE = '2015-07-01'
    END_DATE = '2025-07-31'

    # Run the data fetching function
    fetch_data(TICKERS, START_DATE, END_DATE)
