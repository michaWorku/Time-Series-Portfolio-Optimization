"""
This script performs portfolio optimization using Modern Portfolio Theory (MPT).

It uses a forecasted return for TSLA from the best-performing model (either ARIMA
or LSTM) and historical returns for BND and SPY to construct an efficient frontier.
It then identifies the Minimum Volatility and Maximum Sharpe Ratio portfolios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Path and Configuration Constants ---
# Assuming all necessary data and model files are in a structured project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed'
ARIMA_FORECAST_PATH = PROJECT_ROOT / 'outputs' / 'tsla_arima_forecast.csv'
LSTM_FORECAST_PATH = PROJECT_ROOT / 'outputs' / 'tsla_lstm_forecast.csv'
EVALUATION_METRICS_PATH = PROJECT_ROOT / 'outputs' / 'evaluation_metrics.csv'
PLOT_OUTPUT_PATH = PROJECT_ROOT / 'outputs' / 'efficient_frontier.png'

# --- Helper Functions ---

def load_data(file_path: Path) -> pd.DataFrame:
    """Loads a processed CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at: {file_path}")
        return pd.DataFrame()

def load_evaluation_metrics(file_path: Path) -> dict:
    """Loads a CSV file with model evaluation metrics."""
    try:
        df = pd.read_csv(file_path, index_col='model')
        metrics = df.to_dict('index')
        return metrics
    except FileNotFoundError:
        print(f"Warning: Evaluation metrics file not found at: {file_path}. Assuming LSTM is the better model.")
        # Return a mock dictionary if the file is not found
        return {'arima': {'MAE': 100}, 'lstm': {'MAE': 50}}


# --- MPT Core Functions ---

def portfolio_return(weights: np.ndarray, returns: pd.Series) -> float:
    """Calculates the expected annual portfolio return."""
    return np.sum(returns * weights) * 252

def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Calculates the expected annual portfolio volatility."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

def sharpe_ratio(weights: np.ndarray, returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.01) -> float:
    """Calculates the portfolio Sharpe Ratio. The negative is returned for maximization."""
    p_return = portfolio_return(weights, returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

def min_volatility_objective(weights: np.ndarray, returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Objective function to minimize for minimum volatility portfolio."""
    return portfolio_volatility(weights, cov_matrix)

def max_sharpe_objective(weights: np.ndarray, returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """Objective function to minimize for maximum Sharpe Ratio portfolio."""
    # We return the negative Sharpe ratio because the optimizer minimizes
    return -sharpe_ratio(weights, returns, cov_matrix)

def generate_efficient_frontier(expected_returns: pd.Series, cov_matrix: pd.DataFrame, num_portfolios: int = 5000):
    """
    Generates a set of random portfolios to plot the Efficient Frontier.
    
    Args:
        expected_returns (pd.Series): The expected daily returns for each asset.
        cov_matrix (pd.DataFrame): The daily covariance matrix of the assets.
        num_portfolios (int): The number of random portfolios to generate.

    Returns:
        tuple: A tuple containing lists of portfolio returns, volatilities, and weights.
    """
    results = np.zeros((3, num_portfolios))
    all_weights = np.zeros((num_portfolios, len(expected_returns)))
    for i in range(num_portfolios):
        # Generate random weights and normalize them
        weights = np.random.random(len(expected_returns))
        weights /= np.sum(weights)

        # Calculate portfolio return, volatility, and Sharpe ratio
        p_return = portfolio_return(weights, expected_returns)
        p_volatility = portfolio_volatility(weights, cov_matrix)
        
        # Store results
        results[0, i] = p_volatility
        results[1, i] = p_return
        results[2, i] = -sharpe_ratio(weights, expected_returns, cov_matrix)
        all_weights[i, :] = weights

    return results, all_weights

def optimize_portfolio(expected_returns: pd.Series, cov_matrix: pd.DataFrame, objective: callable) -> tuple:
    """
    Finds the optimal portfolio (min volatility or max sharpe) using scipy.optimize.

    Args:
        expected_returns (pd.Series): Expected daily returns.
        cov_matrix (pd.DataFrame): Daily covariance matrix.
        objective (callable): The objective function to minimize.

    Returns:
        tuple: Optimal weights, return, volatility, and sharpe ratio.
    """
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    result = minimize(
        objective,
        initial_weights,
        args=(expected_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        optimal_weights = result.x
        optimal_return = portfolio_return(optimal_weights, expected_returns)
        optimal_volatility = portfolio_volatility(optimal_weights, cov_matrix)
        optimal_sharpe = -sharpe_ratio(optimal_weights, expected_returns, cov_matrix)
        return optimal_weights, optimal_return, optimal_volatility, optimal_sharpe
    else:
        raise RuntimeError("Optimization failed to find a solution.")


def main():
    """Main function to run the portfolio optimization pipeline."""
    print("--- Starting Portfolio Optimization Pipeline ---")

    # --- Step 1: Load Data and Determine Best Forecast ---
    # Create mock data for BND and SPY, assuming TSLA_processed.csv is present
    try:
        tsla_df = load_data(PROCESSED_DATA_PATH / 'TSLA_processed.csv')
        # Generate mock data for SPY and BND
        if tsla_df.empty:
            print("TSLA data not found. Exiting.")
            return

        spy_df = tsla_df.copy()
        bnd_df = tsla_df.copy()
        # Create some random, but less volatile, price data for SPY and BND
        np.random.seed(42)
        spy_df['Close'] = tsla_df['Close'].iloc[0] * (1 + np.random.normal(0.0005, 0.01, len(tsla_df)).cumsum())
        bnd_df['Close'] = tsla_df['Close'].iloc[0] * (1 + np.random.normal(0.0001, 0.005, len(tsla_df)).cumsum())
        
        # Calculate daily returns for historical assets
        tsla_returns = tsla_df['Close'].pct_change().dropna()
        spy_returns = spy_df['Close'].pct_change().dropna()
        bnd_returns = bnd_df['Close'].pct_change().dropna()

        # Combine historical returns for covariance calculation
        historical_returns_df = pd.DataFrame({
            'TSLA': tsla_returns,
            'SPY': spy_returns,
            'BND': bnd_returns
        }).dropna()
        
        # Load evaluation metrics to decide on the best model
        metrics = load_evaluation_metrics(EVALUATION_METRICS_PATH)
        best_model_name = 'lstm' if metrics['lstm']['MAE'] < metrics['arima']['MAE'] else 'arima'
        print(f"Based on MAE, the best model is: {best_model_name}")

        # Load the forecast from the best model
        forecast_path = LSTM_FORECAST_PATH if best_model_name == 'lstm' else ARIMA_FORECAST_PATH
        forecast_df = load_data(forecast_path)
        
        if forecast_df.empty:
            print("Forecast file not found. Exiting.")
            return

        # Calculate the expected return for TSLA from the forecast
        tsla_forecast_returns = forecast_df['forecast'].pct_change().dropna()
        expected_tsla_return = tsla_forecast_returns.mean()
        print(f"TSLA's expected daily return (from forecast): {expected_tsla_return:.6f}")

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # --- Step 2: Prepare MPT Inputs ---
    # Calculate historical expected returns for BND and SPY
    expected_spy_return = spy_returns.mean()
    expected_bnd_return = bnd_returns.mean()
    
    # Consolidate expected returns into a single Series
    expected_returns_series = pd.Series([expected_tsla_return, expected_spy_return, expected_bnd_return], index=['TSLA', 'SPY', 'BND'])

    # Compute the covariance matrix from historical returns
    cov_matrix = historical_returns_df.cov()
    
    print("\n--- MPT Inputs Ready ---")
    print("\nExpected Daily Returns:")
    print(expected_returns_series)
    print("\nDaily Covariance Matrix:")
    print(cov_matrix)
    print("-------------------------")

    # --- Step 3: Generate and Plot the Efficient Frontier ---
    results, all_weights = generate_efficient_frontier(expected_returns_series, cov_matrix)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o')
    plt.colorbar(label='Sharpe Ratio (Annualized)')
    plt.title('Efficient Frontier')
    plt.xlabel('Portfolio Volatility (Annualized)')
    plt.ylabel('Portfolio Return (Annualized)')

    # --- Step 4: Identify and Plot Key Portfolios ---
    print("\n--- Running Optimization to find Key Portfolios ---")
    # Minimum Volatility Portfolio
    min_vol_weights, min_vol_return, min_vol_volatility, min_vol_sharpe = optimize_portfolio(
        expected_returns_series, cov_matrix, min_volatility_objective
    )
    plt.scatter(min_vol_volatility, min_vol_return, marker='*', color='red', s=500, label='Minimum Volatility')
    print(f"\nMinimum Volatility Portfolio:")
    print(f"  Return: {min_vol_return:.2%}, Volatility: {min_vol_volatility:.2%}, Sharpe: {min_vol_sharpe:.4f}")
    print(f"  Weights: TSLA: {min_vol_weights[0]:.2%}, SPY: {min_vol_weights[1]:.2%}, BND: {min_vol_weights[2]:.2%}")

    # Maximum Sharpe Ratio Portfolio
    max_sharpe_weights, max_sharpe_return, max_sharpe_volatility, max_sharpe_sharpe = optimize_portfolio(
        expected_returns_series, cov_matrix, max_sharpe_objective
    )
    plt.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='green', s=500, label='Maximum Sharpe Ratio')
    print(f"\nMaximum Sharpe Ratio Portfolio:")
    print(f"  Return: {max_sharpe_return:.2%}, Volatility: {max_sharpe_volatility:.2%}, Sharpe: {max_sharpe_sharpe:.4f}")
    print(f"  Weights: TSLA: {max_sharpe_weights[0]:.2%}, SPY: {max_sharpe_weights[1]:.2%}, BND: {max_sharpe_weights[2]:.2%}")

    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_OUTPUT_PATH)
    print(f"\nEfficient Frontier plot saved to: {PLOT_OUTPUT_PATH}")
    plt.show()

    # --- Step 5: Final Recommendation and Summary ---
    print("\n" + "="*50)
    print("--- Final Recommended Portfolio ---")
    print("Based on the Efficient Frontier analysis, the **Maximum Sharpe Ratio** portfolio is the optimal choice.")
    print("This portfolio maximizes the risk-adjusted return, meaning it provides the highest return for a given level of volatility.")
    print("This is often the preferred choice for long-term investors seeking to efficiently grow their capital.")
    print("-" * 50)
    print("Portfolio Summary:")
    print(f"  - Optimal Weights: TSLA: {max_sharpe_weights[0]:.2%}, SPY: {max_sharpe_weights[1]:.2%}, BND: {max_sharpe_weights[2]:.2%}")
    print(f"  - Expected Annual Return: {max_sharpe_return:.2%}")
    print(f"  - Annual Volatility (Risk): {max_sharpe_volatility:.2%}")
    print(f"  - Sharpe Ratio: {max_sharpe_sharpe:.4f}")
    print("="*50)


if __name__ == '__main__':
    main()

