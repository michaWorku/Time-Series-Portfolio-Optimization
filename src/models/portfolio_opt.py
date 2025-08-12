# src/models/portfolio_opt.py
"""
Portfolio optimization helpers for Task 4.

Functions:
- load_price_series: read CSV and return price DataFrame
- tsla_forecast_to_expected_return: convert forecast series -> annualized expected return
- prepare_mpt_inputs: build mu, S (annualized)
- optimize_portfolio: runs PyPortfolioOpt to get max-sharpe & min-vol weights
- build_efficient_frontier: sample frontier points (returns/vols)
- save_results: write CSV/JSON and plots
"""

from typing import Optional, Tuple, Dict, Any
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting

# Constants
TRADING_DAYS = 252


def load_price_series(csv_path: str, symbols: Optional[list] = None) -> pd.DataFrame:
    """
    Load price CSV. CSV must contain Date column and columns named by ticker symbols
    (or 'Adj Close' per ticker).
    Returns a DataFrame of adjusted close prices, indexed by Date (datetime).
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    if symbols:
        missing = [s for s in symbols if s not in df.columns]
        if missing:
            raise KeyError(f"CSV missing columns: {missing}")
        return df[symbols].copy()
    return df.copy()


def tsla_forecast_to_expected_return(
    forecast_series: pd.Series,
    last_price: float,
    horizon_days: int = 180,
    annualize: bool = True
) -> float:
    """
    Convert a forecasted TSLA price path into a single expected annual return.
    - forecast_series: pd.Series (index = future dates) of price forecast (e.g., next 180 business days)
    - last_price: last observed price at time of forecast
    horizon_days: number of days in forecast horizon (business days expected)
    Returns either annualized expected return (default) or simple horizon return.
    """
    if len(forecast_series) == 0:
        raise ValueError("forecast_series empty")

    # Use the final forecast point as the horizon price
    final_price = float(forecast_series.iloc[-1])
    if final_price <= 0 or last_price <= 0:
        raise ValueError("Prices must be > 0")

    # simple cumulative return over horizon
    cum_return = final_price / last_price - 1.0

    if not annualize:
        return cum_return

    # convert to annualized return ((1 + r_h)^(252/h) - 1)
    annualized = (1.0 + cum_return) ** (TRADING_DAYS / max(horizon_days, 1)) - 1.0
    return float(annualized)


def prepare_mpt_inputs(
    prices: pd.DataFrame,
    tsla_expected_return: Optional[float] = None,
    use_log: bool = False
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute the expected returns vector (mu) and annualized sample covariance matrix (S).
    - prices: DataFrame with columns ['TSLA','BND','SPY'] containing adjusted close daily prices
    - tsla_expected_return: optionally override TSLA expected return (annualized)
    - use_log: if True compute log returns instead of pct_change
    Returns: mu (pd.Series), S (pd.DataFrame)
    """
    if not set(["TSLA", "BND", "SPY"]).issubset(prices.columns):
        raise KeyError("prices must contain TSLA, BND, SPY columns")

    # daily returns
    if use_log:
        daily_r = np.log(prices / prices.shift(1)).dropna()
    else:
        daily_r = prices.pct_change().dropna()

    # historical expected returns for BND & SPY (mean historical returns, annualized)
    mu_hist = expected_returns.mean_historical_return(prices[["BND", "SPY"]], frequency=TRADING_DAYS)

    # TSLA: either from tsla_expected_return (annualized) or historical
    if tsla_expected_return is None:
        mu_tsla = expected_returns.mean_historical_return(prices[["TSLA"]], frequency=TRADING_DAYS).iloc[0]
    else:
        mu_tsla = float(tsla_expected_return)

    # combine into mu series
    mu = pd.Series({"TSLA": mu_tsla, "BND": mu_hist.loc["BND"], "SPY": mu_hist.loc["SPY"]})

    # covariance matrix (sample_cov returns daily cov; we multiply by TRADING_DAYS to annualize)
    S_daily = risk_models.sample_cov(prices[["TSLA", "BND", "SPY"]])
    S = S_daily * TRADING_DAYS

    return mu, S


def optimize_portfolio(mu: pd.Series, S: pd.DataFrame, weight_bounds: Tuple[float, float] = (0, 1)) -> Dict[str, Any]:
    """
    Run PyPortfolioOpt optimization and return weights and performance for:
      - Maximum Sharpe Ratio (tangency)
      - Minimum Volatility
      - Equal-weight baseline
    Returns dictionary with weights and metrics.
    """
    # build EF once; will re-initialize for printing after optimization ops
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

    # Max Sharpe (tangency)
    max_sharpe_weights = ef.max_sharpe()
    max_sharpe_clean = ef.clean_weights()
    max_sharpe_perf = ef.portfolio_performance(verbose=False)

    # Need a fresh EF for min vol (since previous operations mutate internal state)
    ef2 = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    min_vol_weights = ef2.min_volatility()
    min_vol_clean = ef2.clean_weights()
    min_vol_perf = ef2.portfolio_performance(verbose=False)

    # Equal weight baseline
    equal_w = {"TSLA": 1/3, "BND": 1/3, "SPY": 1/3}
    # compute metrics for equal weight using formulas
    w_arr = np.array([equal_w["TSLA"], equal_w["BND"], equal_w["SPY"]])
    portfolio_return = float(np.dot(mu.values, w_arr))
    portfolio_vol = float(np.sqrt(np.dot(w_arr.T, np.dot(S.values, w_arr))))
    # assume risk-free 0 for Sharpe; user can adjust
    portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol != 0 else np.nan

    results = {
        "max_sharpe": {
            "weights": max_sharpe_clean,
            "performance": {
                "annual_return": max_sharpe_perf[0],
                "annual_volatility": max_sharpe_perf[1],
                "sharpe_ratio": max_sharpe_perf[2],
            }
        },
        "min_vol": {
            "weights": min_vol_clean,
            "performance": {
                "annual_return": min_vol_perf[0],
                "annual_volatility": min_vol_perf[1],
                "sharpe_ratio": min_vol_perf[2],
            }
        },
        "equal_weight": {
            "weights": equal_w,
            "performance": {
                "annual_return": portfolio_return,
                "annual_volatility": portfolio_vol,
                "sharpe_ratio": portfolio_sharpe,
            }
        }
    }
    return results


def build_efficient_frontier(mu: pd.Series, S: pd.DataFrame, points: int = 50):
    """
    Build frontier (return, volatility) samples for plotting.
    Returns (rets, vols, ef_object) where ef_object is the last EfficientFrontier (used by plotting)
    """
    ef = EfficientFrontier(mu, S)
    # sample target returns between min mu and max mu
    rets = np.linspace(mu.min(), mu.max(), points)
    vols = []
    for target in rets:
        try:
            ef_temp = EfficientFrontier(mu, S)
            w = ef_temp.efficient_return(target)
            perf = ef_temp.portfolio_performance(verbose=False)
            vols.append(perf[1])  # annualized volatility
        except Exception:
            vols.append(np.nan)

    return np.array(rets), np.array(vols)


def save_results(
    out_dir: str,
    prices: pd.DataFrame,
    mu: pd.Series,
    S: pd.DataFrame,
    optimize_results: dict,
    frontier_rets: np.ndarray,
    frontier_vols: np.ndarray
) -> None:
    """
    Save outputs:
    - weights + metrics json
    - mu and S csv
    - frontier plot and saved CSVs
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save mu and covariance
    mu.to_csv(out_path / "expected_returns_mu.csv", header=["mu"])
    S.to_csv(out_path / "covariance_S.csv")

    # Save optimize results
    with open(out_path / "portfolio_results.json", "w") as f:
        json.dump(optimize_results, f, indent=2)

    # Save frontier csv
    frontier_df = pd.DataFrame({"expected_return": frontier_rets, "volatility": frontier_vols})
    frontier_df.to_csv(out_path / "efficient_frontier.csv", index=False)

    # Plot frontier and portfolios
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier_vols, frontier_rets, label="Efficient Frontier", lw=2)

    # marker: max sharpe
    ms = optimize_results["max_sharpe"]["performance"]
    mv = optimize_results["min_vol"]["performance"]

    ax.scatter(ms["annual_volatility"], ms["annual_return"], marker="*", s=150, color="green", label="Max Sharpe")
    ax.scatter(mv["annual_volatility"], mv["annual_return"], marker="X", s=120, color="red", label="Min Volatility")

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Efficient Frontier - TSLA / BND / SPY")
    ax.legend()
    ax.grid(True)
    fig.savefig(out_path / "efficient_frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also save a human-readable CSV of the weights
    weights_df = pd.DataFrame({
        "strategy": ["max_sharpe", "min_vol", "equal_weight"],
        "TSLA": [
            optimize_results["max_sharpe"]["weights"]["TSLA"],
            optimize_results["min_vol"]["weights"]["TSLA"],
            optimize_results["equal_weight"]["weights"]["TSLA"]
        ],
        "BND": [
            optimize_results["max_sharpe"]["weights"]["BND"],
            optimize_results["min_vol"]["weights"]["BND"],
            optimize_results["equal_weight"]["weights"]["BND"]
        ],
        "SPY": [
            optimize_results["max_sharpe"]["weights"]["SPY"],
            optimize_results["min_vol"]["weights"]["SPY"],
            optimize_results["equal_weight"]["weights"]["SPY"]
        ],
        "annual_return": [
            optimize_results["max_sharpe"]["performance"]["annual_return"],
            optimize_results["min_vol"]["performance"]["annual_return"],
            optimize_results["equal_weight"]["performance"]["annual_return"]
        ],
        "annual_volatility": [
            optimize_results["max_sharpe"]["performance"]["annual_volatility"],
            optimize_results["min_vol"]["performance"]["annual_volatility"],
            optimize_results["equal_weight"]["performance"]["annual_volatility"]
        ],
        "sharpe_ratio": [
            optimize_results["max_sharpe"]["performance"]["sharpe_ratio"],
            optimize_results["min_vol"]["performance"]["sharpe_ratio"],
            optimize_results["equal_weight"]["performance"]["sharpe_ratio"]
        ]
    })
    weights_df.to_csv(out_path / "portfolio_weights_comparison.csv", index=False)
