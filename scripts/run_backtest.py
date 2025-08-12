import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_price_data(data_dir):
    prices = {}
    for ticker in ["TSLA", "SPY", "BND"]:
        csv_path = Path(data_dir) / f"{ticker}_processed.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Price file not found: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        prices[ticker] = df["Close"]
    return pd.DataFrame(prices).dropna()

def calculate_portfolio_returns(prices, weights):
    daily_returns = prices.pct_change().dropna()
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    return portfolio_returns

def performance_metrics(portfolio_returns):
    ann_return = np.mean(portfolio_returns) * 252
    ann_vol = np.std(portfolio_returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    total_return = (1 + portfolio_returns).prod() - 1
    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Total Return": total_return
    }

def main(args):
    prices = load_price_data(args.price_data_dir)
    prices = prices.loc[args.start_date:args.end_date]

    # Load weights file from Task 4
    weights_df = pd.read_csv(args.weights_csv)
    if args.strategy_name not in weights_df["strategy"].values:
        raise ValueError(f"Strategy '{args.strategy_name}' not found in {args.weights_csv}")

    strat_row = weights_df[weights_df["strategy"] == args.strategy_name].iloc[0]
    strategy_weights = {
        "TSLA": strat_row["TSLA"],
        "BND": strat_row["BND"],
        "SPY": strat_row["SPY"]
    }

    # Benchmark 60/40 SPY/BND
    benchmark_weights = {"SPY": 0.6, "BND": 0.4, "TSLA": 0.0}

    # Calculate returns
    strat_returns = calculate_portfolio_returns(prices, strategy_weights)
    bench_returns = calculate_portfolio_returns(prices, benchmark_weights)

    # Metrics
    strat_metrics = performance_metrics(strat_returns)
    bench_metrics = performance_metrics(bench_returns)
    metrics_df = pd.DataFrame([strat_metrics, bench_metrics], index=["Strategy", "Benchmark"])
    metrics_df.to_csv(Path(args.out_dir) / "backtest_metrics.csv")

    # Cumulative returns plot
    strat_cum = (1 + strat_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(strat_cum, label=f"{args.strategy_name} Portfolio")
    plt.plot(bench_cum, label="Benchmark 60/40 SPY/BND", linestyle="--")
    plt.title("Cumulative Returns: Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth")
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(args.out_dir) / "cumulative_returns.png")
    plt.close()

    print("\nBacktest Completed. Metrics and plots saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest strategy portfolio vs benchmark")
    parser.add_argument("--price_data_dir", type=str, required=True, help="Directory with processed CSV files for TSLA, SPY, BND")
    parser.add_argument("--weights_csv", type=str, required=True, help="Portfolio weights CSV from Task 4")
    parser.add_argument("--strategy_name", type=str, default="max_sharpe", help="Name of strategy in weights CSV (max_sharpe, min_vol, equal_weight)")
    parser.add_argument("--start_date", type=str, default="2024-08-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-07-31", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--out_dir", type=str, default="outputs/backtest", help="Directory to save outputs")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)
