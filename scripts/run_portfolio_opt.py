# src/run_portfolio_opt.py

import sys
from pathlib import Path
import argparse
from pathlib import Path
import pandas as pd
import re

# Add the project root to the path for module imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.portfolio_opt import (
    tsla_forecast_to_expected_return,
    prepare_mpt_inputs,
    optimize_portfolio,
    build_efficient_frontier,
    save_results
)


def combine_price_data(base_dir: str, tickers=("TSLA", "BND", "SPY")) -> pd.DataFrame:
    """
    Look for CSVs in processed/ or raw/ containing each ticker.
    Merge into single DataFrame (Date index, price columns).
    Works even if date column name differs.
    """
    base_path = Path(base_dir)
    search_dirs = [base_path / "processed", base_path / "raw"]

    combined_df = None

    for ticker in tickers:
        file_found = None
        for directory in search_dirs:
            if directory.exists():
                matches = list(directory.glob(f"*{ticker}*.csv"))
                if matches:
                    file_found = matches[0]
                    break
        if not file_found:
            raise FileNotFoundError(f"No CSV found for {ticker} in processed/ or raw/ under {base_dir}")

        df = pd.read_csv(file_found)

        # Auto-detect date column
        date_col = None
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
                break
        if date_col is None:
            raise ValueError(f"No date column found in {file_found}")

        # Detect price column
        price_col = None
        for col in df.columns:
            if col.lower() in ["adj close", "adj_close"]:
                price_col = col
                break
        if price_col is None:
            for col in df.columns:
                if col.lower() == "close":
                    price_col = col
                    break
        if price_col is None:
            raise ValueError(f"No price column found in {file_found}")

        df[date_col] = pd.to_datetime(df[date_col])
        df = df[[date_col, price_col]].rename(columns={price_col: ticker})
        df.set_index(date_col, inplace=True)

        combined_df = df if combined_df is None else combined_df.join(df, how="outer")

    return combined_df.dropna()


def main():
    parser = argparse.ArgumentParser(description="Portfolio optimization pipeline for TSLA/BND/SPY")
    parser.add_argument("--raw_data_dir", default="data/raw", help="Directory containing raw CSVs for TSLA, BND, SPY")
    parser.add_argument("--tsla_forecast_csv", default=None,
                        help="Optional TSLA forecast CSV (Date, forecast_price). If omitted, TSLA historical return used.")
    parser.add_argument("--forecast_horizon_days", type=int, default=180,
                        help="Horizon (business days) to convert forecast to annual return")
    parser.add_argument("--out_dir", default="outputs/portfolio_opt", help="Directory to save results")
    parser.add_argument("--use_log_returns", action="store_true", help="Use log returns for covariance (optional)")
    args = parser.parse_args()

    # Step 1: Load & combine raw CSV data
    prices = combine_price_data(args.raw_data_dir, tickers=("TSLA", "BND", "SPY"))
    last_price = float(prices["TSLA"].iloc[-1])

    # Step 2: Handle TSLA forecast -> expected return
    tsla_expected_return = None
    if args.tsla_forecast_csv:
        fc = pd.read_csv(args.tsla_forecast_csv, parse_dates=["Date"]).set_index("Date").squeeze()
        if isinstance(fc, pd.DataFrame):
            fc = fc.select_dtypes(include=["number"]).iloc[:, 0]
        tsla_expected_return = tsla_forecast_to_expected_return(
            forecast_series=fc,
            last_price=last_price,
            horizon_days=args.forecast_horizon_days,
            annualize=True
        )
        print(f"Derived TSLA annualized expected return from forecast: {tsla_expected_return:.2%}")
    else:
        print("No TSLA forecast provided; using historical mean for TSLA expected return.")

    # Step 3: Prepare MPT inputs
    mu, S = prepare_mpt_inputs(prices, tsla_expected_return=tsla_expected_return, use_log=args.use_log_returns)

    # Step 4: Optimize
    results = optimize_portfolio(mu, S)

    # Step 5: Efficient Frontier
    frontier_rets, frontier_vols = build_efficient_frontier(mu, S, points=60)

    # Step 6: Save results
    save_results(args.out_dir, prices, mu, S, results, frontier_rets, frontier_vols)

    print(f"Results saved to: {args.out_dir}")
    print("Summary:")
    print("Max Sharpe weights:", results["max_sharpe"]["weights"])
    print("Min Vol weights:", results["min_vol"]["weights"])


if __name__ == "__main__":
    main()
