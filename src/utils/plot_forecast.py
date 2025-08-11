"""
This script provides a reusable helper function to visualize time series forecasts.
Plot forecast vs historical with confidence intervals and diagnostics.
Saves PNG file to outputs/.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def plot_history_and_forecast(
    history: pd.Series,
    forecast: pd.Series,
    ci: Optional[pd.DataFrame] = None,
    title: str = "Forecast vs History",
    xlabel: str = "Date",
    ylabel: str = "Price",
    savepath: str = "outputs/forecast.png",
    forecast_start_date: Optional[pd.Timestamp] = None,
):
    """
    Plots the historical time series data and a new forecast.
    
    Args:
        history (pd.Series): Series of historical prices indexed by datetime.
        forecast (pd.Series): Series of future values. The index can be integers or datetimes.
        ci (Optional[pd.DataFrame]): DataFrame with 'lower' and 'upper' bounds for the confidence interval,
                                     indexed the same as the forecast.
        title (str): The title for the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        savepath (str): The file path to save the plot.
        forecast_start_date (Optional[pd.Timestamp]): If given, maps integer-indexed forecasts to dates.
    """
    plt.figure(figsize=(12, 6))

    # Prepare index for forecast (map integers to dates if possible)
    if not isinstance(forecast.index, pd.DatetimeIndex) and forecast_start_date is not None:
        forecast_index = pd.date_range(start=forecast_start_date, periods=len(forecast), freq="B")  # business days
    else:
        forecast_index = forecast.index

    # Plot history
    plt.plot(history.index, history.values, label="Historical", color="tab:blue")

    # Plot forecast
    plt.plot(forecast_index, forecast.values, label="Forecast", color="tab:red", linestyle="--")

    # Plot confidence intervals if provided
    if ci is not None:
        ci_index = forecast_index if isinstance(ci.index, pd.DatetimeIndex) else ci.index
        if not isinstance(ci.index, pd.DatetimeIndex) and forecast_start_date is not None:
            ci_index = pd.date_range(start=forecast_start_date, periods=len(ci), freq="B")

        plt.fill_between(
            ci_index,
            ci['lower'],
            ci['upper'],
            color="tab:red",
            alpha=0.2,
            label="Confidence Interval"
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Forecast plot saved to: {savepath}")
    plt.show()


def plot_diagnostics(
    history: pd.Series,
    forecast: pd.Series,
    savepath: str = "outputs/forecast_diagnostics.png"
):
    """
    Plots simple diagnostics: distribution of residuals from an overlap region.

    Args:
        history (pd.Series): The historical data.
        forecast (pd.Series): The forecast data.
        savepath (str): The file path to save the plot.
    """
    # If forecast index intersects history index, compute residuals for overlap
    common_index = history.index.intersection(forecast.index)
    
    plt.figure(figsize=(8, 6))

    if len(common_index) > 0:
        residuals = history.loc[common_index].values - forecast.loc[common_index].values
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.title("Residuals distribution (Overlap region)")
        plt.xlabel("Residuals (Actual - Forecast)")
        plt.ylabel("Frequency")
    else:
        plt.text(0.5, 0.5, "No overlapping data to compute residuals.", ha='center', va='center')
        plt.title("Residuals distribution")
        plt.axis('off') # Hide axes for the text-only plot

    plt.tight_layout()
    plt.savefig(savepath)
    print(f"Diagnostics plot saved to: {savepath}")
    plt.show()

