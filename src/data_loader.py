# src/data_loader.py

import yfinance as yf
import pandas as pd


def load_data(ticker: str, start: str, end: str) -> pd.Series:
    """
    Load daily close prices for a given ticker.
    
    Parameters
    ----------
    ticker : str
        Asset ticker (e.g., 'BTC-USD', 'SPY')
    start : str
        Start date in format 'YYYY-MM-DD'
    end : str
        End date in format 'YYYY-MM-DD'
        
    Returns
    -------
    pd.Series
        Cleaned daily close price series
    """
    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        raise ValueError("No data downloaded. Check ticker or date range.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][ticker]
    else:
        close = data["Close"]

    close = close.dropna()
    close.name = ticker

    return close
