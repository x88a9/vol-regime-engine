import numpy as np
import pandas as pd


def rolling_annualized_vol(
    returns: pd.Series,
    window: int = 30,
    trading_days: int = 252
) -> pd.Series:
    """
    Compute rolling annualized realized volatility.

    Parameters
    ----------
    returns : pd.Series
        Log return series
    window : int
        Rolling window length
    trading_days : int
        Annualization factor (default 252)

    Returns
    -------
    pd.Series
        Annualized rolling volatility
    """
    rolling_std = returns.rolling(window=window).std()
    annualized_vol = rolling_std * np.sqrt(trading_days)

    annualized_vol = annualized_vol.dropna()
    annualized_vol.name = f"rolling_vol_{window}"

    return annualized_vol