import numpy as np
import pandas as pd


def compute_log_returns(price_series: pd.Series) -> pd.Series:
    """
    Compute log returns from price series.
    
    r_t = ln(P_t / P_{t-1})
    """
    returns = np.log(price_series / price_series.shift(1))
    returns = returns.dropna()
    returns.name = "log_returns"

    return returns