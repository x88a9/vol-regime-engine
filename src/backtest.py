import pandas as pd
import numpy as np


def compute_equity_curve(
    returns: pd.Series,
    exposure: pd.Series
) -> pd.Series:
    """
    Compute equity curve from returns and exposure.
    """

    aligned_returns, aligned_exposure = returns.align(exposure, join="inner")

    strategy_returns = aligned_returns * aligned_exposure

    equity = (1 + strategy_returns).cumprod()
    equity.name = "equity_curve"

    return equity