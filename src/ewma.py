import numpy as np
import pandas as pd


def compute_ewma_vol(
    returns: pd.Series,
    lambda_: float = 0.94,
    trading_days: int = 252
) -> pd.Series:
    """
    Compute EWMA annualized volatility (RiskMetrics style).
    """

    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[0] ** 2

    for t in range(1, len(returns)):
        var.iloc[t] = (
            lambda_ * var.iloc[t-1]
            + (1 - lambda_) * returns.iloc[t] ** 2
        )

    vol = np.sqrt(var) * np.sqrt(trading_days)
    vol.name = "ewma_vol"

    return vol
