# src/metrics.py

import numpy as np
import pandas as pd


def compute_cagr(equity: pd.Series, trading_days: int = 252) -> float:
    total_return = equity.iloc[-1]
    n_periods = len(equity)
    years = n_periods / trading_days

    return total_return ** (1 / years) - 1


def compute_annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
    return returns.std() * np.sqrt(trading_days)


def compute_sharpe(returns: pd.Series, trading_days: int = 252) -> float:
    mean_return = returns.mean()
    std_return = returns.std()

    return (mean_return / std_return) * np.sqrt(trading_days)


def compute_max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    return drawdown.min()


def compute_calmar(cagr: float, max_dd: float) -> float:
    return cagr / abs(max_dd)
