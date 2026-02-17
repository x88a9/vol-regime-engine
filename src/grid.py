import itertools
import pandas as pd

from src.volatility import rolling_annualized_vol
from src.vol_targeting import compute_vol_target_exposure
from src.metrics import compute_cagr, compute_sharpe, compute_max_drawdown
from src.portfolio import equal_weight_portfolio


def run_vol_grid(returns_dict, vol_windows, target_vols):

    results = []

    for window, target in itertools.product(vol_windows, target_vols):

        strategy_returns = {}

        for asset, returns in returns_dict.items():

            vol = rolling_annualized_vol(returns, window=window)
            exposure = compute_vol_target_exposure(vol, target_vol=target)
            exposure = exposure.shift(1).dropna()

            aligned_returns = returns.align(exposure, join="inner")[0]
            strategy_returns[asset] = aligned_returns * exposure

        portfolio_returns = equal_weight_portfolio(strategy_returns)
        portfolio_equity = (1 + portfolio_returns).cumprod()

        cagr = compute_cagr(portfolio_equity)
        sharpe = compute_sharpe(portfolio_returns)
        max_dd = compute_max_drawdown(portfolio_equity)

        results.append({
            "window": window,
            "target_vol": target,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": max_dd
        })

    return pd.DataFrame(results)
