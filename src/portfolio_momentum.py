import pandas as pd

from src.momentum import compute_momentum_signal
from src.volatility import rolling_annualized_vol
from src.vol_targeting import compute_vol_target_exposure
from src.portfolio import equal_weight_portfolio


def run_portfolio_momentum(
    price_dict,
    returns_dict,
    vol_window=30,
    target_vol=0.3,
    lookback=252
):

    strategy_returns = {}

    for asset in price_dict.keys():

        prices = price_dict[asset]
        returns = returns_dict[asset]

        # Momentum signal
        signal = compute_momentum_signal(prices, lookback=lookback)
        signal = signal.shift(1).dropna()

        # Vol Target
        vol = rolling_annualized_vol(returns, window=vol_window)
        exposure = compute_vol_target_exposure(vol, target_vol=target_vol)
        exposure = exposure.shift(1).dropna()

        combined_exposure = signal * exposure
        aligned_returns = returns.align(combined_exposure, join="inner")[0]

        strategy_returns[asset] = aligned_returns * combined_exposure

    portfolio_returns = equal_weight_portfolio(strategy_returns)
    portfolio_equity = (1 + portfolio_returns).cumprod()

    return portfolio_returns, portfolio_equity
