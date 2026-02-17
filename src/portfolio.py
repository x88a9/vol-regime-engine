import pandas as pd


def equal_weight_portfolio(returns_dict: dict) -> pd.Series:
    """
    Combine asset return series via equal weighting.
    """

    aligned = pd.concat(returns_dict.values(), axis=1).dropna()
    portfolio_returns = aligned.mean(axis=1)

    portfolio_returns.name = "equal_weight_portfolio"

    return portfolio_returns
