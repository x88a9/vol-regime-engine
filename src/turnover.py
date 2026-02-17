import pandas as pd


def compute_turnover(exposure_series: pd.Series) -> pd.Series:
    """
    Absolute change in exposure per period.
    """
    turnover = exposure_series.diff().abs()
    turnover = turnover.fillna(0)
    return turnover


def annualized_turnover(turnover_series: pd.Series, trading_days: int = 252) -> float:
    """
    Annualized turnover estimate.
    """
    return turnover_series.mean() * trading_days
