import pandas as pd


def compute_momentum_signal(
    price_series: pd.Series,
    lookback: int = 252
) -> pd.Series:
    """
    Long/Flat time-series momentum signal.

    Signal = 1 if price > price_{t-lookback}
    Signal = 0 otherwise
    """

    past_price = price_series.shift(lookback)
    signal = (price_series > past_price).astype(int)
    signal.name = "momentum_signal"

    return signal
