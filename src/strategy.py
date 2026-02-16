import pandas as pd


def map_exposure(
    regime_series: pd.Series,
    exposure_map: dict | None = None
) -> pd.Series:
    """
    Map regime labels to exposure levels.
    """

    if exposure_map is None:
        exposure_map = {
            "low_vol": 1.5,
            "neutral": 1.0,
            "high_vol": 0.3
        }

    exposure = regime_series.map(exposure_map)
    exposure.name = "exposure"

    return exposure
