import pandas as pd
import numpy as np


def compute_vol_target_exposure(
    realized_vol: pd.Series,
    target_vol: float = 0.5,
    min_exposure: float = 0.0,
    max_exposure: float = 2.0
) -> pd.Series:
    """
    Compute volatility targeting exposure.

    exposure_t = target_vol / realized_vol_t
    """

    exposure = (target_vol / realized_vol).astype(float)
    exposure = pd.Series(exposure, index=realized_vol.index)

    exposure = exposure.clip(min_exposure, max_exposure)
    exposure.name = "vol_target_exposure"

    return exposure