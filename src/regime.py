import pandas as pd


def compute_z_score(vol_series: pd.Series) -> pd.Series:
    """
    Compute z-score of volatility series.
    """
    mean_vol = vol_series.mean()
    std_vol = vol_series.std()

    z = (vol_series - mean_vol) / std_vol
    z.name = "vol_z_score"

    return z


def classify_regime(z_series: pd.Series, threshold: float = 0.5) -> pd.Series:
    """
    Classify volatility regimes based on z-score threshold.
    
    Regimes:
    z < -threshold  → low_vol
    |z| <= threshold → neutral
    z > threshold   → high_vol
    """
    regime = pd.Series(index=z_series.index, dtype="object")

    regime[z_series < -threshold] = "low_vol"
    regime[(z_series >= -threshold) & (z_series <= threshold)] = "neutral"
    regime[z_series > threshold] = "high_vol"

    regime.name = "regime"

    return regime
