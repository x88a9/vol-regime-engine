from src.data_loader import load_data
from src.returns import compute_log_returns
from src.volatility import rolling_annualized_vol
from src.regime import compute_z_score, classify_regime
from src.strategy import map_exposure
from src.metrics import (
    compute_cagr,
    compute_annualized_vol,
    compute_sharpe,
    compute_max_drawdown,
    compute_calmar
)
from src.vol_targeting import compute_vol_target_exposure
from src.ewma import compute_ewma_vol
from src.momentum import compute_momentum_signal
from src.grid import run_vol_grid
from src.portfolio_momentum import run_portfolio_momentum
from src.turnover import compute_turnover, annualized_turnover
import pandas as pd

total_turnover = 0

if __name__ == "__main__":

    # =========================
    # Load Data
    # =========================
    prices = load_data("BTC-USD", "2018-01-01", "2025-01-01")
    returns = compute_log_returns(prices)

    # =========================
    # Buy & Hold
    # =========================
    bh_equity = (1 + returns).cumprod()
    bh_cagr = compute_cagr(bh_equity)
    bh_vol = compute_annualized_vol(returns)
    bh_sharpe = compute_sharpe(returns)
    bh_max_dd = compute_max_drawdown(bh_equity)
    bh_calmar = compute_calmar(bh_cagr, bh_max_dd)

    # =========================
    # Regime Strategy
    # =========================
    rolling_vol = rolling_annualized_vol(returns, window=30)

    z = compute_z_score(rolling_vol)
    regime = classify_regime(z)

    regime_exposure = map_exposure(regime)
    regime_exposure = regime_exposure.shift(1).dropna()

    regime_returns = returns.align(regime_exposure, join="inner")[0] * regime_exposure
    regime_equity = (1 + regime_returns).cumprod()

    regime_cagr = compute_cagr(regime_equity)
    regime_vol = compute_annualized_vol(regime_returns)
    regime_sharpe = compute_sharpe(regime_returns)
    regime_max_dd = compute_max_drawdown(regime_equity)
    regime_calmar = compute_calmar(regime_cagr, regime_max_dd)

    # =========================
    # Vol Target (Rolling)
    # =========================
    vt_exposure = compute_vol_target_exposure(rolling_vol, target_vol=0.5)
    vt_exposure = vt_exposure.shift(1).dropna()

    vt_returns = returns.align(vt_exposure, join="inner")[0] * vt_exposure
    vt_equity = (1 + vt_returns).cumprod()

    vt_cagr = compute_cagr(vt_equity)
    vt_vol = compute_annualized_vol(vt_returns)
    vt_sharpe = compute_sharpe(vt_returns)
    vt_max_dd = compute_max_drawdown(vt_equity)
    vt_calmar = compute_calmar(vt_cagr, vt_max_dd)

    # =========================
    # Vol Target (EWMA)
    # =========================
    ewma_vol = compute_ewma_vol(returns)

    vt_exposure_ewma = compute_vol_target_exposure(ewma_vol, target_vol=0.5)
    vt_exposure_ewma = vt_exposure_ewma.shift(1).dropna()

    vt_returns_ewma = returns.align(vt_exposure_ewma, join="inner")[0] * vt_exposure_ewma
    vt_equity_ewma = (1 + vt_returns_ewma).cumprod()

    ewma_cagr = compute_cagr(vt_equity_ewma)
    ewma_vol_metric = compute_annualized_vol(vt_returns_ewma)
    ewma_sharpe = compute_sharpe(vt_returns_ewma)
    ewma_max_dd = compute_max_drawdown(vt_equity_ewma)
    ewma_calmar = compute_calmar(ewma_cagr, ewma_max_dd)

    # =========================
    # Momentum (Long / Flat)
    # =========================
    momentum_signal = compute_momentum_signal(prices, lookback=252)
    momentum_signal = momentum_signal.shift(1).dropna()

    mom_returns = returns.align(momentum_signal, join="inner")[0] * momentum_signal
    mom_equity = (1 + mom_returns).cumprod()

    mom_cagr = compute_cagr(mom_equity)
    mom_vol = compute_annualized_vol(mom_returns)
    mom_sharpe = compute_sharpe(mom_returns)
    mom_max_dd = compute_max_drawdown(mom_equity)
    mom_calmar = compute_calmar(mom_cagr, mom_max_dd)

    # =========================
    # Momentum × Vol Target
    # =========================
    combined_exposure = momentum_signal * vt_exposure
    combined_exposure = combined_exposure.dropna()

    combined_returns = returns.align(combined_exposure, join="inner")[0] * combined_exposure
    combined_equity = (1 + combined_returns).cumprod()

    comb_cagr = compute_cagr(combined_equity)
    comb_vol = compute_annualized_vol(combined_returns)
    comb_sharpe = compute_sharpe(combined_returns)
    comb_max_dd = compute_max_drawdown(combined_equity)
    comb_calmar = compute_calmar(comb_cagr, comb_max_dd)

    # =========================
    # PRINT RESULTS
    # =========================
    def print_block(name, cagr, vol, sharpe, max_dd, calmar):
        print("\n==============================")
        print(name)
        print("==============================")
        print("CAGR:", round(cagr, 3))
        print("Vol:", round(vol, 3))
        print("Sharpe:", round(sharpe, 3))
        print("Max DD:", round(max_dd, 3))
        print("Calmar:", round(calmar, 3))

    print_block("Buy & Hold", bh_cagr, bh_vol, bh_sharpe, bh_max_dd, bh_calmar)
    print_block("Regime Strategy", regime_cagr, regime_vol, regime_sharpe, regime_max_dd, regime_calmar)
    print_block("Vol Target (Rolling)", vt_cagr, vt_vol, vt_sharpe, vt_max_dd, vt_calmar)
    print_block("Vol Target (EWMA)", ewma_cagr, ewma_vol_metric, ewma_sharpe, ewma_max_dd, ewma_calmar)
    print_block("Momentum (Long/Flat)", mom_cagr, mom_vol, mom_sharpe, mom_max_dd, mom_calmar)
    print_block("Momentum × Vol Target", comb_cagr, comb_vol, comb_sharpe, comb_max_dd, comb_calmar)


    assets = {
        "BTC": compute_log_returns(load_data("BTC-USD", "2018-01-01", "2026-01-01")),
        "SPY": compute_log_returns(load_data("SPY", "2018-01-01", "2026-01-01")),
        "GLD": compute_log_returns(load_data("GLD", "2018-01-01", "2026-01-01")),
    }

    grid_results = run_vol_grid(
        returns_dict=assets,
        vol_windows=[20, 30, 60],
        target_vols=[0.3, 0.5, 0.7]
    )

    print("\nGRID RESULTS:")
    print(grid_results.sort_values("sharpe", ascending=False))

        
    price_dict = {
        "BTC": load_data("BTC-USD", "2018-01-01", "2026-01-01"),
        "SPY": load_data("SPY", "2018-01-01", "2026-01-01"),
        "GLD": load_data("GLD", "2018-01-01", "2026-01-01"),
    }

    returns_dict = {
        asset: compute_log_returns(price_dict[asset])
        for asset in price_dict
    }

    pm_returns, pm_equity, pm_asset_returns, pm_exposures = run_portfolio_momentum(
        price_dict,
        returns_dict,
        vol_window=30,
        target_vol=0.3,
        lookback=252
    )

    pm_cagr = compute_cagr(pm_equity)
    pm_vol = compute_annualized_vol(pm_returns)
    pm_sharpe = compute_sharpe(pm_returns)
    pm_max_dd = compute_max_drawdown(pm_equity)
    pm_calmar = compute_calmar(pm_cagr, pm_max_dd)

    print_block("Portfolio Momentum × Vol Target", pm_cagr, pm_vol, pm_sharpe, pm_max_dd, pm_calmar)

    for asset, exposure in pm_exposures.items():
        turnover_series = compute_turnover(exposure)
        total_turnover += annualized_turnover(turnover_series)

    avg_turnover = total_turnover / len(pm_exposures)

    print("\n==============================")
    print("Turnover Analysis")
    print("==============================")
    print("Average Annual Turnover:", round(avg_turnover, 2))

    cost_rate = 0.001 

    cost_series = pd.Series(0, index=pm_returns.index)

    for exposure in pm_exposures.values():
        turnover = compute_turnover(exposure)
        turnover = turnover.reindex(pm_returns.index).fillna(0)
        cost_series += turnover * cost_rate

    cost_series = cost_series / len(pm_exposures)

    net_returns = pm_returns - cost_series
    net_equity = (1 + net_returns).cumprod()

    net_cagr = compute_cagr(net_equity)
    net_sharpe = compute_sharpe(net_returns)
    net_max_dd = compute_max_drawdown(net_equity)

    print("\n==============================")
    print("After Costs (10bps)")
    print("==============================")
    print("CAGR:", round(net_cagr, 3))
    print("Sharpe:", round(net_sharpe, 3))
    print("Max DD:", round(net_max_dd, 3))
