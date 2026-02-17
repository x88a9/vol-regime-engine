import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.data_loader import load_data
from src.returns import compute_log_returns
from src.volatility import rolling_annualized_vol
from src.vol_targeting import compute_vol_target_exposure
from src.momentum import compute_momentum_signal
from src.metrics import compute_cagr, compute_sharpe, compute_max_drawdown
from src.grid import run_vol_grid
from src.portfolio_momentum import run_portfolio_momentum


st.set_page_config(layout="wide")
st.title("Alpha-Risk Portfolio Dashboard")

# =============================
# Sidebar
# =============================

asset = st.sidebar.selectbox(
    "Asset",
    ["BTC-USD", "SPY", "GLD", "Portfolio"]
)

lookback = st.sidebar.slider("Momentum Lookback", 63, 365, 252)
vol_window = st.sidebar.slider("Vol Window", 10, 60, 30)
target_vol = st.sidebar.slider("Target Vol", 0.1, 0.7, 0.3, step=0.05)

# =============================
# Strategy Logic
# =============================

if asset != "Portfolio":

    prices = load_data(asset, "2018-01-01", "2025-01-01")
    returns = compute_log_returns(prices)

    signal = compute_momentum_signal(prices, lookback=lookback).shift(1)
    vol = rolling_annualized_vol(returns, window=vol_window)
    exposure = compute_vol_target_exposure(vol, target_vol=target_vol).shift(1)

    combined_exposure = signal * exposure
    aligned_returns = returns.align(combined_exposure, join="inner")[0]

    strategy_returns = aligned_returns * combined_exposure
    equity = (1 + strategy_returns).cumprod()

    bh_equity = (1 + returns).cumprod()

else:

    price_dict = {
        "BTC": load_data("BTC-USD", "2018-01-01", "2025-01-01"),
        "SPY": load_data("SPY", "2018-01-01", "2025-01-01"),
        "GLD": load_data("GLD", "2018-01-01", "2025-01-01"),
    }

    returns_dict = {
        a: compute_log_returns(price_dict[a])
        for a in price_dict
    }

    strategy_returns, equity, _, exposures = run_portfolio_momentum(
        price_dict,
        returns_dict,
        vol_window=vol_window,
        target_vol=target_vol,
        lookback=lookback
    )

    # Portfolio Buy & Hold
    bh_returns = pd.concat(returns_dict.values(), axis=1).mean(axis=1)
    bh_equity = (1 + bh_returns).cumprod()

    combined_exposure = None  # not single series

# =============================
# Metrics
# =============================

cagr = compute_cagr(equity)
sharpe = compute_sharpe(strategy_returns)
max_dd = compute_max_drawdown(equity)

col1, col2, col3 = st.columns(3)
col1.metric("CAGR", f"{cagr:.2%}")
col2.metric("Sharpe", f"{sharpe:.2f}")
col3.metric("Max Drawdown", f"{max_dd:.2%}")

# =============================
# Equity Plot
# =============================

fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Strategy"))
fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity.values,
                         name="Buy & Hold", line=dict(dash="dot")))

fig.update_layout(title="Equity Curve", height=500)
st.plotly_chart(fig, width='stretch')

# =============================
# Drawdown
# =============================

st.subheader("Drawdown")

drawdown = equity / equity.cummax() - 1
bh_drawdown = bh_equity / bh_equity.cummax() - 1

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Strategy"))
fig_dd.add_trace(go.Scatter(x=bh_drawdown.index, y=bh_drawdown.values,
                            name="Buy & Hold", line=dict(dash="dot")))

fig_dd.update_layout(height=400)
st.plotly_chart(fig_dd, width='stretch')

# =============================
# Exposure (single asset only)
# =============================

if asset != "Portfolio":

    st.subheader("Exposure")

    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(
        x=combined_exposure.index,
        y=combined_exposure.values,
        name="Exposure"
    ))

    fig_exp.update_layout(height=300)
    st.plotly_chart(fig_exp, width='stretch')

# =============================
# Rolling Sharpe
# =============================

st.subheader("Rolling 12M Sharpe")
rolling_window = 252

rolling_mean = strategy_returns.rolling(
    window=rolling_window,
    min_periods=200
).mean()

rolling_std = strategy_returns.rolling(
    window=rolling_window,
    min_periods=200
).std()

rolling_sharpe = (rolling_mean / rolling_std) * (252 ** 0.5)


fig_rs = go.Figure()
fig_rs.add_trace(go.Scatter(
    x=rolling_sharpe.index,
    y=rolling_sharpe.values,
    name="Rolling Sharpe"
))

fig_rs.update_layout(height=400)
st.plotly_chart(fig_rs, width='stretch')

st.subheader("Subperiod Performance")

def compute_subperiod_metrics(returns, equity, start, end):
    mask = (returns.index >= start) & (returns.index < end)

    sub_returns = returns.loc[mask]
    sub_equity = equity.loc[mask]

    if len(sub_returns) < 50:
        return None

    return {
        "CAGR": compute_cagr(sub_equity),
        "Sharpe": compute_sharpe(sub_returns),
        "MaxDD": compute_max_drawdown(sub_equity)
    }


periods = [
    ("2018-01-01", "2020-01-01"),
    ("2020-01-01", "2022-01-01"),
    ("2022-01-01", "2025-01-01"),
]

rows = []

for start, end in periods:
    metrics = compute_subperiod_metrics(
        strategy_returns,
        equity,
        pd.to_datetime(start),
        pd.to_datetime(end)
    )

    if metrics:
        rows.append({
            "Period": f"{start[:4]}–{end[:4]}",
            "CAGR": f"{metrics['CAGR']:.2%}",
            "Sharpe": f"{metrics['Sharpe']:.2f}",
            "MaxDD": f"{metrics['MaxDD']:.2%}"
        })

df_sub = pd.DataFrame(rows)

st.dataframe(df_sub, width='stretch')


# =============================
# Heatmap (always portfolio grid)
# =============================

st.subheader("Vol Target Grid Sharpe Heatmap")

assets = {
    "BTC": compute_log_returns(load_data("BTC-USD", "2018-01-01", "2025-01-01")),
    "SPY": compute_log_returns(load_data("SPY", "2018-01-01", "2025-01-01")),
    "GLD": compute_log_returns(load_data("GLD", "2018-01-01", "2025-01-01")),
}

grid_results = run_vol_grid(
    returns_dict=assets,
    vol_windows=[20, 30, 60],
    target_vols=[0.3, 0.5, 0.7]
)

pivot = grid_results.pivot(
    index="target_vol",
    columns="window",
    values="sharpe"
)

fig_heatmap = px.imshow(
    pivot,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Viridis"
)

fig_heatmap.update_layout(
    xaxis_title="Vol Window",
    yaxis_title="Target Vol",
    height=400
)

st.plotly_chart(fig_heatmap, width='stretch')
