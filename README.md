# Alpha-Risk Portfolio Framework

A modular quantitative research framework combining:

- Time-Series Momentum
- Volatility Targeting
- Equal-Weight Multi-Asset Allocation
- Parameter Robustness Analysis
- Transaction Cost Simulation
- Regime Stability Diagnostics

Built as a one-week structured quant research project.
 

## 1. Strategy Architecture

Final exposure is defined as:

Signal × Volatility Target × Allocation

Where:

- **Signal** = Time-series momentum (long / flat)
- **Vol Target** = Dynamic leverage scaling to target annualized volatility
- **Allocation** = Equal-weight across assets (BTC, SPY, GLD)
 

## 2. Implemented Strategies

- Buy & Hold
- Vol Target (Rolling & EWMA)
- Momentum (Long/Flat)
- Momentum × Vol Target
- Multi-Asset Portfolio Momentum × Vol Target
 

## 3. Core Research Questions

1. Does volatility targeting improve risk-adjusted returns?
2. Is performance parameter-robust?
3. Does the model survive different market regimes?
4. What is the turnover and cost sensitivity?
5. Is performance stable across subperiods?
 

## 4. Key Findings

### Multi-Asset Portfolio

Sharpe ≈ 1.09  
Max Drawdown ≈ -32%  
Calmar ≈ 0.57  

Performance remains positive across distinct market regimes.
 

## 5. Parameter Robustness

Grid search performed over:

- Vol Windows: 20 / 30 / 60
- Target Vol: 0.3 / 0.5 / 0.7

Results show plateau behavior rather than isolated performance peaks,
indicating structural robustness rather than parameter overfitting.
 

## 6. Subperiod Stability

| Period     | Sharpe | Max DD |
|------------|--------|--------|
| 2018–2020  | Negative | Moderate |
| 2020–2022  | Strong   | Stable  |
| 2022–2025  | Positive | Controlled |

Momentum underperforms in sideways markets but remains robust in trending environments.
 

## 7. Diagnostics

- Rolling 12M Sharpe
- Drawdown comparison vs Buy & Hold
- Exposure visualization
- Turnover and transaction cost modeling
- Heatmap parameter sensitivity


## 8. Dashboard

Interactive Streamlit interface:



streamlit run app/dashboard.py



## 9. Limitations

- No walk-forward validation yet
- No cross-sectional momentum
- No dynamic asset selection
- No macro regime overlay
