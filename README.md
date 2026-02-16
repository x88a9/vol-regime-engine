# Volatility Regime Engine

A state-based volatility regime detection and adaptive exposure model.

## Objective
Detect volatility regimes using rolling realized volatility and dynamically adjust portfolio exposure.

## Model Structure

1. Data Layer
2. Feature Engineering (Rolling Vol, Z-Score)
3. Regime Classification
4. Strategy Mapping
5. Backtesting Engine
6. Performance Metrics

## Mathematical Framework

Rolling Volatility:
σ_t = sqrt(252) * std(r_{t-n:t})

Z-Score:
Z_t = (σ_t - μ(σ)) / std(σ)

Regime Rules:
Z < -0.5 → Low Vol  
-0.5 ≤ Z ≤ 0.5 → Neutral  
Z > 0.5 → High Vol  

## Roadmap

- [ ] Data pipeline
- [ ] Regime classification
- [ ] Strategy backtest
- [ ] Metrics
- [ ] Local web dashboard
