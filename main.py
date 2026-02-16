from src.data_loader import load_data
from src.returns import compute_log_returns

if __name__ == "__main__":
    prices = load_data("BTC-USD", "2018-01-01", "2024-01-01")
    returns = compute_log_returns(prices)

    print(prices.head())
    print(returns.head())
