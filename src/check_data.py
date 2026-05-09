import pandas as pd

prices = pd.read_parquet("data/sp500_prices.parquet")

print(f"Shape: {prices.shape}")
print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

# Get list of unique tickers
tickers = prices.columns.get_level_values(0).unique()
print(f"\nNumber of tickers: {len(tickers)}")
print(f"First 10 tickers: {list(tickers[:10])}")

# Check a sample ticker
print(f"\nSample data for {tickers[0]}:")
print(prices[tickers[0]].head())