import pandas as pd
import yfinance as yf
import time
import os

# Load the tickers
tickers_df = pd.read_csv("data/sp500_tickers.csv")
tickers = tickers_df['Symbol'].tolist()
tickers = [t.replace('.', '-') for t in tickers]  # yfinance format

print(f"Downloading 5 years of data for {len(tickers)} stocks in batches...\n")

BATCH_SIZE = 25  # small batches = fewer threads = no DNS errors
all_data = {}
failed = []

for i in range(0, len(tickers), BATCH_SIZE):
    batch = tickers[i:i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Batch {batch_num}/{total_batches}: {batch[0]}...{batch[-1]}")

    try:
        data = yf.download(
            batch,
            start="2020-01-01",
            end="2025-01-01",
            group_by='ticker',
            auto_adjust=True,
            threads=False,        # turn OFF threads — this is the key fix
            progress=False
        )
        # Store each ticker's data
        for ticker in batch:
            try:
                ticker_data = data[ticker].dropna(how='all')
                if not ticker_data.empty:
                    all_data[ticker] = ticker_data
                else:
                    failed.append(ticker)
            except (KeyError, Exception):
                failed.append(ticker)
    except Exception as e:
        print(f"  Batch failed: {e}")
        failed.extend(batch)

    time.sleep(1)  # be polite to Yahoo's servers

# Retry failed tickers one at a time (slower but more reliable)
print(f"\nRetrying {len(failed)} failed tickers individually...")
still_failed = []
for ticker in failed:
    try:
        data = yf.download(ticker, start="2020-01-01", end="2025-01-01",
                           auto_adjust=True, threads=False, progress=False)
        if not data.empty:
            all_data[ticker] = data
        else:
            still_failed.append(ticker)
    except Exception:
        still_failed.append(ticker)
    time.sleep(0.3)

# Combine into one big DataFrame with multi-level columns
print(f"\nCombining {len(all_data)} successful tickers...")
combined = pd.concat(all_data, axis=1)

os.makedirs("data", exist_ok=True)
combined.to_parquet("data/sp500_prices.parquet")

print(f"\n✅ Done!")
print(f"   Successful: {len(all_data)}")
print(f"   Failed:     {len(still_failed)}")
print(f"   Saved to:   data/sp500_prices.parquet")
print(f"   Shape:      {combined.shape}")

if still_failed:
    print(f"\nFailed tickers (will exclude from analysis): {still_failed[:20]}{'...' if len(still_failed) > 20 else ''}")