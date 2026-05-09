import pandas as pd
import numpy as np

print("Loading price data...")
prices = pd.read_parquet("data/sp500_prices.parquet")
tickers = prices.columns.get_level_values(0).unique()
print(f"Building features for {len(tickers)} stocks...\n")

all_features = []

for i, ticker in enumerate(tickers):
    if i % 50 == 0:
        print(f"  Processing {i}/{len(tickers)}...")

    df = prices[ticker].copy()
    df = df.dropna(subset=['Close'])
    if len(df) < 250:  # need at least ~1 year of data
        continue

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    feat = pd.DataFrame(index=df.index)
    feat['ticker'] = ticker
    feat['close'] = close

    # === RETURNS ===
    feat['return_1d'] = close.pct_change(1)
    feat['return_5d'] = close.pct_change(5)
    feat['return_21d'] = close.pct_change(21)    # ~1 month
    feat['return_63d'] = close.pct_change(63)    # ~3 months
    feat['return_252d'] = close.pct_change(252)  # ~1 year (momentum)

    # === MOVING AVERAGES ===
    feat['sma_20'] = close.rolling(20).mean()
    feat['sma_50'] = close.rolling(50).mean()
    feat['sma_200'] = close.rolling(200).mean()

    # Price relative to moving averages (% above/below)
    feat['price_vs_sma20'] = (close / feat['sma_20']) - 1
    feat['price_vs_sma50'] = (close / feat['sma_50']) - 1
    feat['price_vs_sma200'] = (close / feat['sma_200']) - 1

    # === VOLATILITY ===
    daily_ret = close.pct_change()
    feat['volatility_21d'] = daily_ret.rolling(21).std() * np.sqrt(252)
    feat['volatility_63d'] = daily_ret.rolling(63).std() * np.sqrt(252)

    # === RSI (Relative Strength Index, 14-day) ===
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    feat['rsi_14'] = 100 - (100 / (1 + rs))

    # === BOLLINGER BAND POSITION ===
    # Where is the price within the 20-day Bollinger Band? 0 = lower band, 1 = upper band
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

    # === VOLUME ===
    feat['volume_ratio_20d'] = volume / volume.rolling(20).mean()

    all_features.append(feat)

# Combine all
print(f"\nCombining features for {len(all_features)} stocks...")
features_df = pd.concat(all_features)
features_df = features_df.reset_index().rename(columns={'Date': 'date'})

# Drop rows where features couldn't be computed (early dates)
features_df = features_df.dropna()

print(f"Final shape: {features_df.shape}")
print(f"Saving to data/features.parquet...")
features_df.to_parquet("data/features.parquet", index=False)

print("\n✅ Done!")
print(f"   Total feature rows: {len(features_df):,}")
print(f"   Features per row:   {features_df.shape[1] - 3}")  # minus date, ticker, close
print(f"   Date range:         {features_df['date'].min().date()} to {features_df['date'].max().date()}")