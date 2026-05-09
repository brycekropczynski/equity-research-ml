import pandas as pd
import numpy as np

print("Loading features...")
features = pd.read_parquet("data/features.parquet")

# Forward 21-day (1-month) return — what we want to predict
print("Computing forward 21-day returns...")
features = features.sort_values(['ticker', 'date'])
features['forward_return_21d'] = features.groupby('ticker')['close'].transform(
    lambda x: x.pct_change(21).shift(-21)
)

# Drop rows where we can't compute forward return (last 21 days of each stock)
features = features.dropna(subset=['forward_return_21d'])

# Convert to a classification target: top 20% = "buy" (1), rest = 0
# We do this CROSS-SECTIONALLY (per date), so we're asking
# "of all stocks today, which ones are likely to outperform their peers next month?"
features['target'] = features.groupby('date')['forward_return_21d'].transform(
    lambda x: (x >= x.quantile(0.80)).astype(int)
)

print(f"\nTarget distribution:")
print(f"  Class 0 (not top quintile): {(features['target'] == 0).sum():,}")
print(f"  Class 1 (top quintile):     {(features['target'] == 1).sum():,}")
print(f"  Class 1 fraction:            {features['target'].mean():.2%}")

print(f"\nForward 21-day return statistics:")
print(features['forward_return_21d'].describe().round(4))

features.to_parquet("data/features_with_target.parquet", index=False)
print(f"\n✅ Saved to data/features_with_target.parquet")
print(f"   Final shape: {features.shape}")