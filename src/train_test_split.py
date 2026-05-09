import pandas as pd

print("Loading features...")
df = pd.read_parquet("data/features_with_target.parquet")

# Chronological split: train on 2020-2023, test on 2024
SPLIT_DATE = "2024-01-01"

train = df[df['date'] < SPLIT_DATE].copy()
test = df[df['date'] >= SPLIT_DATE].copy()

print(f"\nTrain set:")
print(f"  Date range: {train['date'].min().date()} to {train['date'].max().date()}")
print(f"  Rows:       {len(train):,}")
print(f"  Stocks:     {train['ticker'].nunique()}")
print(f"  Class 1 %:  {train['target'].mean():.2%}")

print(f"\nTest set:")
print(f"  Date range: {test['date'].min().date()} to {test['date'].max().date()}")
print(f"  Rows:       {len(test):,}")
print(f"  Stocks:     {test['ticker'].nunique()}")
print(f"  Class 1 %:  {test['target'].mean():.2%}")

train.to_parquet("data/train.parquet", index=False)
test.to_parquet("data/test.parquet", index=False)
print("\n✅ Saved train.parquet and test.parquet")