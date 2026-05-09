import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading predictions...")
preds = pd.read_parquet("data/test_predictions_xgboost.parquet")

# Strategy: every month, pick the top 20 stocks by predicted probability, equal-weight them
# Hold for 21 days, then rebalance

# Use month-start dates as rebalance points
preds['date'] = pd.to_datetime(preds['date'])
preds['year_month'] = preds['date'].dt.to_period('M')

# For each month, find the first trading date and rank stocks
rebalance_dates = preds.groupby('year_month')['date'].min().values
print(f"Rebalancing on {len(rebalance_dates)} dates")

TOP_N = 20
strategy_returns = []

for rebal_date in rebalance_dates:
    # Get predictions for this date
    day_preds = preds[preds['date'] == rebal_date].copy()
    if len(day_preds) < TOP_N:
        continue

    # Pick top N by predicted probability
    top_picks = day_preds.nlargest(TOP_N, 'prediction_proba')

    # Each stock's actual forward 21-day return
    avg_return = top_picks['forward_return_21d'].mean()
    strategy_returns.append({
        'date': rebal_date,
        'strategy_return': avg_return,
        'top_picks': list(top_picks['ticker'].values)
    })

strategy_df = pd.DataFrame(strategy_returns)
strategy_df['date'] = pd.to_datetime(strategy_df['date'])
strategy_df = strategy_df.sort_values('date').reset_index(drop=True)

# Benchmark: equal-weight all stocks (proxy for "the market")
benchmark = preds.groupby('date')['forward_return_21d'].mean().reset_index()
benchmark.columns = ['date', 'benchmark_return']
benchmark = benchmark[benchmark['date'].isin(strategy_df['date'])].reset_index(drop=True)

# Merge
results = strategy_df.merge(benchmark, on='date')

# Cumulative returns (compound)
results['strategy_cumulative'] = (1 + results['strategy_return']).cumprod()
results['benchmark_cumulative'] = (1 + results['benchmark_return']).cumprod()

# === Performance metrics ===
def annualize_return(monthly_returns):
    avg = np.mean(monthly_returns)
    return (1 + avg) ** 12 - 1

def annualize_vol(monthly_returns):
    return np.std(monthly_returns) * np.sqrt(12)

def sharpe_ratio(monthly_returns, rf=0):
    return (annualize_return(monthly_returns) - rf) / annualize_vol(monthly_returns)

def max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

print("\n" + "=" * 50)
print("BACKTEST RESULTS")
print("=" * 50)
print(f"\nPeriod: {results['date'].min().date()} to {results['date'].max().date()}")
print(f"Number of rebalances: {len(results)}")
print(f"Stocks held per period: {TOP_N}")

print(f"\n--- Strategy ---")
print(f"  Total return:       {(results['strategy_cumulative'].iloc[-1] - 1) * 100:+.2f}%")
print(f"  Annualized return:  {annualize_return(results['strategy_return']) * 100:+.2f}%")
print(f"  Annualized vol:     {annualize_vol(results['strategy_return']) * 100:.2f}%")
print(f"  Sharpe ratio:       {sharpe_ratio(results['strategy_return']):.2f}")
print(f"  Max drawdown:       {max_drawdown(results['strategy_cumulative']) * 100:.2f}%")

print(f"\n--- Benchmark (equal-weight all stocks) ---")
print(f"  Total return:       {(results['benchmark_cumulative'].iloc[-1] - 1) * 100:+.2f}%")
print(f"  Annualized return:  {annualize_return(results['benchmark_return']) * 100:+.2f}%")
print(f"  Annualized vol:     {annualize_vol(results['benchmark_return']) * 100:.2f}%")
print(f"  Sharpe ratio:       {sharpe_ratio(results['benchmark_return']):.2f}")
print(f"  Max drawdown:       {max_drawdown(results['benchmark_cumulative']) * 100:.2f}%")

print(f"\n--- Active performance ---")
alpha = annualize_return(results['strategy_return']) - annualize_return(results['benchmark_return'])
print(f"  Annualized alpha:   {alpha * 100:+.2f}%")
print(f"  Win rate vs bench:  {(results['strategy_return'] > results['benchmark_return']).mean():.1%}")

# === The Big Chart ===
fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})

# Top: cumulative returns
axes[0].plot(results['date'], results['strategy_cumulative'], label='ML Strategy (Top 20)',
             linewidth=2.5, color='#2ecc71')
axes[0].plot(results['date'], results['benchmark_cumulative'], label='Benchmark (Equal-Weight)',
             linewidth=2, color='#3498db', linestyle='--')
axes[0].set_title("ML Stock Strategy vs Benchmark — 2024 Backtest",
                  fontsize=15, fontweight='bold')
axes[0].set_ylabel("Cumulative Return (Start = 1.0)")
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].axhline(1, color='black', linewidth=0.5)

# Bottom: monthly difference
diff = (results['strategy_return'] - results['benchmark_return']) * 100
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in diff]
axes[1].bar(results['date'], diff, color=colors, width=20)
axes[1].set_title("Monthly Active Return (Strategy - Benchmark)", fontsize=11)
axes[1].set_ylabel("Excess Return (%)")
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/backtest_chart.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Chart saved to data/backtest_chart.png")

# Save results
results.to_parquet("data/backtest_results.parquet", index=False)