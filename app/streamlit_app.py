import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="S&P 500 ML Equity Research",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# DATA LOADING (cached so it's fast)
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"

@st.cache_data
def load_predictions():
    return pd.read_parquet(DATA_DIR / "test_predictions_xgboost.parquet")

@st.cache_data
def load_backtest():
    return pd.read_parquet(DATA_DIR / "backtest_results.parquet")

@st.cache_data
def load_prices():
    return pd.read_parquet(DATA_DIR / "sp500_prices.parquet")

@st.cache_data
def load_features():
    return pd.read_parquet(DATA_DIR / "features_with_target.parquet")

@st.cache_data
def load_tickers():
    df = pd.read_csv(DATA_DIR / "sp500_tickers.csv")
    df['Symbol'] = df['Symbol'].str.replace('.', '-')
    return df

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📈 ML Equity Research")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate:",
    ["Strategy Overview", "Stock Screener", "Stock Deep Dive"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About:** XGBoost-based equity research tool that ranks S&P 500 stocks "
    "on predicted 1-month forward performance, using technical and momentum features."
)

# ============================================================
# PAGE 1 — STRATEGY OVERVIEW
# ============================================================
if page == "Strategy Overview":
    st.title("ML-Driven S&P 500 Stock Strategy")
    st.markdown(
        "An XGBoost model trained on technical features ranks the S&P 500 "
        "each month and selects the top 20 stocks by predicted forward return."
    )

    backtest = load_backtest()

    # === Key metrics ===
    strategy_total = (backtest['strategy_cumulative'].iloc[-1] - 1) * 100
    benchmark_total = (backtest['benchmark_cumulative'].iloc[-1] - 1) * 100

    def annualize_return(returns):
        return ((1 + returns.mean()) ** 12 - 1) * 100

    def sharpe(returns):
        ann_ret = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        return ann_ret / ann_vol if ann_vol > 0 else 0

    def max_dd(cumulative):
        return ((cumulative - cumulative.cummax()) / cumulative.cummax()).min() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Strategy Return",
        f"{strategy_total:+.1f}%",
        f"{strategy_total - benchmark_total:+.1f}% vs benchmark"
    )
    col2.metric(
        "Annualized Return",
        f"{annualize_return(backtest['strategy_return']):+.1f}%"
    )
    col3.metric(
        "Sharpe Ratio",
        f"{sharpe(backtest['strategy_return']):.2f}"
    )
    col4.metric(
        "Max Drawdown",
        f"{max_dd(backtest['strategy_cumulative']):.1f}%"
    )

    st.markdown("---")

    # === Main chart ===
    st.subheader("Cumulative Returns: Strategy vs Benchmark")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(backtest['date'], backtest['strategy_cumulative'],
            label='ML Strategy (Top 20)', linewidth=2.5, color='#2ecc71')
    ax.plot(backtest['date'], backtest['benchmark_cumulative'],
            label='Benchmark (Equal-Weight)', linewidth=2,
            color='#3498db', linestyle='--')
    ax.set_ylabel("Cumulative Return (Start = 1.0)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.axhline(1, color='black', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # === Monthly excess returns ===
    st.subheader("Monthly Excess Returns")
    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    diff = (backtest['strategy_return'] - backtest['benchmark_return']) * 100
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in diff]
    ax2.bar(backtest['date'], diff, color=colors, width=20)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel("Excess Return (%)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

    # === Methodology ===
    st.markdown("---")
    st.subheader("Methodology")
    st.markdown("""
    **Data:** 5 years of daily OHLCV data for ~480 S&P 500 constituents (Yahoo Finance).

    **Features (13 total):** Multi-horizon returns, moving averages, volatility,
    RSI, Bollinger Band position, volume ratios.

    **Target:** Binary indicator — is this stock in the top 20% of forward 21-day
    returns for its date? *Cross-sectional* ranking prevents regime bias.

    **Model:** XGBoost classifier trained on 2020–2023, tested on 2024.

    **Strategy:** Each month, rank S&P 500 stocks by predicted probability of
    being a top performer; equal-weight the top 20.

    **Important caveats:** Backtest excludes transaction costs and is run on a
    single year of out-of-sample data. The strategy concentrates risk in 20 names
    and would have benefited from 2024's momentum-favorable environment.
    Walk-forward testing across multiple regimes would be needed before
    treating this as a deployable strategy.
    """)

# ============================================================
# PAGE 2 — STOCK SCREENER
# ============================================================
elif page == "Stock Screener":
    st.title("Stock Screener")
    st.markdown("Filter S&P 500 stocks by sector and view the model's current rankings.")

    preds = load_predictions()
    tickers_df = load_tickers()

    # Use the most recent prediction date
    latest_date = preds['date'].max()
    st.caption(f"Showing predictions for: **{pd.to_datetime(latest_date).date()}**")

    latest = preds[preds['date'] == latest_date].copy()
    latest = latest.merge(
        tickers_df[['Symbol', 'Security', 'GICS Sector']],
        left_on='ticker', right_on='Symbol', how='left'
    )

    # === Sidebar filters ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    sectors = sorted(latest['GICS Sector'].dropna().unique())
    selected_sectors = st.sidebar.multiselect(
        "Sectors", sectors, default=sectors
    )
    top_n = st.sidebar.slider("Show top N", 10, 100, 25)

    # === Filter and rank ===
    filtered = latest[latest['GICS Sector'].isin(selected_sectors)].copy()
    filtered = filtered.sort_values('prediction_proba', ascending=False).head(top_n)
    filtered['Rank'] = range(1, len(filtered) + 1)

    # === Display table ===
    display = filtered[[
        'Rank', 'ticker', 'Security', 'GICS Sector',
        'prediction_proba', 'rsi_14', 'volatility_21d',
        'return_252d', 'price_vs_sma200'
    ]].rename(columns={
        'ticker': 'Ticker',
        'Security': 'Company',
        'GICS Sector': 'Sector',
        'prediction_proba': 'Model Score',
        'rsi_14': 'RSI',
        'volatility_21d': 'Vol (21d)',
        'return_252d': '12mo Return',
        'price_vs_sma200': 'Price/SMA200'
    })

    display['Model Score'] = display['Model Score'].apply(lambda x: f"{x:.3f}")
    display['RSI'] = display['RSI'].apply(lambda x: f"{x:.1f}")
    display['Vol (21d)'] = display['Vol (21d)'].apply(lambda x: f"{x:.1%}")
    display['12mo Return'] = display['12mo Return'].apply(lambda x: f"{x:+.1%}")
    display['Price/SMA200'] = display['Price/SMA200'].apply(lambda x: f"{x:+.1%}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"""
    **Model Score** = predicted probability that this stock will be in the
    top 20% of S&P 500 returns over the next 21 trading days.

    Showing **{len(filtered)}** stocks across **{len(selected_sectors)}** sectors.
    """)

# ============================================================
# PAGE 3 — STOCK DEEP DIVE
# ============================================================
elif page == "Stock Deep Dive":
    st.title("Stock Deep Dive")

    prices = load_prices()
    preds = load_predictions()
    tickers_df = load_tickers()

    available_tickers = sorted(preds['ticker'].unique())
    ticker = st.selectbox("Select a ticker:", available_tickers, index=0)

    # Stock info
    info = tickers_df[tickers_df['Symbol'] == ticker]
    if len(info) > 0:
        info = info.iloc[0]
        st.subheader(f"{ticker} — {info['Security']}")
        st.caption(f"Sector: {info['GICS Sector']}")

    # Latest prediction
    latest_pred = preds[preds['ticker'] == ticker].sort_values('date').iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Score", f"{latest_pred['prediction_proba']:.3f}")
    col2.metric("RSI (14d)", f"{latest_pred['rsi_14']:.1f}")
    col3.metric("12mo Return", f"{latest_pred['return_252d']:+.1%}")
    col4.metric("Volatility (21d)", f"{latest_pred['volatility_21d']:.1%}")

    # === Price chart with moving averages ===
    st.subheader("Price History & Moving Averages")
    stock_prices = prices[ticker].dropna(subset=['Close']).copy()
    stock_prices['SMA_50'] = stock_prices['Close'].rolling(50).mean()
    stock_prices['SMA_200'] = stock_prices['Close'].rolling(200).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(stock_prices.index, stock_prices['Close'],
            label='Close', linewidth=1.5, color='#2c3e50')
    ax.plot(stock_prices.index, stock_prices['SMA_50'],
            label='50-day MA', linewidth=1.2, color='#e67e22', alpha=0.8)
    ax.plot(stock_prices.index, stock_prices['SMA_200'],
            label='200-day MA', linewidth=1.2, color='#c0392b', alpha=0.8)
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # === Model score over time ===
    st.subheader("Model Score Over Test Period")
    ticker_history = preds[preds['ticker'] == ticker].sort_values('date')

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(ticker_history['date'], ticker_history['prediction_proba'],
             linewidth=1.8, color='#9b59b6')
    ax2.axhline(0.20, color='gray', linestyle='--', alpha=0.5,
                label='Base rate (20%)')
    ax2.set_ylabel("Predicted Probability")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

    st.caption(
        "The model score represents the predicted probability that this stock "
        "will be in the top quintile of S&P 500 returns over the next month. "
        "Scores above the gray line indicate above-average ranking."
    )