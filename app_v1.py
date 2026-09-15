import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import NSE_UNIVERSE, fetch_price_data
from optimizer import (
    annualize,
    portfolio_performance,
    random_portfolios,
    min_volatility,
    max_sharpe,
    efficient_frontier,
    portfolio_for_target_return,
    portfolio_for_target_risk,
    effective_number_of_assets,
)

st.set_page_config(page_title="NSE Portfolio Optimizer", layout="wide")

st.title("NSE Portfolio Optimizer")
st.caption(
    "Daily prices via yfinance (Yahoo Finance). Mean-variance / Markowitz "
    "optimization. Educational tool, not investment advice."
)

# --------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("1. Universe")
    default_picks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"]
    picked = st.multiselect(
        "NSE stocks",
        options=list(NSE_UNIVERSE.keys()),
        default=default_picks,
        format_func=lambda t: f"{t}  —  {NSE_UNIVERSE[t]}",
    )
    extra_raw = st.text_input(
        "Add other NSE tickers (comma-separated)",
        placeholder="e.g. PERSISTENT.NS, DIXON.NS",
        help="'.NS' is appended automatically if you leave it off.",
    )
    extra_tickers = []
    for t in extra_raw.split(","):
        t = t.strip().upper()
        if not t:
            continue
        if "." not in t:
            t += ".NS"
        extra_tickers.append(t)
    tickers = sorted(set(picked) | set(extra_tickers))
    if extra_tickers:
        st.caption(f"Added: {', '.join(extra_tickers)}")

    st.header("2. History window")
    years_back = st.slider("Years of daily history", 1, 10, 5)
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365 * years_back)

    st.header("3. Assumptions")
    rf_pct = st.number_input(
        "Risk-free rate, annual % — set this to the current "
        "India T-bill / G-Sec yield",
        min_value=0.0, max_value=15.0, value=6.5, step=0.1,
    )
    rf = rf_pct / 100

    allow_short = st.checkbox("Allow short selling", value=False)
    max_weight_pct = st.slider(
        "Max weight per stock, % (diversification cap)",
        min_value=5, max_value=100, value=100, step=5,
        help="Lower this to force the optimizer to spread allocation across "
             "more names instead of concentrating in the single best "
             "risk/return stock.",
    )
    max_weight = max_weight_pct / 100

    n_sim = st.slider("Monte Carlo portfolios to simulate", 500, 20000, 5000, step=500)

    st.header("4. Your target")
    target_mode = st.radio(
        "Optimize for",
        ["Max Sharpe ratio", "Min volatility", "Target return", "Target risk"],
    )
    target_value = None
    if target_mode == "Target return":
        target_value = st.slider("Target annual return, %", 0.0, 60.0, 15.0, step=0.5)
    elif target_mode == "Target risk":
        target_value = st.slider("Target annual volatility, %", 1.0, 60.0, 20.0, step=0.5)

    run = st.button("Run", type="primary", use_container_width=True)

# ------------------------------------------------------------------ main --
if not run:
    st.info("Set your universe and constraints in the sidebar, then hit **Run**.")
    st.stop()

if len(tickers) < 2:
    st.warning("Pick at least two stocks — a single asset has no frontier to trace.")
    st.stop()

if not allow_short and len(tickers) * max_weight < 1.0:
    st.error(
        f"With {len(tickers)} stocks and a {max_weight_pct}% cap each, the "
        f"maximum possible total allocation is {len(tickers) * max_weight_pct}%. "
        "Raise the cap or add more stocks."
    )
    st.stop()

with st.spinner("Pulling daily prices from Yahoo Finance..."):
    prices = fetch_price_data(tuple(tickers), start_date, end_date)

missing = [t for t in tickers if t not in prices.columns]
if missing:
    st.warning(
        f"No usable data for: {', '.join(missing)}. Continuing with the rest. "
        "Usually this means the ticker is wrong/delisted, or Yahoo Finance briefly "
        "rate-limited the request — try again in a minute if you're sure the ticker's right."
    )

if prices.shape[1] < 2:
    st.error("Fewer than two tickers returned usable data. Try a different selection or date range.")
    st.stop()

returns = prices.pct_change().dropna(how="any")
mean_returns, cov_matrix = annualize(returns)
n_assets = prices.shape[1]

bounds_note = (
    f"long-only, max {max_weight_pct}% per stock" if not allow_short
    else f"shorting allowed, |weight| ≤ {max_weight_pct}%"
)
st.caption(
    f"{n_assets} stocks · {prices.index[0].date()} to {prices.index[-1].date()} "
    f"({len(prices)} trading days) · {bounds_note}"
)

# Monte Carlo cloud (always long-only/uncapped — illustrative background)
rets_mc, vols_mc, sharpe_mc, _ = random_portfolios(n_sim, mean_returns, cov_matrix, rf, n_assets)

# Reference portfolios + true frontier (respect the cap / short setting above)
mv_weights = min_volatility(mean_returns, cov_matrix, max_weight, allow_short)
ms_weights = max_sharpe(mean_returns, cov_matrix, rf, max_weight, allow_short)
eq_weights = np.repeat(1 / n_assets, n_assets)
frontier = efficient_frontier(mean_returns, cov_matrix, 40, max_weight, allow_short)

# User-selected portfolio
if target_mode == "Max Sharpe ratio":
    sel_weights = ms_weights
elif target_mode == "Min volatility":
    sel_weights = mv_weights
elif target_mode == "Target return":
    sel_weights, ok = portfolio_for_target_return(mean_returns, cov_matrix, target_value / 100, max_weight, allow_short)
    if not ok:
        st.warning("That return target isn't reachable with this universe/cap — showing the closest feasible portfolio instead.")
else:  # Target risk
    sel_weights, ok = portfolio_for_target_risk(mean_returns, cov_matrix, target_value / 100, max_weight, allow_short)
    if not ok:
        st.warning("That risk target isn't reachable with this universe/cap — showing the closest feasible portfolio instead.")

sel_ret, sel_vol, sel_sharpe = portfolio_performance(sel_weights, mean_returns, cov_matrix, rf)
mv_ret, mv_vol, mv_sharpe = portfolio_performance(mv_weights, mean_returns, cov_matrix, rf)
ms_ret, ms_vol, ms_sharpe = portfolio_performance(ms_weights, mean_returns, cov_matrix, rf)
eq_ret, eq_vol, eq_sharpe = portfolio_performance(eq_weights, mean_returns, cov_matrix, rf)

# ---------------------------------------------------------- frontier plot --
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=vols_mc * 100, y=rets_mc * 100, mode="markers",
    marker=dict(size=4, color=sharpe_mc, colorscale="Viridis", showscale=True,
                colorbar=dict(title="Sharpe")),
    name="Simulated portfolios", opacity=0.45,
))
if frontier:
    f_ret = [f[0] * 100 for f in frontier]
    f_vol = [f[1] * 100 for f in frontier]
    fig.add_trace(go.Scatter(x=f_vol, y=f_ret, mode="lines",
                              line=dict(color="black", width=2), name="Efficient frontier"))
fig.add_trace(go.Scatter(x=[eq_vol * 100], y=[eq_ret * 100], mode="markers",
                          marker=dict(size=11, symbol="circle", color="gray"), name="Equal weight"))
fig.add_trace(go.Scatter(x=[mv_vol * 100], y=[mv_ret * 100], mode="markers",
                          marker=dict(size=14, symbol="diamond", color="#1f77b4"), name="Min volatility"))
fig.add_trace(go.Scatter(x=[ms_vol * 100], y=[ms_ret * 100], mode="markers",
                          marker=dict(size=15, symbol="star", color="gold",
                                      line=dict(width=1, color="black")), name="Max Sharpe"))
fig.add_trace(go.Scatter(x=[sel_vol * 100], y=[sel_ret * 100], mode="markers",
                          marker=dict(size=16, symbol="x", color="red", line=dict(width=2)),
                          name="Your portfolio"))
fig.update_layout(
    xaxis_title="Annualized volatility (%)", yaxis_title="Annualized return (%)",
    height=550, legend=dict(orientation="h", y=-0.15), margin=dict(t=20),
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- results --
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Your portfolio")
    m1, m2 = st.columns(2)
    m1.metric("Expected annual return", f"{sel_ret * 100:.2f}%")
    m2.metric("Annual volatility", f"{sel_vol * 100:.2f}%")
    m3, m4 = st.columns(2)
    m3.metric("Sharpe ratio", f"{sel_sharpe:.2f}")
    m4.metric("Effective # of holdings", f"{effective_number_of_assets(sel_weights):.1f} / {n_assets}")
    st.caption(
        f"Reference — equal weight: {eq_ret*100:.1f}% return, {eq_vol*100:.1f}% vol, "
        f"{eq_sharpe:.2f} Sharpe. Max Sharpe portfolio: {ms_sharpe:.2f} Sharpe."
    )

weights_df = pd.DataFrame({"Ticker": prices.columns, "Weight": sel_weights})
weights_df["Weight %"] = (weights_df["Weight"] * 100).round(2)
weights_df = weights_df.sort_values("Weight", ascending=False)

with col2:
    # Pie only shows meaningful slices — a stock the optimizer assigned ~0%
    # still appears in the table below, it's just not worth a pie slice.
    pie_df = weights_df[weights_df["Weight"] > 0.001]
    fig_pie = go.Figure(go.Pie(labels=pie_df["Ticker"], values=pie_df["Weight %"], hole=0.45))
    fig_pie.update_layout(height=330, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

st.caption(
    "Every stock that made it into this run — including ones the optimizer "
    "gave 0% to. If a ticker you added isn't listed here at all, it failed to "
    "download (see the warning above); if it's listed at 0.00%, it loaded fine "
    "but wasn't part of the optimal mix under your current settings."
)
st.dataframe(weights_df[["Ticker", "Weight %"]].set_index("Ticker"), use_container_width=True)
st.download_button(
    "Download weights as CSV",
    weights_df[["Ticker", "Weight %"]].to_csv(index=False),
    file_name="portfolio_weights.csv",
    mime="text/csv",
)

with st.expander("Correlation between selected stocks"):
    corr = returns.corr()
    fig_corr = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                     colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
    fig_corr.update_layout(height=450, margin=dict(t=10))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(
        "Lower / more negative correlations between holdings mean more genuine "
        "diversification benefit — two stocks that move together add little, "
        "regardless of the weight cap above."
    )
