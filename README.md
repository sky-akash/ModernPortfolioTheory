# NSE Portfolio Optimizer

A Streamlit dashboard that pulls daily price history for NSE (India)
stocks via `yfinance`, then runs a mean-variance (Markowitz) optimization
to plot the efficient frontier and build a portfolio to your spec:

- **Stocks** — pick from a default list of NSE large-caps, or type in any other `.NS` ticker
- **Return** — target an annual return; the app finds the lowest-risk portfolio that hits it
- **Risk** — target an annual volatility; the app finds the highest-return portfolio within it
- **Diversification** — a max-weight-per-stock cap, so the optimizer can't dump everything into one name

It also shows a Max Sharpe portfolio, a Min Volatility portfolio, an
equal-weight reference point, and a correlation heatmap across your
selected stocks.

## How it works

- Prices: `yfinance`, daily `Close` (auto-adjusted for splits/dividends), NSE tickers via the `.NS` suffix
- Returns/covariance: computed from daily returns, annualized (×252 / ×√252)
- The gray cloud in the chart is a Monte Carlo simulation of thousands of
  random long-only portfolios — it's illustrative context, not the answer itself
- The black line is the actual efficient frontier, found by minimizing
  variance at each target return via `scipy.optimize` (SLSQP)
- The diversification cap only applies to the optimized portfolios (frontier,
  Min Vol, Max Sharpe, your target) — not to the Monte Carlo cloud — so you
  can visually see the cost of the constraint as the frontier pulling inward
  from the cloud's edge

This is a teaching/analysis tool, not investment advice. Yahoo Finance data
can have gaps, delays, or the odd bad tick; always sanity-check before
relying on it for anything real.

## Run it locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Host it for free — Streamlit Community Cloud (recommended)

This is the standard free way to run a Streamlit app publicly, and it
deploys straight from GitHub — plain GitHub Pages won't work here since
it only serves static files and can't run a Python backend.

1. Push this folder to a **public** GitHub repo (a `requirements.txt` at
   the repo root, like this one, is required).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick the repo/branch, and set the main file path to `app.py`.
4. Click **Deploy**. You'll get a public URL like `https://<your-app>.streamlit.app`.
5. Every push to the branch redeploys automatically.

Free-tier notes: apps on a shared, limited CPU/RAM pool, and they sleep
after ~12 hours with no traffic (a visit wakes them back up in a few
seconds). Fine for a personal/coursework dashboard.

**Alternative:** [Hugging Face Spaces](https://huggingface.co/spaces) —
pick the "Streamlit" SDK when creating a Space. More generous free
hardware (2 vCPU / 16GB RAM) but sleeps after ~48h idle instead of 12h,
and it's a Space-native git repo rather than a direct GitHub link (you can
still mirror a GitHub repo into it).

## Notes and possible extensions

- The default ticker list is a convenience starting point, not vetted
  against today's index membership — corporate actions (mergers, delistings)
  happen. If a ticker fails to download, the app reports it and continues
  with the rest.
- Shorting is off by default (long-only), matching how most retail
  portfolios actually work — toggle it on in the sidebar if you want it.
- The risk-free rate is a manual input — set it to whatever the current
  Indian T-bill/G-Sec yield is; the app doesn't fetch it automatically.
- Natural next steps if you want to extend this: a historical backtest of
  the optimized weights vs. a Nifty 50 benchmark (`^NSEI` in yfinance),
  sector-level diversification constraints instead of (or alongside) the
  per-stock cap, or caching results to disk so repeated Streamlit Cloud
  cold-starts don't re-hit Yahoo Finance as often.
