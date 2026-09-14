"""
Data loading for NSE (India) stocks via yfinance.

Yahoo Finance addresses NSE-listed stocks with a ".NS" suffix
(e.g. "RELIANCE.NS"). This module exposes a small default universe
of liquid large-caps for the picker, plus a cached fetch function —
the app also lets users type in any other ".NS" ticker.
"""

import pandas as pd
import streamlit as st
import yfinance as yf

# A starting universe, not a recommendation. Edit freely, or add any
# valid NSE ticker via the "custom tickers" box in the app. Corporate
# actions (mergers, delistings, renames) happen — if one of these ever
# fails to download, the app reports it and continues with the rest.
NSE_UNIVERSE = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "BAJFINANCE.NS": "Bajaj Finance",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "ASIANPAINT.NS": "Asian Paints",
    "NESTLEIND.NS": "Nestle India",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "JSWSTEEL.NS": "JSW Steel",
    "POWERGRID.NS": "Power Grid Corp",
    "NTPC.NS": "NTPC",
    "M&M.NS": "Mahindra & Mahindra",
    "TECHM.NS": "Tech Mahindra",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "ONGC.NS": "Oil & Natural Gas Corp",
    "COALINDIA.NS": "Coal India",
    "GRASIM.NS": "Grasim Industries",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "CIPLA.NS": "Cipla",
    "DIVISLAB.NS": "Divi's Laboratories",
    "BRITANNIA.NS": "Britannia Industries",
    "HINDALCO.NS": "Hindalco Industries",
    "TATACONSUM.NS": "Tata Consumer Products",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "SBILIFE.NS": "SBI Life Insurance",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "INDUSINDBK.NS": "IndusInd Bank",
    "EICHERMOT.NS": "Eicher Motors",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(tickers: tuple, start, end) -> pd.DataFrame:
    """
    Download daily adjusted-close prices for a tuple of NSE tickers.

    Returns a wide DataFrame (Date index, one column per ticker) trimmed
    to the common trading history across all requested tickers. Tickers
    that return no usable data are dropped rather than raising — the
    caller can diff the input list against `prices.columns` to see what
    was dropped.
    """
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,      # Close is already split/dividend-adjusted
        progress=False,
        group_by="column",     # top-level columns are price fields (Close, Open, ...)
        multi_level_index=True,  # guarantees raw["Close"] works for 1 or many tickers
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    prices = raw["Close"].copy()
    prices = prices.dropna(axis=1, how="all")   # tickers with zero data
    prices = prices.ffill().dropna(how="any")   # align calendars, trim to common history
    return prices
