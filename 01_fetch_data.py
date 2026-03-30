# ------------------------------------------------------------
#  01_fetch_data.py
#  Pulls all financial data from yfinance automatically
# ------------------------------------------------------------

import yfinance as yf
import pandas as pd
import os
import json
from config import (
    TICKER, PEERS, PRICE_HISTORY_PERIOD,
    COMPANY_NAME, OUTPUT_DIR
)

RAW_DIR = "data/raw/"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_financials(ticker_symbol: str) -> dict:
    """
    Fetches income statement, balance sheet, cash flow,
    key info, and price history for a given ticker.
    Returns a dict of DataFrames.
    """
    print(f"\n Fetching data for: {ticker_symbol}")
    t = yf.Ticker(ticker_symbol)

    data = {}

    #  Financial Statements 
    try:
        data["income_stmt"] = t.financials.T          # rows = years
        print(f"   Income Statement ({len(data['income_stmt'])} periods)")
    except Exception as e:
        print(f"   Income Statement failed: {e}")
        data["income_stmt"] = pd.DataFrame()

    try:
        data["balance_sheet"] = t.balance_sheet.T
        print(f"   Balance Sheet ({len(data['balance_sheet'])} periods)")
    except Exception as e:
        print(f"   Balance Sheet failed: {e}")
        data["balance_sheet"] = pd.DataFrame()

    try:
        data["cashflow"] = t.cashflow.T
        print(f"   Cash Flow Statement ({len(data['cashflow'])} periods)")
    except Exception as e:
        print(f"   Cash Flow Statement failed: {e}")
        data["cashflow"] = pd.DataFrame()

    #  Key Info (P/E, Beta, Market Cap, etc.) 
    try:
        info = t.info
        data["info"] = {
            "longName":              info.get("longName", ticker_symbol),
            "sector":                info.get("sector", "N/A"),
            "industry":              info.get("industry", "N/A"),
            "marketCap":             info.get("marketCap", None),
            "currentPrice":          info.get("currentPrice", None),
            "trailingPE":            info.get("trailingPE", None),
            "forwardPE":             info.get("forwardPE", None),
            "priceToBook":           info.get("priceToBook", None),
            "beta":                  info.get("beta", 1.0),
            "sharesOutstanding":     info.get("sharesOutstanding", None),
            "totalDebt":             info.get("totalDebt", None),
            "totalCash":             info.get("totalCash", None),
            "enterpriseValue":       info.get("enterpriseValue", None),
            "enterpriseToEbitda":    info.get("enterpriseToEbitda", None),
            "dividendYield":         info.get("dividendYield", None),
            "returnOnEquity":        info.get("returnOnEquity", None),
            "debtToEquity":          info.get("debtToEquity", None),
            "revenueGrowth":         info.get("revenueGrowth", None),
            "52WeekHigh":            info.get("fiftyTwoWeekHigh", None),
            "52WeekLow":             info.get("fiftyTwoWeekLow", None),
        }
        print(f"   Company Info (sector: {data['info']['sector']})")
    except Exception as e:
        print(f"   Info fetch failed: {e}")
        data["info"] = {}

    #  Historical Price Data 
    try:
        hist = t.history(period=PRICE_HISTORY_PERIOD)
        data["price_history"] = hist[["Open", "High", "Low", "Close", "Volume"]]
        print(f"   Price History ({len(data['price_history'])} trading days)")
    except Exception as e:
        print(f"   Price History failed: {e}")
        data["price_history"] = pd.DataFrame()

    return data


def fetch_all() -> dict:
    """
    Fetches data for the main ticker and all peers.
    Saves raw CSVs to data/raw/.
    Returns combined dict.
    """
    all_data = {}

    # Fetch main ticker
    all_data[TICKER] = fetch_financials(TICKER)
    _save_raw(TICKER, all_data[TICKER])

    # Fetch peers
    for peer in PEERS:
        all_data[peer] = fetch_financials(peer)
        _save_raw(peer, all_data[peer])

    print(f"\n All data fetched. Raw files saved to {RAW_DIR}")
    return all_data


def _save_raw(ticker_symbol: str, data: dict):
    """Saves DataFrames as CSVs and info dict as JSON."""
    prefix = ticker_symbol.replace(".", "_")

    for key in ["income_stmt", "balance_sheet", "cashflow", "price_history"]:
        if key in data and not data[key].empty:
            path = os.path.join(RAW_DIR, f"{prefix}_{key}.csv")
            data[key].to_csv(path)

    if "info" in data and data["info"]:
        path = os.path.join(RAW_DIR, f"{prefix}_info.json")
        with open(path, "w") as f:
            json.dump(data["info"], f, indent=2, default=str)


if __name__ == "__main__":
    data = fetch_all()
    print(f"\n Current Price of {COMPANY_NAME}: "
          f"₹{data[TICKER]['info'].get('currentPrice', 'N/A')}")
