# ------------------------------------------------------------
#  02_clean.py
#  Cleans raw financials and computes all derived metrics
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import os
from config import TICKER, PEERS

PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def compute_fcf(cashflow_df: pd.DataFrame) -> pd.Series:
    """
    Free Cash Flow = Operating Cash Flow - Capital Expenditure
    CapEx is stored as negative in yfinance cashflow, so we add it.
    """
    if cashflow_df.empty:
        return pd.Series(dtype=float)

    ocf = cashflow_df.get("Operating Cash Flow",
          cashflow_df.get("Total Cash From Operating Activities",
          pd.Series(0, index=cashflow_df.index)))

    capex = cashflow_df.get("Capital Expenditure",
            cashflow_df.get("Capital Expenditures",
            pd.Series(0, index=cashflow_df.index)))

    # CapEx in yfinance is already negative — adding gives FCF
    fcf = ocf + capex
    return fcf.rename("FCF")


def compute_key_ratios(info: dict, income_df: pd.DataFrame,
                       balance_df: pd.DataFrame) -> dict:
    """
    Computes or extracts key valuation and financial ratios.
    """
    ratios = {}

    #  From info dict (market data) 
    ratios["P/E (Trailing)"]   = info.get("trailingPE")
    ratios["P/E (Forward)"]    = info.get("forwardPE")
    ratios["P/B"]              = info.get("priceToBook")
    ratios["EV/EBITDA"]        = info.get("enterpriseToEbitda")
    ratios["Beta"]             = info.get("beta", 1.0)
    ratios["ROE (%)"]          = round(info.get("returnOnEquity", 0) * 100, 2) \
                                 if info.get("returnOnEquity") else None
    ratios["Debt/Equity"]      = info.get("debtToEquity")
    ratios["Dividend Yield (%)"] = round(info.get("dividendYield", 0) * 100, 2) \
                                   if info.get("dividendYield") else None
    ratios["Market Cap (Cr)"]  = round(info.get("marketCap", 0) / 1e7, 2) \
                                 if info.get("marketCap") else None

    #  Revenue CAGR (from income statement) 
    if not income_df.empty and "Total Revenue" in income_df.columns:
        rev = income_df["Total Revenue"].dropna().sort_index()
        if len(rev) >= 2:
            years = (rev.index[-1] - rev.index[0]).days / 365.25
            cagr = (rev.iloc[-1] / rev.iloc[0]) ** (1 / years) - 1
            ratios["Revenue CAGR (%)"] = round(cagr * 100, 2)

    #  Net Profit Margin (most recent year) 
    if not income_df.empty:
        rev_col   = next((c for c in ["Total Revenue", "Revenue"] if c in income_df.columns), None)
        prof_col  = next((c for c in ["Net Income", "Net Income Common Stockholders"]
                          if c in income_df.columns), None)
        if rev_col and prof_col:
            latest = income_df.sort_index().iloc[-1]
            if latest[rev_col] and latest[rev_col] != 0:
                ratios["Net Margin (%)"] = round(
                    latest[prof_col] / latest[rev_col] * 100, 2)

    return ratios


def clean_ticker_data(raw_data: dict) -> dict:
    """
    Takes raw fetched data dict for one ticker.
    Returns cleaned dict with derived metrics added.
    """
    cleaned = {}

    income_df  = raw_data.get("income_stmt", pd.DataFrame())
    balance_df = raw_data.get("balance_sheet", pd.DataFrame())
    cashflow_df = raw_data.get("cashflow", pd.DataFrame())
    price_df   = raw_data.get("price_history", pd.DataFrame())
    info       = raw_data.get("info", {})

    # Sort all by date ascending
    for df in [income_df, balance_df, cashflow_df]:
        if not df.empty:
            df.sort_index(inplace=True)

    # Compute FCF
    cleaned["fcf"] = compute_fcf(cashflow_df)

    # FCF CAGR
    fcf = cleaned["fcf"].dropna()
    if len(fcf) >= 2:
        years = (fcf.index[-1] - fcf.index[0]).days / 365.25
        if fcf.iloc[0] > 0:
            cleaned["fcf_cagr"] = (fcf.iloc[-1] / fcf.iloc[0]) ** (1 / years) - 1
        else:
            cleaned["fcf_cagr"] = None
    else:
        cleaned["fcf_cagr"] = None

    # Key ratios
    cleaned["ratios"] = compute_key_ratios(info, income_df, balance_df)

    # Pass through cleaned statements and price
    cleaned["income_stmt"]  = income_df
    cleaned["balance_sheet"] = balance_df
    cleaned["cashflow"]     = cashflow_df
    cleaned["price_history"] = price_df
    cleaned["info"]         = info

    return cleaned


def clean_all(raw_all: dict) -> dict:
    """Cleans data for all tickers and saves processed CSVs."""
    cleaned_all = {}

    for ticker_symbol, raw_data in raw_all.items():
        print(f" Cleaning {ticker_symbol}...")
        cleaned = clean_ticker_data(raw_data)
        cleaned_all[ticker_symbol] = cleaned

        # Save processed
        prefix = ticker_symbol.replace(".", "_")
        if not cleaned["fcf"].empty:
            cleaned["fcf"].to_csv(
                os.path.join(PROCESSED_DIR, f"{prefix}_fcf.csv"))
        if cleaned["ratios"]:
            pd.Series(cleaned["ratios"]).to_csv(
                os.path.join(PROCESSED_DIR, f"{prefix}_ratios.csv"))

    print(f"\n Cleaned data saved to {PROCESSED_DIR}")
    return cleaned_all


if __name__ == "__main__":
    # For testing standalone: reload raw data from CSVs
    print("Run main.py to execute the full pipeline.")
