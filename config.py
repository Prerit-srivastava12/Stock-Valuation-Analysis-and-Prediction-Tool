# -------------------------------------------------------
#  STOCK VALUATION & PREDICTION TOOL
#  config.py — The ONLY file you need to change
# -------------------------------------------------------

#  TARGET COMPANY 
TICKER = "INFY.NS"           # NSE: add .NS suffix | BSE: add .BO suffix
                             # Examples: "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS"
COMPANY_NAME = "Infosys Ltd"

#  PEER COMPANIES for relative valuation (same sector) 
PEERS = ["TCS.NS", "WIPRO.NS", "HCLTECH.NS"]

#  DATA SETTINGS 
HISTORICAL_YEARS = 5         # Years of financial history to fetch
PRICE_HISTORY_PERIOD = "5y"  # yfinance period: 1y, 2y, 5y, 10y

#  DCF ASSUMPTIONS 
PROJECTION_YEARS = 5         # How many years to project FCF
TERMINAL_GROWTH_RATE = 0.04  # Long-run growth rate (4% = conservative)
RISK_FREE_RATE = 0.072       # Indian 10-yr G-Sec yield (approx 7.2%)
EQUITY_RISK_PREMIUM = 0.055  # India ERP (approx 5.5%)
#  Note: WACC is computed automatically from the above + beta from yfinance

#  SENSITIVITY ANALYSIS RANGES 
WACC_RANGE = [-0.01, 0, +0.01]          # ±1% around computed WACC
GROWTH_RANGE = [-0.005, 0, +0.005]      # ±0.5% around terminal growth

#  ML PREDICTION SETTINGS 
PREDICTION_HORIZONS = [63, 126, 252]    # Trading days: ~3m, 6m, 12m
TEST_SPLIT_RATIO = 0.2                  # 20% of data used for testing
RANDOM_STATE = 42

#  OUTPUT 
OUTPUT_DIR = "outputs/"
REPORT_FILENAME = f"{TICKER.replace('.', '_')}_valuation_report"
