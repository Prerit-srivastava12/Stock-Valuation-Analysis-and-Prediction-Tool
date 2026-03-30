# Stock Valuation & Prediction Tool

This project is an automated stock analysis tool that performs valuation and price prediction using Python.

You just change the company name once, and the entire pipeline runs — data collection, valuation, prediction, and report generation.

---

## What it does

- Fetches financial data using yfinance
- Cleans and processes the data
- Runs DCF valuation
- Compares the company with its peers
- Predicts future stock prices using ML models
- Generates a final report with results

---

## How to run

1. Install dependencies

pip install -r requirements.txt

2. Set the company in config.py

```python
TICKER = "INFY.NS"
COMPANY_NAME = "Infosys Ltd"
PEERS = ["TCS.NS", "WIPRO.NS", "HCLTECH.NS"]
```

3. Run the script
python main.py

---

## Project Structure

stock-valuation-project/

config.py          # only file to edit
main.py            # runs everything

01_fetch_data.py
02_clean.py
03_valuation.py
04_prediction.py
05_report.py

data/
outputs/

requirements.txt
README.md

---

## Valuation

**DCF Model**
Projects free cash flow for 5 years
Uses WACC (CAPM based discount rate)
Terminal value using Gordon Growth

**Relative Valuation**
Compares P/E, P/B, EV/EBITDA with peers
Checks if stock is over/undervalued vs sector

---

## ML Prediction

**Models used**
Linear Regression
Random Forest
Gradient Boosting

**Features:**
Moving averages
RSI
Returns
Volatility
Volume trends

**Evaluation:**
RMSE used to compare models

**Predictions:**
3 months
6 months
12 months

---

##Output

The tool generates:

Text report in outputs/
Charts (DCF, sensitivity, predictions)

Example:

DCF Value: ₹2105
Market Price: ₹1842
Margin of Safety: +12.5%
Conclusion: Undervalued

---

## Tech stack
Python
pandas, numpy
scikit-learn
yfinance

---

## Notes
Data is fetched using yfinance, so values may not always be perfectly accurate
This is for learning and research purposes, not investment advice

---

## Contributors

Prerit Srivastava  
Sumit Kumar Mahto

---

## Roadmap

- [x] Data pipeline using yfinance  
- [x] DCF valuation model + WACC  
- [x] Relative valuation (peer comparison)  
- [x] Sensitivity analysis  
- [x] ML price prediction (LR, RF, GBM)  

- [ ] Add ARIMA / Prophet models  
- [ ] Streamlit dashboard  
- [ ] Multi-stock comparison  
- [ ] Improve model evaluation (RMSE, backtesting)  

- [ ] LSTM model (optional)  
- [ ] Portfolio-level analysis  
