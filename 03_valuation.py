# ------------------------------------------------------------
#  03_valuation.py
#  DCF model + Relative Valuation + Sensitivity Analysis
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from config import (
    TICKER, PEERS, PROJECTION_YEARS, TERMINAL_GROWTH_RATE,
    RISK_FREE_RATE, EQUITY_RISK_PREMIUM,
    WACC_RANGE, GROWTH_RANGE, OUTPUT_DIR
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "charts"), exist_ok=True)


# ─────────────────────────────────────────
#  1. WACC CALCULATION
# ─────────────────────────────────────────

def compute_wacc(info: dict) -> float:
    """
    WACC = (E/V) * Re + (D/V) * Rd * (1 - Tax Rate)
    Re (Cost of Equity) via CAPM: Rf + Beta * ERP
    """
    beta        = info.get("beta", 1.0) or 1.0
    market_cap  = info.get("marketCap", 0) or 0
    total_debt  = info.get("totalDebt", 0) or 0

    # Cost of equity (CAPM)
    cost_of_equity = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM

    if market_cap + total_debt == 0:
        return cost_of_equity   # fallback: all-equity

    # Weights
    E = market_cap
    D = total_debt
    V = E + D
    weight_equity = E / V
    weight_debt   = D / V

    # Cost of debt (approximate: use 8% if unavailable)
    cost_of_debt  = 0.08
    tax_rate      = 0.25   # India corporate tax (approx)

    wacc = (weight_equity * cost_of_equity +
            weight_debt   * cost_of_debt * (1 - tax_rate))

    return round(wacc, 4)


# ─────────────────────────────────────────
#  2. DCF VALUATION
# ─────────────────────────────────────────

def run_dcf(cleaned_data: dict, wacc: float = None,
            terminal_growth: float = None) -> dict:
    """
    Projects FCF for PROJECTION_YEARS, computes Terminal Value,
    discounts everything back to get intrinsic value per share.
    """
    info      = cleaned_data["info"]
    fcf       = cleaned_data["fcf"].dropna()
    fcf_cagr  = cleaned_data.get("fcf_cagr")

    # Fallbacks
    if wacc is None:
        wacc = compute_wacc(info)
    if terminal_growth is None:
        terminal_growth = TERMINAL_GROWTH_RATE
    if fcf_cagr is None or np.isnan(fcf_cagr):
        fcf_cagr = 0.08   # conservative default: 8%

    # Cap unrealistic growth rates
    growth_rate = min(fcf_cagr, 0.25)

    # Base FCF = most recent year
    base_fcf = fcf.iloc[-1] if not fcf.empty else 0

    if base_fcf <= 0:
        return {"error": "Negative or zero FCF — DCF not applicable",
                "wacc": wacc}

    # Project FCFs
    projected_fcf = []
    for year in range(1, PROJECTION_YEARS + 1):
        projected_fcf.append(base_fcf * (1 + growth_rate) ** year)

    # Terminal Value (Gordon Growth Model)
    terminal_value = (projected_fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)

    # Discount to PV
    pv_fcfs = [cf / (1 + wacc) ** (i + 1)
               for i, cf in enumerate(projected_fcf)]
    pv_terminal = terminal_value / (1 + wacc) ** PROJECTION_YEARS

    # Enterprise Value
    enterprise_value = sum(pv_fcfs) + pv_terminal

    # Equity Value = EV - Debt + Cash
    total_debt  = info.get("totalDebt", 0) or 0
    total_cash  = info.get("totalCash", 0) or 0
    equity_value = enterprise_value - total_debt + total_cash

    # Per share
    shares = info.get("sharesOutstanding", 1) or 1
    intrinsic_value = equity_value / shares

    return {
        "wacc":              round(wacc * 100, 2),
        "growth_rate_used":  round(growth_rate * 100, 2),
        "terminal_growth":   round(terminal_growth * 100, 2),
        "base_fcf_cr":       round(base_fcf / 1e7, 2),
        "projected_fcf":     [round(x / 1e7, 2) for x in projected_fcf],
        "pv_fcfs_cr":        [round(x / 1e7, 2) for x in pv_fcfs],
        "pv_terminal_cr":    round(pv_terminal / 1e7, 2),
        "enterprise_value_cr": round(enterprise_value / 1e7, 2),
        "equity_value_cr":   round(equity_value / 1e7, 2),
        "intrinsic_value":   round(intrinsic_value, 2),
        "terminal_value_pct": round(pv_terminal / enterprise_value * 100, 1),
    }


# ─────────────────────────────────────────
#  3. SENSITIVITY ANALYSIS
# ─────────────────────────────────────────

def sensitivity_analysis(cleaned_data: dict) -> pd.DataFrame:
    """
    Creates a sensitivity table: WACC (rows) x Terminal Growth (cols)
    showing intrinsic value per share.
    """
    base_wacc = compute_wacc(cleaned_data["info"])
    rows = {}

    for dw in WACC_RANGE:
        w = base_wacc + dw
        row = {}
        for dg in GROWTH_RANGE:
            g = TERMINAL_GROWTH_RATE + dg
            if w <= g:
                row[f"g={g*100:.1f}%"] = "N/A"
                continue
            result = run_dcf(cleaned_data, wacc=w, terminal_growth=g)
            row[f"g={g*100:.1f}%"] = result.get("intrinsic_value", "N/A")
        rows[f"WACC={w*100:.1f}%"] = row

    return pd.DataFrame(rows).T


# ─────────────────────────────────────────
#  4. RELATIVE VALUATION
# ─────────────────────────────────────────

def relative_valuation(cleaned_all: dict) -> pd.DataFrame:
    """
    Compares key multiples across the main ticker and peers.
    """
    records = []
    all_tickers = [TICKER] + PEERS

    for ticker_symbol in all_tickers:
        if ticker_symbol not in cleaned_all:
            continue
        info    = cleaned_all[ticker_symbol]["info"]
        ratios  = cleaned_all[ticker_symbol]["ratios"]
        records.append({
            "Ticker":       ticker_symbol,
            "Company":      info.get("longName", ticker_symbol)[:20],
            "Price":        info.get("currentPrice"),
            "P/E":          ratios.get("P/E (Trailing)"),
            "P/B":          ratios.get("P/B"),
            "EV/EBITDA":    ratios.get("EV/EBITDA"),
            "ROE (%)":      ratios.get("ROE (%)"),
            "Debt/Equity":  ratios.get("Debt/Equity"),
            "Net Margin (%)": ratios.get("Net Margin (%)"),
        })

    df = pd.DataFrame(records).set_index("Ticker")
    return df


def relative_verdict(rel_df: pd.DataFrame) -> str:
    """
    Checks if main ticker's P/E and EV/EBITDA are above/below sector median.
    """
    if TICKER not in rel_df.index:
        return "Insufficient data for relative verdict."

    target = rel_df.loc[TICKER]
    peers_only = rel_df.drop(index=TICKER, errors="ignore")

    signals = []

    for metric in ["P/E", "EV/EBITDA", "P/B"]:
        if pd.notna(target.get(metric)) and len(peers_only[metric].dropna()) > 0:
            sector_median = peers_only[metric].median()
            val = target[metric]
            pct_diff = (val - sector_median) / sector_median * 100
            direction = "PREMIUM" if pct_diff > 0 else "DISCOUNT"
            signals.append(
                f"  {metric}: {val:.1f}x vs sector median {sector_median:.1f}x "
                f"→ {direction} of {abs(pct_diff):.1f}%"
            )

    return "\n".join(signals) if signals else "No comparable data available."


# ─────────────────────────────────────────
#  5. CHART — DCF WATERFALL
# ─────────────────────────────────────────

def plot_dcf_breakdown(dcf_result: dict, ticker_symbol: str):
    """Bar chart showing PV of projected FCFs + Terminal Value."""
    if "error" in dcf_result:
        return

    years    = [f"Y{i+1}" for i in range(PROJECTION_YEARS)]
    pv_fcfs  = dcf_result["pv_fcfs_cr"]
    tv       = dcf_result["pv_terminal_cr"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2563EB"] * PROJECTION_YEARS + ["#16A34A"]
    bars   = ax.bar(years + ["Terminal\nValue"],
                    pv_fcfs + [tv],
                    color=colors, edgecolor="white", linewidth=0.5)

    ax.set_title(f"DCF Value Breakdown — {ticker_symbol}",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("₹ Crore", fontsize=11)
    ax.set_xlabel("Component", fontsize=11)

    for bar, val in zip(bars, pv_fcfs + [tv]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(pv_fcfs + [tv]) * 0.01,
                f"₹{val:,.0f}Cr", ha="center", va="bottom", fontsize=9)

    patch1 = mpatches.Patch(color="#2563EB", label="PV of Projected FCF")
    patch2 = mpatches.Patch(color="#16A34A", label="PV of Terminal Value")
    ax.legend(handles=[patch1, patch2])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "charts",
                        f"{ticker_symbol.replace('.','_')}_dcf_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"   DCF chart saved: {path}")


def plot_sensitivity_heatmap(sens_df: pd.DataFrame, ticker_symbol: str):
    """Heatmap of intrinsic value across WACC x growth combinations."""
    numeric_df = sens_df.apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(numeric_df.values.astype(float), cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(numeric_df.columns)))
    ax.set_xticklabels(numeric_df.columns)
    ax.set_yticks(range(len(numeric_df.index)))
    ax.set_yticklabels(numeric_df.index)

    for i in range(len(numeric_df.index)):
        for j in range(len(numeric_df.columns)):
            val = numeric_df.iloc[i, j]
            ax.text(j, i, f"₹{val:,.0f}" if not np.isnan(val) else "N/A",
                    ha="center", va="center", fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Intrinsic Value (₹)")
    ax.set_title(f"Sensitivity: WACC vs Terminal Growth — {ticker_symbol}",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "charts",
                        f"{ticker_symbol.replace('.','_')}_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"   Sensitivity heatmap saved: {path}")


# ─────────────────────────────────────────
#  6. MASTER VALUATION FUNCTION
# ─────────────────────────────────────────

def run_full_valuation(cleaned_all: dict) -> dict:
    """
    Runs all valuation modules. Returns results dict.
    """
    print("\n Running Valuation Engine...")

    cleaned = cleaned_all[TICKER]
    info    = cleaned["info"]
    wacc    = compute_wacc(info)

    print(f"  WACC computed: {wacc*100:.2f}%")

    # DCF
    dcf_result = run_dcf(cleaned, wacc=wacc)
    print(f"  DCF Intrinsic Value: ₹{dcf_result.get('intrinsic_value', 'N/A')}")

    # Sensitivity
    sens_df = sensitivity_analysis(cleaned)

    # Relative
    rel_df = relative_valuation(cleaned_all)

    # Charts
    plot_dcf_breakdown(dcf_result, TICKER)
    plot_sensitivity_heatmap(sens_df, TICKER)

    return {
        "dcf":           dcf_result,
        "sensitivity":   sens_df,
        "relative":      rel_df,
        "rel_verdict":   relative_verdict(rel_df),
        "wacc":          wacc,
        "current_price": info.get("currentPrice"),
    }
